"""Prepare Decode GEMV data — static ring weight layout (N8-K-sequential)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tensor_io import load_golden_tensor, save_install_tensor

try:
    from .relayout_gemm import (
        BASE_HW_PARAMS,
        KV_HW_PARAMS,
        install_target_slices,
    )
    from .relayout_gemm_local import relayout_in0_N8M2N4
except ImportError:  # Standalone: python single_op_data/relayout_gemv.py
    from relayout_gemm import (  # type: ignore[no-redef]
        BASE_HW_PARAMS,
        KV_HW_PARAMS,
        install_target_slices,
    )
    from relayout_gemm_local import relayout_in0_N8M2N4  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# GEMV-specific ring orders (different from GEMM)
# ---------------------------------------------------------------------------
_GEMV_RING_28 = [
    0, 12, 13, 15, 17, 19, 21, 23, 25, 27,
    26, 10, 11, 9, 8, 24, 22, 20, 18, 16,
    14, 2, 4, 6, 7, 5, 3, 1,
]
_GEMV_RING_KV = [0, 3, 2, 1]  # 4-slice KV ring (same as GEMM KV ring)


def _build_ring_prev_map(ring_order: list[int]) -> dict[int, int]:
    """Build a map: slice → previous slice in the ring (backwards)."""
    n = len(ring_order)
    ring_next = {ring_order[i]: ring_order[(i + 1) % n] for i in range(n)}
    return {dst: src for src, dst in ring_next.items()}


def _source_sequence(slice_id: int, slice_num: int, ring_prev: dict[int, int]) -> list[int]:
    """Ring step → which slice's K-block to use."""
    src = slice_id
    seq = []
    for _ in range(slice_num):
        seq.append(src)
        src = ring_prev[src]
    return seq


def _relayout_n8_k_seq(matrix: np.ndarray) -> np.ndarray:
    """N8-K-sequential relayout (replaces N8K2N4K for GEMV).

    Input:  (K, N_slice)  fp16 matrix
    Output: 1-D linearized fp16 array: for each N8 group (8 cols),
            iterate all K rows sequentially.
    """
    mat = np.asarray(matrix, dtype=np.float16)
    k_dim, local_n = mat.shape
    if local_n % 8 != 0:
        raise ValueError(f"N_slice={local_n} must be divisible by 8")
    n_groups = local_n // 8
    out = np.empty(k_dim * local_n, dtype=np.float16)
    cursor = 0
    for n_group in range(n_groups):
        col_start = n_group * 8
        col_end = col_start + 8
        for k_idx in range(k_dim):
            out[cursor : cursor + 8] = mat[k_idx, col_start:col_end]
            cursor += 8
    return out


def _load_case_tensors(
    case_entry: dict[str, object], golden_dir: Path
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for raw_entry in case_entry["inputs"]:  # type: ignore[index]
        entry = dict(raw_entry)
        inputs[str(entry["port"])] = load_golden_tensor(golden_dir / str(entry["path"]))
    output_entry = dict(case_entry["output"])  # type: ignore[arg-type]
    return inputs, load_golden_tensor(golden_dir / str(output_entry["path"]))


def _split_weight_streams(linearized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split linearized weight into B/B' via 64-fp16 ping-pong chunks.

    Hardware reads B and B' alternately in chunks of 4×256-bit = 128 bytes
    = 64 fp16 values.  Even chunks → B, odd chunks → B'.
    """
    values = np.asarray(linearized, dtype=np.float16).reshape(-1)
    CHUNK = 64  # fp16 values per ping-pong slice
    if values.size % (2 * CHUNK) != 0:
        raise ValueError(
            f"weight payload must be multiple of {2 * CHUNK} fp16, got {values.size}"
        )
    b_parts: list[np.ndarray] = []
    bp_parts: list[np.ndarray] = []
    for start in range(0, values.size, 2 * CHUNK):
        b_parts.append(values[start : start + CHUNK])
        bp_parts.append(values[start + CHUNK : start + 2 * CHUNK])
    return np.concatenate(b_parts), np.concatenate(bp_parts)


def _write_gemv_slice(
    slice_dir: Path,
    activation: np.ndarray,
    weight_linearized: np.ndarray,
    output: np.ndarray,
) -> None:
    weight_b, weight_bp = _split_weight_streams(weight_linearized)
    save_install_tensor(slice_dir, "matrix_A_linearized_128bit.bin", activation)
    save_install_tensor(slice_dir, "matrix_B_linearized_128bit.bin", weight_b)
    save_install_tensor(slice_dir, "matrix_Bp_linearized_128bit.bin", weight_bp)
    save_install_tensor(slice_dir, "matrix_D_linearized_128bit.bin", output)


def write_gemv_ring_case(
    case_entry: dict[str, object],
    golden_dir: Path,
    install_dir: Path,
    config: dict[str, object],
    op_label: str = "",
) -> None:
    """Static-ring GEMV weight install: K-block remap + N8-K-sequential relayout.

    Unlike GEMM which uses ``reorder_in0_slice_by_ring`` + N8K2N4K, GEMV
    uses a different ring order and a simpler N8-K-sequential layout where
    K blocks are permuted so that sequential reads match ring-step sources.
    """
    inputs, output_tensor = _load_case_tensors(case_entry, golden_dir)
    weight = np.asarray(inputs["B"]).squeeze()
    activation = np.asarray(inputs["A"]).reshape(-1, order="F")
    output = np.asarray(output_tensor).reshape(-1, order="F")
    if weight.ndim != 2:
        raise ValueError(f"decode_gemv_ring weight must be 2D, got {weight.shape}")

    k_size, n_size = weight.shape

    # --- Select HW params and GEMV ring order ---
    kv_dim = int(config["num_key_value_heads"]) * int(config["head_dim"])  # 128
    is_kv_gemv = (n_size == kv_dim and n_size < k_size)

    if is_kv_gemv:
        hw_params = KV_HW_PARAMS
        ring_order = list(_GEMV_RING_KV)
        # KV padding
        kv_pad_k = int(config.get("kv_padding_b", 0))
        if kv_pad_k > k_size:
            weight_pad = np.zeros((kv_pad_k, n_size), dtype=weight.dtype)
            weight_pad[:k_size, :] = weight
            weight = weight_pad
            k_size = kv_pad_k
        kv_pad_a = int(config.get("kv_padding_a", 0))
        if kv_pad_a > 0:
            pad_total = kv_pad_a * hw_params["num_slices"]
            if pad_total > activation.size:
                act_pad = np.zeros(pad_total, dtype=activation.dtype)
                act_pad[:activation.size] = activation
                activation = act_pad
    else:
        hw_params = BASE_HW_PARAMS
        ring_order = list(_GEMV_RING_28)

    logical_slices = hw_params["num_slices"]  # 28 or 4

    if k_size % logical_slices or n_size % logical_slices:
        raise ValueError(
            f"ring GEMV K={k_size}, N={n_size} must divide {logical_slices} slices"
        )
    if activation.size != k_size:
        raise ValueError(
            f"ring GEMV activation size {activation.size} != K={k_size} (after kv_padding)"
        )
    if output.size != n_size:
        raise ValueError("ring GEMV output dimensions do not match the weight N")

    slice_k = k_size // logical_slices
    slice_n = n_size // logical_slices
    entry_id = op_label or str(case_entry.get("instance_id", case_entry.get("id", case_entry.get("name", ""))))
    op_dir = install_dir / entry_id

    # Build ring-prev map for static K-block remapping
    ring_prev = _build_ring_prev_map(ring_order)

    for logical_slice in range(logical_slices):
        k_start = logical_slice * slice_k
        n_start = logical_slice * slice_n

        # --- Weight slice: full K rows, this slice's N columns ---
        weight_slice = np.asarray(
            weight[:, n_start : n_start + slice_n], dtype=np.float16
        )  # (K, N_slice)

        # --- Static K-block remap ---
        seq = _source_sequence(logical_slice, logical_slices, ring_prev)
        remapped = np.empty_like(weight_slice)
        for new_idx, old_idx in enumerate(seq):
            src_start = old_idx * slice_k
            src_end = src_start + slice_k
            dst_start = new_idx * slice_k
            dst_end = dst_start + slice_k
            remapped[dst_start:dst_end, :] = weight_slice[src_start:src_end, :]

        # --- N8-K-sequential relayout ---
        weight_linearized = _relayout_n8_k_seq(remapped)

        # --- Write to physical slices ---
        for physical_slice in install_target_slices(hw_params, logical_slice):
            _write_gemv_slice(
                op_dir / f"slice{physical_slice:02d}",
                activation[k_start : k_start + slice_k],
                weight_linearized,
                output[n_start : n_start + slice_n],
            )


def write_gemv_local_case(
    case_entry: dict[str, object],
    golden_dir: Path,
    install_dir: Path,
    config: dict[str, object],
    op_label: str = "",
) -> None:
    inputs, output_tensor = _load_case_tensors(case_entry, golden_dir)
    weight = np.asarray(inputs["B"])
    activation = np.asarray(inputs["A"])
    output = np.asarray(output_tensor)
    heads = int(config["num_attention_heads"])
    slices_per_head = int(config["slice_per_head"])

    if weight.shape[2] != heads or activation.shape[1] != heads:
        raise ValueError(
            f"local GEMV head mismatch: weight={weight.shape}, activation={activation.shape}"
        )
    k_size = weight.shape[0]
    vector_length = weight.shape[1]
    if k_size % slices_per_head:
        raise ValueError(
            f"local GEMV K={k_size} must divide slices_per_head={slices_per_head}"
        )
    if output.shape[:3] != (vector_length, slices_per_head, heads):
        raise ValueError(
            "local GEMV output must contain one partial vector per head slice; "
            f"got {output.shape}"
        )

    slice_k = k_size // slices_per_head
    entry_id = op_label or str(case_entry.get("instance_id", case_entry.get("id", case_entry.get("name", ""))))
    op_dir = install_dir / entry_id
    for head in range(heads):
        for local_slice in range(slices_per_head):
            global_slice = head * slices_per_head + local_slice
            start = local_slice * slice_k
            end = start + slice_k
            weight_slice = weight[start:end, :, head]
            # Match the existing GEMM-local in0 path: present (N, Kslice)
            # to N8M2N4 before flattening.
            weight_linearized = relayout_in0_N8M2N4(weight_slice.T)
            activation_slice = activation[start:end, head, 0]
            output_slice = output[:, local_slice, head]
            _write_gemv_slice(
                op_dir / f"slice{global_slice:02d}",
                activation_slice,
                weight_linearized,
                output_slice,
            )


def write_gemv_case(
    case_entry: dict[str, object],
    golden_dir: Path,
    install_dir: Path,
    config: dict[str, object],
    op_label: str = "",
) -> None:
    policy = str(case_entry["slice_policy"])
    if policy == "gemv_ring":
        write_gemv_ring_case(case_entry, golden_dir, install_dir, config, op_label)
    elif policy == "gemv_local":
        write_gemv_local_case(case_entry, golden_dir, install_dir, config, op_label)
    else:
        raise ValueError(f"unsupported GEMV slice policy: {policy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relayout one Decode GEMV operator.")
    parser.add_argument("operator", choices=("decode_gemv_ring", "decode_gemv_local"))
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "python_golden_decode" / "manifest.json")
    parser.add_argument("--install-dir", type=Path, default=PROJECT_DIR / "single_op_data" / "install_decode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = next(
        (entry for entry in manifest["operators"] if entry["name"] == args.operator),
        None,
    )
    if selected is None:
        raise ValueError(f"operator {args.operator!r} is not present in {args.manifest}")
    write_gemv_case(selected, args.manifest.parent, args.install_dir, manifest["config"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
