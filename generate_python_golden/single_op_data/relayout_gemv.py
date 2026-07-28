"""Prepare Decode GEMV data while reusing the existing GEMM weight layouts."""

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
except ImportError:  # Standalone: python single_op_data/relayout_gemv.py
    from relayout_gemm import (  # type: ignore[no-redef]
        BASE_HW_PARAMS,
        KV_HW_PARAMS,
        install_target_slices,
    )


# ---------------------------------------------------------------------------
# GEMV-specific B-weight relayout:  n8 k2 k4 kr (K//(r*8)) (N//8)
#
#  ・N-slice first  → [K, slice_n]
#  ・transaction unit = n8k2 = 8 N × 2 K = 16 fp16
#  ・k4 = 4  inner K grouping  (k2 × k4 = k8 per ring transmission)
#  ・kr = ring_size  ring_order reorder dimension (28 for BASE, 4 for KV)
#  ・ring_order reorder on kr dimension  (rolled per logical slice)
#  ・remapping granularity: n8k2k4 = 64 fp16 contiguous
# ---------------------------------------------------------------------------

def relayout_gemv_B_k2n8kr(
    weight_slice: np.ndarray,
    logical_slice: int,
    slice_n: int,
    ring_order: np.ndarray,
) -> np.ndarray:
    """GEMV B-weight relayout:  n8 k2 k4 kr (K//(r*8)) (N//8).

    Parameters
    ----------
    weight_slice : np.ndarray
        N-sliced weight with shape ``(K, slice_n)``.
        K must be divisible by 8*r, slice_n by 8.
    logical_slice : int
        Logical slice index (0 … num_logical_slices-1).  Determines the
        roll of *ring_order* so that each slice sees its "own" K-block
        first, matching the order A data arrives through the ring.
    slice_n : int
        N elements per logical slice.
    ring_order : np.ndarray
        1-D integer array, the ring traversal order (len = ring_size = kr).
    """
    if weight_slice.ndim != 2:
        raise ValueError("weight_slice must be 2-D (K, slice_n)")
    K, sn = weight_slice.shape
    if sn != slice_n:
        raise ValueError(f"weight_slice columns {sn} != slice_n {slice_n}")

    ring_size = len(ring_order)
    K4 = 4                     # k4: inner K grouping  (k2 × k4 = k8 per ring tx)
    if K % (ring_size * K4 * 2) != 0:
        raise ValueError(f"K={K} must be divisible by {ring_size * K4 * 2}")
    if slice_n % 8 != 0:
        raise ValueError(f"slice_n={slice_n} must be divisible by 8")

    K_outer = K // (ring_size * K4 * 2)
    N_outer = slice_n // 8

    # ---- step 1: reshape  [K, slice_n] → [kr, K_inner, k4, k2, N_outer, n8] ----
    #  K = kr × stride + K_inner × 8 + k4 × 2 + k2
    #    where stride = K / ring_size  (e.g. 32 for BASE, 256 for KV)
    #  kr is axis 0 → ring_order reorder.
    #  K_inner = K_outer  is the "round" index within each kr block.
    reshaped = weight_slice.reshape(ring_size, K_outer, K4, 2, N_outer, 8).copy()

    # ---- step 2: ring reorder on kr  (per-slice roll) ----
    ring_arr = np.asarray(ring_order, dtype=np.intp)
    start_pos = int(np.where(ring_arr == logical_slice)[0][0])
    rolled_ring = np.roll(ring_arr, -start_pos)
    reordered = reshaped[rolled_ring, :, :, :, :, :]
    # shape:  [kr, K_inner, k4, k2, N_outer, n8]

    # ---- step 3: transpose to desired memory order ----
    #  Current axes:  0:kr  1:K_inner  2:k4  3:k2  4:N_outer  5:n8
    #  Desired  axes:  N_outer, K_inner, kr, k4, k2, n8
    #  → memory (innermost first):  n8(8), k2(2), k4(4), kr(r), K_inner, N_outer
    #  n8k2k4 = 64 fp16 contiguous → ring_order reorders these 64-fp16 chunks.
    transposed = reordered.transpose(4, 1, 0, 2, 3, 5)
    # shape:  [N//8, K_inner, kr, k4, 2, 8]

    # ---- step 4: flatten ----
    return transposed.reshape(-1)


def _relayout_n8k(weight_slice: np.ndarray) -> np.ndarray:
    """GEMV B-weight relayout: n8k — N grouped by 8, K consecutive within.

    GEMV is vector×matrix (M=1), so the N8M2N4/M8N2M4 matrix layouts are
    unnecessary.  The hardware reads B in n8k order: outer loop over N//8
    groups, inner loops over K then the 8 N elements.

    Parameters
    ----------
    weight_slice : np.ndarray
        Shape ``(K, N)``.  N must be a multiple of 8.
    """
    arr = np.asarray(weight_slice, dtype=np.float16)
    K, N = arr.shape
    if N % 8 != 0:
        raise ValueError(f"n8k requires N divisible by 8, got N={N}")
    result = np.empty(K * N, dtype=arr.dtype)
    idx = 0
    for n_group in range(0, N, 8):      # N//8 groups
        for k in range(K):               # all K elements
            for n_off in range(8):       # 8 consecutive N
                result[idx] = arr[k, n_group + n_off]
                idx += 1
    return result


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
    inputs, output_tensor = _load_case_tensors(case_entry, golden_dir)
    weight = np.asarray(inputs["B"]).squeeze()
    activation = np.asarray(inputs["A"]).reshape(-1, order="F")
    output = np.asarray(output_tensor).reshape(-1, order="F")
    if weight.ndim != 2:
        raise ValueError(f"decode_gemv_ring weight must be 2D, got {weight.shape}")

    k_size, n_size = weight.shape

    # --- Select HW params following prefill OP_SPECS convention ---
    # q_gen: BASE (28 slices)  |  k_gen/v_gen: KV (4 logical → 28 physical)
    kv_dim = int(config["num_key_value_heads"]) * int(config["head_dim"])  # 128
    is_kv_gemv = (n_size == kv_dim and n_size < k_size)
    hw_params = KV_HW_PARAMS if is_kv_gemv else BASE_HW_PARAMS
    logical_slices = hw_params["num_slices"]  # 4 for KV, 28 otherwise
    ring_order = np.array(hw_params["ring_order"], dtype=np.intp)
    ring_size = len(ring_order)  # 28 for BASE, 4 for KV

    # --- kv_padding for K/V: pad K-dim to multiple of ring_size*2 ---
    kv_pad_k = int(config.get("kv_padding_b", 0))
    if is_kv_gemv and kv_pad_k > k_size:
        weight_pad = np.zeros((kv_pad_k, n_size), dtype=weight.dtype)
        weight_pad[:k_size, :] = weight
        weight = weight_pad
        k_size = kv_pad_k

    # --- kv_padding for activation: pad A to kv_padding_a * ring_size ---
    kv_pad_a = int(config.get("kv_padding_a", 0))
    if is_kv_gemv and kv_pad_a > 0:
        pad_total = kv_pad_a * ring_size
        if pad_total > activation.size:
            act_pad = np.zeros(pad_total, dtype=activation.dtype)
            act_pad[: activation.size] = activation
            activation = act_pad

    if k_size % (ring_size * 8) != 0:
        raise ValueError(
            f"ring GEMV K={k_size} must be divisible by {ring_size * 8} (ring_size×8)"
        )
    if n_size % logical_slices != 0:
        raise ValueError(
            f"ring GEMV N={n_size} must divide {logical_slices} logical slices"
        )
    if activation.size != k_size:
        raise ValueError(
            f"ring GEMV activation size {activation.size} != K={k_size}"
        )
    if output.size != n_size:
        raise ValueError("ring GEMV output dimensions do not match the weight N")

    # K split by ring_size; N split by logical slices.
    slice_k = k_size // ring_size        # K per ring slice
    slice_n = n_size // logical_slices   # N per logical slice

    entry_id = op_label or str(case_entry.get("instance_id", case_entry.get("id", case_entry.get("name", ""))))
    op_dir = install_dir / entry_id

    for logical_slice in range(logical_slices):
        n_start = logical_slice * slice_n
        weight_slice = weight[:, n_start : n_start + slice_n]  # [K, slice_n]

        # GEMV-specific B relayout: k2 n8 kr (K//(r*2)) (N//8)
        #  ・N-slice first  → [K, slice_n]
        #  ・reshape to expose kr  → [K//(r*2), r, 2, N//8, 8]
        #  ・ring_order reorder on kr  (per-slice roll)
        #  ・transpose → [N//8, K//(r*2), r, 8, 2] → flatten
        weight_linearized = relayout_gemv_B_k2n8kr(
            weight_slice,
            logical_slice,
            slice_n,
            ring_order,
        )

        # A/B/D all use physical_mapping forward: logical_slice → install_target_slices
        # decides the physical slice; data chunk indexed directly by logical_slice.
        phys_slices = install_target_slices(hw_params, logical_slice)
        for physical_slice in phys_slices:
            a_chunk = activation[logical_slice * slice_k : (logical_slice + 1) * slice_k]
            _write_gemv_slice(
                op_dir / f"slice{physical_slice:02d}",
                a_chunk,
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
        # ---- B (weight) split by K across slices_per_head ----
        for local_slice in range(slices_per_head):
            start = local_slice * slice_k
            end = start + slice_k
            weight_slice = weight[start:end, :, head]          # (K, N)
            weight_linearized = _relayout_n8k(weight_slice)   # n8k: N//8 groups, K inner

            # ---- A (activation) split by K, same as B ----
            # GEMV is vector×matrix: M=1 always (decode single token).
            # Each slice computes partial QK^T on its K-dim chunk;
            # remote_sum (op23) aggregates the 4 partial results.
            # A shape: (slice_k,) = (32,) — matches program JSON [1,1,_HD_SLICE].
            act_slice = activation[start:end, head, 0]        # (slice_k,)
            act_linearized = np.asarray(act_slice, dtype=np.float16).reshape(-1)

            # ---- D (output) split by local_slice ----
            output_slice = output[:, local_slice, head]

            # ---- Physical mapping: logical → physical via BASE_HW_PARAMS ----
            logical_slice = head * slices_per_head + local_slice
            phys_slices = install_target_slices(BASE_HW_PARAMS, logical_slice)
            for physical_slice in phys_slices:
                slice_dir = op_dir / f"slice{physical_slice:02d}"
                slice_dir.mkdir(parents=True, exist_ok=True)

                weight_b, weight_bp = _split_weight_streams(weight_linearized)
                save_install_tensor(slice_dir, "matrix_A_linearized_128bit.bin", act_linearized)
                save_install_tensor(slice_dir, "matrix_B_linearized_128bit.bin", weight_b)
                save_install_tensor(slice_dir, "matrix_Bp_linearized_128bit.bin", weight_bp)
                save_install_tensor(slice_dir, "matrix_D_linearized_128bit.bin", output_slice)


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
