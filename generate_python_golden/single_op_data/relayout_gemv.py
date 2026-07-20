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
        relayout_in0_N8K2N4K,
        reorder_in0_slice_by_ring,
    )
    from .relayout_gemm_local import relayout_in0_N8M2N4
except ImportError:  # Standalone: python single_op_data/relayout_gemv.py
    from relayout_gemm import (  # type: ignore[no-redef]
        BASE_HW_PARAMS,
        KV_HW_PARAMS,
        install_target_slices,
        relayout_in0_N8K2N4K,
        reorder_in0_slice_by_ring,
    )
    from relayout_gemm_local import relayout_in0_N8M2N4  # type: ignore[no-redef]


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
    """Split the linear GEMV weight payload across hardware B/B' streams."""

    values = np.asarray(linearized, dtype=np.float16).reshape(-1)
    if values.size % 2:
        raise ValueError(f"GEMV weight payload must have even length, got {values.size}")
    midpoint = values.size // 2
    return values[:midpoint].copy(), values[midpoint:].copy()


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

    # --- kv_padding for K/V: pad K-dim to multiple of logical_slices ---
    kv_pad_k = int(config.get("kv_padding_b", 0))
    if is_kv_gemv and kv_pad_k > k_size:
        weight_pad = np.zeros((kv_pad_k, n_size), dtype=weight.dtype)
        weight_pad[:k_size, :] = weight
        weight = weight_pad
        k_size = kv_pad_k

    kv_pad_a = int(config.get("kv_padding_a", 0))
    if is_kv_gemv and kv_pad_a > 0:
        pad_total = kv_pad_a * logical_slices
        if pad_total > activation.size:
            act_pad = np.zeros(pad_total, dtype=activation.dtype)
            act_pad[:activation.size] = activation
            activation = act_pad

    if k_size % logical_slices or n_size % logical_slices:
        raise ValueError(
            f"ring GEMV dimensions K={k_size}, N={n_size} must divide {logical_slices} slices"
        )
    if activation.size != k_size:
        raise ValueError(
            f"ring GEMV activation size {activation.size} != K={k_size} (after kv_padding)"
        )
    if output.size != n_size:
        raise ValueError("ring GEMV output dimensions do not match the weight N")

    slice_k = k_size // logical_slices
    slice_n = n_size // logical_slices
    physical_mapping = list(hw_params["physical_mapping"])
    ring_order = list(hw_params["ring_order"])
    entry_id = op_label or str(case_entry.get("instance_id", case_entry.get("id", case_entry.get("name", ""))))
    op_dir = install_dir / entry_id

    for logical_slice in range(logical_slices):
        k_start = logical_slice * slice_k
        n_start = logical_slice * slice_n
        weight_slice = weight[:, n_start : n_start + slice_n]
        weight_ring = reorder_in0_slice_by_ring(
            weight_slice,
            logical_slice,
            logical_slices,
            slice_k,
            ring_order,
        )
        weight_linearized = relayout_in0_N8K2N4K(
            weight_ring,
            k_size,
            slice_n,
            logical_slices,
        )

        # Write to ALL physical slices for this logical slice
        #   Q/FFN:   logical == physical (1 target each)
        #   K/V:     4 logical, each → 7 physical (via install_targets)
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
