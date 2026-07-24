"""Slice Decode vectors without changing the order inside each slice."""

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


def split_vector(values: np.ndarray, slice_count: int) -> list[np.ndarray]:
    vector = np.asarray(values).reshape(-1, order="F")
    if vector.size % slice_count:
        raise ValueError(
            f"vector length {vector.size} is not divisible by slice_count={slice_count}"
        )
    width = vector.size // slice_count
    return [vector[index * width : (index + 1) * width].copy() for index in range(slice_count)]


def split_head_vectors(
    values: np.ndarray,
    heads: int,
    slices_per_head: int,
) -> list[np.ndarray]:
    array = np.asarray(values)
    if array.shape[1] != heads:
        raise ValueError(f"expected head dimension {heads}, got shape {array.shape}")
    length = array.shape[0]
    if length % slices_per_head:
        raise ValueError(
            f"head vector length {length} is not divisible by {slices_per_head}"
        )
    width = length // slices_per_head
    chunks: list[np.ndarray] = []
    for head in range(heads):
        vector = array[:, head, ...].reshape(length, order="F")
        for local_slice in range(slices_per_head):
            start = local_slice * width
            chunks.append(vector[start : start + width].copy())
    return chunks


def split_head_reduction(values: np.ndarray, heads: int, slices_per_head: int) -> list[np.ndarray]:
    array = np.asarray(values)
    expected = (slices_per_head, heads)
    squeezed = array.reshape(array.shape[0], array.shape[1], order="F")
    if squeezed.shape != expected:
        raise ValueError(f"expected head reduction shape {expected}, got {squeezed.shape}")
    return [
        np.asarray([squeezed[local_slice, head]], dtype=squeezed.dtype)
        for head in range(heads)
        for local_slice in range(slices_per_head)
    ]


def split_one_value_per_slice(values: np.ndarray, slice_count: int) -> list[np.ndarray]:
    vector = np.asarray(values).reshape(-1, order="F")
    if vector.size != slice_count:
        raise ValueError(f"expected {slice_count} partial values, found {vector.size}")
    return [np.asarray([value], dtype=vector.dtype) for value in vector]


def split_head_full_vectors(
    values: np.ndarray,
    heads: int,
    slices_per_head: int,
) -> list[np.ndarray]:
    """Prefill Mode B: (N, heads) → each head's N-vector replicated to all
    slices_per_head slices.  No subdivision within a head."""
    array = np.asarray(values)
    if array.ndim == 3:
        # Squeeze trailing 1: (N, heads, 1) → (N, heads)
        array = array.reshape(array.shape[0], array.shape[1])
    if array.ndim != 2 or array.shape[1] != heads:
        raise ValueError(f"Mode B expected shape (N, {heads}), got {array.shape}")
    length = array.shape[0]
    chunks: list[np.ndarray] = []
    for head in range(heads):
        vector = array[:, head].reshape(length, order="F").copy()
        for _ in range(slices_per_head):
            chunks.append(vector.copy())
    return chunks


# ---------------------------------------------------------------------------
# KV element-wise linear (round-robin) physical slice mapping.
# Prefill relayout_rmsnorm uses  global_idx = head * slices_per_head + i,
# NOT the interleaved GEMV install_targets.
# ---------------------------------------------------------------------------
def _kv_elemwise_phys_slices(logical_idx: int, heads: int, slices_per_head: int) -> list[int]:
    """Linear round-robin: logical slice i → physical [i, i+4, i+8, ...]"""
    return [head * slices_per_head + logical_idx for head in range(heads)]


def replicate_scalar(values: np.ndarray, slice_count: int) -> list[np.ndarray]:
    vector = np.asarray(values).reshape(-1, order="F")
    if vector.size != 1:
        raise ValueError(f"expected a scalar tensor, found {vector.size} values")
    return [vector.copy() for _ in range(slice_count)]


def _load_case_tensors(
    case_entry: dict[str, object], golden_dir: Path
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for input_entry in case_entry["inputs"]:  # type: ignore[index]
        entry = dict(input_entry)
        inputs[str(entry["port"])] = load_golden_tensor(golden_dir / str(entry["path"]))
    output_entry = dict(case_entry["output"])  # type: ignore[arg-type]
    output = load_golden_tensor(golden_dir / str(output_entry["path"]))
    return inputs, output


def slice_passthrough_case(
    case_entry: dict[str, object],
    golden_dir: Path,
    config: dict[str, object],
    slice_count_override: int | None = None,
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    policy = str(case_entry["slice_policy"])
    slice_count = slice_count_override or int(config["used_slices"])
    heads = int(config["num_attention_heads"])
    slices_per_head = int(config["slice_per_head"])
    inputs, output = _load_case_tensors(case_entry, golden_dir)

    if policy == "summac_hidden":
        # K/V summac: 4-slice + kv_padding (pad 896→1024, split 4×256,
        # recompute summac per logical slice, replicate via install_targets).
        instance_id = str(case_entry.get("instance_id", ""))
        is_kv_summac = instance_id.startswith("kv_")
        if is_kv_summac:
            from decode_ops import summac_partials as _summac_partials
            logical_slices = slices_per_head  # 4 (same as prefill relayout_rmsnorm)
            kv_pad_a = int(config["kv_padding_a"])  # 256
            padded_total = kv_pad_a * logical_slices  # 1024

            a_flat = np.asarray(inputs["A"], dtype=np.float32).reshape(-1, order="F")
            if a_flat.size < padded_total:
                a_padded = np.zeros(padded_total, dtype=np.float32)
                a_padded[:a_flat.size] = a_flat
            else:
                a_padded = a_flat[:padded_total]

            partials = _summac_partials(a_padded, logical_slices)  # 4 values

            log_input = {"A": split_vector(a_padded, logical_slices)}
            log_output = [np.asarray([partials[i]], dtype=np.float32) for i in range(logical_slices)]

            input_chunks = {"A": [None] * slice_count}
            output_chunks = [None] * slice_count
            for lidx in range(logical_slices):
                for pidx in _kv_elemwise_phys_slices(lidx, heads, slices_per_head):
                    input_chunks["A"][pidx] = log_input["A"][lidx]
                    output_chunks[pidx] = log_output[lidx]
        else:
            input_chunks = {"A": split_vector(inputs["A"], slice_count)}
            output_chunks = split_one_value_per_slice(output, slice_count)
    elif policy == "head_reduction":
        # ── Attention head reduction: Mode B when output is (1, heads, 1) ──
        def _is_mode_b(tensor: np.ndarray) -> bool:
            arr = np.asarray(tensor)
            return (arr.ndim == 3 and arr.shape[1] == heads and arr.shape[2] == 1)
        if _is_mode_b(output):
            input_chunks = {"A": split_head_full_vectors(inputs["A"], heads, slices_per_head)}
            output_chunks = split_head_full_vectors(output, heads, slices_per_head)
        else:
            input_chunks = {
                "A": split_head_vectors(inputs["A"], heads, slices_per_head)
            }
            output_chunks = split_head_reduction(output, heads, slices_per_head)
    elif policy == "scalar_replicate":
        input_chunks = {
            port: replicate_scalar(tensor, slice_count) for port, tensor in inputs.items()
        }
        output_chunks = replicate_scalar(output, slice_count)
    elif policy == "hidden_elementwise":
        # ── Attention chain: Mode B — (N, heads) split by head,
        #     each head's full vector replicated to slice_per_head slices. ──
        # Detect by shape: second dimension == num_heads (e.g. (32,7,1) or (1,7,1))
        def _is_mode_b(tensor: np.ndarray) -> bool:
            arr = np.asarray(tensor)
            if arr.ndim == 3 and arr.shape[1] == heads and arr.shape[2] == 1:
                return True
            if arr.ndim == 2 and arr.shape[1] == heads:
                return True
            return False

        if any(_is_mode_b(t) for t in (list(inputs.values()) + [output])):
            def _mode_b(tensor: np.ndarray) -> list[np.ndarray]:
                flat = np.asarray(tensor).reshape(-1, order="F")
                if flat.size == 1:
                    return [tensor.copy() for _ in range(slice_count)]
                return split_head_full_vectors(tensor, heads, slices_per_head)
            input_chunks = {port: _mode_b(t) for port, t in inputs.items()}
            output_chunks = _mode_b(output)

        # K/V element-wise ops (kv_dim=128): split into 4 logical slices,
        # replicate to 28 physical slices via KV_HW_PARAMS.install_targets.
        # NO kv_padding — 128/4=32 is already integer.
        else:
            kv_dim = int(config["num_key_value_heads"]) * int(config["head_dim"])
            is_kv_elemwise = any(
                np.asarray(t).reshape(-1, order="F").size == kv_dim
                for t in (list(inputs.values()) + [output])
            )
            if is_kv_elemwise:
                logical_slices = slices_per_head  # 4 (same as prefill relayout_rmsnorm)
                def _split_kv_4(tensor: np.ndarray) -> list[np.ndarray]:
                    flat = np.asarray(tensor).reshape(-1, order="F")
                    if flat.size == 1:
                        return [flat.copy() for _ in range(logical_slices)]
                    return split_vector(tensor, logical_slices)
                logical_input_chunks = {port: _split_kv_4(t) for port, t in inputs.items()}
                logical_output_chunks = _split_kv_4(output)

                # Replicate to physical slices (linear round-robin, not GEMV interleaved)
                input_chunks = {}
                for port in inputs:
                    input_chunks[port] = [None] * slice_count
                output_chunks = [None] * slice_count
                for logical_idx in range(logical_slices):
                    for phys_idx in _kv_elemwise_phys_slices(logical_idx, heads, slices_per_head):
                        for port in inputs:
                            input_chunks[port][phys_idx] = logical_input_chunks[port][logical_idx]
                        output_chunks[phys_idx] = logical_output_chunks[logical_idx]
            else:
                def _split_or_replicate(tensor: np.ndarray, sc: int) -> list[np.ndarray]:
                    flat = np.asarray(tensor).reshape(-1, order="F")
                    if flat.size == 1:
                        return replicate_scalar(tensor, sc)
                    return split_vector(tensor, sc)
                input_chunks = {
                    port: _split_or_replicate(tensor, slice_count) for port, tensor in inputs.items()
                }
                output_chunks = _split_or_replicate(output, slice_count)
    elif policy == "remote_sum":
        # ── Attention remote_sum: Mode B when output is (N, heads, 1) ──
        instance_id = str(case_entry.get("instance_id", ""))
        def _is_mode_b(tensor: np.ndarray) -> bool:
            arr = np.asarray(tensor)
            return (arr.ndim == 3 and arr.shape[1] == heads and arr.shape[2] == 1)
        if _is_mode_b(output):
            input_chunks = {"A": split_head_full_vectors(inputs["A"], heads, slices_per_head)}
            output_chunks = split_head_full_vectors(output, heads, slices_per_head)
        # K/V remote_sum: 4 logical slices (slice_per_head), linear round-robin.
        elif instance_id.startswith("kv_"):
            logical_slices = slices_per_head  # 4 (same as prefill relayout_rmsnorm)
            # Golden produces slice_per_head=4 partial values; split one per logical slice.
            log_input = {"A": split_one_value_per_slice(inputs["A"], logical_slices)}
            log_output = [output.copy() for _ in range(logical_slices)]

            input_chunks = {"A": [None] * slice_count}
            output_chunks = [None] * slice_count
            for lidx in range(logical_slices):
                for pidx in _kv_elemwise_phys_slices(lidx, heads, slices_per_head):
                    input_chunks["A"][pidx] = log_input["A"][lidx]
                    output_chunks[pidx] = log_output[lidx]
        else:
            input_chunks = {"A": split_one_value_per_slice(inputs["A"], slice_count)}
            output_chunks = replicate_scalar(output, slice_count)
    else:
        raise ValueError(f"slice policy {policy!r} is not a passthrough policy")

    for port, chunks in input_chunks.items():
        if len(chunks) != slice_count:
            raise AssertionError(f"port {port} produced {len(chunks)} slices")
    if len(output_chunks) != slice_count:
        raise AssertionError(f"output produced {len(output_chunks)} slices")
    return input_chunks, output_chunks


def write_passthrough_case(
    case_entry: dict[str, object],
    golden_dir: Path,
    install_dir: Path,
    config: dict[str, object],
    op_label: str = "",
) -> None:
    input_chunks, output_chunks = slice_passthrough_case(case_entry, golden_dir, config)
    entry_id = op_label or str(case_entry.get("instance_id", case_entry.get("id", case_entry.get("name", ""))))
    instance_id = str(case_entry.get("instance_id", ""))
    op_dir = install_dir / entry_id
    slice_count = len(output_chunks)

    # --- KV RMS norm (kv_mul_scale, kv_mul_cast): 4-slice + kv_padding,
    #     same as K/V GEMV.  Re-read full tensors, pad 896→1024, split 4×256,
    #     replicate via install_targets to 28 physical slices. ---
    if instance_id in ("kv_mul_scale", "kv_mul_cast"):
        heads = int(config["num_attention_heads"])
        slices_per_head = int(config["slice_per_head"])
        logical_slices = slices_per_head  # 4 (same as prefill relayout_rmsnorm)
        padded_total = int(config["kv_padding_a"]) * logical_slices  # 256×4=1024

        # Re-load full golden tensors (before 28-slice split)
        inputs_full, output_full = _load_case_tensors(case_entry, golden_dir)

        def _needs_padding(t: np.ndarray) -> bool:
            """Only pad ports that are the full hidden vector (896-dim), not scalars."""
            return np.asarray(t).size > 1

        def _pad_to_1024(t: np.ndarray) -> np.ndarray:
            flat = np.asarray(t, dtype=t.dtype).reshape(-1, order="F")
            if flat.size >= padded_total:
                return flat[:padded_total]
            padded = np.zeros(padded_total, dtype=flat.dtype)
            padded[:flat.size] = flat
            return padded

        # Pad only the ports that need it; scalars (e.g. RMS scale B port) stay as-is
        inputs_padded: dict[str, np.ndarray] = {}
        for port, t in inputs_full.items():
            if _needs_padding(t):
                inputs_padded[port] = _pad_to_1024(t)
            else:
                inputs_padded[port] = np.asarray(t)

        if _needs_padding(output_full):
            output_padded = _pad_to_1024(output_full)
        else:
            output_padded = np.asarray(output_full)

        # Split padded ports into 4 logical slices; scalar ports are replicated
        log_input: dict[str, list[np.ndarray]] = {}
        for port, t in inputs_padded.items():
            if _needs_padding(t):
                log_input[port] = split_vector(t, logical_slices)
            else:
                log_input[port] = [np.asarray(t).copy() for _ in range(logical_slices)]

        if _needs_padding(output_padded):
            log_output = split_vector(output_padded, logical_slices)
        else:
            log_output = [np.asarray(output_padded).copy() for _ in range(logical_slices)]

        # Replicate to 28 physical slices (linear round-robin, not GEMV interleaved)
        input_chunks = {port: [None] * 28 for port in inputs_padded}
        output_chunks = [None] * 28
        for lidx in range(logical_slices):
            for pidx in _kv_elemwise_phys_slices(lidx, heads, slices_per_head):
                for port in inputs_padded:
                    input_chunks[port][pidx] = log_input[port][lidx]
                output_chunks[pidx] = log_output[lidx]
        slice_count = 28

    for slice_index in range(slice_count):
        slice_dir = op_dir / f"slice{slice_index:02d}"
        for port, chunks in input_chunks.items():
            save_install_tensor(
                slice_dir,
                f"matrix_{port}_linearized_128bit.bin",
                chunks[slice_index],
            )
        save_install_tensor(
            slice_dir,
            "matrix_D_linearized_128bit.bin",
            output_chunks[slice_index],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice one Decode vector operator.")
    parser.add_argument("operator")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "python_golden_decode" / "manifest.json")
    parser.add_argument("--install-dir", type=Path, default=PROJECT_DIR / "single_op_data" / "install_decode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    instances = manifest.get("instances", manifest.get("operators", []))
    selected = next(
        (entry for entry in instances
         if entry.get("instance_id") == args.operator or entry.get("name") == args.operator),
        None,
    )
    if selected is None:
        raise ValueError(f"operator {args.operator!r} is not present in {args.manifest}")
    write_passthrough_case(
        selected,
        golden_dir=args.manifest.parent,
        install_dir=args.install_dir,
        config=manifest["config"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

