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
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    policy = str(case_entry["slice_policy"])
    slice_count = int(config["used_slices"])
    heads = int(config["num_attention_heads"])
    slices_per_head = int(config["slice_per_head"])
    inputs, output = _load_case_tensors(case_entry, golden_dir)

    if policy == "summac_hidden":
        input_chunks = {"A": split_vector(inputs["A"], slice_count)}
        output_chunks = split_one_value_per_slice(output, slice_count)
    elif policy == "head_reduction":
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
        def _split_or_replicate(tensor: np.ndarray) -> list[np.ndarray]:
            flat = np.asarray(tensor).reshape(-1, order="F")
            if flat.size == 1:
                return replicate_scalar(tensor, slice_count)
            return split_vector(tensor, slice_count)
        input_chunks = {
            port: _split_or_replicate(tensor) for port, tensor in inputs.items()
        }
        output_chunks = _split_or_replicate(output)
    elif policy == "remote_sum":
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
    op_dir = install_dir / entry_id
    slice_count = int(config["used_slices"])
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

