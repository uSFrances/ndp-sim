"""Shared tensor I/O helpers for Prefill/Decode golden data.

Golden tensors use the existing project convention: metadata is encoded in the
filename and multidimensional arrays are flattened in Fortran order.  Hardware
install files are already linearized, so they are written in their existing
one-dimensional order and accompanied by 128-bit and decimal text views.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TAG_TO_DTYPE = {
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
    "i32": np.dtype("<i4"),
}
DTYPE_TO_TAG = {dtype: tag for tag, dtype in TAG_TO_DTYPE.items()}

TENSOR_FILE_RE = re.compile(
    r"^(?P<name>.+)_shape(?P<shape>[0-9x]+)_dtype_(?P<dtype>f16|f32|i32)\.bin$"
)


@dataclass(frozen=True)
class TensorFileInfo:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


def canonical_dtype(dtype: np.dtype | type) -> np.dtype:
    value = np.dtype(dtype).newbyteorder("<")
    if value not in DTYPE_TO_TAG:
        raise ValueError(f"unsupported tensor dtype: {np.dtype(dtype)}")
    return value


def dtype_tag(dtype: np.dtype | type) -> str:
    return DTYPE_TO_TAG[canonical_dtype(dtype)]


def tensor_filename(name: str, tensor: np.ndarray) -> str:
    if not name or name.endswith(".bin"):
        raise ValueError(f"tensor name must be a non-empty stem, got: {name!r}")
    shape = "x".join(str(int(dim)) for dim in tensor.shape)
    return f"{name}_shape{shape}_dtype_{dtype_tag(tensor.dtype)}.bin"


def parse_tensor_filename(path: str | Path) -> TensorFileInfo:
    filename = Path(path).name
    match = TENSOR_FILE_RE.match(filename)
    if match is None:
        raise ValueError(f"unrecognized tensor filename: {filename}")
    shape = tuple(int(part) for part in match.group("shape").split("x"))
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid tensor shape in filename: {filename}")
    return TensorFileInfo(
        name=match.group("name"),
        shape=shape,
        dtype=TAG_TO_DTYPE[match.group("dtype")],
    )


def save_golden_tensor(directory: str | Path, name: str, tensor: np.ndarray) -> Path:
    array = np.asarray(tensor)
    dtype = canonical_dtype(array.dtype)
    array = array.astype(dtype, copy=False)
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / tensor_filename(name, array)
    array.ravel(order="F").tofile(output_path)
    return output_path


def load_golden_tensor(path: str | Path) -> np.ndarray:
    tensor_path = Path(path)
    info = parse_tensor_filename(tensor_path)
    data = np.fromfile(tensor_path, dtype=info.dtype)
    expected = int(np.prod(info.shape))
    if data.size != expected:
        raise ValueError(
            f"tensor byte count mismatch for {tensor_path}: "
            f"expected {expected} values, found {data.size}"
        )
    return data.reshape(info.shape, order="F")


def _value_bits(value: np.generic, dtype: np.dtype) -> str:
    if dtype == TAG_TO_DTYPE["f16"]:
        integer = np.asarray(value, dtype=dtype).view(np.uint16).item()
        return f"{integer:016b}"
    integer = np.asarray(value, dtype=dtype).view(np.uint32).item()
    return f"{integer:032b}"


def write_128bit_txt(bin_path: str | Path, dtype: np.dtype | type) -> Path:
    """Write the project's lane-reversed 128-bit text representation."""

    path = Path(bin_path)
    file_dtype = canonical_dtype(dtype)
    values = np.fromfile(path, dtype=file_dtype)
    lanes_per_line = 8 if file_dtype == TAG_TO_DTYPE["f16"] else 4
    remainder = values.size % lanes_per_line
    if remainder:
        values = np.concatenate(
            [values, np.zeros(lanes_per_line - remainder, dtype=file_dtype)]
        )

    txt_path = path.with_suffix(".txt")
    with txt_path.open("w", encoding="ascii", newline="\n") as stream:
        for start in range(0, values.size, lanes_per_line):
            lanes = [
                _value_bits(values[start + lane], file_dtype)
                for lane in range(lanes_per_line)
            ]
            stream.write("".join(reversed(lanes)))
            stream.write("\n")
    return txt_path


def write_decimal_1d_txt(bin_path: str | Path, dtype: np.dtype | type) -> Path:
    path = Path(bin_path)
    file_dtype = canonical_dtype(dtype)
    values = np.fromfile(path, dtype=file_dtype)
    txt_path = path.with_name(f"{path.stem}_decimal_1d.txt")
    with txt_path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            if np.issubdtype(file_dtype, np.integer):
                stream.write(f"{int(value)}\n")
            else:
                stream.write(f"{float(value):.10g}\n")
    return txt_path


def save_install_tensor(
    directory: str | Path,
    filename: str,
    tensor: np.ndarray,
) -> Path:
    """Save an already-linearized install tensor plus its debug views."""

    if not filename.endswith(".bin"):
        raise ValueError(f"install filename must end in .bin: {filename}")
    array = np.asarray(tensor)
    dtype = canonical_dtype(array.dtype)
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    array.astype(dtype, copy=False).reshape(-1, order="C").tofile(output_path)
    write_128bit_txt(output_path, dtype)
    write_decimal_1d_txt(output_path, dtype)
    return output_path

