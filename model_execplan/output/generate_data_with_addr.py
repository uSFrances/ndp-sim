#!/usr/bin/env python3
"""Generate hex matrix data and address-to-data maps from sca_cfg.json."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


MATRIX_PATH_RE = re.compile(
    r"^install/(?P<op>op\d+)/(?P<slice>slice\d+)/"
    r"matrix_(?P<matrix>[A-Z])_linearized_128bit\.txt$"
)


def parse_hex_int(text: str) -> int:
    return int(text.replace("_", ""), 16)


def load_operator_specs(model_json: Path) -> dict[str, dict]:
    data = json.loads(model_json.read_text(encoding="utf-8"))
    return {op["id"]: op for op in data.get("operators", [])}


def infer_dtype(operator_specs: dict[str, dict], op_id: str, matrix: str) -> str:
    operator = operator_specs.get(op_id, {})
    if matrix in ("A", "B"):
        dtype = operator.get("inputs", {}).get(matrix, {}).get("dtype")
        if dtype in ("fp16", "fp32"):
            return dtype
    if matrix == "D":
        dtype = operator.get("output", {}).get("dtype")
        if dtype in ("fp16", "fp32"):
            return dtype

    op_type = operator.get("type", "")
    dtypes = re.findall(r"fp(?:16|32)", op_type.lower())
    if matrix == "A" and dtypes:
        return dtypes[0]
    if matrix == "B" and len(dtypes) >= 3:
        return dtypes[1]
    if matrix == "D" and len(dtypes) >= 2:
        return dtypes[-1]
    return "unknown"


def normalize_hex_word(text: str) -> str:
    raw = text.strip().lower().removeprefix("0x").replace("_", "")
    if not raw:
        raise ValueError("empty hex word")
    int(raw, 16)
    if len(raw) > 32:
        raise ValueError(f"hex word is wider than 128 bits: 0x{raw}")
    return raw.zfill(32)


def binary_line_to_hex(line: str) -> str:
    bits = line.strip().replace("_", "")
    if len(bits) != 128 or any(ch not in "01" for ch in bits):
        raise ValueError("binary line is not exactly 128 bits")
    return f"{int(bits, 2):032x}"


def decimal_tokens_to_hex_words(tokens: list[str], dtype: str) -> list[str]:
    if dtype not in ("fp16", "fp32"):
        raise ValueError(
            f"decimal data needs dtype fp16/fp32, got {dtype!r}"
        )

    chunks: list[bytes] = []
    if dtype == "fp32":
        for token in tokens:
            chunks.append(struct.pack(">f", float(token)))
    else:
        for token in tokens:
            chunks.append(struct.pack(">e", float(token)))

    bytes_per_word = 16
    element_bytes = b"".join(chunks)
    if len(element_bytes) % bytes_per_word != 0:
        raise ValueError(
            f"decimal data byte length {len(element_bytes)} is not divisible by 16"
        )
    return [
        element_bytes[offset : offset + bytes_per_word].hex()
        for offset in range(0, len(element_bytes), bytes_per_word)
    ]


def convert_to_hex_words(input_path: Path, dtype: str) -> list[str]:
    lines = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return []

    first = lines[0].replace("_", "")
    if len(first) == 128 and all(ch in "01" for ch in first):
        return [binary_line_to_hex(line) for line in lines]

    if first.lower().startswith("0x") or (
        len(first) <= 32 and all(ch in "0123456789abcdefABCDEF" for ch in first)
    ):
        return [normalize_hex_word(line) for line in lines]

    tokens: list[str] = []
    for line in lines:
        tokens.extend(token for token in re.split(r"[\s,]+", line) if token)
    return decimal_tokens_to_hex_words(tokens, dtype)


def iter_matrix_entries(sca_cfg: dict):
    for key, value in sca_cfg.items():
        if not isinstance(value, dict):
            continue
        path = value.get("path")
        base_addr = value.get("base_addr")
        if not isinstance(path, str) or not isinstance(base_addr, str):
            continue
        match = MATRIX_PATH_RE.fullmatch(path)
        if match is None:
            continue
        yield key, value, match


def write_outputs(
    output_root: Path,
    relative_path: str,
    base_addr: int,
    dtype: str,
    hex_words: list[str],
) -> tuple[Path, Path]:
    stripped = relative_path.removeprefix("install/")
    hex_output = output_root / stripped
    addr_output = hex_output.with_name(f"{hex_output.stem}_withaddr.txt")
    hex_output.parent.mkdir(parents=True, exist_ok=True)

    hex_output.write_text(
        "".join(f"{word}\n" for word in hex_words),
        encoding="utf-8",
    )

    with addr_output.open("w", encoding="utf-8") as f:
        f.write(f"# dtype: {dtype}\n")
        f.write(f"# base_addr_bytes: 0x{base_addr:08x}\n")
        f.write("# index | byte_addr | word_addr_128b | data_hex\n")
        for index, word in enumerate(hex_words):
            byte_addr = base_addr + index * 16
            f.write(
                f"{index} | 0x{byte_addr:08x} | 0x{byte_addr >> 4:06x} | 0x{word}\n"
            )
    return hex_output, addr_output


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_cfg = script_dir / "layer0_physic" / "sca_cfg.json"
    default_d_cfg = script_dir / "layer0_physic" / "sca_cfg_D.json"
    default_model_json = script_dir.parent / "layer0_physic.json"
    default_output = script_dir / "layer0_physic_datawithaddr"

    parser = argparse.ArgumentParser(
        description=(
            "Read layer0_physic/sca_cfg.json, convert matrix txt data to "
            "128-bit hex words, and generate address-to-data files."
        )
    )
    parser.add_argument(
        "--sca-cfg",
        type=Path,
        action="append",
        help=(
            "sca_cfg JSON to read. Can be passed multiple times. "
            "Defaults to layer0_physic/sca_cfg.json and sca_cfg_D.json if present."
        ),
    )
    parser.add_argument("--model-json", type=Path, default=default_model_json)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--op", help="Only process one op, for example op0")
    parser.add_argument("--slice", help="Only process one slice, for example slice00")
    args = parser.parse_args()

    sca_cfg_paths = (
        [path.resolve() for path in args.sca_cfg]
        if args.sca_cfg
        else [path.resolve() for path in (default_cfg, default_d_cfg) if path.is_file()]
    )
    output_root = args.output_dir.resolve()
    operator_specs = load_operator_specs(args.model_json.resolve())

    processed = 0
    for sca_cfg_path in sca_cfg_paths:
        sca_root = sca_cfg_path.parent
        sca_cfg = json.loads(sca_cfg_path.read_text(encoding="utf-8"))
        for _, value, match in iter_matrix_entries(sca_cfg):
            op_id = match.group("op")
            slice_name = match.group("slice")
            matrix = match.group("matrix")
            if args.op is not None and op_id != args.op:
                continue
            if args.slice is not None and slice_name != args.slice:
                continue

            relative_path = value["path"]
            input_path = sca_root / relative_path
            if not input_path.is_file():
                raise FileNotFoundError(input_path)

            dtype = infer_dtype(operator_specs, op_id, matrix)
            hex_words = convert_to_hex_words(input_path, dtype)
            base_addr = parse_hex_int(value["base_addr"])
            hex_output, addr_output = write_outputs(
                output_root, relative_path, base_addr, dtype, hex_words
            )
            processed += 1
            print(f"{relative_path} -> {hex_output.relative_to(output_root)}")
            print(f"{relative_path} -> {addr_output.relative_to(output_root)}")

    print(f"Processed matrix files: {processed}")
    print("sca_cfg files:")
    for path in sca_cfg_paths:
        print(f"  {path}")
    print(f"Output directory: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
