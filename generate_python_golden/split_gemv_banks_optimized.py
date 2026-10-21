#!/usr/bin/env python3
"""Split ALL 128-bit data files in install_decode_optimized by bank_interleave.

Reads bank_interleave for every op+port from layer0_decode_optimized.json,
then round-robin splits every .txt and .bin data file (A, B, B', D) across
the specified number of banks.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BANK_JSON = BASE_DIR / "model_execplan" / "examples" / "layer0_decode_optimized.json"
INSTALL_DIR = BASE_DIR / "single_op_data" / "install_decode_optimized"

PORT_MAP = {"A": "A", "B": "B", "Bp": "B'", "D": "D", "C": "C"}

WORD_SIZE = 16  # 128-bit


def _split_txt(path: Path, num_banks: int) -> None:
    records = path.read_text(encoding="ascii").splitlines()
    stem = path.stem
    for bid in range(num_banks):
        bank_records = records[bid::num_banks]
        out = path.parent / f"{stem}_{bid}.txt"
        out.write_text("".join(r + "\n" for r in bank_records), encoding="ascii")


def _split_bin(path: Path, num_banks: int) -> None:
    data = path.read_bytes()
    num_words = len(data) // WORD_SIZE
    stem = path.stem
    for bid in range(num_banks):
        chunk = bytearray()
        for i in range(bid, num_words, num_banks):
            chunk.extend(data[i * WORD_SIZE:(i + 1) * WORD_SIZE])
        out = path.parent / f"{stem}_{bid}.bin"
        out.write_bytes(bytes(chunk))


def _detect_port(fname: str) -> str:
    """Extract port name from filename."""
    if fname.startswith("matrix_"):
        parts = fname.split("_")
        if len(parts) >= 2:
            return PORT_MAP.get(parts[1], "")
    else:
        parts = fname.split("_")
        if len(parts) >= 1:
            return PORT_MAP.get(parts[0], "")
    return ""


def main():
    data = json.loads(BANK_JSON.read_text(encoding="utf-8"))

    op_banks: dict[str, dict[str, int]] = {}
    for op in data["operators"]:
        op_id = op["id"]
        port_info = {}
        for p, info in op.get("inputs", {}).items():
            if "bank_interleave" in info:
                port_info[p] = info["bank_interleave"]
        out_info = op.get("output", {})
        if "bank_interleave" in out_info:
            port_info["D"] = out_info["bank_interleave"]
        op_banks[op_id] = port_info

    count = 0
    for f in sorted(INSTALL_DIR.glob("op*/slice*/*")):
        if not f.is_file():
            continue

        op_id = f.parent.parent.name
        fname = f.name
        ext = f.suffix
        if ext not in (".txt", ".bin"):
            continue

        port = _detect_port(fname)
        if not port or op_id not in op_banks:
            continue

        bn = op_banks[op_id].get(port, 1)
        if bn <= 1:
            continue

        if ext == ".txt":
            _split_txt(f, bn)
        else:
            _split_bin(f, bn)
        count += 1

    print(f"Done. {count} files processed.")


if __name__ == "__main__":
    main()
