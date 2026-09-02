#!/usr/bin/env python3
"""Split GEMV B/B' 128-bit .txt files by bank_interleave (round-robin)."""

import json
from pathlib import Path

BANK_JSON = Path(__file__).resolve().parent.parent / "model_execplan" / "examples" / "layer0_decode_bank2_remapped.json"
INSTALL_DIR = Path(__file__).resolve().parent / "single_op_data" / "install_decode"


def main():
    data = json.loads(BANK_JSON.read_text(encoding="utf-8"))

    op_banks: dict[str, dict[str, int]] = {}
    for op in data["operators"]:
        op_banks[op["id"]] = {
            p: info["bank_interleave"]
            for p, info in op.get("inputs", {}).items()
            if "bank_interleave" in info
        }

    port_map = {"Bp": "B'", "B": "B", "A": "A"}
    count = 0

    for f in sorted(INSTALL_DIR.glob("op*/slice*/matrix_B*_linearized_128bit.txt")):
        op_id = f.parent.parent.name
        stem = f.stem  # matrix_B_linearized_128bit or matrix_Bp_linearized_128bit
        port = port_map.get(stem.split("_")[1])
        if op_id not in op_banks or port not in op_banks[op_id]:
            continue

        bn = op_banks[op_id][port]
        records = f.read_text(encoding="ascii").splitlines()

        for bid in range(bn):
            out = f.parent / f"{stem}_{bid}.txt"
            bank_records = records[bid::bn]
            out.write_text("".join(r + "\n" for r in bank_records), encoding="ascii")

        b0 = len(records[0::bn])
        b1 = len(records[1::bn])
        print(f"  {f.relative_to(INSTALL_DIR)}: banks={bn}, total={len(records)}, b0={b0}, b1={b1}")
        count += 1

    print(f"Done. {count} files processed.")


if __name__ == "__main__":
    main()
