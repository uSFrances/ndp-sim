#!/usr/bin/env python3
"""Split 128-bit text records across banks in round-robin order."""

import argparse
import json
from pathlib import Path


MATRIX_NAMES = ("A", "B", "D")


def find_input_files(source_root: Path) -> list[Path]:
    files = []
    for matrix_name in MATRIX_NAMES:
        pattern = f"op*/slice*/matrix_{matrix_name}_linearized_128bit.txt"
        files.extend(source_root.glob(pattern))
    return sorted(files)


def load_bank_interleaves(remapped_json: Path) -> dict[str, int]:
    data = json.loads(remapped_json.read_text(encoding="utf-8"))
    operators = data.get("operators")
    if not isinstance(operators, list):
        raise ValueError(f"{remapped_json} does not contain an operators list")

    bank_interleaves = {}
    for operator in operators:
        op_id = operator.get("id")
        if not isinstance(op_id, str):
            raise ValueError(f"operator without a valid id in {remapped_json}")

        values = []
        for input_value in operator.get("inputs", {}).values():
            if "bank_interleave" in input_value:
                values.append(input_value["bank_interleave"])
        output = operator.get("output", {})
        if "bank_interleave" in output:
            values.append(output["bank_interleave"])

        if not values:
            raise ValueError(f"{op_id} has no bank_interleave value")
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError(
                f"{op_id} bank_interleave values must be positive integers"
            )
        if len(set(values)) != 1:
            raise ValueError(
                f"{op_id} has inconsistent bank_interleave values: {values}"
            )
        bank_interleaves[op_id] = values[0]

    return bank_interleaves


def parse_source_file(source_file: Path) -> tuple[str, str]:
    op_id = source_file.parent.parent.name
    prefix = "matrix_"
    suffix = "_linearized_128bit.txt"
    filename = source_file.name
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"unexpected matrix filename: {source_file}")
    matrix_name = filename[len(prefix) : -len(suffix)]
    return op_id, matrix_name


def split_file(
    source_file: Path,
    destination_dir: Path,
    bank_num: int,
) -> tuple[int, list[int]]:
    records = source_file.read_text(encoding="ascii").splitlines()
    for line_number, record in enumerate(records, start=1):
        if len(record) != 128 or set(record) - {"0", "1"}:
            raise ValueError(
                f"{source_file}:{line_number} is not a 128-bit binary record"
            )

    bank_records = [records[bank_id::bank_num] for bank_id in range(bank_num)]

    destination_dir.mkdir(parents=True, exist_ok=True)
    output_stem = source_file.stem
    counts = []

    for bank_id, selected_records in enumerate(bank_records):
        output_file = destination_dir / f"{output_stem}_{bank_id}.txt"
        output_text = "".join(f"{record}\n" for record in selected_records)
        output_file.write_text(output_text, encoding="ascii")
        counts.append(len(selected_records))

    return len(records), counts


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_source = script_dir / "data" / "layer0_physic" / "install"
    default_destination = script_dir / "data" / "layer0_physic_bank" / "install"
    default_remapped_json = (
        script_dir / "layer0_padding_bankinterleave2_remapped.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Split matrix_A/B/D_linearized_128bit.txt into bank files by "
            "assigning consecutive 128-bit lines round-robin. Bank counts are "
            "read from a remapped JSON file."
        )
    )
    parser.add_argument(
        "--remapped-json",
        type=Path,
        default=default_remapped_json,
        help=f"Remapped JSON containing bank_interleave. Default: {default_remapped_json}",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help=f"Source install directory. Default: {default_source}",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=default_destination,
        help=f"Destination install directory. Default: {default_destination}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the record count written to every bank for every file.",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    destination_root = args.destination.resolve()
    remapped_json = args.remapped_json.resolve()
    if not source_root.is_dir():
        parser.error(f"source directory does not exist: {source_root}")
    if not remapped_json.is_file():
        parser.error(f"remapped JSON does not exist: {remapped_json}")

    source_files = find_input_files(source_root)
    if not source_files:
        parser.error(f"no matrix A/B/D 128-bit .txt files found in {source_root}")

    bank_interleaves = load_bank_interleaves(remapped_json)
    total_records = 0
    output_files = 0

    for source_file in source_files:
        op_id, matrix_name = parse_source_file(source_file)
        if op_id not in bank_interleaves:
            raise ValueError(f"{op_id} is not defined in {remapped_json}")
        bank_num = bank_interleaves[op_id]

        relative_parent = source_file.parent.relative_to(source_root)
        destination_dir = destination_root / relative_parent
        record_count, bank_counts = split_file(
            source_file,
            destination_dir,
            bank_num,
        )
        total_records += record_count
        output_files += bank_num

        if args.verbose:
            counts = ", ".join(
                f"bank{bank_id}={count}"
                for bank_id, count in enumerate(bank_counts)
            )
            print(
                f"{source_file.relative_to(source_root)}: "
                f"bank_interleave={bank_num}, total={record_count}, {counts}"
            )

    print(f"Remapped JSON: {remapped_json}")
    print(f"Source files processed: {len(source_files)}")
    print(f"128-bit records processed: {total_records}")
    print(f"Bank files generated: {output_files}")
    print(f"Destination: {destination_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
