#!/usr/bin/env python3
"""Compare selected op matrix files and write one report per op/slice."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


MATRIX_FILENAME = "matrix_{}_linearized_128bit.txt"


def parse_op_selection(text: str) -> list[int]:
    """Parse selections such as '0-5', 'op0-op5', or '0,2,5-8'."""
    selected: set[int] = set()
    for part in text.lower().replace("op", "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"invalid descending op range: {part!r}"
                )
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise argparse.ArgumentTypeError("at least one op must be selected")
    return sorted(selected)


def parse_matrices(text: str) -> list[str]:
    matrices = list(dict.fromkeys(text.upper().replace(",", "")))
    invalid = [matrix for matrix in matrices if matrix not in "ABD"]
    if not matrices or invalid:
        raise argparse.ArgumentTypeError(
            "matrices must contain only A, B, and D, for example ABD or A,D"
        )
    return matrices


def read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip().replace("_", "")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def to_hex_128(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if len(value) != 128 or re.fullmatch(r"[01]{128}", value) is None:
        return None
    return f"0x{int(value, 2):032x}"


def compare_file(candidate: Path, reference: Path) -> dict:
    if not candidate.is_file() or not reference.is_file():
        return {
            "status": "missing",
            "candidate_exists": candidate.is_file(),
            "reference_exists": reference.is_file(),
            "candidate_path": str(candidate),
            "reference_path": str(reference),
        }

    candidate_lines = read_nonempty_lines(candidate)
    reference_lines = read_nonempty_lines(reference)
    lines = []
    different_line_count = 0
    line_count = max(len(candidate_lines), len(reference_lines))

    for index in range(line_count):
        candidate_value = (
            candidate_lines[index] if index < len(candidate_lines) else None
        )
        reference_value = (
            reference_lines[index] if index < len(reference_lines) else None
        )
        identical = candidate_value == reference_value
        line_result = {
            "line": index + 1,
            "status": "identical" if identical else "different",
            "candidate_hex": to_hex_128(candidate_value),
            "reference_hex": to_hex_128(reference_value),
        }
        if candidate_value is not None and line_result["candidate_hex"] is None:
            line_result["candidate_raw"] = candidate_value
        if reference_value is not None and line_result["reference_hex"] is None:
            line_result["reference_raw"] = reference_value
        lines.append(line_result)
        if not identical:
            different_line_count += 1

    return {
        "status": "identical" if different_line_count == 0 else "different",
        "candidate_line_count": len(candidate_lines),
        "reference_line_count": len(reference_lines),
        "different_line_count": different_line_count,
        "lines": lines,
    }


def slice_names(candidate_op: Path, reference_op: Path) -> list[str]:
    names = {
        path.name
        for op_dir in (candidate_op, reference_op)
        if op_dir.is_dir()
        for path in op_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"slice\d+", path.name)
    }
    return sorted(names, key=lambda name: int(name.removeprefix("slice")))


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    layer_dir = script_dir / "rope"

    parser = argparse.ArgumentParser(
        description="Compare selected op/slice matrix files against install data."
    )
    parser.add_argument(
        "--ops",
        required=True,
        type=parse_op_selection,
        help="Ops to compare, for example 0-5, op0-op5, or 0,2,5-8.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=layer_dir / "rope_output",
        help="Directory containing candidate opX/sliceXX files.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=layer_dir / "install",
        help="Directory containing reference opX/sliceXX files.",
    )
    parser.add_argument(
        "--matrices",
        type=parse_matrices,
        default=parse_matrices("ABD"),
        help="Matrices to compare (default: ABD).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output_compare_rope",
        help="Output directory for opX/sliceXX.json reports.",
    )
    args = parser.parse_args()

    candidate_root = args.candidate_dir.resolve()
    reference_root = args.reference_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    compared_files = 0
    identical_files = 0
    missing_files = 0
    different_files = 0
    generated_slice_reports = 0
    ops_without_slices = []

    for op_index in args.ops:
        op_name = f"op{op_index}"
        candidate_op = candidate_root / op_name
        reference_op = reference_root / op_name
        slices = slice_names(candidate_op, reference_op)
        if not slices:
            ops_without_slices.append(op_name)
            continue

        for slice_name in slices:
            slice_report = {
                "op": op_name,
                "slice": slice_name,
                "matrices": {},
            }
            for matrix in args.matrices:
                filename = MATRIX_FILENAME.format(matrix)
                candidate_path = candidate_op / slice_name / filename
                reference_path = reference_op / slice_name / filename
                if not candidate_path.is_file() and not reference_path.is_file():
                    continue
                result = compare_file(
                    candidate_path,
                    reference_path,
                )
                compared_files += 1
                if result["status"] == "identical":
                    identical_files += 1
                elif result["status"] == "missing":
                    missing_files += 1
                else:
                    different_files += 1
                slice_report["matrices"][f"matrix_{matrix}"] = result

            if slice_report["matrices"]:
                slice_report_path = output_root / op_name / f"{slice_name}.json"
                slice_report_path.parent.mkdir(parents=True, exist_ok=True)
                slice_report_path.write_text(
                    json.dumps(slice_report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                generated_slice_reports += 1

    summary = {
        "selection": {
            "ops": [f"op{index}" for index in args.ops],
            "matrices": args.matrices,
            "candidate_dir": str(candidate_root),
            "reference_dir": str(reference_root),
        },
        "summary": {
            "compared_files": compared_files,
            "identical_files": identical_files,
            "different_files": different_files,
            "missing_files": missing_files,
            "generated_slice_reports": generated_slice_reports,
            "ops_without_slice_directories": ops_without_slices,
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Compared files:  {compared_files}")
    print(f"Identical files: {identical_files}")
    print(f"Different files: {different_files}")
    print(f"Missing files:   {missing_files}")
    print(f"Slice reports:   {generated_slice_reports}")
    print(f"Output: {output_root}")
    return 1 if different_files or missing_files or ops_without_slices else 0


if __name__ == "__main__":
    raise SystemExit(main())
