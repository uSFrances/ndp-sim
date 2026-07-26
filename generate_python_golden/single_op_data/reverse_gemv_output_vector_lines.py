from __future__ import annotations

import argparse
from pathlib import Path


def reverse_line_tokens(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return line

    tokens = stripped.split()
    return " ".join(reversed(tokens)) + "\n"


def process_file(path: Path) -> None:
    original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_lines = [reverse_line_tokens(line) for line in original_lines]
    path.write_text("".join(updated_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reverse token order in each line of all slice gemv_output_vector_fp16.txt files."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run output directory containing sliceXX/gemv_output_vector_fp16.txt files.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    matched_files = sorted(run_dir.glob("slice*/gemv_output_vector_fp16.txt"))
    if not matched_files:
        raise FileNotFoundError(
            f"No slice*/gemv_output_vector_fp16.txt files found under {run_dir}"
        )

    for path in matched_files:
        process_file(path)
        print(f"[OK] Reversed line token order in {path}")


if __name__ == "__main__":
    main()
