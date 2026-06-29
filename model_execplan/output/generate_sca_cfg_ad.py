#!/usr/bin/env python3
"""Generate an A/B/D-only SCA config with actual matrix data lengths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MATRIX_PATH_RE = re.compile(
    r"^install/(?P<op>op\d+)/(?P<slice>slice\d+)/"
    r"matrix_(?P<matrix>[ABD])_linearized_128bit\.txt$"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def normalize_matrix_path(path_text: str) -> str:
    """Normalize older D paths that contain an extra cfg_pkg output prefix."""
    marker = "/install/"
    if path_text.startswith("install/"):
        return path_text
    if marker in path_text:
        return "install/" + path_text.split(marker, 1)[1]
    return path_text


def add_matrix_entries(
    output: dict,
    source: dict,
    data_dir: Path,
    matrices: set[str],
) -> tuple[int, list[str]]:
    added = 0
    missing = []
    for key, value in source.items():
        if not isinstance(value, dict):
            continue
        path_text = value.get("path")
        if not isinstance(path_text, str):
            continue

        path_text = normalize_matrix_path(path_text)
        match = MATRIX_PATH_RE.fullmatch(path_text)
        if match is None or match.group("matrix") not in matrices:
            continue

        data_path = data_dir / path_text
        if not data_path.is_file():
            missing.append(str(data_path))
            continue

        entry = dict(value)
        entry["path"] = path_text
        entry["length"] = count_nonempty_lines(data_path)
        output[key] = entry
        added += 1
    return added, missing


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_layer_dir = script_dir / "layer0_physic"
    default_data_dir = script_dir.parent / "data" / "layer0_physic"

    parser = argparse.ArgumentParser(
        description=(
            "Combine A/B entries from sca_cfg.json and D entries from "
            "sca_cfg_D.json, then calculate length from each matrix txt file."
        )
    )
    parser.add_argument(
        "--layer-dir",
        type=Path,
        default=default_layer_dir,
        help="Directory containing sca_cfg files and install data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: <layer-dir>/sca_cfg_AD.json).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing install/opX/sliceXX matrix data.",
    )
    args = parser.parse_args()

    layer_dir = args.layer_dir.resolve()
    data_dir = args.data_dir.resolve()
    source_path = layer_dir / "sca_cfg.json"
    d_source_path = layer_dir / "sca_cfg_D.json"
    output_path = (
        args.output.resolve()
        if args.output is not None
        else layer_dir / "sca_cfg_AD.json"
    )

    source = load_json(source_path)
    d_source = load_json(d_source_path)

    # Match sca_cfg_D.json: output matrix entries only, without config metadata.
    output = {}

    ab_count, ab_missing = add_matrix_entries(
        output, source, data_dir, {"A", "B"}
    )
    d_count, d_missing = add_matrix_entries(
        output, d_source, data_dir, {"D"}
    )
    missing = ab_missing + d_missing
    if missing:
        print(f"Missing matrix data files: {len(missing)}")
        for path in missing[:20]:
            print(f"  {path}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"A/B entries: {ab_count}")
    print(f"D entries:   {d_count}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
