"""Assemble a self-contained Decode package from ``sca_cfg.json`` references.

``model_execplan`` writes the execution plan, configuration bitstreams and an
SCA manifest, while Decode tensor inputs are produced under
``single_op_data/install_decode``.  This script joins those two outputs without
changing filenames or tensor contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(root: Path, path_text: str) -> tuple[Path, tuple[str, ...]]:
    """Resolve one POSIX manifest path and reject absolute/traversal paths."""

    if "\\" in path_text:
        raise ValueError(f"SCA paths must use forward slashes: {path_text!r}")
    relative = PurePosixPath(path_text)
    parts = relative.parts
    if relative.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe SCA path: {path_text!r}")

    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*parts).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"SCA path escapes package root: {path_text!r}")
    return resolved, parts


def _is_tensor_reference(parts: tuple[str, ...]) -> bool:
    return len(parts) >= 3 and parts[0] == "install" and parts[1].startswith("op")


def assemble_decode_package(
    sca_cfg_path: Path,
    data_root: Path,
    package_root: Path,
    manifest_path: Path | None = None,
) -> Path:
    sca_cfg_path = sca_cfg_path.resolve()
    data_root = data_root.resolve()
    package_root = package_root.resolve()
    if not sca_cfg_path.is_file():
        raise FileNotFoundError(f"SCA config not found: {sca_cfg_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Decode install data not found: {data_root}")

    sca_cfg = json.loads(sca_cfg_path.read_text(encoding="utf-8"))
    if not isinstance(sca_cfg, dict):
        raise ValueError("sca_cfg.json must contain a JSON object")

    # Preflight every reference before copying anything, so a bad manifest
    # cannot leave a partially updated package.
    entries: list[dict[str, Any]] = []
    destinations: set[Path] = set()
    for key, value in sca_cfg.items():
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            continue
        path_text = value["path"]
        destination, parts = resolve_manifest_path(package_root, path_text)
        if destination in destinations:
            raise ValueError(f"duplicate SCA destination: {path_text}")
        destinations.add(destination)

        is_tensor = _is_tensor_reference(parts)
        if is_tensor:
            source = data_root.joinpath(*parts[1:]).resolve()
            if source != data_root and data_root not in source.parents:
                raise ValueError(f"tensor source escapes data root: {path_text!r}")
        else:
            source = destination
        if not source.is_file():
            raise FileNotFoundError(f"SCA source file is missing for {key}: {source}")
        entries.append(
            {
                "key": key,
                "path": path_text,
                "source": source,
                "destination": destination,
                "is_tensor": is_tensor,
            }
        )

    if not entries:
        raise ValueError("sca_cfg.json does not contain any file references")

    copied = 0
    unchanged = 0
    file_records: list[dict[str, Any]] = []
    for entry in entries:
        source: Path = entry["source"]
        destination: Path = entry["destination"]
        source_hash = sha256_file(source)
        if entry["is_tensor"]:
            destination_hash = sha256_file(destination) if destination.is_file() else None
            if destination_hash != source_hash:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied += 1
            else:
                unchanged += 1
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise AssertionError(f"packaged file differs from its source: {entry['path']}")
        file_records.append(
            {
                "key": entry["key"],
                "path": entry["path"],
                "kind": "tensor" if entry["is_tensor"] else "execplan_or_config",
                "size_bytes": destination.stat().st_size,
                "sha256": destination_hash,
            }
        )

    output_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else package_root / "decode_package_manifest.json"
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    tensor_count = sum(bool(entry["is_tensor"]) for entry in entries)
    output_manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "sca_cfg": str(sca_cfg_path),
                "tensor_data_root": str(data_root),
                "package_root": str(package_root),
                "referenced_files": len(entries),
                "tensor_files": tensor_count,
                "copied_tensor_files": copied,
                "unchanged_tensor_files": unchanged,
                "files": file_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[decode-package] "
        f"references={len(entries)} tensors={tensor_count} "
        f"copied={copied} unchanged={unchanged}"
    )
    print(f"[decode-package] wrote self-contained package under {package_root}")
    return output_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the Decode SCA loading package.")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=BASE_DIR / "model_execplan" / "output" / "layer0_decode",
    )
    parser.add_argument(
        "--sca-cfg",
        type=Path,
        default=None,
        help="Defaults to <package-root>/sca_cfg.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BASE_DIR / "single_op_data" / "install_decode",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve()
    sca_cfg = args.sca_cfg or package_root / "sca_cfg.json"
    assemble_decode_package(sca_cfg, args.data_root, package_root, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
