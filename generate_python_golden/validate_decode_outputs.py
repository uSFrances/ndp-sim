"""Validate generated Decode artifacts without requiring a hardware simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assemble_decode_package import resolve_manifest_path, sha256_file
from decode_ops import SUPPORTED_DECODE_OPERATORS, load_decode_config
from single_op_data.decode_passthrough import slice_passthrough_case
from tensor_io import load_golden_tensor


BASE_DIR = Path(__file__).resolve().parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_text_companions(bin_path: Path) -> None:
    text_path = bin_path.with_suffix(".txt")
    decimal_path = bin_path.with_name(f"{bin_path.stem}_decimal_1d.txt")
    _assert(text_path.is_file(), f"missing 128-bit text companion: {text_path}")
    _assert(decimal_path.is_file(), f"missing decimal companion: {decimal_path}")
    with text_path.open("r", encoding="ascii") as stream:
        first_line = stream.readline().strip()
    _assert(len(first_line) == 128, f"expected a 128-bit line in {text_path}")
    _assert(set(first_line) <= {"0", "1"}, f"non-binary character in {text_path}")


def _load_install(path: Path, dtype: np.dtype) -> np.ndarray:
    _assert(path.is_file(), f"missing install tensor: {path}")
    _validate_text_companions(path)
    return np.fromfile(path, dtype=dtype)


def _validate_golden_manifest(manifest: dict[str, object], golden_dir: Path) -> None:
    instances = manifest.get("instances", manifest.get("operators", []))
    _assert(isinstance(instances, list), "Golden manifest instances must be a list")
    actual_types = set()
    for entry in instances:  # type: ignore[assignment]
        actual_types.add(entry.get("op_name", entry.get("name")))
        tensors = [*entry["inputs"], entry["output"]]
        for tensor_entry in tensors:
            path = golden_dir / tensor_entry["path"]
            tensor = load_golden_tensor(path)
            _assert(list(tensor.shape) == tensor_entry["shape"], f"shape mismatch: {path}")
    expected_types = {spec.name for spec in SUPPORTED_DECODE_OPERATORS}
    missing_types = expected_types - actual_types
    _assert(not missing_types, f"Golden manifest missing operator types: {missing_types}")


def _validate_passthrough(
    entry: dict[str, object],
    golden_dir: Path,
    install_dir: Path,
    config: dict[str, object],
) -> int:
    entry_id = str(entry.get("instance_id", entry.get("id", entry.get("name", ""))))
    expected_inputs, expected_outputs = slice_passthrough_case(entry, golden_dir, config)
    checked = 0
    for slice_index in range(int(config["used_slices"])):
        slice_dir = install_dir / entry_id / f"slice{slice_index:02d}"
        for port, chunks in expected_inputs.items():
            expected = chunks[slice_index].reshape(-1, order="C")
            actual = _load_install(
                slice_dir / f"matrix_{port}_linearized_128bit.bin",
                expected.dtype,
            )
            np.testing.assert_array_equal(actual, expected)
            checked += 1
        expected_d = expected_outputs[slice_index].reshape(-1, order="C")
        actual_d = _load_install(
            slice_dir / "matrix_D_linearized_128bit.bin",
            expected_d.dtype,
        )
        np.testing.assert_array_equal(actual_d, expected_d)
        checked += 1
    return checked


def _validate_gemv(
    entry: dict[str, object],
    install_dir: Path,
    config: dict[str, object],
) -> int:
    entry_id = str(entry.get("instance_id", entry.get("id", entry.get("name", ""))))
    policy = str(entry["slice_policy"])
    if policy == "gemv_ring":
        expected_sizes = {"A": 32, "B": 14336, "Bp": 14336, "D": 32}
    elif policy == "gemv_local":
        expected_sizes = {"A": 32, "B": 512, "Bp": 512, "D": 32}
    else:
        raise AssertionError(f"unknown GEMV policy: {policy}")

    checked = 0
    for slice_index in range(int(config["used_slices"])):
        slice_dir = install_dir / entry_id / f"slice{slice_index:02d}"
        for port, size in expected_sizes.items():
            actual = _load_install(
                slice_dir / f"matrix_{port}_linearized_128bit.bin",
                np.dtype("<f2"),
            )
            _assert(actual.size == size, f"unexpected {policy} {port} size in {slice_dir}")
            checked += 1
    return checked


def _validate_execplan(execplan_root: Path, install_dir: Path) -> tuple[int, int]:
    execplan_path = execplan_root / "install" / "execplan.txt"
    sca_path = execplan_root / "sca_cfg.json"
    _assert(execplan_path.is_file(), f"missing execution plan: {execplan_path}")
    _assert(sca_path.is_file(), f"missing execution plan manifest: {sca_path}")
    lines = [line.strip() for line in execplan_path.read_text(encoding="ascii").splitlines() if line.strip()]
    _assert(lines, "execution plan is empty")
    _assert(all(len(line) == 128 and set(line) <= {"0", "1"} for line in lines), "invalid execplan line")
    sca = json.loads(sca_path.read_text(encoding="utf-8"))
    _assert(int(sca["Exec_Length"]) == len(lines), "Exec_Length does not match execplan.txt")

    package_manifest_path = execplan_root / "decode_package_manifest.json"
    _assert(package_manifest_path.is_file(), f"missing package manifest: {package_manifest_path}")
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    records = package_manifest.get("files")
    _assert(isinstance(records, list), "package manifest files must be a list")
    record_by_path = {record["path"]: record for record in records}

    referenced_paths = 0
    tensor_paths = 0
    for key, value in sca.items():
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            continue
        path_text = value["path"]
        destination, parts = resolve_manifest_path(execplan_root, path_text)
        _assert(destination.is_file(), f"SCA path does not exist for {key}: {destination}")
        referenced_paths += 1

        record = record_by_path.get(path_text)
        _assert(record is not None, f"package manifest is missing SCA path: {path_text}")
        destination_hash = sha256_file(destination)
        _assert(record.get("sha256") == destination_hash, f"stale package hash: {path_text}")
        _assert(int(record.get("size_bytes", -1)) == destination.stat().st_size, f"stale package size: {path_text}")

        if len(parts) >= 3 and parts[0] == "install" and parts[1].startswith("op"):
            source = install_dir.resolve().joinpath(*parts[1:]).resolve()
            _assert(source.is_file(), f"package tensor source is missing: {source}")
            _assert(sha256_file(source) == destination_hash, f"package tensor differs from source: {path_text}")
            tensor_paths += 1

    _assert(len(record_by_path) == referenced_paths, "package manifest contains stale/unreferenced files")
    _assert(int(package_manifest.get("referenced_files", -1)) == referenced_paths, "wrong package reference count")
    _assert(int(package_manifest.get("tensor_files", -1)) == tensor_paths, "wrong package tensor count")

    config_root = execplan_root / "config"
    for spec in SUPPORTED_DECODE_OPERATORS:
        op_dir = config_root / spec.op_id
        _assert(any(op_dir.glob("*bitstream_64b.bin")), f"missing 64-bit bitstream for {spec.op_id}")
        _assert(any(op_dir.glob("*bitstream_128b.bin")), f"missing 128-bit bitstream for {spec.op_id}")
    return len(lines), referenced_paths


def validate_decode_outputs(
    config_path: Path,
    golden_manifest_path: Path,
    install_dir: Path,
    execplan_root: Path,
) -> dict[str, int]:
    config = load_decode_config(config_path)
    manifest = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
    _assert(manifest["config"] == config, "Golden manifest config differs from config.json")
    _validate_golden_manifest(manifest, golden_manifest_path.parent)

    install_manifest = json.loads((install_dir / "manifest.json").read_text(encoding="utf-8"))
    install_ops = install_manifest.get("operators", [])
    _assert(len(install_ops) > 0, "install manifest has no operators")

    instances = manifest.get("instances", manifest.get("operators", []))
    checked_tensors = 0
    for entry in instances:
        entry_id = str(entry.get("instance_id", entry.get("id", entry.get("name", ""))))
        op_dir = install_dir / entry_id
        slices = sorted(path for path in op_dir.glob("slice??") if path.is_dir())
        _assert(len(slices) == int(config["used_slices"]), f"wrong slice count for {entry_id}")
        if str(entry["slice_policy"]).startswith("gemv_"):
            checked_tensors += _validate_gemv(entry, install_dir, config)
        else:
            checked_tensors += _validate_passthrough(
                entry, golden_manifest_path.parent, install_dir, config
            )

    execplan_lines, package_files = _validate_execplan(execplan_root, install_dir)
    unique_types = len({entry.get("op_name", entry.get("name")) for entry in instances})
    result = {
        "operator_types": unique_types,
        "layer_instances": len(instances),
        "used_slices": int(config["used_slices"]),
        "install_tensors": checked_tensors,
        "execplan_128bit_lines": execplan_lines,
        "package_files": package_files,
    }
    print(
        "[decode-validate] "
        f"operators={result['operators']} slices={result['slices_per_operator']} "
        f"install_tensors={result['install_tensors']} "
        f"execplan_lines={result['execplan_128bit_lines']} "
        f"package_files={result['package_files']}"
    )
    print("[decode-validate] software artifact validation passed; hardware output comparison skipped")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated Decode artifacts.")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument(
        "--golden-manifest",
        type=Path,
        default=BASE_DIR / "python_golden_decode" / "manifest.json",
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=BASE_DIR / "single_op_data" / "install_decode",
    )
    parser.add_argument(
        "--execplan-root",
        type=Path,
        default=BASE_DIR / "model_execplan" / "output" / "layer0_decode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_decode_outputs(
        args.config,
        args.golden_manifest,
        args.install_dir,
        args.execplan_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
