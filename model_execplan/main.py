from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Keep source layout simple without requiring package installation.
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
OP_CONFIG_ROOT = PROJECT_ROOT / "config"
OP_JSON_ROOT = REPO_ROOT / "jsons"
BITSTREAM_MAIN = REPO_ROOT / "bitstream" / "main.py"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution_plan_generator.json_loader import execution_plan_to_dict
from execution_plan_generator.bank_data_exporter import export_bank_data
from execution_plan_generator.output_writer import (
    write_input_with_baseaddr,
    write_emulator_bundle,
    write_install_manifest,
    write_instruction_outputs,
)
from execution_plan_generator.pipeline import ExecutionPlanPipeline


@dataclass(frozen=True)
class GeneratedConfigRecord:
    op_type: str
    op_dir: Path
    backup_dir: Path | None


def _collect_operator_types(json_file: Path) -> list[str]:
    with json_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    operators = payload.get("operators")
    if not isinstance(operators, list):
        raise ValueError("Input JSON must contain an 'operators' list.")

    op_types: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(operators):
        if not isinstance(item, dict):
            raise ValueError(f"operators[{idx}] must be an object.")
        op_type = item.get("type")
        if not isinstance(op_type, str) or not op_type:
            raise ValueError(f"operators[{idx}].type must be a non-empty string.")
        if op_type not in seen:
            seen.add(op_type)
            op_types.append(op_type)
    return op_types


def _has_required_operator_config_files(op_dir: Path) -> bool:
    if not op_dir.is_dir():
        return False
    has_parsed = (op_dir / "parsed_bitstream.txt").is_file()
    has_bitstream = bool(list(op_dir.glob("*bitstream_64b.bin")) or list(op_dir.glob("*bitstream_128b.bin")))
    return has_parsed and has_bitstream


def _backup_operator_dir(op_dir: Path) -> Path | None:
    if not op_dir.exists():
        return None
    backup_dir = op_dir.parent / f".{op_dir.name}.autogen_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(op_dir, backup_dir)
    return backup_dir


def _rollback_generated_configs(records: list[GeneratedConfigRecord]) -> None:
    for record in records:
        if record.op_dir.exists():
            shutil.rmtree(record.op_dir)
        if record.backup_dir is not None and record.backup_dir.exists():
            shutil.copytree(record.backup_dir, record.op_dir)


def _cleanup_config_backups(records: list[GeneratedConfigRecord]) -> None:
    for record in records:
        if record.backup_dir is not None and record.backup_dir.exists():
            shutil.rmtree(record.backup_dir)


def _decode_subprocess_output(raw: bytes | None) -> str:
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def ensure_operator_configs(json_file: Path, force_regenerate: bool = False) -> list[GeneratedConfigRecord]:
    if not BITSTREAM_MAIN.is_file():
        raise FileNotFoundError(f"bitstream entry not found: {BITSTREAM_MAIN}")

    op_types = _collect_operator_types(json_file)
    generated_records: list[GeneratedConfigRecord] = []
    generated: list[str] = []
    skipped: list[str] = []
    missing_json_templates: list[Path] = []

    for op_type in op_types:
        op_dir = OP_CONFIG_ROOT / op_type
        needs_generation = force_regenerate or (not _has_required_operator_config_files(op_dir))
        if not needs_generation:
            skipped.append(op_type)
            continue

        op_json = OP_JSON_ROOT / f"{op_type}.json"
        if not op_json.is_file():
            missing_json_templates.append(op_json)
            continue

        backup_dir = _backup_operator_dir(op_dir)
        if op_dir.exists():
            shutil.rmtree(op_dir)
        op_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(BITSTREAM_MAIN),
            "--visualize-placement",
            "-c",
            str(op_json),
            "-o",
            str(op_dir),
        ]
        print(f"[op-config] generating: {op_type}")
        try:
            child_env = os.environ.copy()
            child_env.setdefault("PYTHONUTF8", "1")
            child_env.setdefault("PYTHONIOENCODING", "utf-8")
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=True,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            if op_dir.exists():
                shutil.rmtree(op_dir)
            if backup_dir is not None and backup_dir.exists():
                shutil.copytree(backup_dir, op_dir)
            stderr_tail = _decode_subprocess_output(exc.stderr).strip()
            if len(stderr_tail) > 1200:
                stderr_tail = stderr_tail[-1200:]
            raise RuntimeError(
                f"Failed to generate config for operator '{op_type}' with command: {' '.join(cmd)}"
                + (f"\n[subprocess stderr]\n{stderr_tail}" if stderr_tail else "")
            ) from exc
        generated_records.append(
            GeneratedConfigRecord(op_type=op_type, op_dir=op_dir, backup_dir=backup_dir)
        )
        generated.append(op_type)

    if missing_json_templates:
        missing = "\n".join(str(p) for p in missing_json_templates)
        raise FileNotFoundError(
            "Missing operator JSON template file(s) under jsons directory:\n"
            f"{missing}"
        )

    print(
        "[op-config] done: "
        f"generated={len(generated)}, skipped={len(skipped)}, force_regenerate={force_regenerate}"
    )
    return generated_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execution plan generator framework (JSON parsing ready)."
    )
    parser.add_argument("json_file", type=Path, help="Path to input operator graph JSON")
    parser.add_argument(
        "--dump-normalized-json",
        type=Path,
        default=None,
        help="Optional output path for normalized parsed JSON",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Optional output prefix for instruction stream files",
    )
    parser.add_argument(
        "-b",
        "--export-bank-data",
        action="store_true",
        help="Export per-slice per-bank data files to <output_prefix>/Bank_data",
    )
    parser.add_argument(
        "-e",
        "--export-emulator",
        action="store_true",
        help="Export emulator inputs to <output_prefix>/emulator",
    )
    parser.add_argument(
        "-reop","--regenerate-operator-configs",
        action="store_true",
        default=False,
        help=(
            "Enable force regeneration of operator config folders under model_execplan/config "
            "using bitstream/main.py, even if they already exist"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    generated_records = ensure_operator_configs(
        json_file=args.json_file,
        force_regenerate=args.regenerate_operator_configs,
    )

    pipeline = ExecutionPlanPipeline()
    try:
        result = pipeline.build_from_json(args.json_file)
    except Exception:
        _rollback_generated_configs(generated_records)
        _cleanup_config_backups(generated_records)
        raise
    _cleanup_config_backups(generated_records)

    print(f"Parsed operators: {len(result.execution_input.operators)}")
    print(
        f"Default slices mask: {result.execution_input.used_slices:028b} "
        f"(count={result.execution_input.used_slice_count()})"
    )
    for op in result.execution_input.operators:
        input_names = ",".join(op.inputs.keys())
        print(
            f"  - {op.op_id} ({op.op_type}), inputs=[{input_names}], output_shape={op.output.shape}, "
            f"used_slices={op.used_slices:028b} (count={op.used_slice_count()})"
        )
        template = result.templates.get(op.op_id)
        if template is not None:
            print(
                f"    template: initial_size={template.initial_size}, target_size={template.target_size}, "
                f"update_control={template.should_update_control_registers}, "
                f"decoded_regs={len(template.original_register_values)}"
            )

    print(f"Address assignments: {len(result.address_plan.assignments)}")
    for tensor_name, assignment in result.address_plan.assignments.items():
        first_enabled_slice = min(assignment.per_slice_addresses) if assignment.per_slice_addresses else 0
        first_slice = assignment.per_slice_addresses.get(first_enabled_slice, assignment.base_address)
        print(
            f"  - {tensor_name}: shape={assignment.shape}, size={assignment.size_bytes}B, "
            f"base=0x{assignment.base_address:08X}, slice{first_enabled_slice}=0x{first_slice:08X}"
        )

    print(f"Generated commands: {len(result.artifact.commands)}")
    if result.artifact.metadata:
        print(f"Command metadata: {result.artifact.metadata}")
    for idx, cmd in enumerate(result.artifact.commands[:8]):
        print(f"  cmd[{idx}] = 0x{cmd:016X}")

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = PROJECT_ROOT / "output" / args.json_file.stem
    hex_path, explanation_path = write_instruction_outputs(result.artifact, output_prefix)
    manifest_path = write_install_manifest(
        execution_input=result.execution_input,
        address_plan=result.address_plan,
        templates=result.templates,
        artifact=result.artifact,
        output_prefix=output_prefix,
    )
    with_baseaddr_path = write_input_with_baseaddr(
        input_json_path=args.json_file,
        execution_input=result.execution_input,
        address_plan=result.address_plan,
        output_prefix=output_prefix,
    )
    print(f"Instruction stream written to: {hex_path}")
    print(f"Instruction explanation written to: {explanation_path}")
    print(f"Install manifest written to: {manifest_path}")
    print(f"Input with baseaddr written to: {with_baseaddr_path}")

    if args.export_emulator:
        emulator_paths = write_emulator_bundle(
            execution_input=result.execution_input,
            address_plan=result.address_plan,
            output_prefix=output_prefix,
            emulator_suffix=args.json_file.stem,
        )
        print(f"Emulator bundle written to: {output_prefix / f'emulator_{args.json_file.stem}'} ({len(emulator_paths)} files)")

    if args.export_bank_data:
        bank_data_dir = output_prefix / "Bank_data"
        bank_files = export_bank_data(manifest_path=manifest_path, output_dir=bank_data_dir)
        print(
            f"Bank_data exported to: {bank_data_dir} "
            f"(files={len(bank_files)})"
        )

    if args.dump_normalized_json is not None:
        payload = execution_plan_to_dict(result.execution_input)
        args.dump_normalized_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Normalized JSON written to: {args.dump_normalized_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
