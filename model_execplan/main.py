from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep source layout simple without requiring package installation.
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
OP_JSON_ROOT = REPO_ROOT / "jsons"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution_plan_generator.json_loader import execution_plan_to_dict
from execution_plan_generator.bank_data_exporter import export_bank_data, export_combined_bank_data
from execution_plan_generator.output_writer import (
    write_input_with_baseaddr,
    write_emulator_bundle,
    write_install_manifest,
    write_instruction_outputs,
    write_instruction_op_outputs,
)
from execution_plan_generator.pipeline import ExecutionPlanPipeline


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
        "-b",
        "--export-bank-data",
        action="store_true",
        help="Export per-slice per-bank data files to <output_prefix>/Bank_data",
    )
    parser.add_argument(
        "-bc",
        "--bank-combined",
        action="store_true",
        default=False,
        help="Combine all banks into a single file per slice (only with --export-bank-data)",
    )
    parser.add_argument(
        "-lw",
        "--bank-line-width",
        type=int,
        choices=[32, 128],
        default=32,
        help="Bank data output line width in bits: 32 or 128 (only with --export-bank-data, default: 32)",
    )
    parser.add_argument(
        "--bank-output-format",
        choices=["binary", "hex"],
        default="hex",
        help="Bank data output format: binary or hex (default: hex)",
    )
    parser.add_argument(
        "-e",
        "--export-emulator",
        action="store_true",
        help="Export emulator inputs to <output_prefix>/emulator",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_prefix = PROJECT_ROOT / "output" / args.json_file.stem
    pipeline = ExecutionPlanPipeline()
    result = pipeline.build_from_json(args.json_file)

    print(f"Parsed operators: {len(result.execution_input.operators)}")
    for op in result.execution_input.operators:
        input_names = ",".join(op.inputs.keys())
        sfu_note = ""
        template = result.templates.get(op.op_id)
        if template is not None and template.config_sfu_type:
            sfu_note = f", sfu={template.config_sfu_type}"
        print(
            f"  - {op.op_id} ({op.op_type}), inputs=[{input_names}], output_shape={op.output.shape}, "
            f"used_slices={op.used_slices:028b} (count={op.used_slice_count()}){sfu_note}"
        )

    # Show config / SFU reuse across operators of the same type.
    type_ops: dict[str, list[str]] = {}
    for op in result.execution_input.operators:
        type_ops.setdefault(op.op_type, []).append(op.op_id)
    for op_type, op_ids in type_ops.items():
        if len(op_ids) <= 1:
            continue
        cfg_addr = result.address_plan.operator_config_base_addresses.get(op_ids[0])
        sfu_addr = result.address_plan.operator_sfu_config_base_addresses.get(op_ids[0])
        extra = []
        if cfg_addr is not None:
            extra.append(f"config=0x{cfg_addr:08X}")
        if sfu_addr is not None:
            extra.append(f"sfu=0x{sfu_addr:08X}")
        print(f"    [{op_type}] shared by {', '.join(op_ids)}  ({', '.join(extra)})")

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

    hex_path, explanation_path = write_instruction_outputs(result.artifact, output_prefix)
    op_explanation_paths = write_instruction_op_outputs(result.artifact, output_prefix)
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
    # if op_explanation_paths:
    #     print("Per-op instruction-only outputs written to:")
    #     for path in op_explanation_paths:
    #         print(f"  {path}")
    print(f"Install manifest written to: {manifest_path}")
    print(f"Input with baseaddr written to: {with_baseaddr_path}")

    if args.export_emulator:
        emulator_paths = write_emulator_bundle(
            execution_input=result.execution_input,
            address_plan=result.address_plan,
            output_prefix=output_prefix,
            emulator_suffix=args.json_file.stem,
            skip_missing_data=True,
        )
        print(f"Emulator bundle written to: {output_prefix / f'emulator_{args.json_file.stem}'} ({len(emulator_paths)} files)")

    if args.export_bank_data:
        bank_data_dir = output_prefix / "Bank_data"
        export_fn = export_combined_bank_data if args.bank_combined else export_bank_data
        bank_files = export_fn(
            manifest_path=manifest_path,
            output_dir=bank_data_dir,
            line_width_bits=args.bank_line_width,
            output_format=args.bank_output_format,
        )
        print(
            f"Bank_data exported to: {bank_data_dir} "
            f"(files={len(bank_files)}, combined={args.bank_combined})"
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
