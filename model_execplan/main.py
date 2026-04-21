from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep source layout simple without requiring package installation.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution_plan_generator.json_loader import execution_plan_to_dict
from execution_plan_generator.bank_data_exporter import export_bank_data
from execution_plan_generator.output_writer import (
    write_input_with_baseaddr,
    write_install_manifest,
    write_instruction_outputs,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pipeline = ExecutionPlanPipeline()
    result = pipeline.build_from_json(args.json_file)

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
