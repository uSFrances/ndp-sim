from __future__ import annotations

import json
import re
import shutil
from math import ceil
from pathlib import Path
from typing import Callable

from .models import AddressPlan, ExecutionPlanArtifact, ExecutionPlanInput, InputSourceType, OperatorTemplate


_OUTPUT_BASE_ADDR_ROUTER_BY_OP_AND_TYPE: dict[tuple[str, str], Callable[[int], int]] = {
    ("prefill_mul_fp32MN_fp32MN_fp32MN", "rope_slice_xor2"): lambda slice_id: slice_id ^ 0b10,
}

_OUTPUT_BASE_ADDR_ROUTER_BY_TYPE: dict[str, Callable[[int], int]] = {}


def write_instruction_outputs(
    artifact: ExecutionPlanArtifact,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    output_dir = prefix if prefix.suffix == "" else prefix.parent / prefix.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    install_dir = output_dir / "install"
    install_dir.mkdir(parents=True, exist_ok=True)

    binary_path = install_dir / "execplan.txt"
    explanation_path = output_dir / "instructions_explained.txt"

    binary_lines = _to_128bit_lines(artifact.commands)
    binary_path.write_text("\n".join(binary_lines) + "\n", encoding="utf-8")

    explanation_lines: list[str] = []
    current_operator_id: str | None = None
    for idx, command in enumerate(artifact.commands):
        explanation = ""
        if idx < len(artifact.command_explanations):
            explanation = artifact.command_explanations[idx]
        operator_id = _extract_operator_id(explanation)
        if operator_id != current_operator_id:
            if explanation_lines:
                explanation_lines.append("")
            if operator_id is not None:
                explanation_lines.append(f"===== operator {operator_id} =====")
            else:
                explanation_lines.append("===== operator unknown =====")
            current_operator_id = operator_id
        explanation_lines.append(f"{idx:04d}  {command:064b}  {explanation}".rstrip())
    explanation_path.write_text("\n".join(explanation_lines) + "\n", encoding="utf-8")

    return binary_path, explanation_path


def write_install_manifest(
    execution_input: ExecutionPlanInput,
    address_plan: AddressPlan,
    templates: dict[str, OperatorTemplate],
    artifact: ExecutionPlanArtifact,
    output_prefix: str | Path,
    exec_base_addr: int | None = None,
    exec_plan_path: str = "install/execplan.txt",
) -> Path:
    prefix = Path(output_prefix)
    output_dir = prefix if prefix.suffix == "" else prefix.parent / prefix.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    install_dir = output_dir / "install"
    install_dir.mkdir(parents=True, exist_ok=True)
    cfg_pkg_dir = install_dir / "cfg_pkg"
    cfg_pkg_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {}

    exec_length = len(_to_128bit_lines(artifact.commands))
    if exec_base_addr is None:
        exec_base_addr = _compute_exec_base_after_allocations(address_plan)
    payload["Exec_Base"] = _format_hex32_grouped(exec_base_addr)
    payload["Exec_Length"] = exec_length
    payload["ExecutionPlan"] = {
        "base_addr": _format_hex32(exec_base_addr),
        "path": exec_plan_path,
    }

    # Record every operator input/output tensor using flat keys like op0_matrixA.
    for op in execution_input.operators:
        for input_name in op.inputs.keys():
            if input_name == "B'":
                continue
            input_spec = op.inputs.get(input_name)
            if input_spec is not None and input_spec.source is not None:
                if input_spec.source.source_type == InputSourceType.OPERATOR:
                    # Input reuses a previous operator output; avoid duplicate manifest entry.
                    continue
            io_key = f"{op.op_id}.input.{input_name}"
            tensor_name = address_plan.operator_io_to_tensor.get(io_key)
            if tensor_name is None:
                continue
            assignment = address_plan.assignments.get(tensor_name)
            if assignment is None:
                continue

            for slice_id, slice_base_addr in sorted(assignment.per_slice_addresses.items()):
                slice_dir = _format_slice_dir(slice_id)
                payload[f"{op.op_id}_matrix{input_name}_slice{slice_id}"] = {
                    "base_addr": _format_hex32(slice_base_addr),
                    "path": f"install/{op.op_id}/{slice_dir}/matrix_{input_name}_linearized_128bit.bin",
                }

        output_key = f"{op.op_id}.output.D"
        output_tensor_name = address_plan.operator_io_to_tensor.get(output_key)
        if output_tensor_name is None:
            continue
        output_assignment = address_plan.assignments.get(output_tensor_name)
        if output_assignment is None:
            continue
        for slice_id, _slice_base_addr in sorted(output_assignment.per_slice_addresses.items()):
            source_slice_id = _resolve_output_base_addr_source_slice(
                op_type=op.op_type,
                output_type=op.output.special_type,
                write_slice_id=slice_id,
            )
            if source_slice_id not in output_assignment.per_slice_addresses:
                raise ValueError(
                    "Output manifest base_addr source slice is not available in assignment: "
                    f"operator={op.op_id}, write_slice={slice_id}, source_slice={source_slice_id}"
                )
            slice_base_addr = output_assignment.per_slice_addresses[source_slice_id]
            slice_dir = _format_slice_dir(slice_id)
            payload[f"{op.op_id}_matrixD_slice{slice_id}"] = {
                "base_addr": _format_hex32(slice_base_addr),
                "path": f"install/{op.op_id}/{slice_dir}/matrix_D_linearized_128bit.bin",
            }

    for op in execution_input.operators:
        template = templates.get(op.op_id)
        if template is None:
            continue
        config_len = int(template.config_length or 0)
        if config_len <= 0:
            continue
        config_base_addr = address_plan.operator_config_base_addresses.get(op.op_id)
        if config_base_addr is None:
            raise ValueError(
                f"Missing planned config address for operator {op.op_id} with config_length={config_len}"
            )

        src_cfg_path = _resolve_config_bitstream_source(op.op_type, template)
        if src_cfg_path is None:
            raise ValueError(
                "Missing source bitstream file for operator type "
                f"{op.op_type}; expected a *bitstream_128b.bin under config/{op.op_type}/"
            )
        dst_cfg_path = cfg_pkg_dir / src_cfg_path.name
        _copy_overwrite_writable(src_cfg_path, dst_cfg_path)
        cfg_path = f"install/cfg_pkg/{src_cfg_path.name}"
        cfg_item = {
            "base_addr": _format_hex32(config_base_addr),
            "path": cfg_path,
        }
        # Keep config address visible with the same flat naming style as matrix entries.
        payload[f"{op.op_id}_config"] = cfg_item

        sfu_type = template.config_sfu_type
        if sfu_type is not None:
            sfu_len = int(template.sfu_config_length or 0)
            if sfu_len <= 0:
                raise ValueError(
                    f"Invalid SFU config length for operator {op.op_id}: sfu_type={sfu_type}, length={sfu_len}"
                )
            sfu_base_addr = address_plan.operator_sfu_config_base_addresses.get(op.op_id)
            if sfu_base_addr is None:
                raise ValueError(
                    f"Missing planned SFU config address for operator {op.op_id} with sfu_type={sfu_type}"
                )
            src_sfu_path = _resolve_sfu_coeff_source(sfu_type)
            dst_sfu_path = cfg_pkg_dir / src_sfu_path.name
            _copy_overwrite_writable(src_sfu_path, dst_sfu_path)
            payload[f"{op.op_id}_sfu_config"] = {
                "base_addr": _format_hex32(sfu_base_addr),
                "path": f"install/cfg_pkg/{src_sfu_path.name}",
            }

    manifest_path = output_dir / "sca_cfg.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def write_input_with_baseaddr(
    input_json_path: str | Path,
    execution_input: ExecutionPlanInput,
    address_plan: AddressPlan,
    output_prefix: str | Path,
) -> Path:
    input_path = Path(input_json_path)
    with input_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Input JSON must be an object at top level")

    raw_ops = raw.get("operators")
    if raw_ops is None:
        raw_ops = raw.get("ops")

    if isinstance(raw_ops, list):
        op_id_to_raw: dict[str, dict] = {}
        for idx, item in enumerate(raw_ops):
            if not isinstance(item, dict):
                continue
            op_id = item.get("id") or item.get("op_id") or f"op_{idx}"
            if isinstance(op_id, str):
                op_id_to_raw[op_id] = item
    elif isinstance(raw_ops, dict):
        op_id_to_raw = {
            str(k): v for k, v in raw_ops.items() if isinstance(v, dict)
        }
    else:
        raise ValueError("Input JSON must contain operators as list or object")

    for op in execution_input.operators:
        raw_op = op_id_to_raw.get(op.op_id)
        if raw_op is None:
            continue

        raw_inputs = raw_op.get("inputs")
        if not isinstance(raw_inputs, dict):
            raw_inputs = {}
            raw_op["inputs"] = raw_inputs

        for input_name in op.inputs.keys():
            io_key = f"{op.op_id}.input.{input_name}"
            tensor_name = address_plan.operator_io_to_tensor.get(io_key)
            if tensor_name is None:
                continue
            assignment = address_plan.assignments.get(tensor_name)
            if assignment is None:
                continue
            slice0_addr = assignment.per_slice_addresses.get(0, assignment.base_address)

            raw_input_tensor = raw_inputs.get(input_name)
            if not isinstance(raw_input_tensor, dict):
                raw_input_tensor = {}
                raw_inputs[input_name] = raw_input_tensor
            raw_input_tensor["base_addr"] = _format_hex32(slice0_addr)

        output_key = f"{op.op_id}.output.D"
        output_tensor_name = address_plan.operator_io_to_tensor.get(output_key)
        if output_tensor_name is not None:
            output_assignment = address_plan.assignments.get(output_tensor_name)
            if output_assignment is not None:
                output_slice0_addr = output_assignment.per_slice_addresses.get(
                    0,
                    output_assignment.base_address,
                )
                raw_output = raw_op.get("output")
                if not isinstance(raw_output, dict):
                    raw_output = raw_op.get("D")
                if not isinstance(raw_output, dict):
                    raw_output = {}
                    raw_op["output"] = raw_output
                raw_output["base_addr"] = _format_hex32(output_slice0_addr)

    prefix = Path(output_prefix)
    output_dir = prefix if prefix.suffix == "" else prefix.parent / prefix.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_withbaseaddr.json"
    output_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def _resolve_config_bitstream_source(op_type: str, template: OperatorTemplate) -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    op_cfg_dir = project_root / "config" / op_type

    # 1) Try explicit template path first.
    explicit = (template.config_bitstream_path or "").strip()
    if explicit:
        explicit_path = Path(explicit)
        candidates = []
        if explicit_path.is_absolute():
            candidates.append(explicit_path)
        else:
            candidates.append(project_root / explicit_path)
            candidates.append(op_cfg_dir / explicit_path.name)
            if explicit_path.suffix.lower() != ".bin":
                candidates.append(op_cfg_dir / f"{explicit_path.stem}.bin")

        for candidate in candidates:
            if candidate.is_file():
                return candidate

    # 2) Fall back to any 128b bitstream binary in operator folder.
    if op_cfg_dir.is_dir():
        matched = sorted(op_cfg_dir.glob("*bitstream_128b.bin"))
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            preferred = [p for p in matched if op_type in p.name]
            if preferred:
                return sorted(preferred)[0]
            return matched[0]

    return None


def _to_128bit_lines(commands: list[int]) -> list[str]:
    lines: list[str] = []
    idx = 0
    while idx < len(commands):
        first = commands[idx]
        second = commands[idx + 1] if idx + 1 < len(commands) else 0
        # Output order per requirement: second 64-bit first, then first 64-bit.
        lines.append(f"{second:064b}{first:064b}")
        idx += 2
    return lines


def _compute_exec_base_after_allocations(address_plan: AddressPlan) -> int:
    # Track next free byte address from tensor and config allocations.
    max_end_addr = 0

    for assignment in address_plan.assignments.values():
        start_addr = assignment.base_address
        words = ceil(assignment.size_bytes / 16)
        end_addr = start_addr + (words * 16)
        if end_addr > max_end_addr:
            max_end_addr = end_addr

    for op_id, base_addr in address_plan.operator_config_base_addresses.items():
        start_addr = base_addr
        length_rows_64b = int(address_plan.operator_config_lengths.get(op_id, 0) or 0)
        reserved_addr = _align_up(length_rows_64b * 8, 16 * 64)
        end_addr = start_addr + reserved_addr
        if end_addr > max_end_addr:
            max_end_addr = end_addr

    for op_id, base_addr in address_plan.operator_sfu_config_base_addresses.items():
        start_addr = base_addr
        length_rows_64b = int(address_plan.operator_sfu_config_lengths.get(op_id, 0) or 0)
        reserved_addr = _align_up(length_rows_64b * 8, 16 * 64)
        end_addr = start_addr + reserved_addr
        if end_addr > max_end_addr:
            max_end_addr = end_addr

    return _align_up(max_end_addr, 16)


def _extract_operator_id(explanation: str) -> str | None:
    match = re.search(r"operator\s+([^\s,()]+)", explanation)
    if match is None:
        return None
    return match.group(1)


def _format_hex32(value: int) -> str:
    return f"0x{value:08X}"


def _format_hex32_grouped(value: int) -> str:
    return f"0x{(value >> 16) & 0xFFFF:04X}_{value & 0xFFFF:04X}"


def _format_slice_dir(slice_id: int) -> str:
    return f"slice{slice_id:02d}"


def _resolve_sfu_coeff_source(sfu_type: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "config" / "SFU_Coeff" / f"{sfu_type}.txt"
    if not file_path.is_file():
        raise ValueError(f"Missing SFU coeff file for type {sfu_type}: {file_path}")
    return file_path


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _copy_overwrite_writable(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.chmod(0o666)
        except OSError:
            pass
        try:
            dst.unlink()
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot overwrite destination file (possibly opened by another process): {dst}"
            ) from exc

    shutil.copyfile(src, dst)

    # Ensure future runs can overwrite this file on Windows.


def _resolve_output_base_addr_source_slice(
    op_type: str,
    output_type: str | None,
    write_slice_id: int,
) -> int:
    if output_type is None:
        return write_slice_id

    router = _OUTPUT_BASE_ADDR_ROUTER_BY_OP_AND_TYPE.get((op_type, output_type))
    if router is None:
        router = _OUTPUT_BASE_ADDR_ROUTER_BY_TYPE.get(output_type)
    if router is None:
        return write_slice_id

    mapped = router(write_slice_id)
    if not isinstance(mapped, int) or mapped < 0:
        raise ValueError(
            f"Invalid output base_addr source slice mapping: op_type={op_type}, "
            f"output_type={output_type}, write_slice={write_slice_id}, mapped={mapped}"
        )
    return mapped
    try:
        dst.chmod(0o666)
    except OSError:
        pass


