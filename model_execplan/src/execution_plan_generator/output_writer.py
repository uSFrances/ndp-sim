from __future__ import annotations

import json
import re
import shutil
import struct
from math import ceil
from pathlib import Path

from .models import (
    AddressPlan,
    ExecutionPlanArtifact,
    ExecutionPlanInput,
    InputSourceType,
    OperatorSpec,
    OperatorTemplate,
)
from .control_registers import compute_control_register_updates
from .slice_routing import resolve_io_base_addr_source_slice


def write_emulator_bundle(
    execution_input: ExecutionPlanInput,
    address_plan: AddressPlan,
    output_prefix: str | Path,
    emulator_suffix: str | None = None,
) -> list[Path]:
    prefix = Path(output_prefix)
    output_dir = prefix if prefix.suffix == "" else prefix.parent / prefix.stem
    emulator_name = "emulator"
    if emulator_suffix:
        emulator_name = f"{emulator_name}_{emulator_suffix}"
    emulator_dir = output_dir / emulator_name
    emulator_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    op_json_root = repo_root / "jsons"
    written_paths: list[Path] = []

    for op in execution_input.operators:
        op_dir = emulator_dir / op.op_id
        if op_dir.exists():
            shutil.rmtree(op_dir)
        op_dir.mkdir(parents=True, exist_ok=True)

        source_json = op_json_root / f"{op.op_type}.json"
        if not source_json.is_file():
            raise FileNotFoundError(
                f"Missing operator JSON template for emulator export: {source_json}"
            )
        op_payload = _load_json_object(source_json)
        _patch_emulator_operator_json_payload(
            payload=op_payload,
            operator=op,
            address_plan=address_plan,
        )

        for slice_id in op.enabled_slice_ids():
            slice_dir_name = f"slice{slice_id:02d}"
            emulator_slice_dir = op_dir / slice_dir_name
            emulator_slice_dir.mkdir(parents=True, exist_ok=True)

            target_json = emulator_slice_dir / source_json.name
            target_json.write_text(
                json.dumps(op_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written_paths.append(target_json)

            install_slice_dir = output_dir / "install" / op.op_id / slice_dir_name
            data_parts: list[bytes] = []
            for input_name in op.inputs.keys():
                if input_name == "B'":
                    continue
                data_file = install_slice_dir / f"matrix_{input_name}_linearized_128bit.bin"
                if not data_file.is_file():
                    raise FileNotFoundError(
                        f"Missing slice input data for emulator export: {data_file}"
                    )
                data_parts.append(data_file.read_bytes())

            data_path = emulator_slice_dir / "dram_data.bin"
            data_path.write_bytes(b"".join(data_parts))
            written_paths.append(data_path)

    return written_paths


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _patch_emulator_operator_json_payload(
    payload: dict[str, object],
    operator: OperatorSpec,
    address_plan: AddressPlan,
) -> None:
    _patch_stream_base_addrs(payload=payload, operator=operator, address_plan=address_plan)

    # Apply shape-driven control rules for this operator type.
    updates = compute_control_register_updates(
        operator=operator,
        template=OperatorTemplate(op_type=operator.op_type),
        apply_instance_mapping=False,
    )
    for field_key, value in updates.items():
        _apply_control_update_to_operator_json(
            payload=payload,
            operator=operator,
            field_key=field_key,
            value=value,
        )


def _patch_stream_base_addrs(
    payload: dict[str, object],
    operator: OperatorSpec,
    address_plan: AddressPlan,
) -> None:
    stream_engine = payload.get("stream_engine")
    if not isinstance(stream_engine, dict):
        return

    local_base_addrs = _compute_operator_local_base_addrs(operator=operator, address_plan=address_plan)
    for target_name, base_addr in local_base_addrs.items():
        mode = "write" if target_name == "D" else "read"
        stream_key = _find_stream_key_by_target(
            stream_engine=stream_engine,
            target=target_name,
            mode=mode,
        )
        if stream_key is None:
            continue
        stream_node = stream_engine.get(stream_key)
        if isinstance(stream_node, dict):
            stream_node["base_addr"] = _format_base_addr_hex(base_addr)


def _compute_operator_local_base_addrs(
    operator: OperatorSpec,
    address_plan: AddressPlan,
) -> dict[str, int]:
    local_base_addrs: dict[str, int] = {}
    cursor = 0

    def _aligned_size_bytes(tensor_name: str | None) -> int | None:
        if tensor_name is None:
            return None
        assignment = address_plan.assignments.get(tensor_name)
        if assignment is None:
            return None
        return _align_up(assignment.size_bytes, 16)

    for input_name in operator.inputs.keys():
        if input_name == "B'":
            if "B" in local_base_addrs:
                local_base_addrs[input_name] = local_base_addrs["B"]
            continue

        io_key = f"{operator.op_id}.input.{input_name}"
        tensor_name = address_plan.operator_io_to_tensor.get(io_key)
        aligned_size = _aligned_size_bytes(tensor_name)
        if aligned_size is None:
            continue

        local_base_addrs[input_name] = cursor
        cursor += aligned_size

    output_key = f"{operator.op_id}.output.D"
    output_tensor_name = address_plan.operator_io_to_tensor.get(output_key)
    output_aligned_size = _aligned_size_bytes(output_tensor_name)
    if output_aligned_size is not None:
        local_base_addrs["D"] = cursor

    return local_base_addrs


def _apply_control_update_to_operator_json(
    payload: dict[str, object],
    operator: OperatorSpec,
    field_key: str,
    value: int,
) -> None:
    if "." not in field_key:
        return
    instance, config_path = field_key.split(".", maxsplit=1)

    if instance.startswith("iga_lc") and config_path == "dram_loop_configs.end":
        lc_idx = _parse_instance_index(instance, prefix="iga_lc")
        if lc_idx is None:
            return
        dram_loop_configs = payload.get("dram_loop_configs")
        if not isinstance(dram_loop_configs, dict):
            return
        lc_key = f"LC{lc_idx}"
        lc_node = dram_loop_configs.get(lc_key)
        if isinstance(lc_node, dict):
            lc_node["end"] = value
        return

    if instance.startswith("iga_pe") and config_path == "lc_pe_configs.inport1.constant":
        pe_idx = _parse_instance_index(instance, prefix="iga_pe")
        if pe_idx is None:
            return
        pe_configs = payload.get("lc_pe_configs")
        if not isinstance(pe_configs, dict):
            return
        pe_key = f"PE{pe_idx}"
        pe_node = pe_configs.get(pe_key)
        if not isinstance(pe_node, dict):
            return
        inport1 = pe_node.get("inport1")
        if isinstance(inport1, dict):
            inport1["constant"] = value
        return

    if instance.startswith("ga_pe") and config_path == "general_array.PE_array.PE.inport1.constant":
        pe_idx = _parse_instance_index(instance, prefix="ga_pe")
        if pe_idx is None:
            return
        general_array = payload.get("general_array")
        if not isinstance(general_array, dict):
            return
        pe_array = general_array.get("PE_array")
        if not isinstance(pe_array, dict):
            return
        pe_key = _resolve_ga_pe_array_key(pe_array=pe_array, pe_idx=pe_idx)
        if pe_key is None:
            return
        pe_node = pe_array.get(pe_key)
        if not isinstance(pe_node, dict):
            return
        inport1 = pe_node.get("inport1")
        if isinstance(inport1, dict):
            # Keep symbolic form in emulator json for readability/debugging.
            # Derive expression from encoded fp32 value instead of assuming a_k.
            inport1["constant"] = _format_fp32_bits_as_symbolic(value)
        return

    if instance.startswith("rd_stream") or instance.startswith("wr_stream"):
        if config_path != "stream_engine.stream.dim_stride":
            return
        stream_engine = payload.get("stream_engine")
        if not isinstance(stream_engine, dict):
            return
        stream_key = _resolve_stream_key_for_instance(stream_engine, instance)
        if stream_key is None:
            return
        stream_node = stream_engine.get(stream_key)
        if not isinstance(stream_node, dict):
            return
        stream_node["dim_stride"] = _decode_packed_dim_stride(value)


def _parse_instance_index(instance: str, prefix: str) -> int | None:
    if not instance.startswith(prefix):
        return None
    suffix = instance[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _resolve_ga_pe_array_key(pe_array: dict[str, object], pe_idx: int) -> str | None:
    # If PE keys use 2D naming (PErc), map linear index in row-major order.
    grid_keys: list[str] = []
    for key in pe_array.keys():
        if re.fullmatch(r"PE\d\d", key):
            grid_keys.append(key)
    if grid_keys:
        cols = max(int(key[3]) for key in grid_keys) + 1
        row = pe_idx // cols
        col = pe_idx % cols
        grid_key = f"PE{row}{col}"
        if grid_key in pe_array:
            return grid_key

    direct_key = f"PE{pe_idx}"
    if direct_key in pe_array:
        return direct_key

    padded_key = f"PE{pe_idx:02d}"
    if padded_key in pe_array:
        return padded_key

    # Compat path: condensed logical index peRC -> physical PE(2R)(2C).
    if 0 <= pe_idx < 100:
        logical_row = pe_idx // 10
        logical_col = pe_idx % 10
        doubled_key = f"PE{logical_row * 2}{logical_col * 2}"
        if doubled_key in pe_array:
            return doubled_key

    return None


def _decode_fp32_bits(value: int) -> float:
    packed = int(value & 0xFFFF_FFFF).to_bytes(4, byteorder="big", signed=False)
    return float(struct.unpack(">f", packed)[0])


def _format_fp32_bits_as_symbolic(value_bits: int) -> str:
    fp_value = _decode_fp32_bits(value_bits)
    if fp_value == 0.0:
        return "0.0"

    inv = 1.0 / fp_value
    inv_rounded = int(round(inv))
    tolerance = max(1e-6, abs(inv) * 1e-6)
    if inv_rounded != 0 and abs(inv - inv_rounded) <= tolerance:
        sign = "-" if inv_rounded < 0 else ""
        return f"{sign}1.0 / {abs(inv_rounded)}"

    return f"{fp_value:.16g}"


def _find_stream_key_by_target(
    stream_engine: dict[str, object],
    target: str,
    mode: str,
) -> str | None:
    for stream_key, stream_node in stream_engine.items():
        if not isinstance(stream_node, dict):
            continue
        if stream_node.get("target") == target and stream_node.get("mode") == mode:
            return stream_key
    return None


def _resolve_stream_key_for_instance(
    stream_engine: dict[str, object],
    instance: str,
) -> str | None:
    mode = "read" if instance.startswith("rd_stream") else "write"
    ordinal = 0
    if instance != "wr_stream":
        tail = instance.split("stream", maxsplit=1)[-1]
        if tail.isdigit():
            ordinal = int(tail)

    candidates: list[tuple[int, str]] = []
    for stream_key, stream_node in stream_engine.items():
        if not isinstance(stream_node, dict):
            continue
        if stream_node.get("mode") != mode:
            continue
        if not stream_key.startswith("stream"):
            continue
        idx_text = stream_key[len("stream"):]
        if not idx_text.isdigit():
            continue
        candidates.append((int(idx_text), stream_key))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    if ordinal < len(candidates):
        return candidates[ordinal][1]
    return candidates[0][1]


def _decode_packed_dim_stride(packed: int) -> list[int | None]:
    mask = (1 << 20) - 1
    port0 = packed & mask
    port1 = (packed >> 20) & mask
    port2 = (packed >> 40) & mask
    return [
        port2 if port2 != 0 else None,
        port1 if port1 != 0 else None,
        port0 if port0 != 0 else None,
    ]


def _resolve_slice0_input_base_addr(
    operator: OperatorSpec,
    input_name: str,
    manifest: dict[str, object],
) -> int:
    direct_key = f"{operator.op_id}_matrix{input_name}_slice0"
    addr = _extract_manifest_base_addr(manifest, direct_key)
    if addr is not None:
        return _force_slice0_base_addr(addr)

    input_spec = operator.inputs.get(input_name)
    if input_spec is None or input_spec.source is None or input_spec.source.operator_id is None:
        raise ValueError(
            f"Cannot resolve slice00 base_addr for operator {operator.op_id} input {input_name}"
        )

    source_key = f"{input_spec.source.operator_id}_matrixD_slice0"
    source_addr = _extract_manifest_base_addr(manifest, source_key)
    if source_addr is None:
        raise ValueError(
            f"Cannot resolve source output base_addr from manifest key: {source_key}"
        )
    return _force_slice0_base_addr(source_addr)


def _resolve_slice0_output_base_addr(op_id: str, manifest: dict[str, object]) -> int:
    key = f"{op_id}_matrixD_slice0"
    addr = _extract_manifest_base_addr(manifest, key)
    if addr is None:
        raise ValueError(f"Cannot resolve slice00 output base_addr from manifest key: {key}")
    return _force_slice0_base_addr(addr)


def _extract_manifest_base_addr(manifest: dict[str, object], key: str) -> int | None:
    value = manifest.get(key)
    if not isinstance(value, dict):
        return None
    base_addr = value.get("base_addr")
    if not isinstance(base_addr, str):
        return None
    return int(base_addr.replace("_", ""), 16)


def _format_base_addr_binary(addr: int) -> str:
    if not (0 <= addr < (1 << 30)):
        raise ValueError(f"base_addr out of 30-bit range: {addr}")
    bits = f"{addr:030b}"
    return f"0b{bits[0:5]}_{bits[5:7]}_{bits[7:20]}_{bits[20:26]}_{bits[26:30]}"


def _format_base_addr_hex(addr: int) -> str:
    if not (0 <= addr < (1 << 30)):
        raise ValueError(f"base_addr out of 30-bit range: {addr}")
    return f"0x{addr:x}"


def _force_slice0_base_addr(addr: int) -> int:
    # Address format: slave(5), bank(2), row(13), col(6), subword(4).
    # Emulator json always uses slice00 view regardless of current slice.
    return addr & ((1 << 25) - 1)


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

    _write_execplan_binary(artifact.commands, binary_path)

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
                explanation_lines.append(f"")
                explanation_lines.append(f"")
                explanation_lines.append(f"=" * 30 + f" operator {operator_id} " + "=" * 150)
            else:
                explanation_lines.append("===== operator unknown =====")
            current_operator_id = operator_id
        explanation_lines.extend(_format_instruction_explanation_block(idx, command, explanation))
    explanation_path.write_text("\n".join(explanation_lines) + "\n", encoding="utf-8")

    return binary_path, explanation_path


def write_instruction_op_outputs(
    artifact: ExecutionPlanArtifact,
    output_prefix: str | Path,
) -> list[Path]:
    prefix = Path(output_prefix)
    output_dir = prefix if prefix.suffix == "" else prefix.parent / prefix.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    install_dir = output_dir / "install"
    install_dir.mkdir(parents=True, exist_ok=True)

    op_blocks = _collect_instruction_op_command_blocks(artifact)
    written_paths: list[Path] = []
    for op_id, commands in op_blocks:
        output_path = install_dir / f"execplan_{op_id}.txt"
        _write_execplan_binary(commands, output_path)
        written_paths.append(output_path)

    return written_paths


def _collect_instruction_op_command_blocks(artifact: ExecutionPlanArtifact) -> list[tuple[str, list[int]]]:
    preamble_commands: list[int] = []
    blocks: list[tuple[str, list[int]]] = []
    current_operator_id: str | None = None
    current_commands: list[int] = []

    for idx, command in enumerate(artifact.commands):
        explanation = ""
        if idx < len(artifact.command_explanations):
            explanation = artifact.command_explanations[idx]
        operator_id = _extract_operator_id(explanation)

        if operator_id is None:
            if current_operator_id is None:
                preamble_commands.append(command)
            else:
                current_commands.append(command)
            continue

        if operator_id != current_operator_id:
            if current_operator_id is not None:
                blocks.append((current_operator_id, current_commands))
            current_operator_id = operator_id
            current_commands = list(preamble_commands)

        current_commands.append(command)

    if current_operator_id is not None:
        blocks.append((current_operator_id, current_commands))

    return blocks


def _write_execplan_binary(commands: list[int], output_path: Path) -> None:
    binary_lines = _to_128bit_lines(commands)
    output_path.write_text("\n".join(binary_lines) + "\n", encoding="utf-8")


def _format_instruction_explanation_block(idx: int, command: int, explanation: str) -> list[str]:
    header = f"{idx:04d}  <{command:064b}>"
    if not explanation.strip():
        return [header]

    clauses = [part.strip() for part in explanation.split(",") if part.strip()]
    if not clauses:
        return [header]
    indent = " " * 4
    lines = [f"{header} \n{indent}{clauses[0]},"]
    for clause in clauses[1:-1]:
        lines.append(f"{indent}{clause},")
    if len(clauses) > 1:
        lines.append(f"{indent}{clauses[-1]}")
    return lines


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
            # if input_spec is not None and input_spec.source is not None:
            #     if input_spec.source.source_type == InputSourceType.OPERATOR:
            #         # Input reuses a previous operator output; avoid duplicate manifest entry.
            #         continue
            io_key = f"{op.op_id}.input.{input_name}"
            tensor_name = address_plan.operator_io_to_tensor.get(io_key)
            if tensor_name is None:
                continue
            assignment = address_plan.assignments.get(tensor_name)
            if assignment is None:
                continue

            for slice_id, _slice_base_addr in sorted(assignment.per_slice_addresses.items()):
                source_slice_id = resolve_io_base_addr_source_slice(
                    op_type=op.op_type,
                    io_type=input_spec.special_type if input_spec is not None else None,
                    write_slice_id=slice_id,
                    io_role="input",
                    io_name=input_name,
                )
                if source_slice_id not in assignment.per_slice_addresses:
                    raise ValueError(
                        "Input manifest base_addr source slice is not available in assignment: "
                        f"operator={op.op_id}, input={input_name}, write_slice={slice_id}, "
                        f"source_slice={source_slice_id}"
                    )
                slice_base_addr = assignment.per_slice_addresses[source_slice_id]
                slice_dir = _format_slice_dir(slice_id)
                payload[f"{op.op_id}_matrix{input_name}_slice{slice_id}"] = {
                    "base_addr": _format_hex32(slice_base_addr),
                    "path": f"install/{op.op_id}/{slice_dir}/matrix_{input_name}_linearized_128bit.txt",
                }

        output_key = f"{op.op_id}.output.D"
        output_tensor_name = address_plan.operator_io_to_tensor.get(output_key)
        if output_tensor_name is None:
            continue
        output_assignment = address_plan.assignments.get(output_tensor_name)
        if output_assignment is None:
            continue
        for slice_id, _slice_base_addr in sorted(output_assignment.per_slice_addresses.items()):
            source_slice_id = resolve_io_base_addr_source_slice(
                op_type=op.op_type,
                io_type=op.output.special_type,
                write_slice_id=slice_id,
                io_role="output",
                io_name="D",
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
                "path": f"install/{op.op_id}/{slice_dir}/matrix_D_linearized_128bit.txt",
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
                f"{op.op_type}; expected a *bitstream_128b.txt under config/{op.op_type}/"
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
        matched = sorted(op_cfg_dir.glob("*bitstream_128b.txt"))
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



