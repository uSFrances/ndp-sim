from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterable

from .errors import InvalidRegisterWritePlanError
from .models import AddressPlan, ExecutionPlanArtifact, ExecutionPlanInput, InputSourceType, OperatorTemplate
from .register_mapping import (
    PartialRegisterWrite,
    RegisterMappingDB,
    build_masked_register_writes,
    load_register_mapping,
)
from .slice_routing import resolve_io_base_addr_source_slice


@dataclass(frozen=True)
class BaseAddressRegisterMap:
    """Storage-register mapping for operator IO base-address programming.

    Inputs A/B/B'/C are read streams, output D is write stream.
    """

    input_a_instance: str = "rd_stream0"
    input_b_instance: str = "rd_stream1"
    input_bp_instance: str = "rd_stream2"
    input_c_instance: str = "rd_stream3"
    output_d_instance: str = "wr_stream0"


class WriteRegEncoder:
    OPCODE_WRITE_REG = 0b100

    @classmethod
    def encode(cls, write_value: int, write_addr: int, slice_id: int) -> int:
        if not (0 <= write_value < (1 << 32)):
            raise ValueError(f"write_value out of 32-bit range: {write_value}")
        if not (0 <= write_addr < (1 << 14)):
            raise ValueError(f"write_addr out of 14-bit range: {write_addr}")
        if not (0 <= slice_id < (1 << 5)):
            raise ValueError(f"slice_id out of 5-bit range: {slice_id}")

        return (
            (write_value << 32)
            | (write_addr << 18)
            | (0 << 8)
            | (slice_id << 3)
            | cls.OPCODE_WRITE_REG
        )


class LoadConfigEncoder:
    OPCODE_LOAD_CONFIG = 0b000

    @classmethod
    def encode(
        cls,
        config_length: int,
        ddr_config_addr: int,
        config_sfu: bool,
        slice_mask: int,
    ) -> int:
        if not (0 <= config_length < (1 << 8)):
            raise ValueError(f"config_length out of 8-bit range: {config_length}")
        if not (0 <= ddr_config_addr < (1 << 22)):
            raise ValueError(f"ddr_config_addr out of 22-bit range: {ddr_config_addr}")
        if not (0 <= slice_mask < (1 << 28)):
            raise ValueError(f"slice_mask out of 28-bit range: {slice_mask}")

        return (
            (config_length << 56)
            | (ddr_config_addr << 34)
            | (0 << 32)
            | (int(config_sfu) << 31)
            | (slice_mask << 3)
            | cls.OPCODE_LOAD_CONFIG
        )


class StartCompEncoder:
    OPCODE_START_COMP = 0b101

    @classmethod
    def encode(cls, slice_mask: int) -> int:
        if not (0 <= slice_mask < (1 << 28)):
            raise ValueError(f"slice_mask out of 28-bit range: {slice_mask}")
        return (slice_mask << 3) | cls.OPCODE_START_COMP


class ClockEnableEncoder:
    OPCODE_CLOCK_ENABLE = 0b001
    CLOCK_SELECT_ALL = 0b1111

    @classmethod
    def encode(cls, slice_mask: int, clock_select: int = CLOCK_SELECT_ALL) -> int:
        if not (0 <= slice_mask < (1 << 28)):
            raise ValueError(f"slice_mask out of 28-bit range: {slice_mask}")
        if not (0 <= clock_select < (1 << 4)):
            raise ValueError(f"clock_select out of 4-bit range: {clock_select}")
        return (clock_select << 31) | (slice_mask << 3) | cls.OPCODE_CLOCK_ENABLE


class InstructionGenerator:
    """Builds Load_Config, Write_Reg and Start_Comp command streams.

    TODO:
    - generate Load_Config instruction per operator type
    - merge writes with decoded original register values from config stream
    - generate Start_Comp instruction after each operator setup
    """

    def __init__(
        self,
        reg_map: BaseAddressRegisterMap | None = None,
        register_db: RegisterMappingDB | None = None,
    ) -> None:
        self._reg_map = reg_map or BaseAddressRegisterMap()
        self._register_db = register_db or self._load_default_register_db(
            reorder_const_fields=True,
            map_const_fields_to_const_addresses=True,
        )
        # Original-register extraction intentionally ignores const0/1/2 special
        # address remapping and follows normal cumulative Xbit ordering.
        self._original_value_register_db = self._load_default_register_db(
            reorder_const_fields=False,
            map_const_fields_to_const_addresses=False,
        )

        # Extension points for IO base_addr special handling:
        # 1) per-op+type router, 2) global type router.
        self._io_base_addr_router_by_op_and_type: dict[tuple[str, str], Callable[[int], int]] = {}
        self._io_base_addr_router_by_type: dict[str, Callable[[int], int]] = {}

    def generate(
        self,
        execution_input: ExecutionPlanInput,
        address_plan: AddressPlan,
        templates: dict[str, OperatorTemplate],
    ) -> ExecutionPlanArtifact:
        commands: list[int] = []
        command_explanations: list[str] = []
        load_config_count = 0
        write_reg_count = 0
        control_write_count = 0
        start_comp_count = 0
        clock_enable_count = 0
        skipped_unchanged_write_count = 0
        unresolved_control_names: set[str] = set()
        skipped_control_ops = 0

        # Clock_Enable is emitted once per run before all operator-local commands.
        global_slice_mask = 0
        for op in execution_input.operators:
            global_slice_mask |= op.used_slices
        clock_enable_cmd = ClockEnableEncoder.encode(slice_mask=global_slice_mask)
        commands.append(clock_enable_cmd)
        command_explanations.append(
            "Clock_Enable (global, once per run): "
            f"clock_select_bin={ClockEnableEncoder.CLOCK_SELECT_ALL:04b}, "
            f"slice_mask_bin={global_slice_mask:028b}"
        )
        clock_enable_count += 1

        for op in execution_input.operators:
            enabled_slice_ids = op.enabled_slice_ids()
            slice_mask = op.used_slices
            template = templates.get(op.op_id, OperatorTemplate(op_type=op.op_type))
            config_len = int(template.config_length or 0)
            config_base_addr = address_plan.operator_config_base_addresses.get(op.op_id)
            if config_len > 0 and config_base_addr is None:
                raise ValueError(
                    f"Missing planned config address for operator {op.op_id} with config_length={config_len}"
                )
            ddr_config_addr = 0
            if config_base_addr is not None:
                ddr_config_addr = self._pack_load_config_ddr_addr(config_base_addr)
            load_config_cmd = LoadConfigEncoder.encode(
                config_length=config_len,
                ddr_config_addr=ddr_config_addr,
                config_sfu=False,
                slice_mask=slice_mask,
            )
            commands.append(load_config_cmd)
            command_explanations.append(
                f"Load_Config for operator {op.op_id} ({op.op_type}): "
                f"config_length_bin={config_len:08b}, "
                f"ddr_config_addr_bin={ddr_config_addr:022b}, "
                f"config_sfu_bin=0, slice_mask_bin={slice_mask:028b}"
            )
            load_config_count += 1

            sfu_type = template.config_sfu_type
            if sfu_type is not None:
                sfu_len = int(template.sfu_config_length or 0)
                sfu_base_addr = address_plan.operator_sfu_config_base_addresses.get(op.op_id)
                if sfu_len > 0 and sfu_base_addr is None:
                    raise ValueError(
                        f"Missing planned SFU config address for operator {op.op_id} with sfu_config_length={sfu_len}"
                    )
                sfu_ddr_config_addr = 0
                if sfu_base_addr is not None:
                    sfu_ddr_config_addr = self._pack_load_config_ddr_addr(sfu_base_addr)
                load_sfu_cmd = LoadConfigEncoder.encode(
                    config_length=sfu_len,
                    ddr_config_addr=sfu_ddr_config_addr,
                    config_sfu=True,
                    slice_mask=slice_mask,
                )
                commands.append(load_sfu_cmd)
                command_explanations.append(
                    f"Load_Config SFU for operator {op.op_id} ({op.op_type}), sfu_type={sfu_type}: "
                    f"config_length_bin={sfu_len:08b}, "
                    f"ddr_config_addr_bin={sfu_ddr_config_addr:022b}, "
                    f"config_sfu_bin=1, slice_mask_bin={slice_mask:028b}"
                )
                load_config_count += 1

            # Keep operator-local order A/B/B'/C then D for deterministic programming sequence.
            for input_name in ("A", "B", "B'", "C"):
                if input_name not in op.inputs:
                    continue
                io_key = f"{op.op_id}.input.{input_name}"
                tensor_name = address_plan.operator_io_to_tensor.get(io_key)
                if tensor_name is None:
                    continue
                assignment = address_plan.assignments[tensor_name]
                reg_field_key = self._base_addr_field_key_for_input(input_name)
                base_seed_value = self._resolve_base_addr_seed_value(
                    field_key=reg_field_key,
                    assignment_base_address=assignment.base_address,
                    template=template,
                    force_assignment_base=True,
                )
                field_original_value = self._extract_field_value_from_original(
                    field_key=reg_field_key,
                    original_register_values=template.original_register_values,
                )
                if field_original_value is None:
                    field_original_value = 0
                for slice_id in enabled_slice_ids:
                    input_spec = op.inputs[input_name]
                    effective_addr_slice_id = resolve_io_base_addr_source_slice(
                        op_type=op.op_type,
                        io_type=input_spec.special_type,
                        write_slice_id=slice_id,
                        io_role="input",
                        io_name=input_name,
                        router_by_op_and_type=self._io_base_addr_router_by_op_and_type,
                        router_by_type=self._io_base_addr_router_by_type,
                    )
                    slice_base_addr = self._compose_slice_specific_base_addr(
                        field_key=reg_field_key,
                        base_seed_value=base_seed_value,
                        slice_id=effective_addr_slice_id,
                    )
                    reg_writes = self._expand_field_to_register_writes(
                        field_key=reg_field_key,
                        field_value=slice_base_addr,
                        original_register_values=template.original_register_values,
                    )
                    self._validate_register_targets_enabled(
                        template=template,
                        reg_addresses=reg_writes.keys(),
                        op_id=op.op_id,
                        field_key=reg_field_key,
                    )
                    for reg_addr, write_value in reg_writes.items():
                        original_value = template.original_register_values.get(reg_addr, 0)
                        if original_value == write_value:
                            skipped_unchanged_write_count += 1
                            continue
                        commands.append(
                            WriteRegEncoder.encode(
                                write_value=write_value,
                                write_addr=reg_addr,
                                slice_id=slice_id,
                            )
                        )
                        command_explanations.append(
                            f"Write_Reg base address for operator {op.op_id} input {input_name}, "
                            f"register_field={reg_field_key}, slice_bin={slice_id:05b}, "
                            f"reg_addr_bin={reg_addr:014b}, "
                            f"original_value_bin={original_value:032b}, "
                            f"write_value_bin={write_value:032b}, "
                            f"field_value_original_bin={field_original_value:032b}, "
                            f"field_value_write_bin={slice_base_addr:032b}, "
                            f"original_value_hex=0x{original_value:08X}, "
                            f"write_value_hex=0x{write_value:08X}, "
                            f"field_value_original_hex=0x{field_original_value:08X}, "
                            f"field_value_write_hex=0x{slice_base_addr:08X}"
                        )
                        write_reg_count += 1

            output_key = f"{op.op_id}.output.D"
            output_tensor_name = address_plan.operator_io_to_tensor.get(output_key)
            if output_tensor_name is not None:
                output_assignment = address_plan.assignments[output_tensor_name]
                output_field_key = self._base_addr_field_key_for_output_d()
                output_base_seed_value = self._resolve_base_addr_seed_value(
                    field_key=output_field_key,
                    assignment_base_address=output_assignment.base_address,
                    template=template,
                    force_assignment_base=True,
                )
                output_field_original_value = self._extract_field_value_from_original(
                    field_key=output_field_key,
                    original_register_values=template.original_register_values,
                )
                if output_field_original_value is None:
                    output_field_original_value = 0
                enabled_slice_set = set(enabled_slice_ids)
                for slice_id in enabled_slice_ids:
                    effective_addr_slice_id = resolve_io_base_addr_source_slice(
                        op_type=op.op_type,
                        io_type=op.output.special_type,
                        write_slice_id=slice_id,
                        io_role="output",
                        io_name="D",
                        router_by_op_and_type=self._io_base_addr_router_by_op_and_type,
                        router_by_type=self._io_base_addr_router_by_type,
                    )
                    if effective_addr_slice_id not in enabled_slice_set:
                        raise ValueError(
                            "Output base_addr source slice is not enabled by used_slices: "
                            f"operator={op.op_id}, write_slice={slice_id}, source_slice={effective_addr_slice_id}, "
                            f"used_slices_bin={slice_mask:028b}"
                        )
                    slice_base_addr = self._compose_slice_specific_base_addr(
                        field_key=output_field_key,
                        base_seed_value=output_base_seed_value,
                        slice_id=effective_addr_slice_id,
                    )
                    output_reg_writes = self._expand_field_to_register_writes(
                        field_key=output_field_key,
                        field_value=slice_base_addr,
                        original_register_values=template.original_register_values,
                    )
                    self._validate_register_targets_enabled(
                        template=template,
                        reg_addresses=output_reg_writes.keys(),
                        op_id=op.op_id,
                        field_key=output_field_key,
                    )
                    for reg_addr, write_value in output_reg_writes.items():
                        original_value = template.original_register_values.get(reg_addr, 0)
                        if original_value == write_value:
                            skipped_unchanged_write_count += 1
                            continue
                        commands.append(
                            WriteRegEncoder.encode(
                                write_value=write_value,
                                write_addr=reg_addr,
                                slice_id=slice_id,
                            )
                        )
                        command_explanations.append(
                            f"Write_Reg base address for operator {op.op_id} output D, "
                            f"register_field={output_field_key}, slice_bin={slice_id:05b}, "
                            f"source_slice_bin={effective_addr_slice_id:05b}, "
                            f"reg_addr_bin={reg_addr:014b}, "
                            f"original_value_bin={original_value:032b}, "
                            f"write_value_bin={write_value:032b}, "
                            f"field_value_original_bin={output_field_original_value:032b}, "
                            f"field_value_write_bin={slice_base_addr:032b}, "
                            f"original_value_hex=0x{original_value:08X}, "
                            f"write_value_hex=0x{write_value:08X}, "
                            f"field_value_original_hex=0x{output_field_original_value:08X}, "
                            f"field_value_write_hex=0x{slice_base_addr:08X}"
                        )
                        write_reg_count += 1

            if not template.should_update_control_registers:
                skipped_control_ops += 1
            else:
                for reg_name, reg_value in template.control_register_values.items():
                    resolved_reg_name = self._resolve_control_register_field_key(reg_name)
                    if resolved_reg_name is None:
                        unresolved_control_names.add(reg_name)
                        continue
                    # Base-address fields are already emitted in IO base-address stage
                    # with per-slice slave replacement; skip duplicated control writes.
                    if self._is_base_addr_field_key(resolved_reg_name):
                        continue
                    effective_original_values = self._build_effective_original_values_for_field(
                        field_key=resolved_reg_name,
                        original_register_values=template.original_register_values,
                    )
                    reg_writes = self._expand_field_to_register_writes(
                        resolved_reg_name,
                        reg_value,
                        effective_original_values,
                    )
                    field_original_value = self._extract_field_value_from_original(
                        field_key=resolved_reg_name,
                        original_register_values=template.original_register_values,
                    )
                    if field_original_value is None:
                        field_original_value = 0
                    if not reg_writes:
                        unresolved_control_names.add(reg_name)
                        continue
                    self._validate_register_targets_enabled(
                        template=template,
                        reg_addresses=reg_writes.keys(),
                        op_id=op.op_id,
                        field_key=resolved_reg_name,
                    )
                    for slice_id in enabled_slice_ids:
                        for reg_addr, write_value in reg_writes.items():
                            original_value = effective_original_values.get(reg_addr, 0)
                            if original_value == write_value:
                                skipped_unchanged_write_count += 1
                                continue
                            commands.append(
                                WriteRegEncoder.encode(
                                    write_value=write_value,
                                    write_addr=reg_addr,
                                    slice_id=slice_id,
                                )
                            )
                            command_explanations.append(
                                f"Write_Reg control field for operator {op.op_id}, register_field={resolved_reg_name}, "
                                f"slice_bin={slice_id:05b}, reg_addr_bin={reg_addr:014b}, "
                                f"original_value_bin={original_value:032b}, "
                                f"write_value_bin={write_value:032b}, "
                                f"field_value_original_bin={field_original_value:032b}, "
                                f"field_value_write_bin={reg_value:032b}, "
                                f"original_value_hex=0x{original_value:08X}, "
                                f"write_value_hex=0x{write_value:08X}, "
                                f"field_value_original_hex=0x{field_original_value:08X}, "
                                f"field_value_write_hex=0x{reg_value:08X}"
                            )
                            write_reg_count += 1
                            control_write_count += 1

            start_comp_cmd = StartCompEncoder.encode(slice_mask)
            commands.append(start_comp_cmd)
            command_explanations.append(
                f"Start_Comp for operator {op.op_id} ({op.op_type}): slice_mask_bin={slice_mask:028b}"
            )
            start_comp_count += 1

        return ExecutionPlanArtifact(
            commands=commands,
            command_explanations=command_explanations,
            metadata={
                "load_config_count": str(load_config_count),
                "clock_enable_count": str(clock_enable_count),
                "write_reg_count": str(write_reg_count),
                "skipped_unchanged_write_count": str(skipped_unchanged_write_count),
                "control_write_count": str(control_write_count),
                "start_comp_count": str(start_comp_count),
                "skipped_control_ops_same_size": str(skipped_control_ops),
                "parsed_register_fields": str(len(self._register_db.field_bindings)),
                "parsed_register_addresses": str(len(self._register_db.address_to_fields)),
                "unresolved_control_names": ",".join(sorted(unresolved_control_names)),
                "note": (
                    "One global Clock_Enable is emitted before all operators; each operator then emits "
                    "Load_Config, Write_Reg updates, and Start_Comp;"
                    " control-register writes are emitted only when template initial_size differs"
                    " from target size."
                ),
            },
        )

    def _base_addr_field_key_for_input(self, input_name: str) -> str:
        if input_name == "A":
            return f"{self._reg_map.input_a_instance}.stream_engine.stream.base_addr"
        if input_name == "B":
            return f"{self._reg_map.input_b_instance}.stream_engine.stream.base_addr"
        if input_name == "B'":
            return f"{self._reg_map.input_bp_instance}.stream_engine.stream.base_addr"
        if input_name == "C":
            return f"{self._reg_map.input_c_instance}.stream_engine.stream.base_addr"
        raise ValueError(f"Unsupported input name: {input_name}")

    def _base_addr_field_key_for_output_d(self) -> str:
        return f"{self._reg_map.output_d_instance}.stream_engine.stream.base_addr"

    def _is_base_addr_field_key(self, field_key: str) -> bool:
        return field_key.endswith(".stream_engine.stream.base_addr")

    def _pack_load_config_ddr_addr(self, full_addr: int) -> int:
        # Load_Config uses DDR config address in row-aligned compressed form.
        # Address planner emits full address {slave,bank,row,col,subword};
        # drop low 10 bits (col+subword) for compressed field.
        return full_addr >> 10

    def _resolve_base_addr_seed_value(
        self,
        field_key: str,
        assignment_base_address: int,
        template: OperatorTemplate,
        force_assignment_base: bool = False,
    ) -> int:
        # For operator-to-operator edges, input must point to planned producer output
        # address, not to template's decoded original base_addr.
        if force_assignment_base:
            return assignment_base_address

        # Priority 1: explicit base_addr override in control_registers.py
        explicit = template.control_register_values.get(field_key)
        if isinstance(explicit, int):
            return explicit

        # Priority 2: original base_addr decoded from template bitstream
        original = self._extract_field_value_from_original(
            field_key=field_key,
            original_register_values=template.original_register_values,
        )
        if original is not None:
            return original

        # Priority 3: planner fallback when template has no decoded original value
        return assignment_base_address

    def _compose_slice_specific_base_addr(
        self,
        field_key: str,
        base_seed_value: int,
        slice_id: int,
    ) -> int:
        binding = self._register_db.get_field(field_key)
        if binding is None:
            raise KeyError(f"Field not found in register mapping: {field_key}")
        width = binding.field_high - binding.field_low + 1
        if width < 5:
            raise ValueError(f"base_addr field width must be >= 5 bits, got {width}: {field_key}")

        tail_bits = width - 5
        tail_mask = (1 << tail_bits) - 1
        return (slice_id << tail_bits) | (base_seed_value & tail_mask)

    def _extract_field_value_from_original(
        self,
        field_key: str,
        original_register_values: dict[int, int],
    ) -> int | None:
        binding = self._original_value_register_db.get_field(field_key)
        if binding is None:
            return None

        value = 0
        seen_any = False
        for seg in binding.segments:
            overlap_low = max(seg.low, binding.field_low)
            overlap_high = min(seg.high, binding.field_high)
            if overlap_low > overlap_high:
                continue

            chunk_width = overlap_high - overlap_low + 1
            segment_offset = overlap_low - seg.low
            field_offset = overlap_low - binding.field_low

            reg_word = original_register_values.get(seg.address)
            if reg_word is None:
                continue
            seen_any = True
            chunk = (reg_word >> segment_offset) & ((1 << chunk_width) - 1)
            value |= chunk << field_offset

        if not seen_any:
            return None
        return value

    def _build_effective_original_values_for_field(
        self,
        field_key: str,
        original_register_values: dict[int, int],
    ) -> dict[int, int]:
        # Start with decoded register words from parsed bitstream.
        effective = dict(original_register_values)

        # Derive field original value using non-reordered mapping, then project it
        # onto write-target register layout (which may use const-address remapping).
        field_original_value = self._extract_field_value_from_original(
            field_key=field_key,
            original_register_values=original_register_values,
        )
        if field_original_value is None:
            return effective

        binding = self._register_db.get_field(field_key)
        if binding is None:
            return effective

        projected = build_masked_register_writes(binding=binding, field_value=field_original_value)
        for reg_addr, partial in projected.items():
            base = effective.get(reg_addr, 0)
            effective[reg_addr] = (base & ~partial.mask) | partial.value

        return effective

    def _resolve_control_register_writes(
        self,
        reg_name: str,
        reg_value: int,
        original_register_values: dict[int, int],
    ) -> dict[int, int]:
        # Support either fully qualified names: <instance>.<config_name>,
        # or config_name-only if it is unique across parsed instances.
        if "." in reg_name:
            return self._expand_field_to_register_writes(reg_name, reg_value, original_register_values)

        matches = [
            key for key in self._register_db.field_bindings.keys() if key.endswith(f".{reg_name}")
        ]
        if len(matches) != 1:
            return {}
        return self._expand_field_to_register_writes(matches[0], reg_value, original_register_values)

    def _resolve_control_register_field_key(self, reg_name: str) -> str | None:
        if "." in reg_name:
            return reg_name

        matches = [
            key for key in self._register_db.field_bindings.keys() if key.endswith(f".{reg_name}")
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    def _expand_field_to_register_writes(
        self,
        field_key: str,
        field_value: int,
        original_register_values: dict[int, int],
    ) -> dict[int, int]:
        binding = self._register_db.get_field(field_key)
        if binding is None:
            raise KeyError(f"Field not found in register mapping: {field_key}")
        partial_writes = build_masked_register_writes(binding=binding, field_value=field_value)
        return self._merge_partial_writes(partial_writes, original_register_values)

    def _merge_partial_writes(
        self,
        partial_writes: dict[int, PartialRegisterWrite],
        original_register_values: dict[int, int],
    ) -> dict[int, int]:
        merged: dict[int, int] = {}
        for reg_addr, partial in partial_writes.items():
            original = original_register_values.get(reg_addr, 0)
            merged[reg_addr] = (original & ~partial.mask) | partial.value
        return merged

    def _validate_register_targets_enabled(
        self,
        template: OperatorTemplate,
        reg_addresses: Iterable[int],
        op_id: str,
        field_key: str,
    ) -> None:
        if not template.enabled_register_addresses:
            return

        missing = sorted(
            reg_addr
            for reg_addr in reg_addresses
            if reg_addr not in template.enabled_register_addresses
        )
        if not missing:
            return

        missing_bits = ", ".join(f"{addr:014b}" for addr in missing)
        raise InvalidRegisterWritePlanError(
            "Invalid register write plan: target register is not enabled in source bitstream "
            f"(chunk leading bit is 0). op={op_id}, field={field_key}, reg_addr_bin=[{missing_bits}]"
        )

    def _load_default_register_db(
        self,
        reorder_const_fields: bool,
        map_const_fields_to_const_addresses: bool,
    ) -> RegisterMappingDB:
        root = Path(__file__).resolve().parents[2]
        return load_register_mapping(
            register_map_csv=root / "config" / "register_map_with_groups1.csv",
            config_output_csv=root / "config" / "config_output.csv",
            reorder_const_fields=reorder_const_fields,
            map_const_fields_to_const_addresses=map_const_fields_to_const_addresses,
        )

