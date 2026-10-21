from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .errors import ExecutionPlanError
from .models import AddressAssignment, AddressPlan, ExecutionPlanInput, InputSourceType
from .control_registers import _BP_INDEPENDENT_ADDR_OPS


@dataclass
class _AddressCursor:
    bank: int = 0
    row: int = 0
    col: int = 0


class AddressPlanningError(ExecutionPlanError):
    """Raised when address planning fails due to invalid dependency or memory overflow."""


class AddressPlanner:
    """Assign tensor storage addresses with no-free growth strategy.

    Address format: slave(5), bank(2), row(13), col(6), subword(4).
    The planner allocates monotonically increasing addresses and never reclaims memory.

    When *all* tensors in the plan carry ``bank_interleave > 1``, the planner
    splits the 4 physical banks into ``4 // bank_interleave`` independent groups
    and always picks the least-used group.  Within each group the caller only
    sees the first bank of the group so ``base_addr`` always lands on an even
    bank (0 or 2 for interleave 2).
    """

    WORD_BYTES = 16
    ROW_BYTES = 16 * 64
    MAX_SLAVES = 28
    MAX_BANKS = 4
    MAX_ROWS = 8192
    MAX_COLS = 64

    def __init__(self, element_bytes: int = 4) -> None:
        if element_bytes <= 0:
            raise ValueError("element_bytes must be positive.")
        self._element_bytes = element_bytes
        self._dtype_bytes = {
            "fp16": 2,
            "fp32": 4,
            "int8": 1,
            "uint8": 1,
            "int16": 2,
            "uint16": 2,
            "int32": 4,
            "uint32": 4,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        execution_input: ExecutionPlanInput,
        config_lengths_by_op: dict[str, int] | None = None,
        sfu_config_lengths_by_op: dict[str, int] | None = None,
        sfu_types_by_op: dict[str, str] | None = None,
    ) -> AddressPlan:
        interleave = self._resolve_plan_interleave(execution_input)

        return self._plan_interleaved(
            execution_input,
            interleave=interleave,
            config_lengths_by_op=config_lengths_by_op,
            sfu_config_lengths_by_op=sfu_config_lengths_by_op,
            sfu_types_by_op=sfu_types_by_op,
        )

    # ------------------------------------------------------------------
    # Interleave helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_plan_interleave(execution_input: ExecutionPlanInput) -> int:
        """Return the maximum ``bank_interleave`` across every tensor in the plan."""
        max_val = 1
        for op in execution_input.operators:
            for t in op.inputs.values():
                max_val = max(max_val, t.bank_interleave)
            max_val = max(max_val, op.output.bank_interleave)
        return max_val

    def _group_count(self, interleave: int) -> int:
        return self.MAX_BANKS // interleave

    def _group_start_bank(self, group_idx: int, interleave: int) -> int:
        return group_idx * interleave

    # ------------------------------------------------------------------
    # Interleaved plan (bank_interleave > 1)
    # ------------------------------------------------------------------

    def _plan_interleaved(
        self,
        execution_input: ExecutionPlanInput,
        interleave: int,
        config_lengths_by_op: dict[str, int] | None,
        sfu_config_lengths_by_op: dict[str, int] | None,
        sfu_types_by_op: dict[str, str] | None = None,
    ) -> AddressPlan:
        # interleave=4 plans are split into two 2-bank groups (banks 0/1 and
        # 2/3) so that ilv=2 tensors (GEMV B/B') can be pinned to a specific
        # bank pair while ilv=4 tensors (GEMV A/D) span all four banks.
        if interleave >= 4:
            num_groups = 2
            bank_span = 2
        else:
            num_groups = self._group_count(interleave)
            bank_span = interleave
        bank_cursors: list[_AddressCursor] = [
            _AddressCursor() for _ in range(self.MAX_BANKS)
        ]

        assignments: dict[str, AddressAssignment] = {}
        io_map: dict[str, str] = {}
        output_tensor_by_op: dict[str, str] = {}
        operator_config_base_addresses: dict[str, int] = {}
        operator_config_lengths: dict[str, int] = {}
        operator_sfu_config_base_addresses: dict[str, int] = {}
        operator_sfu_config_lengths: dict[str, int] = {}

        # -- tensor data -------------------------------------------------
        # Detect groups of consecutive decode_gemv_local_qkt sub-ops whose
        # outputs must be allocated contiguously so that a downstream op
        # reading the first sub-op's output sees the full concatenated range.
        _QKT_SPLIT_TYPE = "decode_gemv_local_qkt"
        qkt_split_groups: list[list[str]] = []
        _current_qkt_group: list[str] = []
        for _op in execution_input.operators:
            if _op.op_type == _QKT_SPLIT_TYPE:
                _current_qkt_group.append(_op.op_id)
            else:
                if len(_current_qkt_group) > 1:
                    qkt_split_groups.append(_current_qkt_group)
                _current_qkt_group = []
        if len(_current_qkt_group) > 1:
            qkt_split_groups.append(_current_qkt_group)
        # Map: first sub-op id → list of all sub-op ids in its group
        qkt_group_map: dict[str, list[str]] = {}
        for group in qkt_split_groups:
            qkt_group_map[group[0]] = group

        for op in execution_input.operators:
            enabled_slice_ids = op.enabled_slice_ids()
            if not enabled_slice_ids or len(enabled_slice_ids) > self.MAX_SLAVES:
                raise AddressPlanningError(
                    f"Operator {op.op_id} used_slices must enable between 1 and "
                    f"{self.MAX_SLAVES} slices, got mask {op.used_slices}."
                )

            input_group_totals: list[int] = [0] * num_groups
            input_group_max_ilv: list[int] = [0] * num_groups
            input_bank_totals: list[int] = [0] * self.MAX_BANKS
            for input_name, tensor in op.inputs.items():
                io_key = self._io_key(op.op_id, "input", input_name)

                if input_name == "B'" and "B" in op.inputs:
                    if op.op_type not in _BP_INDEPENDENT_ADDR_OPS:
                        b_io_key = self._io_key(op.op_id, "input", "B")
                        b_tensor_name = io_map.get(b_io_key)
                        if b_tensor_name is not None:
                            io_map[io_key] = b_tensor_name
                            continue
                    # else: B' gets its own independent allocation below.

                source = tensor.source
                if source is None:
                    raise AddressPlanningError(f"{io_key} is missing source.")

                if source.source_type == InputSourceType.EXTERNAL:
                    tensor_name = f"{op.op_id}.input.{input_name}"
                    alloc_shape = tensor.shape
                    if op.op_type in _BP_INDEPENDENT_ADDR_OPS and input_name in ("B", "B'"):
                        k, m, n = tensor.shape
                        alloc_shape = (k, m, n // 2)

                    tensor_ilv = tensor.bank_interleave or 1
                    force_group: int | None = None
                    if interleave >= 2 and (op.op_type == "decode_gemv_ring" or op.op_type == "decode_gemv_ring_new"):
                        # A/B must live on different banks:
                        # A → group 0 (banks 0/1), B → group 1 (banks 2/3).
                        # D avoids the largest input (typically B) → group 0.
                        if input_name == "A":
                            force_group = 0  # banks 0/1
                        elif input_name == "B":
                            force_group = 1  # banks 2/3
                        elif input_name == "B'":
                            force_group = 0  # banks 0/1
                    elif (
                        op.op_type == _QKT_SPLIT_TYPE
                        and input_name in ("B", "B'")
                        and any(op.op_id in g for g in qkt_split_groups)
                    ):
                        # QKT sub-op B/B' go to the opposite bank group
                        # from the outputs so outputs stay contiguous.
                        force_group = 0  # banks 0/1
                    # Avoid the group that currently holds the most input data,
                    # so inputs naturally spread across bank groups.
                    input_avoid: int | None = None
                    if num_groups > 1 and any(t > 0 for t in input_group_totals):
                        input_avoid = input_group_totals.index(
                            max(input_group_totals)
                        )
                    assignment, gi, rel_bank, _, _ = (
                        self._allocate_tensor_interleaved(
                            tensor_name=tensor_name,
                            tensor_dtype=tensor.dtype,
                            shape=alloc_shape,
                            enabled_slice_ids=enabled_slice_ids,
                            bank_cursors=bank_cursors,
                            interleave=interleave,
                            bank_span=bank_span,
                            force_group_idx=force_group,
                            tensor_interleave=tensor_ilv,
                            avoid_group=input_avoid,
                        )
                    )
                    input_group_totals[gi] += assignment.size_bytes
                    if tensor_ilv > input_group_max_ilv[gi]:
                        input_group_max_ilv[gi] = tensor_ilv
                    abs_bank_start = self._group_start_bank(gi, bank_span)
                    if tensor_ilv == 1:
                        input_bank_totals[abs_bank_start + rel_bank] += assignment.size_bytes
                    else:
                        for b in range(tensor_ilv):
                            input_bank_totals[abs_bank_start + b] += assignment.size_bytes // tensor_ilv
                    assignments[tensor_name] = assignment
                    io_map[io_key] = tensor_name
                    continue

                source_op_id = source.operator_id
                if source_op_id is None:
                    raise AddressPlanningError(
                        f"{io_key} depends on operator source but operator_id is empty."
                    )
                if source_op_id not in output_tensor_by_op:
                    raise AddressPlanningError(
                        f"{io_key} references unknown/unplanned source operator: "
                        f"{source_op_id}."
                    )
                src_tn = output_tensor_by_op[source_op_id]
                src_assignment = assignments[src_tn]
                tensor_ilv = tensor.bank_interleave or 1

                # If this input's declared size exceeds the source output's
                # allocated size (e.g. padded/expanded inputs like
                # prefill_remote_sum_qkt whose A is 4x the gemv output),
                # allocate an independent region instead of inheriting the
                # source address — otherwise the input would overflow into
                # neighbouring tensors.
                declared_size = self._tensor_size_bytes(
                    tensor_name=f"{op.op_id}.input.{input_name}",
                    tensor_dtype=tensor.dtype,
                    shape=tensor.shape,
                )
                if declared_size > src_assignment.size_bytes:
                    tensor_name = f"{op.op_id}.input.{input_name}"
                    assignment, gi, rel_bank, _, _ = (
                        self._allocate_tensor_interleaved(
                            tensor_name=tensor_name,
                            tensor_dtype=tensor.dtype,
                            shape=tensor.shape,
                            enabled_slice_ids=enabled_slice_ids,
                            bank_cursors=bank_cursors,
                            interleave=interleave,
                            bank_span=bank_span,
                            tensor_interleave=tensor_ilv,
                        )
                    )
                    assignments[tensor_name] = assignment
                    io_map[io_key] = tensor_name
                    input_group_totals[gi] += assignment.size_bytes
                    if tensor_ilv > input_group_max_ilv[gi]:
                        input_group_max_ilv[gi] = tensor_ilv
                    abs_bank_start = self._group_start_bank(gi, bank_span)
                    if tensor_ilv == 1:
                        input_bank_totals[abs_bank_start + rel_bank] += assignment.size_bytes
                    else:
                        for b in range(tensor_ilv):
                            input_bank_totals[abs_bank_start + b] += assignment.size_bytes // tensor_ilv
                    continue

                io_map[io_key] = src_tn
                src_bank = (src_assignment.base_address >> 23) & 0x03
                src_group = src_bank // bank_span
                input_group_totals[src_group] += src_assignment.size_bytes
                if tensor_ilv > input_group_max_ilv[src_group]:
                    input_group_max_ilv[src_group] = tensor_ilv
                if tensor_ilv == 1:
                    input_bank_totals[src_bank] += src_assignment.size_bytes
                else:
                    abs_bank_start = self._group_start_bank(src_group, bank_span)
                    for b in range(tensor_ilv):
                        input_bank_totals[abs_bank_start + b] += src_assignment.size_bytes // tensor_ilv

            # output
            output_name = f"{op.op_id}.output.D"
            output_ilv = op.output.bank_interleave or 1
            output_force: int | None = None
            output_avoid: int | None = None
            output_force_bank: int | None = None

            # For decode_gemv_local_qkt sub-ops that are part of a split
            # group, allocate outputs contiguously.  The first sub-op
            # allocates normally; subsequent sub-ops get a directly
            # computed address right after the previous sub-op's output,
            # bypassing the cursor to guarantee contiguity.
            _is_qkt_subsequent = (
                op.op_type == _QKT_SPLIT_TYPE
                and any(op.op_id in g for g in qkt_split_groups)
                and op.op_id not in qkt_group_map  # not the first in its group
            )
            if _is_qkt_subsequent:
                _group = None
                for g in qkt_split_groups:
                    if op.op_id in g:
                        _group = g
                        break
                if _group is not None:
                    _prev_tn = f"{_group[_group.index(op.op_id) - 1]}.output.D"
                    if _prev_tn in assignments:
                        _prev_a = assignments[_prev_tn]
                        _prev_bank = (_prev_a.base_address >> 23) & 0x03
                        _prev_ilv = output_ilv
                        _prev_pbw = ceil(_prev_a.size_bytes / 16) // max(_prev_ilv, 1)
                        if _prev_pbw == 0:
                            _prev_pbw = 1
                        # Compute the next address right after the previous
                        # output on the same bank(s), but never before the
                        # current cursor (which accounts for inputs already
                        # allocated on this bank).
                        _prev_row = (_prev_a.base_address >> 10) & 0x1FFF
                        _prev_col = (_prev_a.base_address >> 4) & 0x3F
                        _prev_end_flat = _prev_row * self.MAX_COLS + _prev_col + _prev_pbw
                        for _b in range(_prev_ilv):
                            _abs_b = _prev_bank + _b
                            _cur = bank_cursors[_abs_b]
                            _cur_flat = _cur.row * self.MAX_COLS + _cur.col
                            # Use the max of previous output end and current
                            # cursor to avoid overlapping with inputs.
                            _start_flat = max(_prev_end_flat, _cur_flat)
                            _start_row = _start_flat // self.MAX_COLS
                            _start_col = _start_flat % self.MAX_COLS
                            _size_bytes = self._tensor_size_bytes(
                                tensor_name=output_name,
                                tensor_dtype=op.output.dtype,
                                shape=op.output.shape,
                            )
                            _base = self._pack_address(
                                slave=0, bank=_prev_bank, row=_start_row,
                                col=_start_col, subword=0,
                            )
                            _per_slice = {}
                            for sid in enabled_slice_ids:
                                _per_slice[sid] = self._pack_address(
                                    slave=sid, bank=_prev_bank, row=_start_row,
                                    col=_start_col, subword=0,
                                )
                            _assign = AddressAssignment(
                                tensor_name=output_name,
                                base_address=_base,
                                per_slice_addresses=_per_slice,
                                size_bytes=_size_bytes,
                                shape=op.output.shape,
                            )
                            assignments[output_name] = _assign
                            output_tensor_by_op[op.op_id] = output_name
                            io_map[self._io_key(op.op_id, "output", "D")] = output_name
                            # Advance cursor past this output.
                            _new_pbw = ceil(_size_bytes / 16) // max(output_ilv, 1)
                            if _new_pbw == 0:
                                _new_pbw = 1
                            _end_flat = _start_flat + _new_pbw
                            bank_cursors[_abs_b] = _AddressCursor(
                                bank=0,
                                row=_end_flat // self.MAX_COLS,
                                col=_end_flat % self.MAX_COLS,
                            )
                            break  # only need to do this once per bank group
                        continue  # skip normal output allocation

            if output_ilv == 1 and any(t > 0 for t in input_bank_totals):
                # Single-bank output: pick the physical bank with the least
                # input data so D lands on a bank different from every input.
                output_force_bank = min(
                    range(self.MAX_BANKS),
                    key=lambda b: (input_bank_totals[b], b),
                )
            elif sum(input_group_totals) > 0 and num_groups > 1:
                # Multi-bank output: avoid the heaviest group, using max
                # bank_interleave as tiebreaker (higher interleave → worse
                # read/write conflicts).
                avoid_g = max(
                    range(num_groups),
                    key=lambda g: (input_group_totals[g], input_group_max_ilv[g]),
                )
                if (
                    interleave >= 2
                    and op.op_type
                    in ("decode_gemv_ring", "decode_gemv_ring_new")
                ):
                    output_force = 1 - avoid_g
                else:
                    output_avoid = avoid_g
            output_assignment, _, _, _, _ = (
                self._allocate_tensor_interleaved(
                    tensor_name=output_name,
                    tensor_dtype=op.output.dtype,
                    shape=op.output.shape,
                    enabled_slice_ids=enabled_slice_ids,
                    bank_cursors=bank_cursors,
                    interleave=interleave,
                    bank_span=bank_span,
                    force_group_idx=output_force,
                    tensor_interleave=output_ilv,
                    avoid_group=output_avoid,
                    force_bank=output_force_bank,
                )
            )
            assignments[output_name] = output_assignment
            output_tensor_by_op[op.op_id] = output_name
            io_map[self._io_key(op.op_id, "output", "D")] = output_name

        # -- config payloads ------------------------------------------------
        config_lengths_by_op = config_lengths_by_op or {}
        sfu_config_lengths_by_op = sfu_config_lengths_by_op or {}

        # Align every bank cursor to row start.
        for b in range(self.MAX_BANKS):
            bank_cursors[b] = self._align_single_to_row(bank_cursors[b])

        # Pick the least-used bank group for config placement.
        config_group = self._choose_bank_group_from_banks(
            bank_cursors, bank_span
        )
        config_bank = self._group_start_bank(config_group, bank_span)

        config_base = self._pack_address(
            slave=0,
            bank=config_bank,
            row=bank_cursors[config_bank].row,
            col=0,
            subword=0,
        )

        # Each operator gets its own independent config address — no dedup by op_type.
        # SFU configs however are static files shared by sfu_type.
        sfu_types_by_op = sfu_types_by_op or {}
        seen_sfu_types: dict[str, int] = {}
        for op in execution_input.operators:
            config_length_64b = int(config_lengths_by_op.get(op.op_id, 0) or 0)
            if config_length_64b < 0:
                raise AddressPlanningError(
                    f"Operator {op.op_id} config_length must be non-negative, "
                    f"got {config_length_64b}."
                )
            if config_length_64b == 0:
                operator_config_lengths[op.op_id] = 0
            else:
                operator_config_base_addresses[op.op_id] = config_base
                operator_config_lengths[op.op_id] = config_length_64b

                config_bytes = config_length_64b * 8
                reserved_bytes = self._align_up(config_bytes, self.ROW_BYTES)
                config_base = self._align_up(
                    config_base + reserved_bytes, self.ROW_BYTES
                )

            sfu_config_length_64b = int(sfu_config_lengths_by_op.get(op.op_id, 0) or 0)
            if sfu_config_length_64b < 0:
                raise AddressPlanningError(
                    f"Operator {op.op_id} sfu config_length must be non-negative, "
                    f"got {sfu_config_length_64b}."
                )
            if sfu_config_length_64b > 0:
                sfu_type = sfu_types_by_op.get(op.op_id, "")
                if sfu_type and sfu_type in seen_sfu_types:
                    # Share SFU address with the first operator of this sfu_type.
                    operator_sfu_config_base_addresses[op.op_id] = seen_sfu_types[sfu_type]
                    operator_sfu_config_lengths[op.op_id] = sfu_config_length_64b
                else:
                    operator_sfu_config_base_addresses[op.op_id] = config_base
                    operator_sfu_config_lengths[op.op_id] = sfu_config_length_64b
                    if sfu_type:
                        seen_sfu_types[sfu_type] = config_base
                    sfu_config_bytes = sfu_config_length_64b * 8
                    sfu_reserved_bytes = self._align_up(sfu_config_bytes, self.ROW_BYTES)
                    config_base = self._align_up(
                        config_base + sfu_reserved_bytes, self.ROW_BYTES
                    )
            else:
                operator_sfu_config_lengths[op.op_id] = 0

        return AddressPlan(
            assignments=assignments,
            operator_io_to_tensor=io_map,
            operator_config_base_addresses=operator_config_base_addresses,
            operator_config_lengths=operator_config_lengths,
            operator_sfu_config_base_addresses=operator_sfu_config_base_addresses,
            operator_sfu_config_lengths=operator_sfu_config_lengths,
        )

    # ------------------------------------------------------------------
    # Original flat plan (bank_interleave = 1, backward-compatible)
    # ------------------------------------------------------------------

    def _plan_flat(
        self,
        execution_input: ExecutionPlanInput,
        config_lengths_by_op: dict[str, int] | None,
        sfu_config_lengths_by_op: dict[str, int] | None,
        sfu_types_by_op: dict[str, str] | None = None,
    ) -> AddressPlan:
        cursor = _AddressCursor()
        assignments: dict[str, AddressAssignment] = {}
        io_map: dict[str, str] = {}
        output_tensor_by_op: dict[str, str] = {}
        operator_config_base_addresses: dict[str, int] = {}
        operator_config_lengths: dict[str, int] = {}
        operator_sfu_config_base_addresses: dict[str, int] = {}
        operator_sfu_config_lengths: dict[str, int] = {}

        for op in execution_input.operators:
            enabled_slice_ids = op.enabled_slice_ids()
            if not enabled_slice_ids or len(enabled_slice_ids) > self.MAX_SLAVES:
                raise AddressPlanningError(
                    f"Operator {op.op_id} used_slices must enable between 1 and "
                    f"{self.MAX_SLAVES} slices, got mask {op.used_slices}."
                )
            for input_name, tensor in op.inputs.items():
                io_key = self._io_key(op.op_id, "input", input_name)

                if input_name == "B'" and "B" in op.inputs:
                    if op.op_type not in _BP_INDEPENDENT_ADDR_OPS:
                        b_io_key = self._io_key(op.op_id, "input", "B")
                        b_tensor_name = io_map.get(b_io_key)
                        if b_tensor_name is not None:
                            io_map[io_key] = b_tensor_name
                            continue
                    # else: B' gets its own independent allocation below.

                source = tensor.source
                if source is None:
                    raise AddressPlanningError(f"{io_key} is missing source.")

                if source.source_type == InputSourceType.EXTERNAL:
                    tensor_name = f"{op.op_id}.input.{input_name}"
                    alloc_shape = tensor.shape
                    if op.op_type in _BP_INDEPENDENT_ADDR_OPS and input_name in ("B", "B'"):
                        k, m, n = tensor.shape
                        alloc_shape = (k, m, n // 2)
                    assignment, cursor = self._allocate_tensor_flat(
                        tensor_name=tensor_name,
                        tensor_dtype=tensor.dtype,
                        shape=alloc_shape,
                        enabled_slice_ids=enabled_slice_ids,
                        cursor=cursor,
                    )
                    assignments[tensor_name] = assignment
                    io_map[io_key] = tensor_name
                    continue

                source_op_id = source.operator_id
                if source_op_id is None:
                    raise AddressPlanningError(
                        f"{io_key} depends on operator source but operator_id is empty."
                    )
                if source_op_id not in output_tensor_by_op:
                    raise AddressPlanningError(
                        f"{io_key} references unknown/unplanned source operator: "
                        f"{source_op_id}."
                    )
                io_map[io_key] = output_tensor_by_op[source_op_id]

            output_name = f"{op.op_id}.output.D"
            output_assignment, cursor = self._allocate_tensor_flat(
                tensor_name=output_name,
                tensor_dtype=op.output.dtype,
                shape=op.output.shape,
                enabled_slice_ids=enabled_slice_ids,
                cursor=cursor,
            )
            assignments[output_name] = output_assignment
            output_tensor_by_op[op.op_id] = output_name
            io_map[self._io_key(op.op_id, "output", "D")] = output_name

        config_lengths_by_op = config_lengths_by_op or {}
        sfu_config_lengths_by_op = sfu_config_lengths_by_op or {}
        cursor = self._align_cursor_to_row(cursor)
        config_cursor_addr = self._pack_address(
            slave=0,
            bank=cursor.bank,
            row=cursor.row,
            col=0,
            subword=0,
        )
        # Each operator gets its own independent config address — no dedup by op_type.
        # SFU configs however are static files shared by sfu_type.
        sfu_types_by_op = sfu_types_by_op or {}
        seen_sfu_types: dict[str, int] = {}
        for op in execution_input.operators:
            config_length_64b = int(config_lengths_by_op.get(op.op_id, 0) or 0)
            if config_length_64b < 0:
                raise AddressPlanningError(
                    f"Operator {op.op_id} config_length must be non-negative, "
                    f"got {config_length_64b}."
                )
            if config_length_64b == 0:
                operator_config_lengths[op.op_id] = 0
            else:
                operator_config_base_addresses[op.op_id] = config_cursor_addr
                operator_config_lengths[op.op_id] = config_length_64b
                config_bytes = config_length_64b * 8
                reserved_bytes = self._align_up(config_bytes, self.ROW_BYTES)
                config_cursor_addr = self._align_up(
                    config_cursor_addr + reserved_bytes, self.ROW_BYTES
                )

            sfu_config_length_64b = int(sfu_config_lengths_by_op.get(op.op_id, 0) or 0)
            if sfu_config_length_64b < 0:
                raise AddressPlanningError(
                    f"Operator {op.op_id} sfu config_length must be non-negative, "
                    f"got {sfu_config_length_64b}."
                )
            if sfu_config_length_64b > 0:
                sfu_type = sfu_types_by_op.get(op.op_id, "")
                if sfu_type and sfu_type in seen_sfu_types:
                    # Share SFU address with the first operator of this sfu_type.
                    operator_sfu_config_base_addresses[op.op_id] = seen_sfu_types[sfu_type]
                    operator_sfu_config_lengths[op.op_id] = sfu_config_length_64b
                else:
                    operator_sfu_config_base_addresses[op.op_id] = config_cursor_addr
                    operator_sfu_config_lengths[op.op_id] = sfu_config_length_64b
                    if sfu_type:
                        seen_sfu_types[sfu_type] = config_cursor_addr
                    sfu_config_bytes = sfu_config_length_64b * 8
                    sfu_reserved_bytes = self._align_up(sfu_config_bytes, self.ROW_BYTES)
                    config_cursor_addr = self._align_up(
                        config_cursor_addr + sfu_reserved_bytes, self.ROW_BYTES
                    )
            else:
                operator_sfu_config_lengths[op.op_id] = 0

        return AddressPlan(
            assignments=assignments,
            operator_io_to_tensor=io_map,
            operator_config_base_addresses=operator_config_base_addresses,
            operator_config_lengths=operator_config_lengths,
            operator_sfu_config_base_addresses=operator_sfu_config_base_addresses,
            operator_sfu_config_lengths=operator_sfu_config_lengths,
        )

    # ------------------------------------------------------------------
    # Interleaved allocation helpers (per-bank cursors)
    # ------------------------------------------------------------------

    def _choose_bank_group_from_banks(
        self,
        bank_cursors: list[_AddressCursor],
        span: int,
        avoid_group: int | None = None,
    ) -> int:
        """Return the index of the least-used bank group.

        Uses the maximum cursor position across each group's banks as the
        load metric, so a group whose banks have advanced least is preferred.
        """
        num_groups = self.MAX_BANKS // span
        best: int | None = None
        best_key: tuple[int, int, int] | None = None
        for gi in range(num_groups):
            if avoid_group is not None and num_groups > 1 and gi == avoid_group:
                continue
            group_start = self._group_start_bank(gi, span)
            max_cursor = max(
                (bank_cursors[b] for b in range(group_start, group_start + span)),
                key=lambda c: (c.row, c.col),
            )
            key = (max_cursor.row, max_cursor.col, gi)
            if best_key is None or key < best_key:
                best_key = key
                best = gi
        if best is None:
            best = 0
        return best

    def _advance_single(
        self, cursor: _AddressCursor, words: int
    ) -> _AddressCursor:
        """Advance a per-bank cursor by *words* within a single bank."""
        current = cursor.row * self.MAX_COLS + cursor.col
        nxt = current + words
        if nxt > self.MAX_ROWS * self.MAX_COLS:
            raise AddressPlanningError("Address space exhausted on bank.")
        return _AddressCursor(
            bank=0, row=nxt // self.MAX_COLS, col=nxt % self.MAX_COLS
        )

    def _align_single_to_row(self, cursor: _AddressCursor) -> _AddressCursor:
        if cursor.col == 0:
            return cursor
        return self._advance_single(cursor, self.MAX_COLS - cursor.col)

    def _allocate_tensor_interleaved(
        self,
        tensor_name: str,
        tensor_dtype: str,
        shape: tuple[int, int, int],
        enabled_slice_ids: list[int],
        bank_cursors: list[_AddressCursor],
        interleave: int,
        bank_span: int | None = None,
        force_group_idx: int | None = None,
        tensor_interleave: int | None = None,
        avoid_group: int | None = None,
        force_bank: int | None = None,
    ) -> tuple[AddressAssignment, int, int, int, int]:
        span = bank_span if bank_span is not None else interleave
        size_bytes = self._tensor_size_bytes(
            tensor_name=tensor_name,
            tensor_dtype=tensor_dtype,
            shape=shape,
        )
        ilv_for_size = tensor_interleave if tensor_interleave is not None else interleave
        per_bank_bytes = size_bytes // ilv_for_size
        words = ceil(per_bank_bytes / self.WORD_BYTES)

        spans_all_groups = ilv_for_size >= interleave and span < interleave

        # Determine which physical banks this tensor occupies.
        if force_bank is not None:
            banks_to_use = [force_bank]
            gi = force_bank // span
        elif spans_all_groups:
            banks_to_use = list(range(self.MAX_BANKS))
            gi = 0
        elif force_group_idx is not None:
            gi = force_group_idx
            group_start = self._group_start_bank(gi, span)
            banks_to_use = list(range(group_start, group_start + span))
        elif ilv_for_size == 1:
            # Single-bank tensor: choose the least-used bank, respecting
            # avoid_group.
            gi = self._choose_bank_group_from_banks(
                bank_cursors, span, avoid_group
            )
            group_start = self._group_start_bank(gi, span)
            best_b = group_start
            best_key = (bank_cursors[group_start].row, bank_cursors[group_start].col)
            for b in range(group_start + 1, group_start + span):
                key = (bank_cursors[b].row, bank_cursors[b].col)
                if key < best_key:
                    best_key = key
                    best_b = b
            banks_to_use = [best_b]
        else:
            # Multi-bank tensor: choose the least-used group.
            gi = self._choose_bank_group_from_banks(
                bank_cursors, span, avoid_group
            )
            group_start = self._group_start_bank(gi, span)
            banks_to_use = list(range(group_start, group_start + span))

        # Starting position = the most-advanced cursor across all banks
        # this tensor will occupy.  This guarantees no overlap with prior
        # allocations on any of those banks.
        start_cursor = max(
            (bank_cursors[b] for b in banks_to_use),
            key=lambda c: (c.row, c.col),
        )
        row = start_cursor.row
        col = start_cursor.col
        bank = banks_to_use[0]

        base_address = self._pack_address(
            slave=0, bank=bank, row=row, col=col, subword=0
        )

        per_slice_addresses: dict[int, int] = {}
        for slice_id in enabled_slice_ids:
            per_slice_addresses[slice_id] = self._pack_address(
                slave=slice_id, bank=bank, row=row, col=col, subword=0
            )

        # Advance every occupied bank's cursor by the same amount.
        new_cursor = self._advance_single(start_cursor, words)
        for b in banks_to_use:
            bank_cursors[b] = new_cursor

        return (
            AddressAssignment(
                tensor_name=tensor_name,
                base_address=base_address,
                per_slice_addresses=per_slice_addresses,
                size_bytes=size_bytes,
                shape=shape,
            ),
            gi,
            0,
            row,
            col,
        )

    # ------------------------------------------------------------------
    # Flat allocation helper (original, renamed from _allocate_tensor)
    # ------------------------------------------------------------------

    def _allocate_tensor_flat(
        self,
        tensor_name: str,
        tensor_dtype: str,
        shape: tuple[int, int, int],
        enabled_slice_ids: list[int],
        cursor: _AddressCursor,
    ) -> tuple[AddressAssignment, _AddressCursor]:
        size_bytes = self._tensor_size_bytes(
            tensor_name=tensor_name,
            tensor_dtype=tensor_dtype,
            shape=shape,
        )
        words = ceil(size_bytes / self.WORD_BYTES)

        bank, row, col = cursor.bank, cursor.row, cursor.col
        base_address = self._pack_address(
            slave=0, bank=bank, row=row, col=col, subword=0
        )

        per_slice_addresses: dict[int, int] = {}
        for slice_id in enabled_slice_ids:
            per_slice_addresses[slice_id] = self._pack_address(
                slave=slice_id,
                bank=bank,
                row=row,
                col=col,
                subword=0,
            )

        next_bank, next_row, next_col = self._advance(
            bank=bank, row=row, col=col, words=words
        )
        next_cursor = _AddressCursor(
            bank=next_bank, row=next_row, col=next_col
        )

        return (
            AddressAssignment(
                tensor_name=tensor_name,
                base_address=base_address,
                per_slice_addresses=per_slice_addresses,
                size_bytes=size_bytes,
                shape=shape,
            ),
            next_cursor,
        )

    # ------------------------------------------------------------------
    # Shared helpers (used by both allocation modes)
    # ------------------------------------------------------------------

    def _tensor_size_bytes(
        self,
        tensor_name: str,
        tensor_dtype: str,
        shape: tuple[int, int, int],
    ) -> int:
        k, m, n = shape
        element_bytes = self._element_bytes_for_tensor(
            tensor_name=tensor_name,
            tensor_dtype=tensor_dtype,
        )
        return k * m * n * element_bytes

    def _element_bytes_for_tensor(self, tensor_name: str, tensor_dtype: str) -> int:
        _ = tensor_name
        normalized = (tensor_dtype or "fp32").strip().lower()
        if normalized == "pf32":
            normalized = "fp32"
        return self._dtype_bytes.get(normalized, self._element_bytes)

    def _advance(self, bank: int, row: int, col: int, words: int) -> tuple[int, int, int]:
        current = self._flatten(bank=bank, row=row, col=col)
        nxt = current + words
        capacity = self.MAX_BANKS * self.MAX_ROWS * self.MAX_COLS
        if nxt > capacity:
            raise AddressPlanningError("Address space exhausted during tensor allocation.")
        return self._unflatten(nxt)

    def _align_up(self, value: int, alignment: int) -> int:
        if alignment <= 0:
            raise AddressPlanningError("alignment must be positive")
        return ((value + alignment - 1) // alignment) * alignment

    def _flatten(self, bank: int, row: int, col: int) -> int:
        return ((bank * self.MAX_ROWS) + row) * self.MAX_COLS + col

    def _unflatten(self, flat: int) -> tuple[int, int, int]:
        bank_rows = self.MAX_ROWS * self.MAX_COLS
        bank = flat // bank_rows
        rem = flat % bank_rows
        row = rem // self.MAX_COLS
        col = rem % self.MAX_COLS
        if bank >= self.MAX_BANKS:
            raise AddressPlanningError("Address space overflow: bank index exceeds range.")
        return bank, row, col

    def _align_cursor_to_row(self, cursor: _AddressCursor) -> _AddressCursor:
        if cursor.col == 0:
            return cursor
        next_bank, next_row, next_col = self._advance(
            bank=cursor.bank,
            row=cursor.row,
            col=cursor.col,
            words=self.MAX_COLS - cursor.col,
        )
        return _AddressCursor(bank=next_bank, row=next_row, col=next_col)

    def _pack_address(self, slave: int, bank: int, row: int, col: int, subword: int) -> int:
        if not (0 <= slave < 32):
            raise AddressPlanningError(f"Invalid slave id: {slave}")
        if not (0 <= bank < self.MAX_BANKS):
            raise AddressPlanningError(f"Invalid bank index: {bank}")
        if not (0 <= row < self.MAX_ROWS):
            raise AddressPlanningError(f"Invalid row index: {row}")
        if not (0 <= col < self.MAX_COLS):
            raise AddressPlanningError(f"Invalid col index: {col}")
        if not (0 <= subword < 16):
            raise AddressPlanningError(f"Invalid subword index: {subword}")

        return (slave << 25) | (bank << 23) | (row << 10) | (col << 4) | subword

    def _io_key(self, op_id: str, io_type: str, tensor_name: str) -> str:
        return f"{op_id}.{io_type}.{tensor_name}"
