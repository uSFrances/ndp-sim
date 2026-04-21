from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .errors import ExecutionPlanError
from .models import AddressAssignment, AddressPlan, ExecutionPlanInput, InputSourceType


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

    def plan(
        self,
        execution_input: ExecutionPlanInput,
        config_lengths_by_op: dict[str, int] | None = None,
        sfu_config_lengths_by_op: dict[str, int] | None = None,
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
                    f"Operator {op.op_id} used_slices must enable between 1 and {self.MAX_SLAVES} slices, got mask {op.used_slices}."
                )
            for input_name, tensor in op.inputs.items():
                io_key = self._io_key(op.op_id, "input", input_name)

                # B' shares the same allocated address space as B for this operator.
                if input_name == "B'" and "B" in op.inputs:
                    b_io_key = self._io_key(op.op_id, "input", "B")
                    b_tensor_name = io_map.get(b_io_key)
                    if b_tensor_name is not None:
                        io_map[io_key] = b_tensor_name
                        continue

                source = tensor.source
                if source is None:
                    raise AddressPlanningError(f"{io_key} is missing source.")

                if source.source_type == InputSourceType.EXTERNAL:
                    tensor_name = f"{op.op_id}.input.{input_name}"
                    assignment, cursor = self._allocate_tensor(
                        tensor_name=tensor_name,
                        tensor_dtype=tensor.dtype,
                        shape=tensor.shape,
                        enabled_slice_ids=enabled_slice_ids,
                        cursor=cursor,
                    )
                    assignments[tensor_name] = assignment
                    io_map[io_key] = tensor_name
                    continue

                source_op_id = source.operator_id
                if source_op_id is None:
                    raise AddressPlanningError(f"{io_key} depends on operator source but operator_id is empty.")
                if source_op_id not in output_tensor_by_op:
                    raise AddressPlanningError(
                        f"{io_key} references unknown/unplanned source operator: {source_op_id}."
                    )
                io_map[io_key] = output_tensor_by_op[source_op_id]

            output_name = f"{op.op_id}.output.D"
            output_assignment, cursor = self._allocate_tensor(
                tensor_name=output_name,
                tensor_dtype=op.output.dtype,
                shape=op.output.shape,
                enabled_slice_ids=enabled_slice_ids,
                cursor=cursor,
            )
            assignments[output_name] = output_assignment
            output_tensor_by_op[op.op_id] = output_name
            io_map[self._io_key(op.op_id, "output", "D")] = output_name

        # Place config payloads after tensor data.
        # config_length unit is 64-bit row count; each config is placed at BANK-row start
        # and reserves a whole number of rows so the next config also begins on a row boundary.
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
        for op in execution_input.operators:
            config_length_64b = int(config_lengths_by_op.get(op.op_id, 0) or 0)
            if config_length_64b < 0:
                raise AddressPlanningError(
                    f"Operator {op.op_id} config_length must be non-negative, got {config_length_64b}."
                )

            if config_length_64b == 0:
                operator_config_lengths[op.op_id] = 0
            else:
                base_address = config_cursor_addr
                operator_config_base_addresses[op.op_id] = base_address
                # Keep metadata value as original 64-bit row count.
                operator_config_lengths[op.op_id] = config_length_64b

                config_bytes = config_length_64b * 8
                reserved_bytes = self._align_up(config_bytes, self.ROW_BYTES)
                config_cursor_addr = self._align_up(base_address + reserved_bytes, self.ROW_BYTES)

            sfu_config_length_64b = int(sfu_config_lengths_by_op.get(op.op_id, 0) or 0)
            if sfu_config_length_64b < 0:
                raise AddressPlanningError(
                    f"Operator {op.op_id} sfu config_length must be non-negative, got {sfu_config_length_64b}."
                )
            if sfu_config_length_64b > 0:
                sfu_base_address = config_cursor_addr
                operator_sfu_config_base_addresses[op.op_id] = sfu_base_address
                operator_sfu_config_lengths[op.op_id] = sfu_config_length_64b

                sfu_config_bytes = sfu_config_length_64b * 8
                sfu_reserved_bytes = self._align_up(sfu_config_bytes, self.ROW_BYTES)
                config_cursor_addr = self._align_up(sfu_base_address + sfu_reserved_bytes, self.ROW_BYTES)
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

    def _allocate_tensor(
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
        base_address = self._pack_address(slave=0, bank=bank, row=row, col=col, subword=0)

        per_slice_addresses: dict[int, int] = {}
        for slice_id in enabled_slice_ids:
            per_slice_addresses[slice_id] = self._pack_address(
                slave=slice_id,
                bank=bank,
                row=row,
                col=col,
                subword=0,
            )

        next_bank, next_row, next_col = self._advance(bank=bank, row=row, col=col, words=words)
        next_cursor = _AddressCursor(bank=next_bank, row=next_row, col=next_col)

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
