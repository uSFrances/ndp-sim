from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .hardware import HardwareSpec


@dataclass(frozen=True)
class AddressTransform:
    name: str
    logical_bit_labels: List[str]
    physical_bit_labels: List[str]
    permutation: List[int]
    interleave_bank_count: int = 1

    @classmethod
    def identity(
        cls,
        logical_bit_labels: Sequence[str],
        name: str = "identity",
        interleave_bank_count: int = 1,
    ) -> "AddressTransform":
        labels = list(logical_bit_labels)
        return cls(
            name=name,
            logical_bit_labels=labels,
            physical_bit_labels=list(labels),
            permutation=list(range(len(labels))),
            interleave_bank_count=interleave_bank_count,
        )

    @classmethod
    def from_edge_result(
        cls,
        edge_result: Mapping[str, object],
        mode: str,
        hw: HardwareSpec,
    ) -> "AddressTransform":
        logical_labels = list(edge_result.get("consumer_visible_outer_bits") or [])
        producer_labels = list(edge_result.get("producer_visible_outer_bits") or logical_labels)
        if mode == "baseline":
            return cls.identity(logical_labels, name="baseline_identity")
        permutation = list(edge_result.get("permutation") or list(range(hw.remap_bits)))
        transform = cls(
            name=f"{mode}_remap",
            logical_bit_labels=logical_labels,
            physical_bit_labels=producer_labels,
            permutation=permutation,
            interleave_bank_count=(hw.bank_count_per_slice if mode == "remap_interleave" else 1),
        )
        return transform

    def apply(self, logical_addr: int, hw: HardwareSpec) -> int:
        logical_addr = int(logical_addr)
        if self.interleave_bank_count > 1:
            logical_addr = _interleave_logical_addr(logical_addr, self.interleave_bank_count)
        physical_bits = [0] * hw.remap_bits
        for in_idx in range(min(len(self.logical_bit_labels), hw.remap_bits)):
            out_idx = self.permutation[in_idx] if in_idx < len(self.permutation) else in_idx
            if out_idx >= hw.remap_bits:
                continue
            physical_bits[out_idx] = (logical_addr >> in_idx) & 1
        physical_ordinal = 0
        for bit_idx, bit in enumerate(physical_bits):
            physical_ordinal |= int(bit) << bit_idx
        return physical_ordinal

    def to_dict(self) -> Dict[str, object]:
        matrix = []
        width = len(self.logical_bit_labels)
        height = max(max(self.permutation, default=-1) + 1, width)
        for row in range(height):
            row_values = [0] * width
            for col in range(width):
                if col < len(self.permutation) and self.permutation[col] == row:
                    row_values[col] = 1
            matrix.append(row_values)
        return {
            "name": self.name,
            "logical_bit_labels": list(self.logical_bit_labels),
            "physical_bit_labels": list(self.physical_bit_labels),
            "permutation": list(self.permutation),
            "matrix": matrix,
            "interleave_bank_count": self.interleave_bank_count,
        }


def decode_physical_address(physical_addr: int, hw: HardwareSpec) -> Dict[str, int]:
    addr_units = int(physical_addr) >> hw.subword_bits
    col_mask = (1 << hw.column_bits) - 1
    row_mask = (1 << hw.row_bits) - 1
    bank_mask = (1 << hw.bank_bits) - 1
    col_id = addr_units & col_mask
    addr_units >>= hw.column_bits
    row_id = addr_units & row_mask
    addr_units >>= hw.row_bits
    bank_id = addr_units & bank_mask
    addr_units >>= hw.bank_bits
    slice_id = addr_units & ((1 << hw.slave_bits) - 1)
    return {
        "slice_id": slice_id,
        "bank_id": bank_id,
        "row_id": row_id,
        "col_id": col_id,
    }


def compose_physical_address(
    *,
    base_addr: int,
    logical_addr: int,
    transform: AddressTransform,
    hw: HardwareSpec,
) -> int:
    physical_ordinal = transform.apply(logical_addr, hw)
    return int(base_addr) + (physical_ordinal << hw.subword_bits)


def encode_physical_address(
    *,
    slice_id: int,
    bank_id: int,
    row_id: int,
    col_id: int,
    hw: HardwareSpec,
) -> int:
    addr_units = (
        (int(slice_id) << (hw.bank_bits + hw.row_bits + hw.column_bits))
        | (int(bank_id) << (hw.row_bits + hw.column_bits))
        | (int(row_id) << hw.column_bits)
        | int(col_id)
    )
    return addr_units << hw.subword_bits


def _interleave_logical_addr(logical_addr: int, bank_count: int) -> int:
    shift = max(1, (max(1, bank_count) - 1).bit_length())
    return ((logical_addr // max(1, bank_count)) << shift) | (logical_addr % max(1, bank_count))
