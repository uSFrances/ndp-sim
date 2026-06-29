from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .hardware import HardwareSpec


@dataclass(frozen=True)
class AddressTransform:
    name: str
    logical_bit_labels: List[str]
    physical_bit_labels: List[str]
    permutation: List[int]
    layout_permutation: Optional[List[int]] = None
    physical_permutation: Optional[List[int]] = None
    composed_permutation: Optional[List[int]] = None
    transform_role: str = "input"
    transform_source: str = "identity"
    interleave_bank_count: int = 1
    placement_policy: str = "identity"
    mode: str = "baseline"

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
            layout_permutation=list(range(len(labels))),
            physical_permutation=list(range(len(labels))),
            composed_permutation=list(range(len(labels))),
            transform_role="identity",
            transform_source="identity",
            interleave_bank_count=interleave_bank_count,
            placement_policy="identity",
            mode="baseline" if name == "identity" else name,
        )

    @classmethod
    def from_edge_result(
        cls,
        edge_result: Mapping[str, object],
        mode: str,
        hw: HardwareSpec,
        transform_role: str = "input",
    ) -> "AddressTransform":
        return build_transform_from_edge_result(
            edge_result=edge_result,
            mode=mode,
            hw=hw,
            transform_role=transform_role,
        )

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
            "mode": self.mode,
            "placement_policy": self.placement_policy,
            "logical_bit_labels": list(self.logical_bit_labels),
            "physical_bit_labels": list(self.physical_bit_labels),
            "permutation": list(self.permutation),
            "layout_permutation": list(self.layout_permutation or self.permutation),
            "physical_permutation": list(self.physical_permutation or self.permutation),
            "composed_permutation": list(self.composed_permutation or self.permutation),
            "transform_role": self.transform_role,
            "transform_source": self.transform_source,
            "matrix": matrix,
            "interleave_bank_count": self.interleave_bank_count,
        }


def build_transform_from_edge_result(
    edge_result: Mapping[str, object],
    mode: str,
    hw: HardwareSpec,
    transform_role: str = "input",
) -> AddressTransform:
    is_output = transform_role == "output"
    logical_labels = list(
        edge_result.get("producer_visible_outer_bits" if is_output else "consumer_visible_outer_bits") or []
    )
    if mode == "baseline":
        identity = list(range(hw.remap_bits))
        return AddressTransform(
            name=f"baseline_{transform_role}_identity",
            logical_bit_labels=logical_labels,
            physical_bit_labels=list(logical_labels),
            permutation=identity,
            layout_permutation=identity,
            physical_permutation=identity,
            composed_permutation=identity,
            transform_role=transform_role,
            transform_source="identity",
            placement_policy="identity",
            mode=mode,
        )

    layout_permutation = list(
        edge_result.get("internal_layout_permutation")
        or edge_result.get("layout_permutation")
        or edge_result.get("internal_permutation")
        or edge_result.get("permutation")
        or []
    )
    physical_permutation = list(
        edge_result.get("internal_physical_permutation")
        or edge_result.get("physical_permutation")
        or list(range(hw.remap_bits))
    )
    composed_permutation = list(
        edge_result.get("internal_composed_permutation")
        or edge_result.get("composed_permutation")
        or edge_result.get("internal_permutation")
        or edge_result.get("permutation")
        or []
    )
    output_permutation = list(
        edge_result.get("internal_output_permutation")
        or edge_result.get("output_permutation")
        or composed_permutation
    )
    input_permutation = list(
        edge_result.get("internal_input_permutation")
        or edge_result.get("input_permutation")
        or physical_permutation
    )
    if not layout_permutation:
        layout_permutation = list(range(hw.remap_bits))
    if not composed_permutation:
        composed_permutation = list(layout_permutation)
    placement_policy = str(edge_result.get("physical_placement_policy", "preserve_layout_only"))
    interleave_bank_count = 1
    selected_permutation = output_permutation if is_output else input_permutation
    selected_policy = placement_policy
    transform_source = "solver"
    if mode == "layout_remap":
        selected_permutation = layout_permutation if is_output else list(range(hw.remap_bits))
        selected_policy = "preserve_layout_only"
        transform_source = "P_layout" if is_output else "identity"
    elif mode == "oracle_interleave":
        selected_permutation = output_permutation if is_output else input_permutation
        selected_policy = "oracle_interleave"
        interleave_bank_count = hw.bank_count_per_slice
        transform_source = "oracle"
    else:
        transform_source = "P_out" if is_output else "P_in"
    physical_labels = _dram_bit_labels(hw)
    selected_labels = _selected_physical_labels(
        physical_labels=physical_labels,
        permutation=selected_permutation,
        visible_bit_count=len(logical_labels),
    )
    return AddressTransform(
        name=f"{mode}_{transform_role}_remap",
        logical_bit_labels=logical_labels,
        physical_bit_labels=selected_labels,
        permutation=selected_permutation,
        layout_permutation=layout_permutation,
        physical_permutation=physical_permutation,
        composed_permutation=composed_permutation,
        transform_role=transform_role,
        transform_source=transform_source,
        interleave_bank_count=interleave_bank_count,
        placement_policy=selected_policy,
        mode=mode,
    )


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


def _selected_physical_labels(
    *,
    physical_labels: Sequence[str],
    permutation: Sequence[int],
    visible_bit_count: int,
) -> List[str]:
    labels: List[str] = []
    for logical_idx in range(visible_bit_count):
        mapped = int(permutation[logical_idx]) if logical_idx < len(permutation) else logical_idx
        if 0 <= mapped < len(physical_labels):
            labels.append(str(physical_labels[mapped]))
        else:
            labels.append(f"bit_{mapped}")
    return labels


def _dram_bit_labels(hw: HardwareSpec) -> List[str]:
    labels: List[str] = []
    labels.extend(f"col_bit_{idx}" for idx in range(hw.column_bits))
    labels.extend(f"row_bit_{idx}" for idx in range(hw.row_bits))
    labels.extend(f"bank_bit_{idx}" for idx in range(hw.bank_bits))
    labels.extend(f"slice_bit_{idx}" for idx in range(hw.slave_bits))
    return labels[: hw.remap_bits]
