import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .hardware import HardwareSpec


EXTENT_RE = re.compile(r"^(?P<axis>[A-Za-z_][A-Za-z0-9_]*)(?://(?P<divisor>[1-9][0-9]*))?$")


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class FactorSpec:
    name: str
    parent_axis: str
    extent_expr: object
    kind: str

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FactorSpec":
        return cls(
            name=str(data["name"]),
            parent_axis=str(data["parent_axis"]),
            extent_expr=data["extent"],
            kind=str(data.get("kind", "tile")),
        )

    def extent_value(self, shape_bindings: Dict[str, int]) -> int:
        expr = self.extent_expr
        if isinstance(expr, int):
            return expr
        if isinstance(expr, str):
            compact = expr.replace(" ", "")
            if compact.isdigit():
                return int(compact)
            match = EXTENT_RE.match(compact)
            if not match:
                raise LayoutError(f"Unsupported extent expression '{expr}' for factor '{self.name}'.")
            axis = match.group("axis")
            if axis not in shape_bindings:
                raise LayoutError(f"Axis '{axis}' in factor '{self.name}' is missing from shape bindings.")
            divisor = int(match.group("divisor") or "1")
            axis_size = shape_bindings[axis]
            if axis_size % divisor != 0:
                raise LayoutError(
                    f"Axis '{axis}' with size {axis_size} is not divisible by {divisor} "
                    f"for factor '{self.name}'."
                )
            return axis_size // divisor
        raise LayoutError(f"Unsupported extent type for factor '{self.name}': {type(expr)!r}")


@dataclass(frozen=True)
class BoundFactor:
    name: str
    parent_axis: str
    extent: int
    kind: str

    @property
    def bits(self) -> int:
        if self.extent <= 0:
            raise LayoutError(f"Factor '{self.name}' has non-positive extent {self.extent}.")
        if self.extent & (self.extent - 1):
            raise LayoutError(f"Factor '{self.name}' extent {self.extent} is not a power of two.")
        return int(math.log2(self.extent))


@dataclass(frozen=True)
class AxisRefinement:
    segment_bit_sizes: Tuple[int, ...]


@dataclass(frozen=True)
class RefinedAxisSegment:
    axis: str
    segment_index: int
    bit_size: int
    low_bit: int
    high_bit: int
    canonical_name: str


@dataclass(frozen=True)
class RefinedLayoutBit:
    axis: str
    bit_index: int
    label: str
    segment_label: str
    canonical_factor_name: str


@dataclass(frozen=True)
class BlockNormalizedLayout:
    all_bits: Tuple[RefinedLayoutBit, ...]
    outer_bits: Tuple[RefinedLayoutBit, ...]
    inner_bits: Tuple[RefinedLayoutBit, ...]


@dataclass(frozen=True)
class BoundLayout:
    dtype: str
    logical_shape: Dict[str, int]
    ordered_factors: Tuple[BoundFactor, ...]


@dataclass(frozen=True)
class LayoutPartition:
    dtype: str
    outer_factors: Tuple[BoundFactor, ...]
    intra_factors: Tuple[BoundFactor, ...]
    block_elements: int
    outer_bits: int
    intra_bits: int


@dataclass(frozen=True)
class LayoutSpec:
    dtype: str
    logical_shape: Dict[str, str]
    factors: Tuple[FactorSpec, ...]
    linear_order: Tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LayoutSpec":
        factors = tuple(FactorSpec.from_dict(item) for item in data["factors"])
        return cls(
            dtype=str(data["dtype"]),
            logical_shape={str(k): str(v) for k, v in dict(data["logical_shape"]).items()},
            factors=factors,
            linear_order=tuple(str(name) for name in data["linear_order"]),
        )

    def bind(self, shape_bindings: Dict[str, int]) -> BoundLayout:
        unresolved = set(self.logical_shape.values()) - set(shape_bindings.keys())
        if unresolved:
            raise LayoutError(f"Missing shape bindings for axes: {sorted(unresolved)}")

        logical_shape = {
            axis_name: shape_bindings[binding_name]
            for axis_name, binding_name in self.logical_shape.items()
        }
        factor_map = {factor.name: factor for factor in self.factors}

        ordered_factors: List[BoundFactor] = []
        seen = set()
        for name in self.linear_order:
            if name not in factor_map:
                raise LayoutError(f"Factor '{name}' is referenced in linear_order but not defined.")
            factor = factor_map[name]
            ordered_factors.append(
                BoundFactor(
                    name=factor.name,
                    parent_axis=factor.parent_axis,
                    extent=factor.extent_value(shape_bindings),
                    kind=factor.kind,
                )
            )
            seen.add(name)

        missing = [factor.name for factor in self.factors if factor.name not in seen]
        if missing:
            raise LayoutError(f"Factors missing from linear_order: {missing}")

        self._validate_axis_coverage(logical_shape, ordered_factors)
        return BoundLayout(
            dtype=self.dtype,
            logical_shape=logical_shape,
            ordered_factors=tuple(ordered_factors),
        )

    @staticmethod
    def _validate_axis_coverage(logical_shape: Dict[str, int], ordered_factors: Sequence[BoundFactor]) -> None:
        per_axis: Dict[str, int] = {axis: 1 for axis in logical_shape}
        for factor in ordered_factors:
            if factor.parent_axis not in logical_shape:
                raise LayoutError(
                    f"Factor '{factor.name}' references unknown logical axis '{factor.parent_axis}'."
                )
            per_axis[factor.parent_axis] *= factor.extent

        for axis, expected in logical_shape.items():
            if per_axis[axis] != expected:
                raise LayoutError(
                    f"Factors covering axis '{axis}' multiply to {per_axis[axis]}, expected {expected}."
                )


def partition_layout(bound: BoundLayout, hardware: HardwareSpec) -> LayoutPartition:
    block_elements = hardware.block_elements(bound.dtype)
    intra_product = 1
    split_index = len(bound.ordered_factors)

    for idx in range(len(bound.ordered_factors) - 1, -1, -1):
        next_product = intra_product * bound.ordered_factors[idx].extent
        if next_product > block_elements:
            break
        intra_product = next_product
        split_index = idx
        if intra_product == block_elements:
            break

    if intra_product != block_elements:
        raise LayoutError(
            f"Fastest factors do not pack exactly into one 128-bit block for dtype {bound.dtype}: "
            f"expected {block_elements} elements, got {intra_product}."
        )

    outer_factors = tuple(bound.ordered_factors[:split_index])
    intra_factors = tuple(bound.ordered_factors[split_index:])
    outer_bits = sum(factor.bits for factor in outer_factors)
    intra_bits = sum(factor.bits for factor in intra_factors)
    expected_intra_bits = int(math.log2(block_elements))
    if intra_bits != expected_intra_bits:
        raise LayoutError(
            f"Internal block factors contribute {intra_bits} bits, expected {expected_intra_bits}."
        )
    return LayoutPartition(
        dtype=bound.dtype,
        outer_factors=outer_factors,
        intra_factors=intra_factors,
        block_elements=block_elements,
        outer_bits=outer_bits,
        intra_bits=intra_bits,
    )


def refined_axis_segments(
    logical_shape: Dict[str, int],
    source_factors: Sequence[BoundFactor],
    target_factors: Sequence[BoundFactor],
) -> Dict[str, AxisRefinement]:
    refinements: Dict[str, AxisRefinement] = {}
    axes = sorted(logical_shape)
    source_by_axis = _group_factors_by_axis(source_factors)
    target_by_axis = _group_factors_by_axis(target_factors)
    for axis in axes:
        axis_size = logical_shape[axis]
        if axis_size <= 0 or axis_size & (axis_size - 1):
            raise LayoutError(f"Logical axis '{axis}' size {axis_size} is not a power of two.")
        total_bits = int(math.log2(axis_size))
        src_boundaries = _factor_boundaries(source_by_axis.get(axis, ()))
        tgt_boundaries = _factor_boundaries(target_by_axis.get(axis, ()))
        if src_boundaries[-1] != total_bits or tgt_boundaries[-1] != total_bits:
            raise LayoutError(f"Axis '{axis}' is not fully covered during refinement.")
        merged = sorted(set(src_boundaries + tgt_boundaries))
        segment_sizes: List[int] = []
        prev = 0
        for boundary in merged:
            segment_sizes.append(boundary - prev)
            prev = boundary
        refinements[axis] = AxisRefinement(segment_bit_sizes=tuple(segment_sizes))
    return refinements


def build_refined_axis_segment_defs(
    logical_shape: Dict[str, int],
    refinements: Dict[str, AxisRefinement],
) -> Dict[str, Tuple[RefinedAxisSegment, ...]]:
    segment_defs: Dict[str, Tuple[RefinedAxisSegment, ...]] = {}
    for axis, refinement in refinements.items():
        axis_size = logical_shape[axis]
        total_bits = int(math.log2(axis_size))
        cursor = total_bits
        per_axis: List[RefinedAxisSegment] = []
        for segment_index, bit_size in enumerate(refinement.segment_bit_sizes):
            high_bit = cursor - 1
            low_bit = high_bit - bit_size + 1
            per_axis.append(
                RefinedAxisSegment(
                    axis=axis,
                    segment_index=segment_index,
                    bit_size=bit_size,
                    low_bit=low_bit,
                    high_bit=high_bit,
                    canonical_name=_canonical_segment_name(axis, total_bits, low_bit, high_bit),
                )
            )
            cursor = low_bit
        if cursor != 0:
            raise LayoutError(f"Axis '{axis}' refinement leaves {cursor} unmatched bits.")
        segment_defs[axis] = tuple(per_axis)
    return segment_defs


def expand_factors_to_segments(
    factors: Sequence[BoundFactor],
    refinements: Dict[str, AxisRefinement],
) -> List[Tuple[str, int]]:
    grouped = _group_factors_by_axis(factors)
    per_axis_segment_index = {axis: 0 for axis in refinements}
    expanded: List[Tuple[str, int]] = []
    for factor in factors:
        segment_sizes = refinements[factor.parent_axis].segment_bit_sizes
        cursor = per_axis_segment_index[factor.parent_axis]
        bits_left = factor.bits
        while bits_left > 0:
            current_segment_bits = segment_sizes[cursor]
            if current_segment_bits > bits_left:
                raise LayoutError(
                    f"Refinement overflow on factor '{factor.name}' for axis '{factor.parent_axis}'."
                )
            label = f"{factor.parent_axis}:seg{cursor}"
            expanded.append((label, current_segment_bits))
            bits_left -= current_segment_bits
            cursor += 1
        per_axis_segment_index[factor.parent_axis] = cursor

    for axis, used in per_axis_segment_index.items():
        expected = len(refinements[axis].segment_bit_sizes)
        if grouped.get(axis) and used != expected:
            raise LayoutError(f"Axis '{axis}' did not consume all refined segments.")
    return expanded


def expand_factors_to_segment_defs(
    factors: Sequence[BoundFactor],
    refinements: Dict[str, AxisRefinement],
    segment_defs: Dict[str, Tuple[RefinedAxisSegment, ...]],
) -> List[RefinedAxisSegment]:
    per_axis_segment_index = {axis: 0 for axis in refinements}
    expanded: List[RefinedAxisSegment] = []
    for factor in factors:
        axis_segment_defs = segment_defs[factor.parent_axis]
        cursor = per_axis_segment_index[factor.parent_axis]
        bits_left = factor.bits
        while bits_left > 0:
            segment_def = axis_segment_defs[cursor]
            if segment_def.bit_size > bits_left:
                raise LayoutError(
                    f"Refinement overflow on factor '{factor.name}' for axis '{factor.parent_axis}'."
                )
            expanded.append(segment_def)
            bits_left -= segment_def.bit_size
            cursor += 1
        per_axis_segment_index[factor.parent_axis] = cursor
    return expanded


def refined_layout_bits(
    logical_shape: Dict[str, int],
    factors: Sequence[BoundFactor],
    refinements: Dict[str, AxisRefinement],
) -> List[RefinedLayoutBit]:
    segment_defs = build_refined_axis_segment_defs(logical_shape, refinements)
    ordered_segments = expand_factors_to_segment_defs(factors, refinements, segment_defs)
    bits: List[RefinedLayoutBit] = []
    for segment in reversed(ordered_segments):
        for bit_index in range(segment.low_bit, segment.high_bit + 1):
            bits.append(
                RefinedLayoutBit(
                    axis=segment.axis,
                    bit_index=bit_index,
                    label=f"{segment.axis}:bit{bit_index + 1}",
                    segment_label=f"{segment.axis}:seg{segment.segment_index}",
                    canonical_factor_name=segment.canonical_name,
                )
            )
    return bits


def normalize_bound_layout_to_block(
    bound: BoundLayout,
    refinements: Dict[str, AxisRefinement],
    inner_bit_count: int,
) -> BlockNormalizedLayout:
    bits = tuple(refined_layout_bits(bound.logical_shape, bound.ordered_factors, refinements))
    if inner_bit_count < 0 or inner_bit_count > len(bits):
        raise LayoutError(
            f"Inner bit count {inner_bit_count} is out of range for layout with {len(bits)} bits."
        )
    return BlockNormalizedLayout(
        all_bits=bits,
        inner_bits=bits[:inner_bit_count],
        outer_bits=bits[inner_bit_count:],
    )


def segment_sequence_to_bits(segments: Iterable[Tuple[str, int]]) -> List[str]:
    labels: List[str] = []
    for segment_label, segment_bits in reversed(list(segments)):
        for bit_index in range(segment_bits):
            labels.append(f"{segment_label}:bit{bit_index}")
    return labels


def _group_factors_by_axis(factors: Sequence[BoundFactor]) -> Dict[str, Tuple[BoundFactor, ...]]:
    grouped: Dict[str, List[BoundFactor]] = {}
    for factor in factors:
        grouped.setdefault(factor.parent_axis, []).append(factor)
    return {axis: tuple(values) for axis, values in grouped.items()}


def _factor_boundaries(factors: Sequence[BoundFactor]) -> List[int]:
    boundaries: List[int] = []
    cursor = 0
    for factor in factors:
        cursor += factor.bits
        boundaries.append(cursor)
    if not boundaries:
        boundaries.append(0)
    return boundaries


def _canonical_segment_name(axis: str, total_bits: int, low_bit: int, high_bit: int) -> str:
    segment_bits = high_bit - low_bit + 1
    extent = 1 << segment_bits
    if low_bit == 0:
        return f"{axis.lower()}{extent}"
    if high_bit == total_bits - 1:
        return f"{axis}_outer{1 << low_bit}"
    return f"{axis.lower()}{extent}"
