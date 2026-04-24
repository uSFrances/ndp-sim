from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from .hardware import HardwareSpec
from .layout import (
    BoundFactor,
    BoundLayout,
    BlockNormalizedLayout,
    LayoutError,
    LayoutSpec,
    expand_factors_to_segments,
    normalize_bound_layout_to_block,
    partition_layout,
    refined_axis_segments,
    refined_layout_bits,
    segment_sequence_to_bits,
)


@dataclass(frozen=True)
class EdgeSolveResult:
    status: str
    permutation: List[int]
    reason: Optional[str]
    reason_code: Optional[str]
    producer: str
    consumer: str
    tensor_name: str
    write_reg_required: bool = False
    write_reg_hint: Optional[str] = None
    shape_bindings: Optional[Dict[str, int]] = None
    producer_bound_layout: Optional[Dict[str, object]] = None
    consumer_bound_layout: Optional[Dict[str, object]] = None
    producer_axis_aliases: Optional[Dict[str, str]] = None
    consumer_axis_aliases: Optional[Dict[str, str]] = None
    producer_visible_outer_bits: Optional[List[str]] = None
    consumer_visible_outer_bits: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "permutation": self.permutation,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "producer": self.producer,
            "consumer": self.consumer,
            "tensor_name": self.tensor_name,
            "write_reg_required": self.write_reg_required,
            "write_reg_hint": self.write_reg_hint,
            "shape_bindings": self.shape_bindings,
            "producer_bound_layout": self.producer_bound_layout,
            "consumer_bound_layout": self.consumer_bound_layout,
            "producer_axis_aliases": self.producer_axis_aliases,
            "consumer_axis_aliases": self.consumer_axis_aliases,
            "producer_visible_outer_bits": self.producer_visible_outer_bits,
            "consumer_visible_outer_bits": self.consumer_visible_outer_bits,
        }


@dataclass(frozen=True)
class SolvePath:
    producer_outer_bits: List[str]
    consumer_outer_bits: List[str]
    write_reg_required: bool = False
    write_reg_hint: Optional[str] = None


@dataclass(frozen=True)
class BoundLayoutVariant:
    bound: BoundLayout
    write_reg_required: bool = False
    write_reg_hint: Optional[str] = None


def solve_edge(
    producer_layout: LayoutSpec,
    consumer_layout: LayoutSpec,
    shape_bindings: Dict[str, int],
    memory_dtype: Optional[str] = None,
    hw_cfg: Optional[HardwareSpec] = None,
    producer: str = "",
    consumer: str = "",
    tensor_name: str = "",
    producer_axis_aliases: Optional[Dict[str, str]] = None,
    consumer_axis_aliases: Optional[Dict[str, str]] = None,
) -> EdgeSolveResult:
    hardware = hw_cfg or HardwareSpec()
    try:
        producer_bound = producer_layout.bind(shape_bindings)
        consumer_bound = consumer_layout.bind(shape_bindings)
    except LayoutError as exc:
        code = "unsupported extent" if "power of two" in str(exc) else "factor mismatch"
        return _failure(code, str(exc), hardware, producer, consumer, tensor_name)

    resolved_dtype = memory_dtype or producer_bound.dtype
    if resolved_dtype != producer_bound.dtype or resolved_dtype != consumer_bound.dtype:
        return _failure(
            "dtype/block packing",
            "Producer and consumer memory layout dtype do not match the tensor memory dtype.",
            hardware,
            producer,
            consumer,
            tensor_name,
        )

    try:
        _validate_factor_bits(producer_bound)
        _validate_factor_bits(consumer_bound)
    except LayoutError as exc:
        return _failure(
            "unsupported extent",
            str(exc),
            hardware,
            producer,
            consumer,
            tensor_name,
        )

    producer_bound = _canonicalize_bound_layout(producer_bound, producer_axis_aliases or {})
    consumer_bound = _canonicalize_bound_layout(consumer_bound, consumer_axis_aliases or {})

    solve_path = _select_solve_path(producer_bound, consumer_bound, hardware)
    if solve_path is None:
        return _failure(
            "factor mismatch",
            "Producer and consumer outer factors do not refine to the same bit slices.",
            hardware,
            producer,
            consumer,
            tensor_name,
        )

    producer_outer_bits = solve_path.producer_outer_bits
    consumer_outer_bits = solve_path.consumer_outer_bits

    if len(producer_outer_bits) > hardware.remap_bits:
        return _failure(
            "non-permutation reorder",
            f"Outer layout requires {len(producer_outer_bits)} remap bits, exceeding hardware limit {hardware.remap_bits}.",
            hardware,
            producer,
            consumer,
            tensor_name,
        )

    source_indices = {label: idx for idx, label in enumerate(producer_outer_bits)}
    permutation = list(range(hardware.remap_bits))
    for idx, label in enumerate(consumer_outer_bits):
        permutation[idx] = source_indices[label]

    for idx in range(len(consumer_outer_bits), hardware.remap_bits):
        permutation[idx] = idx

    return EdgeSolveResult(
        status="ok",
        permutation=permutation,
        reason=None,
        reason_code=None,
        producer=producer,
        consumer=consumer,
        tensor_name=tensor_name,
        write_reg_required=solve_path.write_reg_required,
        write_reg_hint=solve_path.write_reg_hint,
        shape_bindings=dict(shape_bindings),
        producer_bound_layout=_serialize_bound_layout(producer_layout.bind(shape_bindings)),
        consumer_bound_layout=_serialize_bound_layout(consumer_layout.bind(shape_bindings)),
        producer_axis_aliases=dict(producer_axis_aliases or {}),
        consumer_axis_aliases=dict(consumer_axis_aliases or {}),
        producer_visible_outer_bits=list(producer_outer_bits),
        consumer_visible_outer_bits=list(consumer_outer_bits),
    )


def _select_solve_path(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    candidates = (
        _try_gemm_fp16_pre_write_reg_bits(producer_bound, consumer_bound, hardware),
        _try_block_normalized_bits(producer_bound, consumer_bound, hardware),
        _try_identical_layout_bits(producer_bound, consumer_bound, hardware),
        _try_single_axis_contiguous_bits(producer_bound, consumer_bound, hardware),
        _try_legacy_outer_bits(producer_bound, consumer_bound, hardware),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        producer_bits = candidate.producer_outer_bits
        consumer_bits = candidate.consumer_outer_bits
        if Counter(producer_bits) != Counter(consumer_bits):
            continue
        return candidate
    return None


def _try_gemm_fp16_pre_write_reg_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    if producer_bound.dtype != "fp16" or consumer_bound.dtype != "fp16":
        return None
    if producer_bound.logical_shape != consumer_bound.logical_shape:
        return None

    target_factors = _gemm_fp16_pre_write_reg_target_factors(consumer_bound.ordered_factors)
    if target_factors is None:
        return None

    target_bound = BoundLayout(
        dtype=consumer_bound.dtype,
        logical_shape=consumer_bound.logical_shape,
        ordered_factors=target_factors,
    )

    try:
        remap_refinements = refined_axis_segments(
            producer_bound.logical_shape,
            producer_bound.ordered_factors,
            target_bound.ordered_factors,
        )
        producer_refined_bits = refined_layout_bits(
            producer_bound.logical_shape,
            producer_bound.ordered_factors,
            remap_refinements,
        )
        target_refined_bits = refined_layout_bits(
            target_bound.logical_shape,
            target_bound.ordered_factors,
            remap_refinements,
        )
        write_reg_refinements = refined_axis_segments(
            target_bound.logical_shape,
            target_bound.ordered_factors,
            consumer_bound.ordered_factors,
        )
        write_reg_source_bits = refined_layout_bits(
            target_bound.logical_shape,
            target_bound.ordered_factors,
            write_reg_refinements,
        )
        write_reg_target_bits = refined_layout_bits(
            consumer_bound.logical_shape,
            consumer_bound.ordered_factors,
            write_reg_refinements,
        )
    except LayoutError:
        return None

    if len(producer_refined_bits) != len(target_refined_bits):
        return None

    inner_bit_count = _visible_subword_layout_bits(target_bound.dtype, hardware)
    target_inner_labels = _preferred_inner_bit_labels(target_refined_bits, inner_bit_count)
    producer_norm = _normalize_bits_by_label(producer_refined_bits, target_inner_labels)
    target_norm = _normalize_bits_by_label(target_refined_bits, target_inner_labels)

    producer_outer_labels = _normalized_bit_labels(producer_norm.outer_bits)
    target_outer_labels = _normalized_bit_labels(target_norm.outer_bits)
    if Counter(producer_outer_labels) != Counter(target_outer_labels):
        return None
    if len(producer_outer_labels) > hardware.remap_bits:
        return None

    write_reg_window_bits = inner_bit_count + _gemm_fp16_exchange_bits(target_bound, consumer_bound)
    write_reg_hint = _preferred_inner_reorder_hint(
        write_reg_source_bits[:write_reg_window_bits],
        write_reg_target_bits[:write_reg_window_bits],
    )
    if write_reg_hint is None:
        return None

    return SolvePath(
        producer_outer_bits=producer_outer_labels,
        consumer_outer_bits=target_outer_labels,
        write_reg_required=True,
        write_reg_hint=write_reg_hint,
    )


def _try_block_normalized_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    if producer_bound.dtype != consumer_bound.dtype:
        return None
    if producer_bound.logical_shape != consumer_bound.logical_shape:
        return None

    try:
        refinements = refined_axis_segments(
            producer_bound.logical_shape,
            producer_bound.ordered_factors,
            consumer_bound.ordered_factors,
        )
        producer_refined_bits = refined_layout_bits(
            producer_bound.logical_shape,
            producer_bound.ordered_factors,
            refinements,
        )
        consumer_refined_bits = refined_layout_bits(
            consumer_bound.logical_shape,
            consumer_bound.ordered_factors,
            refinements,
        )
    except LayoutError:
        return None

    if len(producer_refined_bits) != len(consumer_refined_bits):
        return None

    min_inner_bits = _visible_subword_layout_bits(producer_bound.dtype, hardware)
    producer_pref_bits = _preferred_inner_bit_labels(producer_refined_bits, min_inner_bits)
    consumer_pref_bits = _preferred_inner_bit_labels(consumer_refined_bits, min_inner_bits)
    consumer_pref_set = set(consumer_pref_bits)
    inner_bit_labels = [label for label in producer_pref_bits if label in consumer_pref_set]

    producer_norm = _normalize_bits_by_label(producer_refined_bits, inner_bit_labels)
    consumer_norm = _normalize_bits_by_label(consumer_refined_bits, inner_bit_labels)

    producer_outer_labels = _normalized_bit_labels(producer_norm.outer_bits)
    consumer_outer_labels = _normalized_bit_labels(consumer_norm.outer_bits)
    if Counter(producer_outer_labels) != Counter(consumer_outer_labels):
        return None

    if len(producer_outer_labels) > hardware.remap_bits:
        return None

    producer_inner_labels = _normalized_bit_labels(producer_norm.inner_bits)
    consumer_inner_labels = _normalized_bit_labels(consumer_norm.inner_bits)
    write_reg_required = (
        producer_pref_bits != consumer_pref_bits
        or producer_inner_labels != consumer_inner_labels
    )
    write_reg_hint = None
    if write_reg_required:
        write_reg_hint = _merge_write_reg_hints(
            _preferred_inner_reorder_hint(
                producer_refined_bits[:min_inner_bits],
                consumer_refined_bits[:min_inner_bits],
            ),
            _normalized_inner_reorder_hint(producer_norm, consumer_norm),
        )

    return SolvePath(
        producer_outer_bits=producer_outer_labels,
        consumer_outer_bits=consumer_outer_labels,
        write_reg_required=write_reg_required,
        write_reg_hint=write_reg_hint,
    )


def _try_identical_layout_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    if producer_bound.dtype != consumer_bound.dtype:
        return None
    if producer_bound.logical_shape != consumer_bound.logical_shape:
        return None

    producer_bits = _layout_bit_labels(producer_bound)
    consumer_bits = _layout_bit_labels(consumer_bound)
    if producer_bits != consumer_bits:
        return None

    producer_factors = [
        (factor.name, factor.parent_axis, factor.extent, factor.kind)
        for factor in producer_bound.ordered_factors
    ]
    consumer_factors = [
        (factor.name, factor.parent_axis, factor.extent, factor.kind)
        for factor in consumer_bound.ordered_factors
    ]
    if producer_factors != consumer_factors:
        return None

    low_bits = _visible_subword_layout_bits(producer_bound.dtype, hardware)
    if len(producer_bits) < low_bits or len(consumer_bits) < low_bits:
        return None

    return SolvePath(
        producer_outer_bits=producer_bits[low_bits:],
        consumer_outer_bits=consumer_bits[low_bits:],
        write_reg_required=False,
        write_reg_hint=None,
    )


def _try_single_axis_contiguous_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    if len(producer_bound.logical_shape) != 1 or len(consumer_bound.logical_shape) != 1:
        return None
    if producer_bound.dtype != consumer_bound.dtype:
        return None
    if producer_bound.logical_shape != consumer_bound.logical_shape:
        return None

    producer_bits = _layout_bit_labels(producer_bound)
    consumer_bits = _layout_bit_labels(consumer_bound)
    if producer_bits != consumer_bits:
        return None

    low_bits = _visible_subword_layout_bits(producer_bound.dtype, hardware)
    if len(producer_bits) < low_bits or len(consumer_bits) < low_bits:
        return None

    return SolvePath(
        producer_outer_bits=producer_bits[low_bits:],
        consumer_outer_bits=consumer_bits[low_bits:],
        write_reg_required=False,
        write_reg_hint=None,
    )


def _try_partition_based_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    return None


def _try_partition_with_write_reg_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    return None


def _validate_factor_bits(bound: BoundLayout) -> None:
    for factor in bound.ordered_factors:
        _ = factor.bits


def _try_legacy_outer_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    try:
        refinements = refined_axis_segments(
            producer_bound.logical_shape,
            producer_bound.ordered_factors,
            consumer_bound.ordered_factors,
        )
        producer_bits = segment_sequence_to_bits(
            expand_factors_to_segments(producer_bound.ordered_factors, refinements)
        )
        consumer_bits = segment_sequence_to_bits(
            expand_factors_to_segments(consumer_bound.ordered_factors, refinements)
        )
    except LayoutError:
        return None

    producer_low_bits = _visible_subword_layout_bits(producer_bound.dtype, hardware)
    consumer_low_bits = _visible_subword_layout_bits(consumer_bound.dtype, hardware)
    if producer_low_bits != consumer_low_bits:
        return None
    low_bits = producer_low_bits
    if len(producer_bits) < low_bits or len(consumer_bits) < low_bits:
        return None
    return SolvePath(
        producer_outer_bits=producer_bits[low_bits:],
        consumer_outer_bits=consumer_bits[low_bits:],
    )


def _try_flexible_refined_bits(
    producer_bound: BoundLayout,
    consumer_bound: BoundLayout,
    hardware: HardwareSpec,
) -> Optional[SolvePath]:
    return None


def _candidate_intrablock_variants(bound: BoundLayout, hardware: HardwareSpec) -> List[BoundLayoutVariant]:
    variants: List[BoundLayoutVariant] = [BoundLayoutVariant(bound=bound)]
    seen = {tuple(f.name for f in bound.ordered_factors)}
    max_suffix = min(4, len(bound.ordered_factors))

    for suffix_len in range(2, max_suffix + 1):
        prefix = tuple(bound.ordered_factors[:-suffix_len])
        suffix = tuple(bound.ordered_factors[-suffix_len:])
        for permuted_suffix in permutations(suffix):
            if permuted_suffix == suffix:
                continue
            factor_key = tuple(f.name for f in prefix + permuted_suffix)
            if factor_key in seen:
                continue
            candidate = BoundLayout(
                dtype=bound.dtype,
                logical_shape=dict(bound.logical_shape),
                ordered_factors=prefix + permuted_suffix,
            )
            try:
                partition_layout(candidate, hardware)
            except LayoutError:
                continue
            seen.add(factor_key)
            variants.append(
                BoundLayoutVariant(
                    bound=candidate,
                    write_reg_required=True,
                    write_reg_hint=_factor_reorder_hint(suffix, permuted_suffix),
                )
            )

    return variants


def _factor_reorder_hint(
    source_factors: Sequence[BoundFactor],
    target_factors: Sequence[BoundFactor],
) -> Optional[str]:
    source_names = [factor.name for factor in source_factors]
    target_names = [factor.name for factor in target_factors]
    if source_names == target_names:
        return None
    return f"reorder({','.join(source_names)})->({','.join(target_names)})"


def _intra_factor_reorder_hint(
    source_factors: Sequence[BoundFactor],
    target_factors: Sequence[BoundFactor],
) -> Optional[str]:
    if len(source_factors) == len(target_factors):
        source_signature = [(factor.parent_axis, factor.extent) for factor in source_factors]
        target_signature = [(factor.parent_axis, factor.extent) for factor in target_factors]
        if source_signature == target_signature:
            return None
    source_names = [factor.name for factor in source_factors]
    target_names = [factor.name for factor in target_factors]
    if source_names == target_names:
        return None
    return f"reorder({','.join(source_names)})->({','.join(target_names)})"


def _merge_write_reg_hints(*hints: Optional[str]) -> Optional[str]:
    filtered = [hint for hint in hints if hint]
    if not filtered:
        return None
    return " ; ".join(dict.fromkeys(filtered))


def _canonicalize_bound_layout(bound: BoundLayout, axis_aliases: Dict[str, str]) -> BoundLayout:
    if not axis_aliases:
        return bound

    logical_shape: Dict[str, int] = {}
    for axis, extent in bound.logical_shape.items():
        canonical_axis = axis_aliases.get(axis, axis)
        existing = logical_shape.get(canonical_axis)
        if existing is not None and existing != extent:
            raise LayoutError(
                f"Axis alias '{canonical_axis}' maps to conflicting extents {existing} and {extent}."
            )
        logical_shape[canonical_axis] = extent

    ordered_factors = tuple(
        BoundFactor(
            name=factor.name,
            parent_axis=axis_aliases.get(factor.parent_axis, factor.parent_axis),
            extent=factor.extent,
            kind=factor.kind,
        )
        for factor in bound.ordered_factors
    )
    return BoundLayout(
        dtype=bound.dtype,
        logical_shape=logical_shape,
        ordered_factors=ordered_factors,
    )


def _layout_bit_labels(bound: BoundLayout) -> List[str]:
    factor_labels = _factor_axis_bit_labels(bound.logical_shape, bound.ordered_factors)
    labels: List[str] = []
    for _, bit_labels in reversed(factor_labels):
        labels.extend(bit_labels)
    return labels


def _factor_axis_bit_labels(
    logical_shape: Dict[str, int],
    ordered_factors: Sequence[BoundFactor],
) -> List[Tuple[str, List[str]]]:
    next_high_bit = {
        axis: logical_shape[axis].bit_length() - 1 for axis in logical_shape
    }
    labeled_factors: List[Tuple[str, List[str]]] = []
    for factor in ordered_factors:
        high_bit = next_high_bit[factor.parent_axis]
        low_bit = high_bit - factor.bits + 1
        bit_labels = [
            f"{factor.parent_axis}:bit{bit_index}"
            for bit_index in range(low_bit, high_bit + 1)
        ]
        labeled_factors.append((factor.name, bit_labels))
        next_high_bit[factor.parent_axis] = low_bit - 1
    return labeled_factors


def _failure(reason_code, reason, hardware, producer, consumer, tensor_name):
    return EdgeSolveResult(
        status="unimplemented",
        permutation=list(range(hardware.remap_bits)),
        reason=reason,
        reason_code=reason_code,
        producer=producer,
        consumer=consumer,
        tensor_name=tensor_name,
        write_reg_required=False,
        write_reg_hint=None,
    )


def _serialize_bound_layout(bound: BoundLayout) -> Dict[str, object]:
    return {
        "dtype": bound.dtype,
        "logical_shape": dict(bound.logical_shape),
        "ordered_factors": [
            {
                "name": factor.name,
                "parent_axis": factor.parent_axis,
                "extent": factor.extent,
                "kind": factor.kind,
                "bits": factor.bits,
            }
            for factor in bound.ordered_factors
        ],
    }


def _visible_subword_layout_bits(dtype: str, hardware: HardwareSpec) -> int:
    return hardware.block_elements(dtype).bit_length() - 1


def _normalized_bit_labels(bits) -> List[str]:
    return [bit.label for bit in bits]


def _normalized_inner_reorder_hint(
    producer_norm: BlockNormalizedLayout,
    consumer_norm: BlockNormalizedLayout,
) -> Optional[str]:
    source_groups = _bit_group_order(producer_norm.inner_bits)
    target_groups = _bit_group_order(consumer_norm.inner_bits)
    if source_groups == target_groups:
        return None
    return f"reorder({','.join(source_groups)})->({','.join(target_groups)})"


def _preferred_inner_reorder_hint(
    producer_inner_bits,
    consumer_inner_bits,
) -> Optional[str]:
    producer_labels = _normalized_bit_labels(producer_inner_bits)
    consumer_labels = _normalized_bit_labels(consumer_inner_bits)
    if producer_labels == consumer_labels:
        return None
    source_groups = _bit_group_order(producer_inner_bits)
    target_groups = _bit_group_order(consumer_inner_bits)
    if source_groups == target_groups:
        return None
    return f"reorder({','.join(source_groups)})->({','.join(target_groups)})"


def _bit_group_order(bits) -> List[str]:
    groups: List[str] = []
    for bit in bits:
        if not groups or groups[-1] != bit.canonical_factor_name:
            groups.append(bit.canonical_factor_name)
    return groups


def _preferred_inner_bit_labels(bits, min_inner_bits: int) -> List[str]:
    return [bit.label for bit in bits[:min_inner_bits]]


def _gemm_fp16_pre_write_reg_target_factors(
    factors: Sequence[BoundFactor],
) -> Optional[Tuple[BoundFactor, ...]]:
    if len(factors) < 2:
        return None
    penultimate = factors[-2]
    last = factors[-1]
    if penultimate.extent != 8 or last.extent != 2:
        return None
    if penultimate.parent_axis == last.parent_axis:
        return None
    return tuple(factors[:-2]) + (last, penultimate)


def _gemm_fp16_exchange_bits(target_bound: BoundLayout, consumer_bound: BoundLayout) -> int:
    if not target_bound.ordered_factors or not consumer_bound.ordered_factors:
        return 0
    return min(target_bound.ordered_factors[-2].bits, consumer_bound.ordered_factors[-1].bits)


def _normalize_bits_by_label(bits, inner_bit_labels: Sequence[str]) -> BlockNormalizedLayout:
    inner_bit_set = set(inner_bit_labels)
    inner_bits = tuple(bit for bit in bits if bit.label in inner_bit_set)
    outer_bits = tuple(bit for bit in bits if bit.label not in inner_bit_set)
    return BlockNormalizedLayout(
        all_bits=tuple(bits),
        inner_bits=inner_bits,
        outer_bits=outer_bits,
    )
