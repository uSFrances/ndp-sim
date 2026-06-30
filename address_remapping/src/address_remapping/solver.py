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
    layout_permutation: Optional[List[int]] = None
    physical_permutation: Optional[List[int]] = None
    composed_permutation: Optional[List[int]] = None
    output_permutation: Optional[List[int]] = None
    input_permutation: Optional[List[int]] = None
    layout_matrix: Optional[List[List[int]]] = None
    physical_matrix: Optional[List[List[int]]] = None
    composed_matrix: Optional[List[List[int]]] = None
    output_matrix: Optional[List[List[int]]] = None
    input_matrix: Optional[List[List[int]]] = None
    bank_interleave_count: int = 1
    physical_bit_labels: Optional[List[str]] = None
    physical_field_labels: Optional[List[str]] = None
    physical_placement_policy: str = "identity"
    producer_port: Optional[str] = None
    consumer_port: Optional[str] = None
    producer_op_type: Optional[str] = None
    consumer_op_type: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        layout_permutation_public = _export_new_to_old_permutation(
            self.layout_permutation or self.permutation,
            len(self.layout_permutation or self.permutation),
        )
        physical_permutation_public = _export_new_to_old_permutation(
            self.physical_permutation or _identity_permutation(len(layout_permutation_public)),
            len(self.physical_permutation or _identity_permutation(len(layout_permutation_public))),
        )
        composed_permutation_public = _export_new_to_old_permutation(
            self.composed_permutation or self.permutation,
            len(self.composed_permutation or self.permutation),
        )
        output_permutation_public = _export_new_to_old_permutation(
            self.output_permutation or composed_permutation_public,
            len(self.output_permutation or composed_permutation_public),
        )
        input_permutation_public = _export_new_to_old_permutation(
            self.input_permutation or physical_permutation_public,
            len(self.input_permutation or physical_permutation_public),
        )
        layout_permutation_internal = _invert_new_to_old_permutation(
            layout_permutation_public,
            len(layout_permutation_public),
        )
        physical_permutation_internal = _invert_new_to_old_permutation(
            physical_permutation_public,
            len(physical_permutation_public),
        )
        composed_permutation_internal = _invert_new_to_old_permutation(
            composed_permutation_public,
            len(composed_permutation_public),
        )
        output_permutation_internal = _invert_new_to_old_permutation(
            output_permutation_public,
            len(output_permutation_public),
        )
        input_permutation_internal = _invert_new_to_old_permutation(
            input_permutation_public,
            len(input_permutation_public),
        )
        return {
            "status": self.status,
            "permutation": list(output_permutation_public),
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
            "layout_permutation": list(layout_permutation_public),
            "physical_permutation": list(physical_permutation_public),
            "composed_permutation": list(composed_permutation_public),
            "output_permutation": list(output_permutation_public),
            "input_permutation": list(input_permutation_public),
            "internal_permutation": list(output_permutation_internal),
            "internal_layout_permutation": layout_permutation_internal,
            "internal_physical_permutation": physical_permutation_internal,
            "internal_composed_permutation": composed_permutation_internal,
            "internal_output_permutation": output_permutation_internal,
            "internal_input_permutation": input_permutation_internal,
            "layout_matrix": self.layout_matrix or _permutation_matrix(layout_permutation_internal),
            "physical_matrix": self.physical_matrix or _permutation_matrix(physical_permutation_internal),
            "composed_matrix": self.composed_matrix or _permutation_matrix(composed_permutation_internal),
            "output_matrix": self.output_matrix or _permutation_matrix(output_permutation_internal),
            "input_matrix": self.input_matrix or _permutation_matrix(input_permutation_internal),
            "bank_interleave_count": int(self.bank_interleave_count),
            "physical_bit_labels": list(self.physical_bit_labels or []),
            "physical_field_labels": list(self.physical_field_labels or []),
            "physical_placement_policy": self.physical_placement_policy,
            "producer_port": self.producer_port,
            "consumer_port": self.consumer_port,
            "producer_op_type": self.producer_op_type,
            "consumer_op_type": self.consumer_op_type,
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
    bank_interleave_count: int = 1,
    producer_port: Optional[str] = None,
    consumer_port: Optional[str] = None,
    producer_op_type: Optional[str] = None,
    consumer_op_type: Optional[str] = None,
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

    layout_permutation = _solve_layout_permutation(
        producer_outer_bits=producer_outer_bits,
        consumer_outer_bits=consumer_outer_bits,
        hardware=hardware,
    )
    physical_result = _solve_physical_permutation(
        layout_permutation=layout_permutation,
        visible_bit_count=len(consumer_outer_bits),
        bank_interleave_count=bank_interleave_count,
        hardware=hardware,
    )
    if physical_result["status"] != "ok":
        return _failure(
            str(physical_result["reason_code"]),
            str(physical_result["reason"]),
            hardware,
            producer,
            consumer,
            tensor_name,
            producer_port=producer_port,
            consumer_port=consumer_port,
            producer_op_type=producer_op_type,
            consumer_op_type=consumer_op_type,
            bank_interleave_count=bank_interleave_count,
        )
    physical_permutation = list(physical_result["permutation"])
    composed_permutation = _compose_permutations(layout_permutation, physical_permutation)
    input_permutation = list(physical_permutation)
    output_permutation = list(composed_permutation)

    return EdgeSolveResult(
        status="ok",
        permutation=output_permutation,
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
        layout_permutation=layout_permutation,
        physical_permutation=physical_permutation,
        composed_permutation=composed_permutation,
        output_permutation=output_permutation,
        input_permutation=input_permutation,
        layout_matrix=_permutation_matrix(layout_permutation),
        physical_matrix=_permutation_matrix(physical_permutation),
        composed_matrix=_permutation_matrix(composed_permutation),
        output_matrix=_permutation_matrix(output_permutation),
        input_matrix=_permutation_matrix(input_permutation),
        bank_interleave_count=int(bank_interleave_count),
        physical_bit_labels=_dram_bit_labels(hardware),
        physical_field_labels=list(physical_result["field_labels"]),
        physical_placement_policy=str(physical_result["placement_policy"]),
        producer_port=producer_port,
        consumer_port=consumer_port,
        producer_op_type=producer_op_type,
        consumer_op_type=consumer_op_type,
    )


def solve_external_input_physical_remap(
    consumer_layout: LayoutSpec,
    shape_bindings: Dict[str, int],
    *,
    hw_cfg: Optional[HardwareSpec] = None,
    consumer: str = "",
    tensor_name: str = "",
    bank_interleave_count: int = 1,
    consumer_port: Optional[str] = None,
    consumer_op_type: Optional[str] = None,
    consumer_axis_aliases: Optional[Dict[str, str]] = None,
) -> EdgeSolveResult:
    hardware = hw_cfg or HardwareSpec()
    try:
        consumer_bound = consumer_layout.bind(shape_bindings)
    except LayoutError as exc:
        code = "unsupported extent" if "power of two" in str(exc) else "factor mismatch"
        return _failure(
            code,
            str(exc),
            hardware,
            "external",
            consumer,
            tensor_name,
            producer_port="external",
            consumer_port=consumer_port,
            producer_op_type="external",
            consumer_op_type=consumer_op_type,
            bank_interleave_count=bank_interleave_count,
        )

    consumer_bound = _canonicalize_bound_layout(consumer_bound, consumer_axis_aliases or {})
    consumer_bits = _layout_bit_labels_relaxed(consumer_bound)
    low_bits = _visible_subword_layout_bits(consumer_bound.dtype, hardware)
    if len(consumer_bits) < low_bits:
        return _failure(
            "unsupported extent",
            "Consumer layout exposes fewer bits than the visible subword width.",
            hardware,
            "external",
            consumer,
            tensor_name,
            producer_port="external",
            consumer_port=consumer_port,
            producer_op_type="external",
            consumer_op_type=consumer_op_type,
            bank_interleave_count=bank_interleave_count,
        )

    consumer_outer_bits = consumer_bits[low_bits:]
    if len(consumer_outer_bits) > hardware.remap_bits:
        return _failure(
            "non-permutation reorder",
            f"External input layout requires {len(consumer_outer_bits)} remap bits, exceeding hardware limit {hardware.remap_bits}.",
            hardware,
            "external",
            consumer,
            tensor_name,
            producer_port="external",
            consumer_port=consumer_port,
            producer_op_type="external",
            consumer_op_type=consumer_op_type,
            bank_interleave_count=bank_interleave_count,
        )

    layout_permutation = list(range(hardware.remap_bits))
    physical_result = _solve_physical_permutation(
        layout_permutation=layout_permutation,
        visible_bit_count=len(consumer_outer_bits),
        bank_interleave_count=bank_interleave_count,
        hardware=hardware,
    )
    if physical_result["status"] != "ok":
        return _failure(
            str(physical_result["reason_code"]),
            str(physical_result["reason"]),
            hardware,
            "external",
            consumer,
            tensor_name,
            producer_port="external",
            consumer_port=consumer_port,
            producer_op_type="external",
            consumer_op_type=consumer_op_type,
            bank_interleave_count=bank_interleave_count,
        )

    physical_permutation = list(physical_result["permutation"])
    return EdgeSolveResult(
        status="ok",
        permutation=list(physical_permutation),
        reason=None,
        reason_code=None,
        producer="external",
        consumer=consumer,
        tensor_name=tensor_name,
        write_reg_required=False,
        write_reg_hint=None,
        shape_bindings=dict(shape_bindings),
        producer_bound_layout=None,
        consumer_bound_layout=_serialize_bound_layout_relaxed(consumer_bound),
        producer_axis_aliases={},
        consumer_axis_aliases=dict(consumer_axis_aliases or {}),
        producer_visible_outer_bits=list(consumer_outer_bits),
        consumer_visible_outer_bits=list(consumer_outer_bits),
        layout_permutation=list(layout_permutation),
        physical_permutation=list(physical_permutation),
        composed_permutation=list(physical_permutation),
        output_permutation=list(physical_permutation),
        input_permutation=list(physical_permutation),
        layout_matrix=_permutation_matrix(layout_permutation),
        physical_matrix=_permutation_matrix(physical_permutation),
        composed_matrix=_permutation_matrix(physical_permutation),
        output_matrix=_permutation_matrix(physical_permutation),
        input_matrix=_permutation_matrix(physical_permutation),
        bank_interleave_count=int(bank_interleave_count),
        physical_bit_labels=_dram_bit_labels(hardware),
        physical_field_labels=list(physical_result["field_labels"]),
        physical_placement_policy=str(physical_result["placement_policy"]),
        producer_port="external",
        consumer_port=consumer_port,
        producer_op_type="external",
        consumer_op_type=consumer_op_type,
    )


def solve_terminal_output_physical_remap(
    producer_layout: LayoutSpec,
    shape_bindings: Dict[str, int],
    *,
    hw_cfg: Optional[HardwareSpec] = None,
    producer: str = "",
    tensor_name: str = "",
    bank_interleave_count: int = 1,
    producer_port: Optional[str] = None,
    producer_op_type: Optional[str] = None,
    producer_axis_aliases: Optional[Dict[str, str]] = None,
) -> EdgeSolveResult:
    hardware = hw_cfg or HardwareSpec()
    try:
        producer_bound = producer_layout.bind(shape_bindings)
    except LayoutError as exc:
        code = "unsupported extent" if "power of two" in str(exc) else "factor mismatch"
        return _failure(
            code,
            str(exc),
            hardware,
            producer,
            "terminal_output",
            tensor_name,
            producer_port=producer_port,
            consumer_port="terminal_output",
            producer_op_type=producer_op_type,
            consumer_op_type="terminal_output",
            bank_interleave_count=bank_interleave_count,
        )

    producer_bound = _canonicalize_bound_layout(producer_bound, producer_axis_aliases or {})
    producer_bits = _layout_bit_labels_relaxed(producer_bound)
    low_bits = _visible_subword_layout_bits(producer_bound.dtype, hardware)
    if len(producer_bits) < low_bits:
        return _failure(
            "unsupported extent",
            "Producer layout exposes fewer bits than the visible subword width.",
            hardware,
            producer,
            "terminal_output",
            tensor_name,
            producer_port=producer_port,
            consumer_port="terminal_output",
            producer_op_type=producer_op_type,
            consumer_op_type="terminal_output",
            bank_interleave_count=bank_interleave_count,
        )

    producer_outer_bits = producer_bits[low_bits:]
    if len(producer_outer_bits) > hardware.remap_bits:
        return _failure(
            "non-permutation reorder",
            f"Terminal output layout requires {len(producer_outer_bits)} remap bits, exceeding hardware limit {hardware.remap_bits}.",
            hardware,
            producer,
            "terminal_output",
            tensor_name,
            producer_port=producer_port,
            consumer_port="terminal_output",
            producer_op_type=producer_op_type,
            consumer_op_type="terminal_output",
            bank_interleave_count=bank_interleave_count,
        )

    layout_permutation = list(range(hardware.remap_bits))
    physical_result = _solve_physical_permutation(
        layout_permutation=layout_permutation,
        visible_bit_count=len(producer_outer_bits),
        bank_interleave_count=bank_interleave_count,
        hardware=hardware,
    )
    if physical_result["status"] != "ok":
        return _failure(
            str(physical_result["reason_code"]),
            str(physical_result["reason"]),
            hardware,
            producer,
            "terminal_output",
            tensor_name,
            producer_port=producer_port,
            consumer_port="terminal_output",
            producer_op_type=producer_op_type,
            consumer_op_type="terminal_output",
            bank_interleave_count=bank_interleave_count,
        )

    physical_permutation = list(physical_result["permutation"])
    return EdgeSolveResult(
        status="ok",
        permutation=list(physical_permutation),
        reason=None,
        reason_code=None,
        producer=producer,
        consumer="terminal_output",
        tensor_name=tensor_name,
        write_reg_required=False,
        write_reg_hint=None,
        shape_bindings=dict(shape_bindings),
        producer_bound_layout=_serialize_bound_layout_relaxed(producer_bound),
        consumer_bound_layout=None,
        producer_axis_aliases=dict(producer_axis_aliases or {}),
        consumer_axis_aliases={},
        producer_visible_outer_bits=list(producer_outer_bits),
        consumer_visible_outer_bits=list(producer_outer_bits),
        layout_permutation=list(layout_permutation),
        physical_permutation=list(physical_permutation),
        composed_permutation=list(physical_permutation),
        output_permutation=list(physical_permutation),
        input_permutation=list(physical_permutation),
        layout_matrix=_permutation_matrix(layout_permutation),
        physical_matrix=_permutation_matrix(physical_permutation),
        composed_matrix=_permutation_matrix(physical_permutation),
        output_matrix=_permutation_matrix(physical_permutation),
        input_matrix=_permutation_matrix(physical_permutation),
        bank_interleave_count=int(bank_interleave_count),
        physical_bit_labels=_dram_bit_labels(hardware),
        physical_field_labels=list(physical_result["field_labels"]),
        physical_placement_policy=str(physical_result["placement_policy"]),
        producer_port=producer_port,
        consumer_port="terminal_output",
        producer_op_type=producer_op_type,
        consumer_op_type="terminal_output",
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


def _layout_bit_labels_relaxed(bound: BoundLayout) -> List[str]:
    factor_labels = _factor_axis_bit_labels(bound.logical_shape, bound.ordered_factors, allow_non_power_of_two=True)
    labels: List[str] = []
    for _, bit_labels in reversed(factor_labels):
        labels.extend(bit_labels)
    return labels


def _factor_axis_bit_labels(
    logical_shape: Dict[str, int],
    ordered_factors: Sequence[BoundFactor],
    allow_non_power_of_two: bool = False,
) -> List[Tuple[str, List[str]]]:
    next_high_bit = {
        axis: logical_shape[axis].bit_length() - 1 for axis in logical_shape
    }
    labeled_factors: List[Tuple[str, List[str]]] = []
    for factor in ordered_factors:
        high_bit = next_high_bit[factor.parent_axis]
        bit_count = _factor_bit_width(factor.extent, allow_non_power_of_two=allow_non_power_of_two)
        low_bit = high_bit - bit_count + 1
        bit_labels = [
            f"{factor.parent_axis}:bit{bit_index}"
            for bit_index in range(low_bit, high_bit + 1)
        ]
        labeled_factors.append((factor.name, bit_labels))
        next_high_bit[factor.parent_axis] = low_bit - 1
    return labeled_factors


def _factor_bit_width(extent: int, *, allow_non_power_of_two: bool = False) -> int:
    if extent <= 0:
        raise LayoutError(f"Factor extent {extent} must be positive.")
    if allow_non_power_of_two:
        return 0 if extent <= 1 else int(extent - 1).bit_length()
    if extent & (extent - 1):
        raise LayoutError(f"Factor extent {extent} is not a power of two.")
    return int(extent).bit_length() - 1


def _failure(
    reason_code,
    reason,
    hardware,
    producer,
    consumer,
    tensor_name,
    *,
    producer_port: Optional[str] = None,
    consumer_port: Optional[str] = None,
    producer_op_type: Optional[str] = None,
    consumer_op_type: Optional[str] = None,
    bank_interleave_count: int = 1,
):
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
        layout_permutation=list(range(hardware.remap_bits)),
        physical_permutation=list(range(hardware.remap_bits)),
        composed_permutation=list(range(hardware.remap_bits)),
        output_permutation=list(range(hardware.remap_bits)),
        input_permutation=list(range(hardware.remap_bits)),
        layout_matrix=_permutation_matrix(list(range(hardware.remap_bits))),
        physical_matrix=_permutation_matrix(list(range(hardware.remap_bits))),
        composed_matrix=_permutation_matrix(list(range(hardware.remap_bits))),
        output_matrix=_permutation_matrix(list(range(hardware.remap_bits))),
        input_matrix=_permutation_matrix(list(range(hardware.remap_bits))),
        bank_interleave_count=int(bank_interleave_count),
        physical_bit_labels=_dram_bit_labels(hardware),
        physical_field_labels=[],
        physical_placement_policy="identity",
        producer_port=producer_port,
        consumer_port=consumer_port,
        producer_op_type=producer_op_type,
        consumer_op_type=consumer_op_type,
    )


def _identity_permutation(width: int) -> List[int]:
    return list(range(int(width)))


def _export_new_to_old_permutation(
    new_to_old: Sequence[int],
    active_width: int,
) -> List[int]:
    return _legalize_new_to_old_permutation(new_to_old, active_width)


def _legalize_new_to_old_permutation(
    new_to_old: Sequence[int],
    active_width: int,
) -> List[int]:
    width = len(new_to_old)
    explicit_new_bits = list(range(min(active_width, width)))
    explicit_old_bits = [int(new_to_old[new_bit]) for new_bit in explicit_new_bits]
    legal = [-1] * width
    used_old_bits = set()
    for new_bit, old_bit in zip(explicit_new_bits, explicit_old_bits):
        if 0 <= old_bit < width and legal[new_bit] == -1 and old_bit not in used_old_bits:
            legal[new_bit] = old_bit
            used_old_bits.add(old_bit)

    remaining_old_bits = [bit for bit in range(width) if bit not in used_old_bits]
    remaining_new_bits = [bit for bit in range(width) if legal[bit] < 0]
    for new_bit, old_bit in zip(remaining_new_bits, remaining_old_bits):
        legal[new_bit] = old_bit
    return [int(bit) for bit in legal]


def _invert_new_to_old_permutation(
    new_to_old: Sequence[int],
    active_width: int,
) -> List[int]:
    full_new_to_old = _legalize_new_to_old_permutation(new_to_old, active_width)
    inverse = [0] * len(full_new_to_old)
    for new_bit, old_bit in enumerate(full_new_to_old):
        inverse[int(old_bit)] = int(new_bit)
    return inverse


def _permutation_matrix(permutation: Sequence[int]) -> List[List[int]]:
    width = len(permutation)
    matrix: List[List[int]] = []
    for row in range(width):
        row_values = [0] * width
        for col, mapped in enumerate(permutation):
            if int(mapped) == row:
                row_values[col] = 1
        matrix.append(row_values)
    return matrix


def _solve_layout_permutation(
    producer_outer_bits: Sequence[str],
    consumer_outer_bits: Sequence[str],
    hardware: HardwareSpec,
) -> List[int]:
    source_indices = {label: idx for idx, label in enumerate(producer_outer_bits)}
    explicit_mapping: Dict[int, int] = {}
    for idx, label in enumerate(consumer_outer_bits):
        explicit_mapping[int(idx)] = int(source_indices[label])
    return _legalize_explicit_new_to_old_mapping(
        width=hardware.remap_bits,
        explicit_mapping=explicit_mapping,
    )


def _solve_physical_permutation(
    *,
    layout_permutation: Sequence[int],
    visible_bit_count: int,
    bank_interleave_count: int,
    hardware: HardwareSpec,
) -> Dict[str, object]:
    target_bank_count = int(bank_interleave_count or 1)
    if target_bank_count not in {1, 2, 4}:
        return {
            "status": "unimplemented",
            "reason_code": "unsupported bank interleave count",
            "reason": f"Unsupported bank_interleave_count={target_bank_count}; expected one of 1, 2, or 4.",
        }
    if target_bank_count > hardware.bank_count_per_slice:
        return {
            "status": "unimplemented",
            "reason_code": "bank interleave exceeds hardware",
            "reason": (
                f"Target bank_interleave_count={target_bank_count} exceeds hardware bank count "
                f"{hardware.bank_count_per_slice}."
            ),
        }

    required_bank_bits = 0 if target_bank_count <= 1 else target_bank_count.bit_length() - 1
    if visible_bit_count < required_bank_bits:
        return {
            "status": "unimplemented",
            "reason_code": "insufficient remap bits for bank interleave",
            "reason": (
                f"Need at least {required_bank_bits} visible remap bits for {target_bank_count}-bank interleave, "
                f"but only {visible_bit_count} are available."
            ),
        }

    layout_positions = [int(layout_permutation[idx]) for idx in range(visible_bit_count)]
    if target_bank_count == 1:
        return {
            "status": "ok",
            "permutation": list(range(hardware.remap_bits)),
            "field_labels": [],
            "placement_policy": "preserve_layout_only",
        }

    try:
        target_positions, field_labels = _physical_target_positions(
            visible_bit_count=visible_bit_count,
            required_bank_bits=required_bank_bits,
            hardware=hardware,
        )
    except LayoutError as exc:
        return {
            "status": "unimplemented",
            "reason_code": "infeasible physical placement",
            "reason": str(exc),
        }
    physical_mapping: Dict[int, int] = {}
    ranking = sorted(enumerate(layout_positions), key=lambda item: (item[1], item[0]))
    for rank, (_, layout_position) in enumerate(ranking):
        if rank >= len(target_positions):
            break
        physical_mapping[int(target_positions[rank])] = int(layout_position)
    physical_permutation = _legalize_explicit_new_to_old_mapping(
        width=hardware.remap_bits,
        explicit_mapping=physical_mapping,
    )
    return {
        "status": "ok",
        "permutation": physical_permutation,
        "field_labels": field_labels,
        "placement_policy": f"bank_interleave_{target_bank_count}",
    }


def _physical_target_positions(
    *,
    visible_bit_count: int,
    required_bank_bits: int,
    hardware: HardwareSpec,
) -> Tuple[List[int], List[str]]:
    bank_positions = list(
        range(
            hardware.column_bits + hardware.row_bits,
            hardware.column_bits + hardware.row_bits + hardware.bank_bits,
        )
    )
    col_positions = list(range(hardware.column_bits))
    row_positions = list(range(hardware.column_bits, hardware.column_bits + hardware.row_bits))
    selected_positions = bank_positions[:required_bank_bits] + col_positions + row_positions
    if len(selected_positions) < visible_bit_count:
        raise LayoutError(
            "Physical placement within a slice does not have enough non-slice positions for the requested bank interleave."
        )
    labels = _dram_bit_labels(hardware)
    target_positions = selected_positions[:visible_bit_count]
    return target_positions, [labels[position] for position in target_positions]


def _compose_permutations(
    layout_permutation: Sequence[int],
    physical_permutation: Sequence[int],
) -> List[int]:
    width = max(len(layout_permutation), len(physical_permutation))
    composed = list(range(width))
    legal_layout = _legalize_new_to_old_permutation(layout_permutation, width)
    legal_physical = _legalize_new_to_old_permutation(physical_permutation, width)
    for physical_new_bit in range(width):
        layout_old_bit = int(legal_physical[physical_new_bit])
        composed[physical_new_bit] = int(legal_layout[layout_old_bit])
    return composed


def _legalize_explicit_new_to_old_mapping(
    *,
    width: int,
    explicit_mapping: Dict[int, int],
) -> List[int]:
    legal = [-1] * int(width)
    used_old_bits = set()
    for new_bit, old_bit in sorted(explicit_mapping.items()):
        if 0 <= int(new_bit) < width and 0 <= int(old_bit) < width and int(old_bit) not in used_old_bits:
            legal[int(new_bit)] = int(old_bit)
            used_old_bits.add(int(old_bit))
    remaining_old_bits = [bit for bit in range(width) if bit not in used_old_bits]
    remaining_new_bits = [bit for bit in range(width) if legal[bit] < 0]
    for new_bit, old_bit in zip(remaining_new_bits, remaining_old_bits):
        legal[new_bit] = old_bit
    return [int(bit) for bit in legal]


def _dram_bit_labels(hardware: HardwareSpec) -> List[str]:
    labels: List[str] = []
    labels.extend(f"col_bit_{idx}" for idx in range(hardware.column_bits))
    labels.extend(f"row_bit_{idx}" for idx in range(hardware.row_bits))
    labels.extend(f"bank_bit_{idx}" for idx in range(hardware.bank_bits))
    labels.extend(f"slice_bit_{idx}" for idx in range(hardware.slave_bits))
    return labels[: hardware.remap_bits]


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


def _serialize_bound_layout_relaxed(bound: BoundLayout) -> Dict[str, object]:
    return {
        "dtype": bound.dtype,
        "logical_shape": dict(bound.logical_shape),
        "ordered_factors": [
            {
                "name": factor.name,
                "parent_axis": factor.parent_axis,
                "extent": factor.extent,
                "kind": factor.kind,
                "bits": _factor_bit_width(factor.extent, allow_non_power_of_two=True),
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
