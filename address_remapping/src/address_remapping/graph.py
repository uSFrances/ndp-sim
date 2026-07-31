from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

from .hardware import HardwareSpec, SolverConfig
from .layout import LayoutSpec
from .model_parser import expand_model_spec
from .solver import EdgeSolveResult, solve_edge


@dataclass(frozen=True)
class PortSpec:
    name: str
    layout: LayoutSpec

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, object]) -> "PortSpec":
        layout_obj = data["layout"] if isinstance(data["layout"], LayoutSpec) else LayoutSpec.from_dict(data["layout"])
        return cls(name=name, layout=layout_obj)


@dataclass(frozen=True)
class OpSpec:
    name: str
    inputs: Dict[str, PortSpec]
    outputs: Dict[str, PortSpec]
    op_type: str = ""

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, object]) -> "OpSpec":
        if "inputs" in data or "outputs" in data:
            inputs = {
                str(port_name): PortSpec.from_dict(str(port_name), port_data)
                for port_name, port_data in dict(data.get("inputs", {})).items()
            }
            outputs = {
                str(port_name): PortSpec.from_dict(str(port_name), port_data)
                for port_name, port_data in dict(data.get("outputs", {})).items()
            }
            return cls(name=name, inputs=inputs, outputs=outputs, op_type=str(data.get("op_type", "")))

        inputs = {
            str(tensor_name): PortSpec(name=str(tensor_name), layout=LayoutSpec.from_dict(layout_data))
            for tensor_name, layout_data in dict(data.get("input_layouts", {})).items()
        }
        outputs = {
            str(tensor_name): PortSpec(name=str(tensor_name), layout=LayoutSpec.from_dict(layout_data))
            for tensor_name, layout_data in dict(data.get("output_layouts", {})).items()
        }
        return cls(name=name, inputs=inputs, outputs=outputs, op_type=str(data.get("op_type", "")))


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: Dict[str, str]
    resolved_shape: Dict[str, int]
    base_addr: int | None = None

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, object]) -> "TensorSpec":
        return cls(
            name=name,
            dtype=str(data["dtype"]),
            shape={str(k): str(v) for k, v in dict(data.get("shape", {})).items()},
            resolved_shape={str(k): int(v) for k, v in dict(data.get("resolved_shape", {})).items()},
            base_addr=int(data["base_addr"]) if data.get("base_addr") is not None else None,
        )



def solve_graph(
    graph_spec: Dict[str, object],
    hw_cfg: HardwareSpec = None,
    solver_cfg: SolverConfig | None = None,
) -> List[EdgeSolveResult]:
    hardware = hw_cfg or HardwareSpec()
    solver_config = solver_cfg or SolverConfig()
    expanded = expand_model_spec(graph_spec) if "model" in graph_spec else graph_spec
    ops = {
        op_name: OpSpec.from_dict(str(op_name), op_data)
        for op_name, op_data in dict(expanded["ops"]).items()
    }
    raw_tensors = dict(expanded.get("tensors", {}))

    results: List[EdgeSolveResult] = []
    for edge in expanded["edges"]:
        producer = str(edge["producer"])
        consumer = str(edge["consumer"])
        producer_port = str(edge.get("producer_port", edge.get("producer_tensor")))
        consumer_port = str(edge.get("consumer_port", edge.get("consumer_tensor")))
        tensor_name = str(edge.get("tensor", edge.get("tensor_name", producer_port)))

        producer_spec = ops[producer].outputs[producer_port]
        consumer_spec = ops[consumer].inputs[consumer_port]
        tensor_data = raw_tensors.get(tensor_name)
        if tensor_data is None:
            raise RuntimeError(f"Edge tensor '{tensor_name}' is not declared in graph_spec['tensors'].")
        try:
            tensor_spec = TensorSpec.from_dict(str(tensor_name), tensor_data)
        except KeyError as exc:
            raise RuntimeError(
                f"Edge tensor '{tensor_name}' is missing required field '{exc.args[0]}' in graph_spec['tensors']."
            ) from exc

        producer_layout = _graph_level_producer_layout_override(
            producer_spec.layout,
            edge=edge,
            ops=ops,
        )
        consumer_axis_aliases_override = _edge_consumer_axis_aliases_override(edge)
        consumer_layout = _edge_consumer_layout_override(
            consumer_spec.layout,
            consumer_axis_aliases_override,
        )

        shape_bindings = _shape_bindings_for_layouts(
            tensor_spec,
            producer_layout,
            consumer_layout,
        )
        _validate_tensor_shape(
            tensor_spec,
            producer_layout,
            consumer_layout,
            shape_bindings,
        )
        producer_axis_aliases = _axis_aliases_for_layout(tensor_spec, producer_layout)
        consumer_axis_aliases = _axis_aliases_for_layout(
            tensor_spec,
            consumer_layout,
        )
        results.append(
            _apply_graph_level_write_reg_annotations(
                solve_edge(
                    producer_layout=producer_layout,
                    consumer_layout=consumer_layout,
                    shape_bindings=shape_bindings,
                    memory_dtype=tensor_spec.dtype,
                    hw_cfg=hardware,
                    producer=producer,
                    consumer=consumer,
                    tensor_name=tensor_name,
                    producer_axis_aliases=producer_axis_aliases,
                    consumer_axis_aliases=consumer_axis_aliases,
                    bank_interleave_count=solver_config.bank_interleave_count(
                        ops[consumer].op_type,
                        consumer_port,
                    ),
                    producer_port=producer_port,
                    consumer_port=consumer_port,
                    producer_op_type=ops[producer].op_type,
                    consumer_op_type=ops[consumer].op_type,
                ),
                edge=edge,
                ops=ops,
            )
        )
    return results


def _graph_level_producer_layout_override(
    producer_layout: LayoutSpec,
    edge: Dict[str, object],
    ops: Dict[str, OpSpec],
) -> LayoutSpec:
    producer = str(edge["producer"])
    consumer = str(edge["consumer"])
    producer_port = str(edge.get("producer_port", ""))
    consumer_port = str(edge.get("consumer_port", ""))
    producer_spec = ops[producer]
    consumer_spec = ops[consumer]
    if not (
        producer_spec.op_type == "ring_gemm_fp16_fp16_fp16"
        and producer_port == "out"
        and consumer_spec.op_type == "prefill_add_V_fp16MN_fp32N_fp16MN"
        and consumer_port == "inB"
    ):
        return producer_layout
    return _reorder_layout_linear_order(producer_layout, "n8", "m8")


def _apply_graph_level_write_reg_annotations(
    result: EdgeSolveResult,
    edge: Dict[str, object],
    ops: Dict[str, OpSpec],
) -> EdgeSolveResult:
    if result.status != "ok":
        return result

    producer = str(edge["producer"])
    consumer = str(edge["consumer"])
    consumer_port = str(edge.get("consumer_port", ""))
    tensor_name = str(edge.get("tensor", edge.get("tensor_name", "")))
    producer_spec = ops[producer]
    consumer_spec = ops[consumer]
    if (
        producer_spec.op_type == "ring_gemm_fp16_fp16_fp16"
        and str(edge.get("producer_port", "")) == "out"
        and consumer_spec.op_type == "prefill_add_V_fp16MN_fp32N_fp16MN"
        and consumer_port == "inB"
    ):
        # The graph-level n8/m8 override is a physical register reorder.
        result = replace(
            result,
            write_reg_required=True,
            write_reg_hint=_merge_write_reg_hints(
                result.write_reg_hint,
                "reorder(n8,m8)->(m8,n8)",
            ),
        )
    if _is_rope_slice_exchange_edge(consumer_spec, consumer_port, tensor_name):
        hint = _merge_write_reg_hints(
            result.write_reg_hint,
            "cross_slice_base_addr_exchange(rope_half_swap_for_local_add)",
        )
        result = replace(
            result,
            write_reg_required=True,
            write_reg_hint=hint,
        )

    return result


def _is_rope_slice_exchange_edge(consumer_spec: OpSpec, consumer_port: str, tensor_name: str) -> bool:
    if consumer_spec.op_type != "prefill_add_fp32MN_fp32MN_fp16MN":
        return False
    if consumer_port != "inB":
        return False
    return tensor_name in {"q_sin_fp32", "k_sin_fp32"}


def _merge_write_reg_hints(*hints: str) -> str:
    filtered = [hint for hint in hints if hint]
    return " ; ".join(dict.fromkeys(filtered))


def _reorder_layout_linear_order(layout: LayoutSpec, first: str, second: str) -> LayoutSpec:
    order = list(layout.linear_order)
    if first not in order or second not in order:
        return layout
    first_index = order.index(first)
    second_index = order.index(second)
    order[first_index], order[second_index] = order[second_index], order[first_index]
    return LayoutSpec(
        dtype=layout.dtype,
        logical_shape=dict(layout.logical_shape),
        factors=layout.factors,
        linear_order=tuple(order),
    )



def _validate_tensor_shape(
    tensor_spec: TensorSpec,
    producer_layout: LayoutSpec,
    consumer_layout: LayoutSpec,
    shape_bindings: Dict[str, int],
    consumer_axis_aliases_override: Dict[str, str] | None = None,
) -> None:
    _validate_layout_compatibility(tensor_spec, producer_layout, "producer", shape_bindings)
    _validate_layout_compatibility(
        tensor_spec,
        consumer_layout,
        "consumer",
        shape_bindings,
        axis_aliases_override=consumer_axis_aliases_override,
    )
    if set(tensor_spec.resolved_shape) != set(tensor_spec.shape):
        raise RuntimeError(
            f"Tensor '{tensor_spec.name}' resolved shape axes {tensor_spec.resolved_shape} do not match symbolic axes {tensor_spec.shape}."
        )


def _validate_layout_compatibility(
    tensor_spec: TensorSpec,
    layout: LayoutSpec,
    role: str,
    shape_bindings: Dict[str, int],
    axis_aliases_override: Dict[str, str] | None = None,
) -> None:
    tensor_axes = list(tensor_spec.shape.keys())
    layout_axes = list(layout.logical_shape.keys())
    if len(tensor_axes) != len(layout_axes):
        raise RuntimeError(
            f"Tensor '{tensor_spec.name}' rank {len(tensor_axes)} does not match {role} layout rank {len(layout_axes)}."
        )
    axis_pairs = _compatible_axis_pairs(tensor_axes, layout_axes, axis_aliases_override=axis_aliases_override)
    tensor_extents = [tensor_spec.resolved_shape[tensor_axis] for tensor_axis, _layout_axis in axis_pairs]
    layout_extents = [shape_bindings[axis] for axis in layout_axes]
    if tensor_extents != layout_extents:
        raise RuntimeError(
            f"Tensor '{tensor_spec.name}' extents {tensor_extents} do not match {role} layout extents {layout_extents}."
        )


def _shape_bindings_for_layouts(
    tensor_spec: TensorSpec,
    producer_layout: LayoutSpec,
    consumer_layout: LayoutSpec,
    consumer_axis_aliases_override: Dict[str, str] | None = None,
) -> Dict[str, int]:
    bindings = dict(tensor_spec.resolved_shape)
    tensor_axes = list(tensor_spec.shape.keys())
    for layout, axis_aliases_override in (
        (producer_layout, None),
        (consumer_layout, consumer_axis_aliases_override),
    ):
        layout_axes = list(layout.logical_shape.keys())
        if len(layout_axes) != len(tensor_axes):
            continue
        axis_pairs = _compatible_axis_pairs(
            tensor_axes,
            layout_axes,
            axis_aliases_override=axis_aliases_override,
        )
        for tensor_axis, layout_axis in axis_pairs:
            value = tensor_spec.resolved_shape[tensor_axis]
            existing = bindings.get(layout_axis)
            if existing is not None and existing != value:
                raise RuntimeError(
                    f"Tensor '{tensor_spec.name}' cannot alias axis '{layout_axis}' to conflicting extents {existing} and {value}."
                )
            bindings[layout_axis] = value
    return bindings


def _axis_aliases_for_layout(
    tensor_spec: TensorSpec,
    layout: LayoutSpec,
    axis_aliases_override: Dict[str, str] | None = None,
) -> Dict[str, str]:
    tensor_axes = list(tensor_spec.shape.keys())
    layout_axes = list(layout.logical_shape.keys())
    if len(tensor_axes) != len(layout_axes):
        return {}
    return {
        layout_axis: tensor_axis
        for tensor_axis, layout_axis in _compatible_axis_pairs(
            tensor_axes,
            layout_axes,
            axis_aliases_override=axis_aliases_override,
        )
    }


def _compatible_axis_pairs(
    tensor_axes: List[str],
    layout_axes: List[str],
    axis_aliases_override: Dict[str, str] | None = None,
) -> List[tuple[str, str]]:
    if axis_aliases_override:
        pairs: List[tuple[str, str]] = []
        for layout_axis in layout_axes:
            tensor_axis = axis_aliases_override.get(layout_axis)
            if tensor_axis is None:
                raise RuntimeError(
                    f"Missing explicit axis alias for layout axis '{layout_axis}'."
                )
            pairs.append((tensor_axis, layout_axis))
        return pairs
    if set(tensor_axes) == set(layout_axes):
        return [(layout_axis, layout_axis) for layout_axis in layout_axes]
    return list(zip(tensor_axes, layout_axes))


def _edge_consumer_axis_aliases_override(edge: Dict[str, object]) -> Dict[str, str] | None:
    raw = edge.get("consumer_axis_aliases_override")
    if not isinstance(raw, dict):
        return None
    return {str(layout_axis): str(tensor_axis) for layout_axis, tensor_axis in raw.items()}


def _edge_consumer_layout_override(
    consumer_layout: LayoutSpec,
    consumer_axis_aliases_override: Dict[str, str] | None,
) -> LayoutSpec:
    if not consumer_axis_aliases_override:
        return consumer_layout
    axis_name_map = dict(consumer_axis_aliases_override)
    renamed_logical_shape = {
        axis_name_map.get(axis, axis): _rename_axis_expr(extent, axis_name_map)
        for axis, extent in consumer_layout.logical_shape.items()
    }
    renamed_factors = tuple(
        replace(
            factor,
            parent_axis=axis_name_map.get(factor.parent_axis, factor.parent_axis),
            extent_expr=_rename_axis_expr(factor.extent_expr, axis_name_map),
        )
        for factor in consumer_layout.factors
    )
    return LayoutSpec(
        dtype=consumer_layout.dtype,
        logical_shape=renamed_logical_shape,
        factors=renamed_factors,
        linear_order=consumer_layout.linear_order,
    )


def _rename_axis_expr(expr: object, axis_name_map: Dict[str, str]) -> object:
    if not isinstance(expr, str):
        return expr
    renamed = expr
    for old_axis, new_axis in axis_name_map.items():
        renamed = re.sub(rf"\b{re.escape(old_axis)}\b", new_axis, renamed)
    return renamed



def load_graph_file(path: str) -> Dict[str, object]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8-sig")
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return json.loads(raw)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML graph files.") from exc
        return yaml.safe_load(raw)
    raise RuntimeError(f"Unsupported graph file type '{suffix}'.")
