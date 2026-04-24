from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

from .hardware import HardwareSpec
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



def solve_graph(graph_spec: Dict[str, object], hw_cfg: HardwareSpec = None) -> List[EdgeSolveResult]:
    hardware = hw_cfg or HardwareSpec()
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

        shape_bindings = _shape_bindings_for_layouts(tensor_spec, producer_layout, consumer_spec.layout)
        _validate_tensor_shape(tensor_spec, producer_layout, consumer_spec.layout, shape_bindings)
        producer_axis_aliases = _axis_aliases_for_layout(tensor_spec, producer_layout)
        consumer_axis_aliases = _axis_aliases_for_layout(tensor_spec, consumer_spec.layout)
        results.append(
            _apply_graph_level_write_reg_annotations(
                solve_edge(
                    producer_layout=producer_layout,
                    consumer_layout=consumer_spec.layout,
                    shape_bindings=shape_bindings,
                    memory_dtype=tensor_spec.dtype,
                    hw_cfg=hardware,
                    producer=producer,
                    consumer=consumer,
                    tensor_name=tensor_name,
                    producer_axis_aliases=producer_axis_aliases,
                    consumer_axis_aliases=consumer_axis_aliases,
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
    tensor_name = str(edge.get("tensor", edge.get("tensor_name", "")))
    producer = str(edge["producer"])
    consumer = str(edge["consumer"])
    producer_spec = ops[producer]
    consumer_spec = ops[consumer]
    if not (
        tensor_name == "v_proj_fp16"
        and producer_spec.op_type == "ring_gemm_fp16_fp16_fp16"
        and consumer_spec.op_type == "prefill_add_V_MN_N_fp16_fp32_fp16"
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

    consumer = str(edge["consumer"])
    consumer_port = str(edge.get("consumer_port", ""))
    tensor_name = str(edge.get("tensor", edge.get("tensor_name", "")))
    consumer_spec = ops[consumer]
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

    if _is_v_row_writeback_edge(edge=edge, ops=ops):
        hint = _merge_write_reg_hints(
            result.write_reg_hint,
            "row_writeback(reorder(n8,m8)->(m8,n8))",
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


def _is_v_row_writeback_edge(edge: Dict[str, object], ops: Dict[str, OpSpec]) -> bool:
    tensor_name = str(edge.get("tensor", edge.get("tensor_name", "")))
    producer = str(edge["producer"])
    consumer = str(edge["consumer"])
    return (
        tensor_name == "v_proj_fp16"
        and ops[producer].op_type == "ring_gemm_fp16_fp16_fp16"
        and ops[consumer].op_type == "prefill_add_V_MN_N_fp16_fp32_fp16"
    )


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
) -> None:
    _validate_layout_compatibility(tensor_spec, producer_layout, "producer", shape_bindings)
    _validate_layout_compatibility(tensor_spec, consumer_layout, "consumer", shape_bindings)
    if set(tensor_spec.resolved_shape) != set(tensor_spec.shape):
        raise RuntimeError(
            f"Tensor '{tensor_spec.name}' resolved shape axes {tensor_spec.resolved_shape} do not match symbolic axes {tensor_spec.shape}."
        )


def _validate_layout_compatibility(
    tensor_spec: TensorSpec,
    layout: LayoutSpec,
    role: str,
    shape_bindings: Dict[str, int],
) -> None:
    tensor_axes = list(tensor_spec.shape.keys())
    layout_axes = list(layout.logical_shape.keys())
    if len(tensor_axes) != len(layout_axes):
        raise RuntimeError(
            f"Tensor '{tensor_spec.name}' rank {len(tensor_axes)} does not match {role} layout rank {len(layout_axes)}."
        )
    tensor_extents = [tensor_spec.resolved_shape[axis] for axis in tensor_axes]
    layout_extents = [shape_bindings[axis] for axis in layout_axes]
    if tensor_extents != layout_extents:
        raise RuntimeError(
            f"Tensor '{tensor_spec.name}' extents {tensor_extents} do not match {role} layout extents {layout_extents}."
        )


def _shape_bindings_for_layouts(
    tensor_spec: TensorSpec,
    producer_layout: LayoutSpec,
    consumer_layout: LayoutSpec,
) -> Dict[str, int]:
    bindings = dict(tensor_spec.resolved_shape)
    tensor_axes = list(tensor_spec.shape.keys())
    for layout in (producer_layout, consumer_layout):
        layout_axes = list(layout.logical_shape.keys())
        if len(layout_axes) != len(tensor_axes):
            continue
        axis_pairs = _compatible_axis_pairs(tensor_axes, layout_axes)
        for tensor_axis, layout_axis in axis_pairs:
            value = tensor_spec.resolved_shape[tensor_axis]
            existing = bindings.get(layout_axis)
            if existing is not None and existing != value:
                raise RuntimeError(
                    f"Tensor '{tensor_spec.name}' cannot alias axis '{layout_axis}' to conflicting extents {existing} and {value}."
                )
            bindings[layout_axis] = value
    return bindings


def _axis_aliases_for_layout(tensor_spec: TensorSpec, layout: LayoutSpec) -> Dict[str, str]:
    tensor_axes = list(tensor_spec.shape.keys())
    layout_axes = list(layout.logical_shape.keys())
    if len(tensor_axes) != len(layout_axes):
        return {}
    return {
        layout_axis: tensor_axis
        for tensor_axis, layout_axis in _compatible_axis_pairs(tensor_axes, layout_axes)
    }


def _compatible_axis_pairs(tensor_axes: List[str], layout_axes: List[str]) -> List[tuple[str, str]]:
    return list(zip(tensor_axes, layout_axes))



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
