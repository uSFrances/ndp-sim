from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .graph import load_graph_file, solve_graph
from .hardware import HardwareSpec
from .model_parser import evaluate_expr
from .registry import RegisteredOp, build_default_registry
from .solver import EdgeSolveResult


EXTERNAL_TO_CANONICAL_OP = {
    "prefill_add_V_fp16MN_fp32N_fp16MN": "prefill_add_V_fp16MN_fp32N_fp16MN",
    "prefill_add_fp16MN_fp32N_fp32MN": "prefill_add_fp16MN_fp32N_fp32MN",
    "prefill_gemm_local": "gemm_local_fp16_fp16_fp16",
    "prefill_gemm_local_qkt": "gemm_local_qkt_fp16_fp16_fp32",
    "prefill_gemm_ring_4slice": "ring_gemm_fp16_fp16_fp16",
    "prefill_mul_fp32MN_fp32N_fp16MN": "prefill_mul_fp32MN_fp32N_fp16MN",
    "prefill_remote_sum_4slice_fp32MN_fp32MN": "prefill_remote_sum_fp32MN_fp32MN_2d",
    "prefill_remote_sum_4slice_fp16MN_fp32MN": "prefill_remote_sum_fp16MN_fp32MN",
    "prefill_summac_fp32MN_fp32MN": "prefill_summac",
    "prefill_remote_sum_fp32MN_fp32MN": "prefill_remote_sum_Mfp32_Mfp32",
    "prefill_mac_SFU_fp32MN_fp32MN": "prefill_mac_SFU",
    "prefill_max_fp32MN_fp32MN": "prefill_max",
    "prefill_silu_fp32MN_fp16MN": "prefill_silu_fp16MN_fp32MN",
    "prefill_sub_SFU_fp32MN_fp32M_fp32MN": "prefill_sub_SFU_fp32MN_fp32MN_fp32MN",
    "prefill_sum_rec_fp32MN_fp32MN": "prefill_sum_rec_fp32MN_fp32MN",
}

EXTERNAL_TO_INTERNAL_PORT = {
    "A": "inA",
    "B": "inB",
    "C": "inC",
}

EXTERNAL_PORT_OVERRIDES = {}

RING_GEMM_EXTERNAL_TYPE = "prefill_gemm_ring_4slice"
LOCAL_GEMM_EXTERNAL_TYPE = "prefill_gemm_local"
LOCAL_GEMM_EXTERNAL_TYPES = {LOCAL_GEMM_EXTERNAL_TYPE, "prefill_gemm_local_qkt"}


class RmsNormBridgeError(ValueError):
    pass


def build_expanded_graph_from_external_execplan(
    payload: Mapping[str, object],
    registry: Mapping[str, RegisteredOp] | None = None,
    require_base_addrs: bool = False,
) -> Dict[str, object]:
    resolved_registry = registry or build_default_registry()
    operators = payload.get("operators")
    if not isinstance(operators, list):
        raise RmsNormBridgeError("External execplan payload must define an 'operators' list.")
    external_values = _resolve_external_values(payload)

    normalized_ops: List[Dict[str, object]] = []
    produced_tensors: Dict[str, Dict[str, object]] = {}
    external_tensors: Dict[str, Dict[str, object]] = {}
    op_ids: set[str] = set()

    for operator in operators:
        if not isinstance(operator, Mapping):
            raise RmsNormBridgeError("Each external operator entry must be a mapping.")
        op_id = str(operator.get("id", "")).strip()
        if not op_id:
            raise RmsNormBridgeError("Each external operator must define a non-empty 'id'.")
        if op_id in op_ids:
            raise RmsNormBridgeError(f"Duplicate external operator id '{op_id}'.")
        op_ids.add(op_id)

        external_type = str(operator.get("type", "")).strip()
        canonical_type = _canonical_operator_type(external_type, resolved_registry)
        registered_op = resolved_registry[canonical_type]
        external_to_internal_ports = _external_to_internal_ports(external_type)

        external_inputs = operator.get("inputs")
        if not isinstance(external_inputs, Mapping):
            raise RmsNormBridgeError(f"Operator '{op_id}' must define an 'inputs' mapping.")

        filtered_external_inputs = _filtered_external_inputs(external_type, external_inputs)

        if len(filtered_external_inputs) != len(registered_op.ordered_inputs):
            raise RmsNormBridgeError(
                f"Operator '{op_id}' ({external_type}) expects {len(registered_op.ordered_inputs)} inputs, "
                f"got {len(filtered_external_inputs)}."
            )

        local_input_shapes: Dict[str, Dict[str, str]] = {}
        local_input_resolved: Dict[str, Dict[str, int]] = {}
        normalized_inputs: Dict[str, Dict[str, object]] = {}
        input_sources: Dict[str, object] = {}
        input_tensor_names: Dict[str, str] = {}
        input_base_addrs: Dict[str, int | None] = {}
        input_axis_views: Dict[str, object] = {}

        for external_port, internal_port in external_to_internal_ports.items():
            if internal_port not in registered_op.input_ports:
                continue
            external_input = filtered_external_inputs.get(external_port)
            if not isinstance(external_input, Mapping):
                raise RmsNormBridgeError(f"Operator '{op_id}' input '{external_port}' must be a mapping.")

            port_template = registered_op.input_ports[internal_port]
            port_layout = port_template.layout_builder(port_template.memory_dtype)
            local_shape, local_resolved, debug_info = _normalize_external_shape(
                external_shape=external_input.get("shape"),
                external_type=external_type,
                internal_port=internal_port,
                expected_axes=list(port_layout.logical_shape.keys()),
                op_id=op_id,
                port_name=external_port,
                values=external_values,
            )
            local_input_shapes[internal_port] = local_shape
            local_input_resolved[internal_port] = local_resolved
            normalized_inputs[internal_port] = {
                **port_template.build(),
                "shape": local_shape,
                "resolved_shape": local_resolved,
                "external_shape_debug": debug_info,
            }

            source = _normalize_source_ref(external_input.get("source"))
            input_sources[internal_port] = source
            input_axis_views[internal_port] = external_input.get("axis_view")
            input_base_addrs[internal_port] = _parse_optional_base_addr(
                external_input.get("base_addr"),
                op_id=op_id,
                port_name=external_port,
                require_base_addr=require_base_addrs,
            )
            if _is_external_source(source):
                tensor_name = f"{op_id}__{internal_port}__external"
                input_tensor_names[internal_port] = tensor_name
                external_tensors[tensor_name] = {
                    "dtype": port_template.memory_dtype,
                    "shape": local_shape,
                    "resolved_shape": local_resolved,
                    "base_addr": input_base_addrs[internal_port],
                }

        inferred_output_shape = registered_op.shape_resolver(local_input_shapes)
        inferred_output_resolved = _resolve_output_shape(inferred_output_shape, local_input_resolved)

        external_output = operator.get("output")
        if not isinstance(external_output, Mapping):
            raise RmsNormBridgeError(f"Operator '{op_id}' must define an 'output' mapping.")
        output_port = registered_op.ordered_outputs[0]
        output_template = registered_op.output_ports[output_port]
        output_layout = output_template.layout_builder(output_template.memory_dtype)
        normalized_output_shape, normalized_output_resolved, output_debug = _normalize_external_shape(
            external_shape=external_output.get("shape"),
            external_type=external_type,
            internal_port=output_port,
            expected_axes=list(output_layout.logical_shape.keys()),
            op_id=op_id,
            port_name="output",
            values=external_values,
        )
        _validate_external_output_shape(
            op_id=op_id,
            external_type=external_type,
            inferred_shape=inferred_output_shape,
            inferred_resolved=inferred_output_resolved,
            normalized_shape=normalized_output_shape,
            normalized_resolved=normalized_output_resolved,
        )
        output_shape = normalized_output_shape
        output_resolved = normalized_output_resolved

        output_base_addr = _parse_optional_base_addr(
            external_output.get("base_addr"),
            op_id=op_id,
            port_name="output",
            require_base_addr=require_base_addrs,
        )

        tensor_name = f"{op_id}__out"
        produced_tensors[op_id] = {
            "name": tensor_name,
            "dtype": output_template.memory_dtype,
            "shape": output_shape,
            "resolved_shape": output_resolved,
            "base_addr": output_base_addr,
        }
        normalized_ops.append(
            {
                "id": op_id,
                "external_type": external_type,
                "canonical_type": canonical_type,
                "hardware_measured": operator.get("hardware_measured"),
                "input_dtypes": {
                    port_name: port_template.memory_dtype
                    for port_name, port_template in registered_op.input_ports.items()
                },
                "inputs": normalized_inputs,
                "input_sources": input_sources,
                "input_tensor_names": input_tensor_names,
                "input_axis_views": input_axis_views,
                "input_base_addrs": input_base_addrs,
                "output": {
                    output_port: {
                        **output_template.build(),
                        "shape": output_shape,
                        "resolved_shape": output_resolved,
                        "external_shape_debug": output_debug,
                        "source_tensor": tensor_name,
                    }
                },
            }
        )

    edges: List[Dict[str, object]] = []
    ops: Dict[str, object] = {}
    tensors: Dict[str, object] = {
        tensor_spec["name"]: {
            "dtype": tensor_spec["dtype"],
            "shape": tensor_spec["shape"],
            "resolved_shape": tensor_spec["resolved_shape"],
            "base_addr": tensor_spec["base_addr"],
        }
        for tensor_spec in produced_tensors.values()
    }
    tensors.update(external_tensors)

    for normalized_op in normalized_ops:
        op_id = str(normalized_op["id"])
        op_type = str(normalized_op["canonical_type"])
        resolved_inputs = {}
        input_consumer_axis_aliases_overrides: Dict[str, Dict[str, str] | None] = {}
        for internal_port, input_spec in dict(normalized_op["inputs"]).items():
            resolved_input = dict(input_spec)
            source = normalized_op["input_sources"].get(internal_port)
            if _is_external_source(source):
                resolved_input["source_tensor"] = normalized_op["input_tensor_names"][internal_port]
                input_consumer_axis_aliases_overrides[internal_port] = None
            else:
                source_op_id = str(source)
                if source_op_id not in produced_tensors:
                    raise RmsNormBridgeError(
                        f"Operator '{op_id}' input '{internal_port}' references unknown producer '{source_op_id}'."
                    )
                producer_tensor = produced_tensors[source_op_id]
                consumer_axis_aliases_override = _normalize_edge_axis_view(
                    raw_axis_view=normalized_op["input_axis_views"].get(internal_port),
                    producer_shape=dict(producer_tensor["shape"]),
                    producer_resolved=dict(producer_tensor["resolved_shape"]),
                    consumer_shape=dict(input_spec["shape"]),
                    consumer_resolved=dict(input_spec["resolved_shape"]),
                    producer=source_op_id,
                    consumer=op_id,
                    port_name=internal_port,
                )
                input_consumer_axis_aliases_overrides[internal_port] = consumer_axis_aliases_override
                _validate_connected_tensor_shape(
                    producer_shape=dict(producer_tensor["shape"]),
                    producer_resolved=dict(producer_tensor["resolved_shape"]),
                    consumer_shape=dict(input_spec["shape"]),
                    consumer_resolved=dict(input_spec["resolved_shape"]),
                    producer=source_op_id,
                    consumer=op_id,
                    port_name=internal_port,
                    consumer_axis_aliases_override=consumer_axis_aliases_override,
                )
                _validate_connected_base_addr(
                    producer_base_addr=producer_tensor["base_addr"],
                    consumer_base_addr=normalized_op["input_base_addrs"].get(internal_port),
                    producer=source_op_id,
                    consumer=op_id,
                    port_name=internal_port,
                    require_base_addrs=require_base_addrs,
                )
                resolved_input["source_tensor"] = producer_tensor["name"]
            resolved_inputs[internal_port] = resolved_input

        ops[op_id] = {
            "op_type": op_type,
            "call_kwargs": {},
            "inputs": resolved_inputs,
            "outputs": normalized_op["output"],
        }
        if "hardware_measured" in normalized_op:
            ops[op_id]["hardware_measured"] = normalized_op["hardware_measured"]

        for internal_port, source in dict(normalized_op["input_sources"]).items():
            if _is_external_source(source):
                continue
            source_op_id = str(source)
            producer_tensor = produced_tensors[source_op_id]
            edges.append(
                {
                    "tensor": producer_tensor["name"],
                    "producer": source_op_id,
                    "producer_port": "out",
                    "consumer": op_id,
                    "consumer_port": internal_port,
                    "consumer_axis_aliases_override": input_consumer_axis_aliases_overrides.get(internal_port),
                }
            )

    return {
        "shape_bindings": {},
        "params": {},
        "ops": ops,
        "tensors": tensors,
        "edges": edges,
    }


def build_expanded_graph_from_external_rmsnorm(
    payload: Mapping[str, object],
    registry: Mapping[str, RegisteredOp] | None = None,
    require_base_addrs: bool = False,
) -> Dict[str, object]:
    return build_expanded_graph_from_external_execplan(
        payload,
        registry=registry,
        require_base_addrs=require_base_addrs,
    )


def is_external_execplan_payload(payload: Mapping[str, object]) -> bool:
    return isinstance(payload.get("operators"), list)


def is_external_rmsnorm_payload(payload: Mapping[str, object]) -> bool:
    return is_external_execplan_payload(payload)


def normalize_graph_spec(
    payload: Mapping[str, object],
    registry: Mapping[str, RegisteredOp] | None = None,
    require_base_addrs: bool = False,
) -> Dict[str, object]:
    if is_external_execplan_payload(payload):
        return build_expanded_graph_from_external_execplan(
            payload,
            registry=registry,
            require_base_addrs=require_base_addrs,
        )
    return dict(payload)


def fill_external_remapping(
    payload: Mapping[str, object],
    hw_cfg: HardwareSpec | None = None,
    registry: Mapping[str, RegisteredOp] | None = None,
) -> Dict[str, object]:
    filled, _results = fill_external_remapping_with_results(
        payload,
        hw_cfg=hw_cfg,
        registry=registry,
    )
    return filled


def fill_external_remapping_with_results(
    payload: Mapping[str, object],
    hw_cfg: HardwareSpec | None = None,
    registry: Mapping[str, RegisteredOp] | None = None,
) -> tuple[Dict[str, object], List[EdgeSolveResult]]:
    hardware = hw_cfg or HardwareSpec()
    expanded = build_expanded_graph_from_external_execplan(payload, registry=registry)
    results = solve_graph(expanded, hardware)
    result_by_key = {
        (
            str(edge["producer"]),
            str(edge["consumer"]),
            str(edge["consumer_port"]),
        ): result
        for edge, result in zip(expanded["edges"], results)
    }
    default_permutation = list(range(hardware.remap_bits))

    filled = copy.deepcopy(dict(payload))
    operators = filled.get("operators")
    if not isinstance(operators, list):
        raise RmsNormBridgeError("External execplan payload must define an 'operators' list.")
    operator_by_id = {
        str(operator.get("id", "")): operator
        for operator in operators
        if isinstance(operator, dict) and str(operator.get("id", "")).strip()
    }
    output_remappings: Dict[str, List[int]] = {}
    output_writeregs: Dict[str, Dict[str, object]] = {}
    output_write_reg_hints: Dict[str, str] = {}

    for operator in operators:
        op_id = str(operator["id"])
        external_type = str(operator.get("type", "")).strip()
        external_to_internal_ports = _external_to_internal_ports(external_type)
        inputs = operator.get("inputs", {})
        for external_port, input_data in dict(inputs).items():
            if not isinstance(input_data, dict):
                continue
            source = _normalize_source_ref(input_data.get("source"))
            if _is_external_source(source):
                input_data["remapping"] = None
                if _uses_input_write_reg_hint_field(external_type, input_data):
                    input_data["write_reg_hint"] = _fixed_input_write_reg_hint(
                        external_type=external_type,
                        external_port=external_port,
                    )
                input_data.pop("writereg", None)
                continue

            internal_port = external_to_internal_ports.get(external_port)
            if internal_port is None:
                raise RmsNormBridgeError(
                    f"Operator '{op_id}' uses unsupported external input port '{external_port}'."
                )
            key = (str(source), op_id, internal_port)
            result = result_by_key.get(key)
            if result is None:
                raise RmsNormBridgeError(
                    f"Missing solve result for edge {source} -> {op_id} port {external_port}."
                )
            _validate_bridgeable_result(result)
            if result.write_reg_required and _write_reg_belongs_on_producer_output(
                external_type=external_type,
                external_port=external_port,
            ):
                producer_op_id = str(source)
                output_writereg = {
                    "required": True,
                    "hint": result.write_reg_hint,
                }
                previous_writereg = output_writeregs.get(producer_op_id)
                if previous_writereg is not None and previous_writereg != output_writereg:
                    raise RmsNormBridgeError(
                        f"Producer '{producer_op_id}' feeds multiple consumers requiring different output writereg "
                        "hints; cannot encode a single output.writereg in the external payload."
                    )
                output_writeregs[producer_op_id] = output_writereg
                if _uses_input_write_reg_hint_field(external_type, input_data):
                    input_data["write_reg_hint"] = None
                input_data.pop("writereg", None)
            elif result.write_reg_required and _write_reg_hint_belongs_on_producer_output(
                external_type=external_type,
                external_port=external_port,
            ):
                producer_op_id = str(source)
                previous_hint = output_write_reg_hints.get(producer_op_id)
                if previous_hint is not None and previous_hint != result.write_reg_hint:
                    raise RmsNormBridgeError(
                        f"Producer '{producer_op_id}' feeds multiple consumers requiring different output "
                        "write_reg_hint values; cannot encode a single output.write_reg_hint in the external payload."
                    )
                output_write_reg_hints[producer_op_id] = str(result.write_reg_hint)
                if _uses_input_write_reg_hint_field(external_type, input_data):
                    input_data["write_reg_hint"] = None
                input_data.pop("writereg", None)
            elif _uses_input_write_reg_hint_field(external_type, input_data):
                fixed_hint = _fixed_input_write_reg_hint(
                    external_type=external_type,
                    external_port=external_port,
                )
                input_data["write_reg_hint"] = (
                    fixed_hint if fixed_hint is not None
                    else (result.write_reg_hint if result.write_reg_required else None)
                )
                input_data.pop("writereg", None)
            elif result.write_reg_required:
                input_data["writereg"] = {
                    "required": True,
                    "hint": result.write_reg_hint,
                }
            else:
                input_data.pop("writereg", None)

            input_data["remapping"] = None

            if result.permutation == default_permutation:
                continue

            producer_op_id = str(source)
            previous = output_remappings.get(producer_op_id)
            if previous is not None and previous != result.permutation:
                raise RmsNormBridgeError(
                    f"Producer '{producer_op_id}' feeds multiple consumers requiring different remappings; "
                    "cannot encode a single output.remapping in the external payload."
                )
            output_remappings[producer_op_id] = list(result.permutation)

    for producer_op_id, permutation in output_remappings.items():
        operator = operator_by_id.get(producer_op_id)
        if operator is None:
            raise RmsNormBridgeError(f"Missing producer operator '{producer_op_id}' while filling remapping output.")
        output = operator.get("output")
        if not isinstance(output, dict):
            raise RmsNormBridgeError(f"Operator '{producer_op_id}' must define an 'output' mapping.")
        output["remapping"] = list(permutation)
    for producer_op_id, writereg in output_writeregs.items():
        operator = operator_by_id.get(producer_op_id)
        if operator is None:
            raise RmsNormBridgeError(f"Missing producer operator '{producer_op_id}' while filling writereg output.")
        output = operator.get("output")
        if not isinstance(output, dict):
            raise RmsNormBridgeError(f"Operator '{producer_op_id}' must define an 'output' mapping.")
        output["writereg"] = dict(writereg)
    for producer_op_id, write_reg_hint in output_write_reg_hints.items():
        operator = operator_by_id.get(producer_op_id)
        if operator is None:
            raise RmsNormBridgeError(f"Missing producer operator '{producer_op_id}' while filling output hint.")
        output = operator.get("output")
        if not isinstance(output, dict):
            raise RmsNormBridgeError(f"Operator '{producer_op_id}' must define an 'output' mapping.")
        output["write_reg_hint"] = write_reg_hint

    return filled, results


def _write_reg_belongs_on_producer_output(external_type: str, external_port: str) -> bool:
    return (
        external_type == "prefill_add_V_fp16MN_fp32N_fp16MN"
        and external_port == "A"
    )


def _write_reg_hint_belongs_on_producer_output(external_type: str, external_port: str) -> bool:
    return (
        external_type == "prefill_add_V_fp16MN_fp32N_fp16MN"
        and external_port == "B"
    )


def _uses_input_write_reg_hint_field(external_type: str, input_data: Mapping[str, object]) -> bool:
    return (
        (external_type == RING_GEMM_EXTERNAL_TYPE or external_type in LOCAL_GEMM_EXTERNAL_TYPES)
        and "write_reg_hint" in input_data
    )


def _fixed_input_write_reg_hint(external_type: str, external_port: str) -> str | None:
    if external_type == RING_GEMM_EXTERNAL_TYPE and external_port == "B":
        return "reorder(n8,k2)->(k2,n8)"
    return None


def fill_external_rmsnorm_remapping(
    payload: Mapping[str, object],
    hw_cfg: HardwareSpec | None = None,
    registry: Mapping[str, RegisteredOp] | None = None,
) -> Dict[str, object]:
    return fill_external_remapping(payload, hw_cfg=hw_cfg, registry=registry)


def fill_external_remapping_file(
    input_path: str,
    output_path: str | None = None,
    hw_cfg: HardwareSpec | None = None,
    registry: Mapping[str, RegisteredOp] | None = None,
) -> Path:
    payload = load_graph_file(input_path)
    filled = fill_external_remapping(payload, hw_cfg=hw_cfg, registry=registry)
    destination = Path(output_path) if output_path else default_rmsnorm_output_path(input_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(filled, indent=2) + "\n", encoding="utf-8")
    return destination


def fill_external_rmsnorm_remapping_file(
    input_path: str,
    output_path: str | None = None,
    hw_cfg: HardwareSpec | None = None,
    registry: Mapping[str, RegisteredOp] | None = None,
) -> Path:
    return fill_external_remapping_file(
        input_path,
        output_path=output_path,
        hw_cfg=hw_cfg,
        registry=registry,
    )


def default_rmsnorm_output_path(input_path: str) -> Path:
    source = Path(input_path)
    return source.with_name(f"{source.stem}_remapped{source.suffix}")


def _normalize_external_shape(
    external_shape: object,
    external_type: str,
    internal_port: str,
    expected_axes: Sequence[str],
    op_id: str,
    port_name: str,
    values: Mapping[str, int],
) -> tuple[Dict[str, str], Dict[str, int], Dict[str, object]]:
    if not isinstance(external_shape, list):
        raise RmsNormBridgeError(
            f"Operator '{op_id}' port '{port_name}' shape must be a list of integers, got {external_shape!r}."
        )
    try:
        dims = [_evaluate_external_dim(dim, values) for dim in external_shape]
    except ValueError as exc:
        raise RmsNormBridgeError(
            f"Operator '{op_id}' port '{port_name}' has invalid shape {external_shape!r}: {exc}"
        ) from exc
    axes = list(expected_axes)
    logical_shape, resolved_shape = _map_external_execplan_shape(
        dims=dims,
        external_type=external_type,
        internal_port=internal_port,
        expected_axes=axes,
        op_id=op_id,
        port_name=port_name,
    )
    debug_info = {
        "external_dims": list(dims),
        "expected_axes": list(axes),
        "mapped_shape": dict(logical_shape),
        "mapped_resolved_shape": dict(resolved_shape),
    }
    return logical_shape, resolved_shape, debug_info


def _map_external_execplan_shape(
    dims: Sequence[int],
    external_type: str,
    internal_port: str,
    expected_axes: Sequence[str],
    op_id: str,
    port_name: str,
) -> tuple[Dict[str, str], Dict[str, int]]:
    axes = list(expected_axes)

    if external_type == RING_GEMM_EXTERNAL_TYPE:
        return _map_ring_gemm_shape(dims, internal_port, axes, op_id, port_name)
    if external_type in LOCAL_GEMM_EXTERNAL_TYPES:
        return _map_local_gemm_shape(dims, internal_port, axes, op_id, port_name)

    if (
        external_type == "prefill_mac_SFU_fp32MN_fp32MN"
        and internal_port == "inA"
        and axes == ["M"]
    ):
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects mac_SFU input shape [factor, 1, M], got {list(dims)}."
            )
        return {"M": "M"}, {"M": int(dims[2])}

    if len(axes) == 2 and axes[0] == "M":
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects external execplan 2D shape [1, M, X], got {list(dims)}."
            )
        second_axis = axes[1]
        return {"M": "M", second_axis: second_axis}, {"M": int(dims[1]), second_axis: int(dims[2])}

    if axes == ["M"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects external execplan vector-M shape [1, M, *], got {list(dims)}."
            )
        non_unit_extents = [int(dim) for dim in dims[1:] if int(dim) != 1]
        m_extent = max(non_unit_extents) if non_unit_extents else 1
        return {"M": "M"}, {"M": m_extent}

    if axes == ["N"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects external execplan vector-N shape [1, 1, N], got {list(dims)}."
            )
        return {"N": "N"}, {"N": int(dims[2])}

    if len(dims) == len(axes):
        return {axis: axis for axis in axes}, {axis: int(dims[idx]) for idx, axis in enumerate(axes)}

    raise RmsNormBridgeError(
        f"Operator '{op_id}' port '{port_name}' uses unsupported logical axes {axes} for external bridge."
    )


def _map_ring_gemm_shape(
    dims: Sequence[int],
    internal_port: str,
    axes: Sequence[str],
    op_id: str,
    port_name: str,
) -> tuple[Dict[str, str], Dict[str, int]]:
    if internal_port == "inA" and list(axes) == ["M", "K"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects ring GEMM A shape [K, M, 1], got {list(dims)}."
            )
        return {"M": "M", "K": "K"}, {"M": int(dims[1]), "K": int(dims[0])}
    if internal_port == "inB" and list(axes) == ["K", "N"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects ring GEMM B shape [K, 1, N], got {list(dims)}."
            )
        return {"K": "K", "N": "N"}, {"K": int(dims[0]), "N": int(dims[2])}
    if internal_port == "out" and list(axes) == ["M", "N"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects ring GEMM output shape [1, M, N], got {list(dims)}."
            )
        return {"M": "M", "N": "N"}, {"M": int(dims[1]), "N": int(dims[2])}
    raise RmsNormBridgeError(
        f"Operator '{op_id}' port '{port_name}' uses unsupported ring GEMM axes {list(axes)}."
    )


def _map_local_gemm_shape(
    dims: Sequence[int],
    internal_port: str,
    axes: Sequence[str],
    op_id: str,
    port_name: str,
) -> tuple[Dict[str, str], Dict[str, int]]:
    if internal_port == "inA" and list(axes) == ["M", "K"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects local GEMM A shape [M, 1, K], got {list(dims)}."
            )
        return {"M": "M", "K": "K"}, {"M": int(dims[0]), "K": int(dims[2])}
    if internal_port == "inB" and list(axes) == ["K", "N"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects local GEMM B shape [1, N, K], got {list(dims)}."
            )
        return {"K": "K", "N": "N"}, {"K": int(dims[2]), "N": int(dims[1])}
    if internal_port == "out" and list(axes) == ["M", "N"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects local GEMM output shape [1, M, N], got {list(dims)}."
            )
        return {"M": "M", "N": "N"}, {"M": int(dims[1]), "N": int(dims[2])}
    raise RmsNormBridgeError(
        f"Operator '{op_id}' port '{port_name}' uses unsupported local GEMM axes {list(axes)}."
    )


def _validate_external_output_shape(
    op_id: str,
    external_type: str,
    inferred_shape: Mapping[str, str],
    inferred_resolved: Mapping[str, int],
    normalized_shape: Mapping[str, str],
    normalized_resolved: Mapping[str, int],
) -> None:
    if dict(inferred_shape) != dict(normalized_shape):
        raise RmsNormBridgeError(
            f"Operator '{op_id}' ({external_type}) output logical axes {dict(normalized_shape)} do not match "
            f"inferred axes {dict(inferred_shape)}."
        )
    if dict(inferred_resolved) != dict(normalized_resolved):
        raise RmsNormBridgeError(
            f"Operator '{op_id}' ({external_type}) output extents {dict(normalized_resolved)} do not match "
            f"inferred extents {dict(inferred_resolved)}."
        )


def _validate_connected_tensor_shape(
    producer_shape: Mapping[str, str],
    producer_resolved: Mapping[str, int],
    consumer_shape: Mapping[str, str],
    consumer_resolved: Mapping[str, int],
    producer: str,
    consumer: str,
    port_name: str,
    consumer_axis_aliases_override: Mapping[str, str] | None = None,
) -> None:
    producer_axes = list(producer_shape.keys())
    consumer_axes = list(consumer_shape.keys())
    if len(producer_axes) != len(consumer_axes):
        raise RmsNormBridgeError(
            f"Edge {producer}->{consumer} port '{port_name}' rank mismatch: "
            f"producer axes {producer_axes}, consumer axes {consumer_axes}."
        )
    if consumer_axis_aliases_override:
        producer_extents = [
            int(producer_resolved[str(consumer_axis_aliases_override[axis])])
            for axis in consumer_axes
        ]
    else:
        producer_extents = [int(producer_resolved[axis]) for axis in producer_axes]
    consumer_extents = [int(consumer_resolved[axis]) for axis in consumer_axes]
    if producer_extents != consumer_extents:
        raise RmsNormBridgeError(
            f"Edge {producer}->{consumer} port '{port_name}' extent mismatch after bridge normalization: "
            f"producer {producer_axes}={producer_extents}, consumer {consumer_axes}={consumer_extents}."
        )


def _normalize_edge_axis_view(
    raw_axis_view: object,
    producer_shape: Mapping[str, str],
    producer_resolved: Mapping[str, int],
    consumer_shape: Mapping[str, str],
    consumer_resolved: Mapping[str, int],
    producer: str,
    consumer: str,
    port_name: str,
) -> Dict[str, str] | None:
    if raw_axis_view is None:
        return None
    if not isinstance(raw_axis_view, Mapping):
        raise RmsNormBridgeError(
            f"Edge {producer}->{consumer} port '{port_name}' axis_view must be a mapping."
        )
    raw_mapping = raw_axis_view.get("producer_to_consumer")
    if not isinstance(raw_mapping, Mapping):
        raise RmsNormBridgeError(
            f"Edge {producer}->{consumer} port '{port_name}' axis_view must define 'producer_to_consumer'."
        )
    producer_axes = list(producer_shape.keys())
    consumer_axes = list(consumer_shape.keys())
    producer_to_consumer = {str(k): str(v) for k, v in dict(raw_mapping).items()}
    if set(producer_to_consumer.keys()) != set(producer_axes):
        raise RmsNormBridgeError(
            f"Edge {producer}->{consumer} port '{port_name}' axis_view producer axes "
            f"{sorted(producer_to_consumer.keys())} do not match producer axes {sorted(producer_axes)}."
        )
    if set(producer_to_consumer.values()) != set(consumer_axes):
        raise RmsNormBridgeError(
            f"Edge {producer}->{consumer} port '{port_name}' axis_view consumer axes "
            f"{sorted(producer_to_consumer.values())} do not match consumer axes {sorted(consumer_axes)}."
        )
    consumer_to_producer = {consumer_axis: producer_axis for producer_axis, consumer_axis in producer_to_consumer.items()}
    for consumer_axis, producer_axis in consumer_to_producer.items():
        if int(producer_resolved[producer_axis]) != int(consumer_resolved[consumer_axis]):
            raise RmsNormBridgeError(
                f"Edge {producer}->{consumer} port '{port_name}' axis_view extent mismatch: "
                f"producer axis '{producer_axis}'={int(producer_resolved[producer_axis])}, "
                f"consumer axis '{consumer_axis}'={int(consumer_resolved[consumer_axis])}."
            )
    return consumer_to_producer


def _parse_optional_base_addr(
    raw_value: object,
    op_id: str,
    port_name: str,
    require_base_addr: bool,
) -> int | None:
    if raw_value is None:
        if require_base_addr:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' must define 'base_addr' for performance analysis."
            )
        return None
    if isinstance(raw_value, int):
        base_addr = raw_value
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        try:
            base_addr = int(text, 0)
        except ValueError as exc:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' has invalid base_addr {raw_value!r}."
            ) from exc
    else:
        raise RmsNormBridgeError(
            f"Operator '{op_id}' port '{port_name}' has unsupported base_addr type {type(raw_value)!r}."
        )
    if base_addr < 0:
        raise RmsNormBridgeError(f"Operator '{op_id}' port '{port_name}' base_addr must be non-negative.")
    return base_addr


def _validate_connected_base_addr(
    producer_base_addr: int | None,
    consumer_base_addr: int | None,
    producer: str,
    consumer: str,
    port_name: str,
    require_base_addrs: bool,
) -> None:
    if producer_base_addr is None or consumer_base_addr is None:
        if require_base_addrs:
            raise RmsNormBridgeError(
                f"Edge {producer}->{consumer} port '{port_name}' requires matching producer/consumer base_addr values."
            )
        return
    if producer_base_addr != consumer_base_addr:
        raise RmsNormBridgeError(
            f"Edge {producer}->{consumer} port '{port_name}' base_addr mismatch: "
            f"producer has {producer_base_addr:#x}, consumer has {consumer_base_addr:#x}."
        )


def _resolve_output_shape(
    output_shape: Mapping[str, str],
    local_input_resolved: Mapping[str, Mapping[str, int]],
) -> Dict[str, int]:
    available: Dict[str, int] = {}
    for input_resolved in local_input_resolved.values():
        available.update({str(axis): int(value) for axis, value in dict(input_resolved).items()})

    resolved: Dict[str, int] = {}
    for axis_name, source_axis in dict(output_shape).items():
        key = str(source_axis)
        if key not in available:
            raise RmsNormBridgeError(
                f"Cannot resolve output axis '{axis_name}' from input axis '{source_axis}'."
            )
        resolved[str(axis_name)] = available[key]
    return resolved


def _validate_bridgeable_result(result: EdgeSolveResult) -> None:
    if result.status != "ok":
        raise RmsNormBridgeError(
            f"Edge {result.producer}->{result.consumer} ({result.tensor_name}) could not be solved: "
            f"{result.reason_code}: {result.reason}"
        )


def _is_external_source(source: object) -> bool:
    return isinstance(source, Mapping) and str(source.get("type", "")).strip() == "external"


def _resolve_external_values(payload: Mapping[str, object]) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for key, value in dict(payload.get("params", {})).items():
        if isinstance(value, int):
            values[str(key)] = value
    if payload.get("used_slices") is not None:
        values["used_slices"] = int(payload["used_slices"])
    return values


def _evaluate_external_dim(dim: object, values: Mapping[str, int]) -> int:
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        return int(evaluate_expr(dim, dict(values)))
    raise ValueError(f"unsupported dimension type {type(dim)!r}")


def _normalize_source_ref(source: object) -> object:
    if _is_external_source(source):
        return source
    if isinstance(source, Mapping):
        source_type = str(source.get("type", "")).strip()
        if source_type:
            return {"type": "external"} if source_type != "external" else source
    return source


def _external_to_internal_ports(external_type: str) -> Dict[str, str]:
    mapping = dict(EXTERNAL_TO_INTERNAL_PORT)
    mapping.update(EXTERNAL_PORT_OVERRIDES.get(external_type, {}))
    return mapping


def _filtered_external_inputs(
    external_type: str,
    external_inputs: Mapping[str, object],
) -> Dict[str, object]:
    filtered = dict(external_inputs)
    if external_type == RING_GEMM_EXTERNAL_TYPE or external_type in LOCAL_GEMM_EXTERNAL_TYPES:
        filtered.pop("B'", None)
    return filtered


def _canonical_operator_type(
    external_type: str,
    registry: Mapping[str, RegisteredOp],
) -> str:
    if external_type in registry:
        return external_type
    canonical_type = EXTERNAL_TO_CANONICAL_OP.get(external_type)
    if canonical_type is None or canonical_type not in registry:
        raise RmsNormBridgeError(f"Unsupported external operator type '{external_type}'.")
    return canonical_type


def _internal_to_external_port(port_name: str) -> str:
    for external_port, internal_port in EXTERNAL_TO_INTERNAL_PORT.items():
        if internal_port == port_name:
            return external_port
    raise RmsNormBridgeError(f"Unsupported internal port name '{port_name}'.")
