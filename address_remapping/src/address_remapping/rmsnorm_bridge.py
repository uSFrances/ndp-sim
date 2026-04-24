from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .graph import load_graph_file, solve_graph
from .hardware import HardwareSpec
from .registry import RegisteredOp, build_default_registry
from .solver import EdgeSolveResult


EXTERNAL_TO_CANONICAL_OP = {
    "prefill_summac_fp32MN_fp32MN": "prefill_summac",
    "prefill_remote_sum_fp32MN_fp32MN": "prefill_remote_sum_Mfp32_Mfp32",
    "prefill_mac_SFU_fp32MN_fp32MN": "prefill_mac_SFU",
    "prefill_max_fp32MN_fp32MN": "prefill_max",
    "prefill_sum_rec_fp32MN_fp32MN": "prefill_sum_rec",
}

EXTERNAL_TO_INTERNAL_PORT = {
    "A": "inA",
    "B": "inB",
    "C": "inC",
}


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

        external_inputs = operator.get("inputs")
        if not isinstance(external_inputs, Mapping):
            raise RmsNormBridgeError(f"Operator '{op_id}' must define an 'inputs' mapping.")

        if len(external_inputs) != len(registered_op.ordered_inputs):
            raise RmsNormBridgeError(
                f"Operator '{op_id}' ({external_type}) expects {len(registered_op.ordered_inputs)} inputs, "
                f"got {len(external_inputs)}."
            )

        local_input_shapes: Dict[str, Dict[str, str]] = {}
        local_input_resolved: Dict[str, Dict[str, int]] = {}
        normalized_inputs: Dict[str, Dict[str, object]] = {}
        input_sources: Dict[str, object] = {}
        input_tensor_names: Dict[str, str] = {}
        input_base_addrs: Dict[str, int | None] = {}

        for external_port, internal_port in EXTERNAL_TO_INTERNAL_PORT.items():
            if internal_port not in registered_op.input_ports:
                continue
            external_input = external_inputs.get(external_port)
            if not isinstance(external_input, Mapping):
                raise RmsNormBridgeError(f"Operator '{op_id}' input '{external_port}' must be a mapping.")

            port_template = registered_op.input_ports[internal_port]
            port_layout = port_template.layout_builder(port_template.memory_dtype)
            local_shape, local_resolved = _normalize_external_shape(
                external_shape=external_input.get("shape"),
                expected_axes=list(port_layout.logical_shape.keys()),
                op_id=op_id,
                port_name=external_port,
            )
            local_input_shapes[internal_port] = local_shape
            local_input_resolved[internal_port] = local_resolved
            normalized_inputs[internal_port] = {
                **port_template.build(),
                "shape": local_shape,
                "resolved_shape": local_resolved,
            }

            source = external_input.get("source")
            input_sources[internal_port] = source
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
        normalized_output_shape, normalized_output_resolved = _normalize_external_shape(
            external_shape=external_output.get("shape"),
            expected_axes=list(output_layout.logical_shape.keys()),
            op_id=op_id,
            port_name="output",
        )
        if normalized_output_shape != inferred_output_shape:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' output shape {normalized_output_shape} does not match inferred shape "
                f"{inferred_output_shape}."
            )
        if normalized_output_resolved != inferred_output_resolved:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' resolved output shape {normalized_output_resolved} does not match inferred "
                f"shape {inferred_output_resolved}."
            )

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
            "shape": inferred_output_shape,
            "resolved_shape": inferred_output_resolved,
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
                "input_base_addrs": input_base_addrs,
                "output": {
                    output_port: {
                        **output_template.build(),
                        "shape": inferred_output_shape,
                        "resolved_shape": inferred_output_resolved,
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
        for internal_port, input_spec in dict(normalized_op["inputs"]).items():
            resolved_input = dict(input_spec)
            source = normalized_op["input_sources"].get(internal_port)
            if _is_external_source(source):
                resolved_input["source_tensor"] = normalized_op["input_tensor_names"][internal_port]
            else:
                source_op_id = str(source)
                if source_op_id not in produced_tensors:
                    raise RmsNormBridgeError(
                        f"Operator '{op_id}' input '{internal_port}' references unknown producer '{source_op_id}'."
                    )
                producer_tensor = produced_tensors[source_op_id]
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
            consumer_input = resolved_inputs[internal_port]
            expected_dtype = str(normalized_op["input_dtypes"][internal_port])
            if producer_tensor["dtype"] != expected_dtype:
                raise RmsNormBridgeError(
                    f"Operator '{op_id}' input '{internal_port}' expects dtype {expected_dtype} but producer "
                    f"'{source_op_id}' outputs {producer_tensor['dtype']}."
                )
            if producer_tensor["resolved_shape"] != consumer_input["resolved_shape"]:
                raise RmsNormBridgeError(
                    f"Operator '{op_id}' input '{internal_port}' resolved shape {consumer_input['resolved_shape']} "
                    f"does not match producer '{source_op_id}' output shape {producer_tensor['resolved_shape']}."
                )
            edges.append(
                {
                    "tensor": producer_tensor["name"],
                    "producer": source_op_id,
                    "producer_port": "out",
                    "consumer": op_id,
                    "consumer_port": internal_port,
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
            _internal_to_external_port(str(edge["consumer_port"])),
        ): result
        for edge, result in zip(expanded["edges"], results)
    }
    default_permutation = list(range(hardware.remap_bits))

    filled = copy.deepcopy(dict(payload))
    operators = filled.get("operators")
    if not isinstance(operators, list):
        raise RmsNormBridgeError("External execplan payload must define an 'operators' list.")

    for operator in operators:
        op_id = str(operator["id"])
        inputs = operator.get("inputs", {})
        for external_port, input_data in dict(inputs).items():
            if not isinstance(input_data, dict):
                continue
            source = input_data.get("source")
            if _is_external_source(source):
                input_data["remapping"] = None
                input_data.pop("writereg", None)
                continue

            key = (str(source), op_id, external_port)
            result = result_by_key.get(key)
            if result is None:
                raise RmsNormBridgeError(
                    f"Missing solve result for edge {source} -> {op_id} port {external_port}."
                )
            _validate_bridgeable_result(result)
            if result.write_reg_required:
                input_data["writereg"] = {
                    "required": True,
                    "hint": result.write_reg_hint,
                }
            else:
                input_data.pop("writereg", None)

            if result.permutation == default_permutation:
                if "remapping" in input_data:
                    input_data["remapping"] = None
                continue
            input_data["remapping"] = result.permutation

    return filled, results


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
    expected_axes: Sequence[str],
    op_id: str,
    port_name: str,
) -> tuple[Dict[str, str], Dict[str, int]]:
    if not isinstance(external_shape, list) or not all(isinstance(dim, int) for dim in external_shape):
        raise RmsNormBridgeError(
            f"Operator '{op_id}' port '{port_name}' shape must be a list of integers, got {external_shape!r}."
        )
    dims = [int(dim) for dim in external_shape]
    if not dims or dims[0] != 1:
        raise RmsNormBridgeError(
            f"Operator '{op_id}' port '{port_name}' shape must start with batch dimension 1, got {dims}."
        )

    axes = list(expected_axes)
    if axes == ["M", "N"]:
        if len(dims) != 3:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects shape [1, M, N], got {dims}."
            )
        return {"M": "M", "N": "N"}, {"M": dims[1], "N": dims[2]}

    if axes == ["M"]:
        if len(dims) < 2:
            raise RmsNormBridgeError(
                f"Operator '{op_id}' port '{port_name}' expects at least [1, M], got {dims}."
            )
        return {"M": "M"}, {"M": dims[-1]}

    if len(dims) == len(axes) + 1:
        return {axis: axis for axis in axes}, {axis: dims[idx + 1] for idx, axis in enumerate(axes)}

    raise RmsNormBridgeError(
        f"Operator '{op_id}' port '{port_name}' uses unsupported logical axes {axes} for external bridge."
    )


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
