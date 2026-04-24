import ast
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .registry import RegisteredOp, build_default_registry


CALL_RE = re.compile(
    r"^(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<op>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)\s*$"
)


class ModelSpecError(ValueError):
    pass


def expand_model_spec(spec: Dict[str, object], registry: Dict[str, RegisteredOp] = None) -> Dict[str, object]:
    resolved_registry = registry or build_default_registry()
    shape_bindings = resolve_shape_bindings(spec)
    params = {str(k): int(v) for k, v in dict(spec.get("params", {})).items()}
    tensors = {
        str(name): _normalize_tensor_spec(name, data)
        for name, data in dict(spec.get("tensors", {})).items()
    }

    ops: Dict[str, object] = {}
    edges: List[Dict[str, object]] = []
    produced_by: Dict[str, Dict[str, str]] = {}

    for index, statement in enumerate(spec["model"]):
        lhs, op_name, args, kwargs = parse_call(str(statement))
        if op_name not in resolved_registry:
            raise ModelSpecError(f"Operator '{op_name}' is not registered.")
        op = resolved_registry[op_name]
        if len(args) != len(op.ordered_inputs):
            raise ModelSpecError(
                f"Operator '{op_name}' expects {len(op.ordered_inputs)} inputs, got {len(args)} in '{statement}'."
            )

        op_instance_name = f"{op_name}_{index}"
        inputs = {}
        outputs = {}
        local_input_shapes: Dict[str, Dict[str, str]] = {}
        local_input_resolved: Dict[str, Dict[str, int]] = {}

        for port_name, tensor_name in zip(op.ordered_inputs, args):
            if tensor_name not in tensors:
                raise ModelSpecError(f"Tensor '{tensor_name}' used before declaration or production.")
            tensor_spec = tensors[tensor_name]
            local_shape, local_resolved = resolve_local_tensor_view(
                tensor_name=tensor_name,
                tensor_spec=tensor_spec,
                op_name=op_name,
                port_name=port_name,
                call_kwargs=kwargs,
                shape_bindings=shape_bindings,
                params=params,
                resolved_inputs=local_input_resolved,
            )
            inputs[port_name] = {
                **op.input_ports[port_name].build(),
                "shape": local_shape,
                "resolved_shape": local_resolved,
                "source_tensor": tensor_name,
            }
            local_input_shapes[port_name] = local_shape
            local_input_resolved[port_name] = local_resolved
            producer = produced_by.get(tensor_name)
            if producer is not None:
                edges.append(
                    {
                        "tensor": tensor_name,
                        "producer": producer["op"],
                        "producer_port": producer["port"],
                        "consumer": op_instance_name,
                        "consumer_port": port_name,
                    }
                )

        try:
            output_shape = op.shape_resolver(local_input_shapes)
        except ValueError as exc:
            raise ModelSpecError(str(exc)) from exc
        output_resolved = _resolve_output_shape(output_shape, local_input_resolved)
        output_port_name = op.ordered_outputs[0]
        outputs[output_port_name] = {
            **op.output_ports[output_port_name].build(),
            "shape": output_shape,
            "resolved_shape": output_resolved,
            "source_tensor": lhs,
        }
        existing_tensor = dict(tensors.get(lhs, {}))
        if existing_tensor.get("dtype") is not None and existing_tensor["dtype"] != op.output_ports[output_port_name].memory_dtype:
            raise ModelSpecError(
                f"Tensor '{lhs}' declares dtype {existing_tensor['dtype']} but model produces {op.output_ports[output_port_name].memory_dtype}."
            )
        if existing_tensor.get("shape") is not None and dict(existing_tensor["shape"]) != output_shape:
            raise ModelSpecError(
                f"Tensor '{lhs}' declares shape {existing_tensor['shape']} but model produces {output_shape}."
            )
        tensors[lhs] = {
            "dtype": op.output_ports[output_port_name].memory_dtype,
            "shape": output_shape,
            "resolved_shape": output_resolved,
            "is_local": True,
            "source": {"op": op_instance_name, "port": output_port_name},
            "base_addr": existing_tensor.get("base_addr"),
        }
        produced_by[lhs] = {"op": op_instance_name, "port": output_port_name}

        ops[op_instance_name] = {
            "op_type": op_name,
            "call_kwargs": kwargs,
            "inputs": inputs,
            "outputs": outputs,
            "hardware_measured": _resolve_op_hardware_measured(
                spec=spec,
                op_instance_name=op_instance_name,
                lhs=lhs,
                op_name=op_name,
                statement_index=index,
            ),
        }

    return {
        "shape_bindings": shape_bindings,
        "params": params,
        "tensors": tensors,
        "ops": ops,
        "edges": edges,
    }


def _resolve_op_hardware_measured(
    spec: Dict[str, object],
    op_instance_name: str,
    lhs: str,
    op_name: str,
    statement_index: int,
) -> Optional[float]:
    raw = spec.get("hardware_measured")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, dict):
        raise ModelSpecError(
            "Top-level 'hardware_measured' must be a number or a mapping keyed by op instance/name/index."
        )

    candidates = (
        op_instance_name,
        lhs,
        op_name,
        str(statement_index),
        f"model[{statement_index}]",
    )
    for key in candidates:
        if key in raw:
            value = raw[key]
            if not isinstance(value, (int, float)):
                raise ModelSpecError(f"hardware_measured['{key}'] must be numeric.")
            return float(value)
    return None


def resolve_shape_bindings(spec: Dict[str, object]) -> Dict[str, int]:
    bindings = {str(k): int(v) for k, v in dict(spec["shape_bindings"]).items()}
    params = {str(k): int(v) for k, v in dict(spec.get("params", {})).items()}
    merged = dict(bindings)
    merged.update(params)
    return merged


def parse_call(statement: str) -> Tuple[str, str, List[str], Dict[str, str]]:
    match = CALL_RE.match(statement.strip())
    if not match:
        raise ModelSpecError(f"Unsupported model statement '{statement}'.")
    lhs = match.group("lhs")
    op_name = match.group("op")
    raw_args = match.group("args").strip()
    args: List[str] = []
    kwargs: Dict[str, str] = {}
    if raw_args:
        for item in _split_arguments(raw_args):
            if "=" in item:
                key, value = item.split("=", 1)
                kwargs[key.strip()] = value.strip()
            else:
                args.append(item.strip())
    return lhs, op_name, args, kwargs


def resolve_local_tensor_view(
    tensor_name: str,
    tensor_spec: Dict[str, object],
    op_name: str,
    port_name: str,
    call_kwargs: Dict[str, str],
    shape_bindings: Dict[str, int],
    params: Dict[str, int],
    resolved_inputs: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    if tensor_spec.get("is_local"):
        return dict(tensor_spec["shape"]), dict(tensor_spec["resolved_shape"])

    global_shape = dict(tensor_spec["shape"])
    partition = dict(tensor_spec.get("partition", {}))
    local_shape = {axis: axis for axis in global_shape}
    local_resolved: Dict[str, int] = {}
    for axis, expr in global_shape.items():
        global_value = evaluate_expr(expr, shape_bindings)
        spec = partition.get(axis)
        local_resolved[axis] = _resolve_partitioned_axis(
            axis=axis,
            global_value=global_value,
            partition_spec=spec,
            op_name=op_name,
            port_name=port_name,
            call_kwargs=call_kwargs,
            params=params,
            resolved_inputs=resolved_inputs,
            tensor_name=tensor_name,
        )
    return local_shape, local_resolved


def evaluate_expr(expr: object, values: Dict[str, int]) -> int:
    if isinstance(expr, int):
        return expr
    if not isinstance(expr, str):
        raise ModelSpecError(f"Unsupported shape expression type {type(expr)!r}.")
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body, values)


def _eval_ast(node: ast.AST, values: Dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ModelSpecError(f"Unknown identifier '{node.id}' in shape expression.")
        return values[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, values)
        right = _eval_ast(node.right, values)
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
    raise ModelSpecError(f"Unsupported shape expression '{ast.unparse(node)}'.")


def _resolve_partitioned_axis(
    axis: str,
    global_value: int,
    partition_spec: object,
    op_name: str,
    port_name: str,
    call_kwargs: Dict[str, str],
    params: Dict[str, int],
    resolved_inputs: Dict[str, Dict[str, int]],
    tensor_name: str,
) -> int:
    if partition_spec is None:
        return global_value
    if isinstance(partition_spec, str):
        if partition_spec == "full":
            return global_value
        raise ModelSpecError(f"Unsupported partition directive '{partition_spec}' for tensor '{tensor_name}'.")
    if "by_scope" in partition_spec:
        scope = _resolve_call_scope(call_kwargs)
        if scope not in partition_spec["by_scope"]:
            raise ModelSpecError(
                f"Tensor '{tensor_name}' axis '{axis}' does not define partition for scope={scope}."
            )
        divisor_param = str(partition_spec["by_scope"][scope])
        divisor = int(params.get(divisor_param, 0))
        if divisor <= 0:
            raise ModelSpecError(f"Parameter '{divisor_param}' must be positive for tensor '{tensor_name}'.")
        if global_value % divisor != 0:
            raise ModelSpecError(
                f"Tensor '{tensor_name}' axis '{axis}' size {global_value} is not divisible by {divisor_param}={divisor}."
            )
        return global_value // divisor
    if "follow" in partition_spec:
        follow = str(partition_spec["follow"])
        ref_port, ref_axis = follow.split(":", 1)
        if ref_port not in resolved_inputs or ref_axis not in resolved_inputs[ref_port]:
            raise ModelSpecError(
                f"Tensor '{tensor_name}' axis '{axis}' cannot follow '{follow}' before that input is resolved."
            )
        return resolved_inputs[ref_port][ref_axis]
    raise ModelSpecError(f"Unsupported partition spec for tensor '{tensor_name}' axis '{axis}': {partition_spec}.")


def _normalize_scope(scope: Optional[str]) -> str:
    normalized = (scope or "").strip().strip("'\"")
    if normalized not in {"cluster", "global"}:
        raise ModelSpecError("Scope must be one of: cluster, global.")
    return normalized


def _resolve_call_scope(call_kwargs: Dict[str, str]) -> str:
    for key in ("ring_scope", "reduce_scope", "scope"):
        raw_scope = call_kwargs.get(key)
        if raw_scope is not None:
            return _normalize_scope(raw_scope)
    raise ModelSpecError(
        "Partitioned tensors with by_scope require one of ring_scope=..., reduce_scope=..., or scope=...."
    )


def _resolve_output_shape(output_shape: Dict[str, str], local_input_resolved: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    available: Dict[str, int] = {}
    for port_shape in local_input_resolved.values():
        available.update(port_shape)
    for axis_name, expr in output_shape.items():
        if expr not in available:
            raise ModelSpecError(f"Cannot resolve output axis '{axis_name}' from expression '{expr}'.")
        resolved[axis_name] = available[expr]
    return resolved


def _normalize_tensor_spec(name: str, data: Dict[str, object]) -> Dict[str, object]:
    normalized = {
        "partition": {str(k): v for k, v in dict(data.get("partition", {})).items()},
        "scope": data.get("scope"),
        "is_local": False,
        "base_addr": int(data["base_addr"]) if "base_addr" in data else None,
    }
    if "dtype" in data:
        normalized["dtype"] = str(data["dtype"])
    if "shape" in data:
        normalized["shape"] = {str(k): str(v) for k, v in dict(data.get("shape", {})).items()}
    return normalized


def _split_arguments(raw_args: str) -> Sequence[str]:
    items: List[str] = []
    current: List[str] = []
    depth = 0
    for char in raw_args:
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items
