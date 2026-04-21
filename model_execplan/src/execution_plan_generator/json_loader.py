from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import JsonFormatError
from .models import ExecutionPlanInput, InputSource, InputSourceType, OperatorSpec, TensorSpec


INPUT_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "A": ("A",),
    "B": ("B",),
    "B'": ("B'", "Bp", "B_prime", "Bprime"),
    "C": ("C",),
}


def load_execution_plan_json(file_path: str | Path) -> ExecutionPlanInput:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise JsonFormatError("Top-level JSON must be an object.")

    used_slices = _parse_used_slices(raw)
    operators = _parse_operators(_pick(raw, "operators", "ops", "算子列表"), used_slices)
    if not operators:
        raise JsonFormatError("No operators found in JSON.")

    return ExecutionPlanInput(used_slices=used_slices, operators=operators)


def execution_plan_to_dict(plan: ExecutionPlanInput) -> dict[str, Any]:
    return {
        "used_slices": _format_slice_mask(plan.used_slices),
        "operators": [
            {
                "id": op.op_id,
                "type": op.op_type,
                "used_slices": _format_slice_mask(op.used_slices),
                "inputs": {
                    name: {
                        "shape": list(spec.shape),
                        "dtype": spec.dtype,
                        "remapping": list(spec.remapping) if spec.remapping is not None else None,
                        "source": {
                            "type": spec.source.source_type.value if spec.source else None,
                            "operator_id": spec.source.operator_id if spec.source else None,
                        },
                    }
                    for name, spec in op.inputs.items()
                },
                "output": {
                    "shape": list(op.output.shape),
                    "dtype": op.output.dtype,
                    "remapping": list(op.output.remapping) if op.output.remapping is not None else None,
                },
            }
            for op in plan.operators
        ],
    }


def _parse_used_slices(raw: dict[str, Any]) -> int:
    value = _pick(raw, "used_slices", "slices", "使用slices数", "使用slice数")
    if value is None:
        return (1 << 28) - 1
    return _parse_slice_mask_value(value, field_name="used_slices")


def _parse_operators(raw_ops: Any, default_used_slices: int) -> list[OperatorSpec]:
    if raw_ops is None:
        raise JsonFormatError("Missing operators section.")

    operators: list[OperatorSpec] = []
    if isinstance(raw_ops, list):
        for idx, item in enumerate(raw_ops):
            if not isinstance(item, dict):
                raise JsonFormatError(f"operators[{idx}] must be an object.")
            operators.append(
                _parse_operator(item, fallback_id=f"op_{idx}", default_used_slices=default_used_slices)
            )
    elif isinstance(raw_ops, dict):
        for key, item in raw_ops.items():
            if not isinstance(item, dict):
                raise JsonFormatError(f"operators[{key}] must be an object.")
            operators.append(
                _parse_operator(item, fallback_id=str(key), default_used_slices=default_used_slices)
            )
    else:
        raise JsonFormatError("operators must be a list or object.")

    return operators


def _parse_operator(raw_op: dict[str, Any], fallback_id: str, default_used_slices: int) -> OperatorSpec:
    op_id = _pick(raw_op, "id", "op_id", "算子编号") or fallback_id
    if not isinstance(op_id, str) or not op_id.strip():
        raise JsonFormatError("Operator id must be a non-empty string.")

    op_type = _pick(raw_op, "type", "op_type", "算子类型")
    if not isinstance(op_type, str) or not op_type.strip():
        raise JsonFormatError(f"Operator {op_id}: type is required.")

    raw_used_slices = _pick(raw_op, "used_slices", "slices", "使用slices数", "使用slice数")
    used_slices = (
        default_used_slices
        if raw_used_slices is None
        else _parse_slice_mask_value(raw_used_slices, field_name=f"Operator {op_id}: used_slices")
    )

    inputs = _parse_inputs(raw_op, op_id)
    output = _parse_output(raw_op, op_id)
    return OperatorSpec(op_id=op_id, op_type=op_type, used_slices=used_slices, inputs=inputs, output=output)


def _parse_inputs(raw_op: dict[str, Any], op_id: str) -> dict[str, TensorSpec]:
    raw_inputs = _pick(raw_op, "inputs", "输入")

    if raw_inputs is None:
        # Also support flat A/B/B'/C keys at operator level.
        raw_inputs = {
            canonical: _pick(raw_op, *aliases, *(f"输入{a}" for a in aliases))
            for canonical, aliases in INPUT_NAME_ALIASES.items()
        }

    if not isinstance(raw_inputs, dict):
        raise JsonFormatError(f"Operator {op_id}: inputs must be an object.")

    inputs: dict[str, TensorSpec] = {}
    for canonical, aliases in INPUT_NAME_ALIASES.items():
        raw_tensor = _pick(raw_inputs, *aliases, *(f"输入{a}" for a in aliases))
        if raw_tensor is None:
            continue
        inputs[canonical] = _parse_input_tensor(raw_tensor, op_id, canonical)

    if not inputs:
        raise JsonFormatError(f"Operator {op_id}: at least one input is required.")

    return inputs


def _parse_input_tensor(raw_tensor: Any, op_id: str, name: str) -> TensorSpec:
    if not isinstance(raw_tensor, dict):
        raise JsonFormatError(f"Operator {op_id}: input {name} must be an object.")

    shape = _parse_shape(raw_tensor, f"Operator {op_id}: input {name}")
    dtype = _parse_dtype(raw_tensor, f"Operator {op_id}: input {name}")
    remapping = _parse_remapping(raw_tensor, f"Operator {op_id}: input {name}")
    source = _parse_source(_pick(raw_tensor, "source", "输入来源", "来源"), op_id, name)

    return TensorSpec(shape=shape, dtype=dtype, source=source, remapping=remapping)


def _parse_output(raw_op: dict[str, Any], op_id: str) -> TensorSpec:
    raw_output = _pick(raw_op, "output", "D", "输出D", "输出")
    if raw_output is None:
        raise JsonFormatError(f"Operator {op_id}: output is required.")
    if not isinstance(raw_output, dict):
        raise JsonFormatError(f"Operator {op_id}: output must be an object.")

    shape = _parse_shape(raw_output, f"Operator {op_id}: output")
    dtype = _parse_dtype(raw_output, f"Operator {op_id}: output")
    remapping = _parse_remapping(raw_output, f"Operator {op_id}: output")
    return TensorSpec(shape=shape, dtype=dtype, remapping=remapping)


def _parse_dtype(raw_tensor: dict[str, Any], location: str) -> str:
    raw_dtype = _pick(raw_tensor, "dtype", "data_type", "数据类型")
    if raw_dtype is None:
        return "fp32"
    if not isinstance(raw_dtype, str):
        raise JsonFormatError(f"{location} dtype must be a string.")

    normalized = raw_dtype.strip().lower()
    aliases = {
        "pf32": "fp32",
        "float32": "fp32",
        "float16": "fp16",
        "int8": "int8",
        "uint8": "uint8",
        "int16": "int16",
        "uint16": "uint16",
        "int32": "int32",
        "uint32": "uint32",
        "fp16": "fp16",
        "fp32": "fp32",
    }
    mapped = aliases.get(normalized)
    if mapped is None:
        raise JsonFormatError(
            f"{location} dtype '{raw_dtype}' is not supported. "
            "Supported: fp16, fp32, int8, uint8, int16, uint16, int32, uint32."
        )
    return mapped


def _parse_shape(raw_tensor: dict[str, Any], location: str) -> tuple[int, int, int]:
    raw_shape = _pick(raw_tensor, "shape", "tensor_shape", "tensor形状")
    if raw_shape is None:
        raise JsonFormatError(f"{location} missing shape.")
    if not isinstance(raw_shape, list) or len(raw_shape) != 3:
        raise JsonFormatError(f"{location} shape must be [K, M, N].")
    if not all(isinstance(x, int) and x > 0 for x in raw_shape):
        raise JsonFormatError(f"{location} shape values must be positive integers.")
    return (raw_shape[0], raw_shape[1], raw_shape[2])


def _parse_remapping(raw_tensor: dict[str, Any], location: str) -> tuple[int, ...] | None:
    raw_remapping = _pick(raw_tensor, "remapping", "重映射")
    if raw_remapping is None:
        return None
    if not isinstance(raw_remapping, list):
        raise JsonFormatError(
            f"{location} remapping must be null or a list of integers in [0, 25]."
        )
    if len(raw_remapping) != 26:
        raise JsonFormatError(
            f"{location} remapping must contain exactly 26 elements, got {len(raw_remapping)}."
        )

    out: list[int] = []
    for idx, value in enumerate(raw_remapping):
        if not isinstance(value, int) or not (0 <= value <= 25):
            raise JsonFormatError(
                f"{location} remapping[{idx}] must be an integer in [0, 25]."
            )
        out.append(value)
    return tuple(out)


def _parse_source(raw_source: Any, op_id: str, input_name: str) -> InputSource:
    if raw_source is None:
        raise JsonFormatError(f"Operator {op_id}: input {input_name} missing source.")

    if isinstance(raw_source, str):
        lower = raw_source.strip().lower()
        if lower in {"external", "input", "外部输入"}:
            return InputSource(source_type=InputSourceType.EXTERNAL)
        # If a plain string is provided and not external, treat it as dependency op id.
        return InputSource(source_type=InputSourceType.OPERATOR, operator_id=raw_source)

    if isinstance(raw_source, int):
        return InputSource(source_type=InputSourceType.OPERATOR, operator_id=str(raw_source))

    if not isinstance(raw_source, dict):
        raise JsonFormatError(f"Operator {op_id}: input {input_name} has invalid source format.")

    source_type_raw = _pick(raw_source, "type", "source_type", "来源类型")
    source_op_id = _pick(
        raw_source,
        "operator_id",
        "op_id",
        "source_op_id",
        "算子编号",
        "来自算子",
    )

    if isinstance(source_type_raw, str):
        lower = source_type_raw.strip().lower()
        if lower in {"external", "input", "外部输入"}:
            return InputSource(source_type=InputSourceType.EXTERNAL)
        if lower in {"operator", "op", "来自算子"}:
            if source_op_id is None:
                raise JsonFormatError(
                    f"Operator {op_id}: input {input_name} source type is operator but operator_id is missing."
                )
            return InputSource(source_type=InputSourceType.OPERATOR, operator_id=str(source_op_id))

    # Backward-compat: if dict has op id but no explicit type, infer as operator.
    if source_op_id is not None:
        return InputSource(source_type=InputSourceType.OPERATOR, operator_id=str(source_op_id))

    raise JsonFormatError(f"Operator {op_id}: input {input_name} source is invalid.")


def _pick(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _parse_slice_mask_value(value: Any, field_name: str) -> int:
    if isinstance(value, int):
        mask = value
    elif isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0b"):
            text = text[2:]
        if not text or any(ch not in {"0", "1"} for ch in text):
            raise JsonFormatError(f"{field_name} must be a binary string like 0b1111.")
        mask = int(text, 2)
    else:
        raise JsonFormatError(f"{field_name} must be a binary string like 0b1111.")

    if mask <= 0:
        raise JsonFormatError(f"{field_name} must enable at least one slice.")
    if mask >= (1 << 28):
        raise JsonFormatError(f"{field_name} must fit within 28 bits.")
    return mask


def _format_slice_mask(mask: int) -> str:
    return f"0b{mask:028b}"
