from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .config_stream_decoder import decode_initial_register_state, load_template_config_stream
from .control_registers import compute_control_register_updates
from .models import ExecutionPlanInput, OperatorSpec, OperatorTemplate
from .register_mapping import RegisterMappingDB, load_register_mapping


ControlRegisterFn = Callable[[OperatorSpec, OperatorTemplate], dict[str, int]]


class OperatorTemplateManager:
    """Load per-op base metadata from one shared config file."""

    def __init__(
        self,
        base_info_path: str | Path | None = None,
        register_db: RegisterMappingDB | None = None,
        control_register_fn: ControlRegisterFn | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self._project_root = root
        self._base_info_path = (
            Path(base_info_path) if base_info_path is not None else root / "config" / "operator_base_info.json"
        )
        self._register_db = register_db or load_register_mapping(
            register_map_csv=root / "config" / "register_map_with_groups1.csv",
            config_output_csv=root / "config" / "config_output.csv",
        )
        self._control_register_fn = control_register_fn or compute_control_register_updates
        self._base_info = self._load_base_info()

    def load_template(
        self,
        op_type: str,
        target_io_sizes: dict[str, tuple[int, int, int]],
    ) -> OperatorTemplate:
        raw = self._base_info.get(op_type)
        if raw is None:
            return OperatorTemplate(
                op_type=op_type,
                target_io_sizes=target_io_sizes,
                target_size=target_io_sizes.get("D"),
                should_update_control_registers=True,
            )

        original_register_values: dict[int, int] = {}
        enabled_register_addresses: frozenset[int] = frozenset()
        config_stream_raw = raw.get("initial_config_stream")
        if config_stream_raw is not None:
            if not isinstance(config_stream_raw, dict):
                raise ValueError("initial_config_stream must be an object")
            config_stream = load_template_config_stream(
                config_stream_raw,
                self._base_info_path.parent,
                self._register_db,
            )
            decoded_state = decode_initial_register_state(config_stream, self._register_db)
            original_register_values = decoded_state.register_values
            enabled_register_addresses = frozenset(decoded_state.enabled_register_addresses)

        initial_size_raw = raw.get("initial_size")
        initial_io_sizes = _parse_io_sizes(initial_size_raw) if initial_size_raw is not None else {}
        initial_size_d = initial_io_sizes.get("D")
        target_size_d = target_io_sizes.get("D")
        should_update = _should_update_control_registers(initial_io_sizes, target_io_sizes)
        detected_config_length = self._detect_config_length_from_bitstream64(op_type)
        sfu_type = self._parse_sfu_type(op_type=op_type, raw_value=raw.get("config_sfu"))
        sfu_config_length = self._detect_sfu_config_length(sfu_type) if sfu_type is not None else None

        return OperatorTemplate(
            op_type=op_type,
            config_length=detected_config_length,
            ddr_config_addr=_optional_int(raw.get("ddr_config_addr")),
            config_bitstream_addr=_optional_addr(raw.get("config_bitstream_addr")),
            config_bitstream_path=_optional_str(raw.get("config_bitstream_path")),
            initial_io_sizes=initial_io_sizes,
            target_io_sizes=target_io_sizes,
            initial_size=initial_size_d,
            target_size=target_size_d,
            should_update_control_registers=should_update,
            original_register_values=original_register_values,
            enabled_register_addresses=enabled_register_addresses,
            config_sfu_type=sfu_type,
            sfu_config_length=sfu_config_length,
        )

    def _parse_sfu_type(self, op_type: str, raw_value: object) -> str | None:
        if raw_value is None:
            return None

        aliases = {
            "gelu": "GELU",
            "rec": "REC",
            "rec_sqrt": "REC_SQRT",
            "relu": "ReLU",
            "sigmoid": "Sigmoid",
            "sigmiod": "Sigmoid",
            "silu": "SiLU",
            "sqrt": "SQRT",
            "tanh": "Tanh",
            "ex": "Ex",
        }

        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
            mapped = aliases.get(normalized)
            if mapped is None:
                supported = "GELU, REC, REC_SQRT, ReLU, Sigmoid, SiLU, SQRT, Tanh, Ex"
                raise ValueError(
                    f"Unsupported config_sfu type '{raw_value}' for operator {op_type}. Supported: {supported}."
                )
            return mapped

        if isinstance(raw_value, bool):
            if not raw_value:
                return None
            inferred = self._infer_sfu_type_from_op_type(op_type)
            if inferred is not None:
                return inferred
            # Backward-compatible fallback when legacy bool is still present.
            return "GELU"

        raise ValueError(f"config_sfu for operator {op_type} must be null, bool, or string type")

    def _infer_sfu_type_from_op_type(self, op_type: str) -> str | None:
        name = op_type.lower()
        if "rec_sqrt" in name:
            return "REC_SQRT"
        if "silu" in name:
            return "SiLU"
        if "sigmoid" in name or "sigmiod" in name:
            return "Sigmoid"
        if "relu" in name:
            return "ReLU"
        if "tanh" in name:
            return "Tanh"
        if "gelu" in name:
            return "GELU"
        if "ex" in name:
            return "Ex"
        if "sqrt" in name:
            return "SQRT"
        if "rec" in name:
            return "REC"
        return None

    def _detect_sfu_config_length(self, sfu_type: str) -> int:
        sfu_file = self._project_root / "config" / "SFU_Coeff" / f"{sfu_type}.txt"
        if not sfu_file.is_file():
            raise ValueError(f"Missing SFU coeff file for type {sfu_type}: {sfu_file}")
        return self._count_non_empty_lines(sfu_file)

    def _detect_config_length_from_bitstream64(self, op_type: str) -> int:
        op_dir = self._project_root / "config" / op_type
        exact_64 = op_dir / f"{op_type}_bitstream_64b.bin"
        if exact_64.is_file():
            return self._count_non_empty_lines(exact_64)

        exact_128 = op_dir / f"{op_type}_bitstream_128b.bin"
        if exact_128.is_file():
            return self._count_non_empty_lines(exact_128) * 2

        wildcard_64 = sorted(op_dir.glob("*bitstream_64b.bin")) if op_dir.is_dir() else []
        if len(wildcard_64) == 1:
            return self._count_non_empty_lines(wildcard_64[0])

        wildcard_128 = sorted(op_dir.glob("*bitstream_128b.bin")) if op_dir.is_dir() else []
        if len(wildcard_128) == 1:
            return self._count_non_empty_lines(wildcard_128[0]) * 2

        raise ValueError(
            "Missing config bitstream file for automatic config_length detection under "
            f"{op_dir}. Expected <op_type>_bitstream_64b.bin or <op_type>_bitstream_128b.bin."
        )

    def _count_non_empty_lines(self, file_path: Path) -> int:
        line_count = 0
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
        return line_count

    def adjust_for_operator(
        self,
        execution_input: ExecutionPlanInput,
    ) -> dict[str, OperatorTemplate]:
        templates: dict[str, OperatorTemplate] = {}
        for op in execution_input.operators:
            target_io_sizes = {name: tensor.shape for name, tensor in op.inputs.items()}
            target_io_sizes["D"] = op.output.shape
            base_template = self.load_template(op.op_type, target_io_sizes=target_io_sizes)
            control_values = (
                self._control_register_fn(op, base_template)
                if base_template.should_update_control_registers
                else {}
            )
            templates[op.op_id] = OperatorTemplate(
                op_type=base_template.op_type,
                config_length=base_template.config_length,
                ddr_config_addr=base_template.ddr_config_addr,
                config_bitstream_addr=base_template.config_bitstream_addr,
                config_bitstream_path=base_template.config_bitstream_path,
                initial_io_sizes=base_template.initial_io_sizes,
                target_io_sizes=base_template.target_io_sizes,
                initial_size=base_template.initial_size,
                target_size=base_template.target_size,
                should_update_control_registers=base_template.should_update_control_registers,
                original_register_values=base_template.original_register_values,
                enabled_register_addresses=base_template.enabled_register_addresses,
                control_register_values=control_values,
                config_sfu_type=base_template.config_sfu_type,
                sfu_config_length=base_template.sfu_config_length,
            )
        return templates

    def _load_base_info(self) -> dict[str, dict[str, object]]:
        if not self._base_info_path.exists():
            return {}

        with self._base_info_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        operators = raw.get("operators")
        if not isinstance(operators, dict):
            raise ValueError("operator_base_info.json must contain an 'operators' object")

        normalized: dict[str, dict[str, object]] = {}
        for op_type, info in operators.items():
            if not isinstance(op_type, str) or not isinstance(info, dict):
                raise ValueError("operator_base_info.json entries must be {op_type: object}")
            normalized[op_type] = info
        return normalized


def _parse_size(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("size must be [K, M, N]")
    if not all(isinstance(v, int) and v > 0 for v in value):
        raise ValueError("size values must be positive integers")
    return value[0], value[1], value[2]


def _parse_io_sizes(value: object) -> dict[str, tuple[int, int, int]]:
    if not isinstance(value, dict):
        raise ValueError("initial_size must be an object like {'A':[...],'B':[...],\"B'\":[...],'C':[...],'D':[...]}.")
    out: dict[str, tuple[int, int, int]] = {}
    for name, shape in value.items():
        if name not in {"A", "B", "B'", "C", "D"}:
            raise ValueError(f"initial_size key must be one of A/B/B'/C/D, got: {name}")
        out[name] = _parse_size(shape)
    if "D" not in out:
        raise ValueError("initial_size must include D shape")
    return out


def _should_update_control_registers(
    initial_io_sizes: dict[str, tuple[int, int, int]],
    target_io_sizes: dict[str, tuple[int, int, int]],
) -> bool:
    if not initial_io_sizes:
        return True
    keys = set(initial_io_sizes.keys()) | set(target_io_sizes.keys())
    for key in keys:
        if initial_io_sizes.get(key) != target_io_sizes.get(key):
            return True
    return False


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("Expected integer field in template metadata")
    return value


def _optional_addr(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Address field must be non-negative")
        return value
    if isinstance(value, str):
        text = value.strip().replace("_", "")
        base = 16 if text.lower().startswith("0x") else 10
        parsed = int(text, base)
        if parsed < 0:
            raise ValueError("Address field must be non-negative")
        return parsed
    raise ValueError("Expected address field as int or hex string")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected string field in template metadata")
    return value
