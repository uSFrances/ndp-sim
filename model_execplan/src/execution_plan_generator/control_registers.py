from __future__ import annotations

import json
import re
from pathlib import Path

from .models import OperatorSpec, OperatorTemplate


BASE_ADDR_BITS = 30
REMAPPING_ENTRY_BITS = 5
REMAPPING_ENTRY_COUNT = 26


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_REVIEW_CACHE: dict[str, dict[str, str]] = {}


def _fit_u20(value: int) -> int:
    if not (0 <= value < (1 << 20)):
        raise ValueError(f"dim_stride element out of 20-bit range: {value}")
    return value


def _fit_i16(value: int) -> int:
    """Return 16-bit two's complement encoding for a signed int16 value."""
    if not (-(1 << 15) <= value < (1 << 15)):
        raise ValueError(f"value out of int16 range: {value}")
    return value & 0xFFFF


def pack_address_remapping(remapping: tuple[int, ...]) -> int:
    """Pack 26 remapping entries into one 130-bit unsigned integer.

    Bit layout uses low-to-high order: remapping[i] occupies bits [5*i+4:5*i].
    """

    if len(remapping) != REMAPPING_ENTRY_COUNT:
        raise ValueError(
            "remapping length must be exactly "
            f"{REMAPPING_ENTRY_COUNT}, got {len(remapping)}"
        )

    packed = 0
    for idx, value in enumerate(remapping):
        if not (0 <= value <= 25):
            raise ValueError(f"remapping[{idx}] out of range [0,25]: {value}")
        packed |= value << (idx * REMAPPING_ENTRY_BITS)
    return packed

def pack_dim_stride(port0: int, port1: int, port2: int) -> int:
    port0 = _fit_u20(port0)
    port1 = _fit_u20(port1)
    port2 = _fit_u20(port2)
    return (port2 << 40) | (port1 << 20) | port0


def parse_base_addr(value: int | str, bits: int = BASE_ADDR_BITS) -> int:
    """Parse base address from int/str and enforce exact target bit width.

    Supported string forms:
    - binary with separators: 0b00000_00_0000000000000_000000_0000
    - plain binary digits
    - hex string: 0x1234
    """

    if bits <= 0:
        raise ValueError("bits must be positive")

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip().replace("_", "")
        if text.lower().startswith("0b"):
            raw = text[2:]
            if not raw or any(ch not in {"0", "1"} for ch in raw):
                raise ValueError(f"Invalid binary base_addr literal: {value}")
            if len(raw) != bits:
                raise ValueError(
                    f"Binary base_addr must have exactly {bits} bits, got {len(raw)}: {value}"
                )
            parsed = int(raw, 2)
        elif text.lower().startswith("0x"):
            parsed = int(text, 16)
        else:
            if any(ch not in {"0", "1"} for ch in text):
                raise ValueError(f"Invalid base_addr literal: {value}")
            if len(text) != bits:
                raise ValueError(
                    f"Binary base_addr must have exactly {bits} bits, got {len(text)}: {value}"
                )
            parsed = int(text, 2)
    else:
        raise TypeError("base_addr must be int or str")

    if not (0 <= parsed < (1 << bits)):
        raise ValueError(f"base_addr out of {bits}-bit range: {parsed}")
    return parsed


def _to_iga_lc_instance(resource: str) -> str | None:
    match = re.fullmatch(r"LC(\d+)", resource)
    if match is None:
        return None
    return f"iga_lc{int(match.group(1))}"


def _to_iga_row_lc_instance(resource: str) -> str | None:
    match = re.fullmatch(r"ROW_LC(\d+)", resource)
    if match is None:
        return None
    return f"iga_row_lc{int(match.group(1))}"


def _to_iga_col_lc_instance(resource: str) -> str | None:
    match = re.fullmatch(r"COL_LC(\d+)", resource)
    if match is None:
        return None
    return f"iga_col_lc{int(match.group(1))}"


def _to_iga_pe_instance(resource: str) -> str | None:
    match = re.fullmatch(r"PE(\d+)", resource)
    if match is None:
        return None
    return f"iga_pe{int(match.group(1))}"


def _to_stream_instance(resource: str) -> str | None:
    read_match = re.fullmatch(r"READ_STREAM(\d+)", resource)
    if read_match is not None:
        return f"rd_stream{int(read_match.group(1))}"

    write_match = re.fullmatch(r"WRITE_STREAM(\d+)", resource)
    if write_match is not None:
        return f"wr_stream{int(write_match.group(1))}"

    return None


def _parse_mapping_node_to_instance(node: str, resource: str) -> tuple[str, str] | None:
    # DRAM loop controllers: DRAM_LC.LC0 -> LC3
    match = re.fullmatch(r"DRAM_LC\.LC(\d+)", node)
    if match is not None:
        physical = _to_iga_lc_instance(resource)
        if physical is None:
            return None
        logical = f"iga_lc{int(match.group(1))}"
        return logical, physical

    # Buffer row loops: GROUP0.ROW_LC -> ROW_LC4
    match = re.fullmatch(r"GROUP(\d+)\.ROW_LC", node)
    if match is not None:
        physical = _to_iga_row_lc_instance(resource)
        if physical is None:
            return None
        logical = f"iga_row_lc{int(match.group(1))}"
        return logical, physical

    # Buffer col loops: GROUP0.COL_LC -> COL_LC4
    match = re.fullmatch(r"GROUP(\d+)\.COL_LC", node)
    if match is not None:
        physical = _to_iga_col_lc_instance(resource)
        if physical is None:
            return None
        logical = f"iga_col_lc{int(match.group(1))}"
        return logical, physical

    # LC PE: LC_PE.PE0 -> PE6
    match = re.fullmatch(r"LC_PE\.PE(\d+)", node)
    if match is not None:
        physical = _to_iga_pe_instance(resource)
        if physical is None:
            return None
        logical = f"iga_pe{int(match.group(1))}"
        return logical, physical

    # Stream engines: STREAM.stream0 -> READ_STREAM0 / WRITE_STREAM0
    match = re.fullmatch(r"STREAM\.stream(\d+)", node)
    if match is not None:
        physical = _to_stream_instance(resource)
        if physical is None:
            return None

        stream_idx = int(match.group(1))
        if physical.startswith("rd_stream"):
            logical = f"rd_stream{stream_idx}"
        else:
            logical = f"wr_stream{stream_idx}"
        return logical, physical

    return None


def _load_operator_instance_mapping(op_type: str) -> dict[str, str]:
    cached = _MAPPING_REVIEW_CACHE.get(op_type)
    if cached is not None:
        return cached

    mapping_path = _PROJECT_ROOT / "config" / op_type / "mapping_review.json"
    if not mapping_path.is_file():
        _MAPPING_REVIEW_CACHE[op_type] = {}
        return _MAPPING_REVIEW_CACHE[op_type]

    with mapping_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    node_to_resource = payload.get("node_to_resource")
    if not isinstance(node_to_resource, list):
        _MAPPING_REVIEW_CACHE[op_type] = {}
        return _MAPPING_REVIEW_CACHE[op_type]

    instance_mapping: dict[str, str] = {}
    for item in node_to_resource:
        if not isinstance(item, dict):
            continue
        node = item.get("node")
        resource = item.get("resource")
        if not isinstance(node, str) or not isinstance(resource, str):
            continue

        parsed = _parse_mapping_node_to_instance(node=node, resource=resource)
        if parsed is None:
            continue
        logical_instance, physical_instance = parsed
        instance_mapping[logical_instance] = physical_instance

    # Allow using unindexed wr_stream in register update functions.
    if "wr_stream" not in instance_mapping and "wr_stream0" in instance_mapping:
        instance_mapping["wr_stream"] = instance_mapping["wr_stream0"]

    _MAPPING_REVIEW_CACHE[op_type] = instance_mapping
    return instance_mapping


def _apply_instance_mapping_to_updates(
    updates: dict[str, int],
    instance_mapping: dict[str, str],
) -> dict[str, int]:
    if not updates or not instance_mapping:
        return updates

    mapped_updates: dict[str, int] = {}
    for field_key, value in updates.items():
        if "." not in field_key:
            mapped_key = field_key
        else:
            instance_name, config_name = field_key.split(".", maxsplit=1)
            # Hardware currently only has one write stream instance: wr_stream0.
            # Keep this normalization independent from per-operator mapping_review.
            if instance_name in {"wr_stream", "wr_stream0"}:
                mapped_instance = "wr_stream0"
            else:
                mapped_instance = instance_mapping.get(instance_name, instance_name)
            mapped_key = f"{mapped_instance}.{config_name}"

        existing = mapped_updates.get(mapped_key)
        if existing is not None and existing != value:
            raise ValueError(
                "Conflicting mapped control register values for "
                f"{mapped_key}: {existing} vs {value}"
            )
        mapped_updates[mapped_key] = value

    return mapped_updates

def _compute_prefill_max_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Example handler for max operator control-register updates.

    This function is intentionally conservative: it reads shapes and returns no-op
    updates by default. Replace the returned dict with real register-value logic
    once max control rules are finalized.
    """

    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {

        "iga_lc0.dram_loop_configs.end": a_m // 8 if a_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_gemm_local_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for gemm_local control register logic."""

    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    input_b_prime = operator.inputs.get("B'")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    b_prime_shape = input_b_prime.shape if input_b_prime is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_m, a_n, a_k) = a_shape if a_shape is not None else (None, None, None)
    (b_m, b_n, b_k) = b_shape if b_shape is not None else (None, None, None)
    (b_prime_m, b_prime_n, b_prime_k) = b_prime_shape if b_prime_shape is not None else (None, None, None)

    return {
        "iga_lc0.dram_loop_configs.end": a_m // 32 if a_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n // 32 if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": a_k // 2 if a_k is not None else 0,
        "iga_lc4.dram_loop_configs.end": a_k // 4 if a_k is not None else 0,
        "iga_pe0.lc_pe_configs.inport1.constant": _fit_i16(2 * a_k) if a_k is not None else 0,
        "iga_pe1.lc_pe_configs.inport1.constant": _fit_i16(2 * a_k) if a_k is not None else 0,
        "iga_pe3.lc_pe_configs.inport1.constant": _fit_i16(a_n // 2) if a_n is not None else 0,
    }

def _compute_prefill_summac_fp16MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for summac control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": a_m // 8 if a_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n // 2 if a_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 16,
            port2 = 16,
        ),
    }

def _compute_prefill_summac_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for summac control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": a_m // 8 if a_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
    }


def _compute_prefill_sum_rec_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for sum_rec control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": a_m // 8 if a_m is not None else 0,
        "iga_lc2.dram_loop_configs.end": a_n if a_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_mac_SFU_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for mac_SFU control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": a_n // 8 if a_n is not None else 0,
    }


def _compute_prefill_mul_fp32MN_fp32M_fp16MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_mul_fp32MN_fp32M_fp16MN control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_n if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = 0,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_mul_fp32MN_fp32M_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_mul_fp32MN_fp32M_fp32MN control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_n if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = 32 * (a_n or 0),
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = 0,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = 32 * (d_n or 0),
            port2 = 32,
        ),
    }

def _compute_prefill_add_fp32MN_fp16MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_add_fp32MN_fp16MN_fp32MN control register logic."""
    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    (b_k, b_m, b_n) = b_shape if b_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": b_n // 2 if b_n is not None else 0,
        "iga_lc3.dram_loop_configs.end": d_n // 2 if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (b_n or 0) * 32,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_add_fp32MN_fp32MN_fp16MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_add_fp32MN_fp32MN_fp16MN control register logic."""
    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    (b_k, b_m, b_n) = b_shape if b_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": b_n if b_n is not None else 0,
        "iga_lc3.dram_loop_configs.end": d_n // 2 if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (b_n or 0) * 32,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_mul_fp32MN_fp16MN_fp16MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_mul_fp32MN_fp16MN_fp16MN control register logic."""
    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    (b_k, b_m, b_n) = b_shape if b_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": b_n // 2 if b_n is not None else 0,
        "iga_lc3.dram_loop_configs.end": d_n // 2 if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (b_n or 0) * 32,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_silu_fp32MN_fp16MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_silu_fp32MN_fp16MN control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n // 2 if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_n if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_sub_SFU_fp32MN_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_sub_SFU_fp32MN_fp32MN_fp32MN control register logic."""
    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    (b_k, b_m, b_n) = b_shape if b_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": b_n if b_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = 0,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }


def _compute_quant_from_buffer_int32MN_uint8MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for quant_from_buffer_int32MN_uint8MN control register logic."""
    input_a = operator.inputs.get("A")

    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)

    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_n // 4 if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }


def _compute_add_dequant_uint8CWH_uint8CWH_fp32CWH_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for add_dequant_uint8CWH_uint8CWH_fp32CWH control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_c, d_w, d_h) = d_shape
    (a_c, a_w, a_h) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_c // 16 if d_c is not None else 0,
        "iga_lc1.dram_loop_configs.end": d_w  if d_w is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_h  if d_h is not None else 0,
        "iga_lc3.dram_loop_configs.end": d_w  if d_w is not None else 0,
        "iga_lc4.dram_loop_configs.end": d_h  if d_h is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = (a_h or 0) * (a_w or 0) * 16,
            port1 = (a_h or 0) * 16,
            port2 = 16,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = (a_h or 0) * (a_w or 0) * 16,
            port1 = (a_h or 0) * 16,
            port2 = 16,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = (a_h or 0) * (a_w or 0) * 16,
            port1 = (a_h or 0) * 64,
            port2 = 64,
        ),
    }

def _compute_prefill_remote_sum_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for remote_sum control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": a_n // 8 if a_n is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_m if a_m is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = 32 * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_mul_fp32MN_fp32N_fp16MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_mul_fp32MN_fp32N_fp16MN control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_n // 4 if d_m is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_m // 32 if a_n is not None else 0,
        "iga_lc5.dram_loop_configs.end": d_m // 64 if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_m or 0) * 4,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_m or 0) * 2,
            port2 = 16,
        ),
    }

def _compute_prefill_add_fp16MN_fp32N_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for prefill_add_fp16MN_fp32N_fp32MN control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_n // 4 if d_m is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_m // 64 if a_n is not None else 0,
        "iga_lc5.dram_loop_configs.end": d_m // 32 if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_m or 0) * 2,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_m or 0) * 4,
            port2 = 16,
        ),
    }

def _compute_decode_mul_fp32N_fp32N_fp16N_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for decode_mul_fp32N_fp32N_fp16N control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc1.dram_loop_configs.end": d_n // 8 if d_m is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_n // 16 if d_n is not None else 0,
    }

def _compute_decode_add_fp16N_fp32N_fp32N_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for decode_add_fp16N_fp32N_fp32N control register logic."""
    input_a = operator.inputs.get("A")
    a_shape = input_a.shape if input_a is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    return {
        "iga_lc1.dram_loop_configs.end": d_n // 16 if d_m is not None else 0,
        "iga_lc2.dram_loop_configs.end": d_n // 8 if d_n is not None else 0,
    }

def _compute_prefill_gemm_ring_4slice_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for gemm_ring_4slice control register logic."""
    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    (b_k, b_m, b_n) = b_shape if b_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": a_m // 32 if a_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n // 32 if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": a_k // 2 if a_k is not None else 0,
        "iga_lc4.dram_loop_configs.end": b_k // 4 if b_k is not None else 0,
        "iga_pe0.lc_pe_configs.inport1.constant": _fit_i16(2 * a_k) if a_k is not None else 0,
        "iga_pe1.lc_pe_configs.inport1.constant": _fit_i16(2 * a_k) if a_k is not None else 0,
        "iga_pe3.lc_pe_configs.inport1.constant": _fit_i16(a_n // 2) if a_n is not None else 0,
    }

def _compute_prefill_add_fp32MN_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,) -> dict[str, int]:
    """Placeholder for prefill_add_fp32MN_fp32MN_fp32MN control register logic."""
    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    (b_k, b_m, b_n) = b_shape if b_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": b_n if b_n is not None else 0,
        "iga_lc3.dram_loop_configs.end": d_n if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (b_n or 0) * 32,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }

def _compute_prefill_mul_fp32MN_fp32MN_fp32MN_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,) -> dict[str, int]:
    """Placeholder for prefill_mul_fp32MN_fp32MN_fp32MN control register logic."""
    input_a = operator.inputs.get("A")
    input_b = operator.inputs.get("B")
    a_shape = input_a.shape if input_a is not None else None
    b_shape = input_b.shape if input_b is not None else None
    d_shape = operator.output.shape
    (d_k, d_m, d_n) = d_shape
    (a_k, a_m, a_n) = a_shape if a_shape is not None else (None, None, None)
    (b_k, b_m, b_n) = b_shape if b_shape is not None else (None, None, None)
    return {
        "iga_lc0.dram_loop_configs.end": d_m // 8 if d_m is not None else 0,
        "iga_lc1.dram_loop_configs.end": a_n if a_n is not None else 0,
        "iga_lc2.dram_loop_configs.end": b_n if b_n is not None else 0,
        "iga_lc3.dram_loop_configs.end": d_n if d_n is not None else 0,
        "rd_stream0.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (a_n or 0) * 32,
            port2 = 32,
        ),
        "rd_stream1.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (b_n or 0) * 32,
            port2 = 32,
        ),
        "wr_stream.stream_engine.stream.dim_stride": pack_dim_stride(
            port0 = 0,
            port1 = (d_n or 0) * 32,
            port2 = 32,
        ),
    }

OP_CONTROL_REGISTER_FN = {
    "prefill_max_fp32MN_fp32MN": _compute_prefill_max_fp32MN_fp32MN_control_register_updates,
    "prefill_gemm_local": _compute_prefill_gemm_local_control_register_updates,
    "prefill_summac_fp16MN_fp32MN": _compute_prefill_summac_fp16MN_fp32MN_control_register_updates,
    "prefill_summac_fp32MN_fp32MN": _compute_prefill_summac_fp32MN_fp32MN_control_register_updates,
    "prefill_sum_rec_fp32MN_fp32MN": _compute_prefill_sum_rec_fp32MN_fp32MN_control_register_updates,
    "prefill_mac_SFU_fp32MN_fp32MN": _compute_prefill_mac_SFU_fp32MN_fp32MN_control_register_updates,
    "prefill_mul_fp32MN_fp32M_fp16MN": _compute_prefill_mul_fp32MN_fp32M_fp16MN_control_register_updates,  
    "prefill_add_fp32MN_fp16MN_fp32MN": _compute_prefill_add_fp32MN_fp16MN_fp32MN_control_register_updates,
    "prefill_add_fp32MN_fp32MN_fp16MN": _compute_prefill_add_fp32MN_fp32MN_fp16MN_control_register_updates,
    "prefill_mul_fp32MN_fp16MN_fp16MN": _compute_prefill_mul_fp32MN_fp16MN_fp16MN_control_register_updates,
    "prefill_silu_fp32MN_fp16MN": _compute_prefill_silu_fp32MN_fp16MN_control_register_updates,
    "prefill_sub_SFU_fp32MN_fp32MN_fp32MN": _compute_prefill_sub_SFU_fp32MN_fp32MN_fp32MN_control_register_updates,
    "quant_from_buffer_int32MN_uint8MN": _compute_quant_from_buffer_int32MN_uint8MN_control_register_updates,
    "add_dequant_uint8CWH_uint8CWH_fp32CWH": _compute_add_dequant_uint8CWH_uint8CWH_fp32CWH_control_register_updates,
    "prefill_remote_sum_fp32MN_fp32MN": _compute_prefill_remote_sum_fp32MN_fp32MN_control_register_updates,
    "prefill_mul_fp32MN_fp32N_fp16MN": _compute_prefill_mul_fp32MN_fp32N_fp16MN_control_register_updates,
    "prefill_mul_fp32MN_fp32M_fp32MN": _compute_prefill_mul_fp32MN_fp32M_fp32MN_control_register_updates,
    "prefill_add_fp16MN_fp32N_fp32MN": _compute_prefill_add_fp16MN_fp32N_fp32MN_control_register_updates,
    "decode_mul_fp32N_fp32N_fp16N": _compute_decode_mul_fp32N_fp32N_fp16N_control_register_updates,
    "decode_add_fp16N_fp32N_fp32N": _compute_decode_add_fp16N_fp32N_fp32N_control_register_updates,
    "prefill_gemm_ring_4slice": _compute_prefill_gemm_ring_4slice_control_register_updates,
    "prefill_add_fp32MN_fp32MN_fp32MN": _compute_prefill_add_fp32MN_fp32MN_fp32MN_control_register_updates,
    "prefill_mul_fp32MN_fp32MN_fp32MN": _compute_prefill_mul_fp32MN_fp32MN_fp32MN_control_register_updates,


}


def _append_remapping_register_updates(
    operator: OperatorSpec,
    updates: dict[str, int],
) -> None:
    field_by_tensor_name = {
        "A": "rd_stream0.stream_engine.stream.address_remapping",
        "B": "rd_stream1.stream_engine.stream.address_remapping",
        "B'": "rd_stream2.stream_engine.stream.address_remapping",
        "C": "rd_stream3.stream_engine.stream.address_remapping",
    }

    for tensor_name, field_key in field_by_tensor_name.items():
        tensor = operator.inputs.get(tensor_name)
        if tensor is None or tensor.remapping is None:
            continue
        updates[field_key] = pack_address_remapping(tensor.remapping)

    if operator.output.remapping is not None:
        updates["wr_stream0.stream_engine.stream.address_remapping"] = pack_address_remapping(
            operator.output.remapping
        )


def compute_control_register_updates(
    operator: OperatorSpec,
    template: OperatorTemplate,
) -> dict[str, int]:
    """Placeholder for shape-driven control register computation.

    Return format:
    {
        "<instance>.<config_name>": value,
        ...
    }

    Register names should follow the naming used in register_map_with_groups1.csv.
    This function is intentionally left as a stable extension point for per-op logic.
    """

    updates: dict[str, int] = {}

    handler = OP_CONTROL_REGISTER_FN.get(operator.op_type)
    if handler is not None:
        updates.update(handler(operator, template))

    _append_remapping_register_updates(operator, updates)
    instance_mapping = _load_operator_instance_mapping(operator.op_type)
    return _apply_instance_mapping_to_updates(updates, instance_mapping)
