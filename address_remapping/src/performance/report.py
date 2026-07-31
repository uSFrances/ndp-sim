"""
Compare per-op roofline-only estimates against measured cycles.

This script intentionally avoids the full address-level performance pipeline,
so it does not require base_addr fields in external execplan graphs.

Recommended usage:

    cd H:\\dev\\projects\\ndp-sim\\address_remapping
    $env:PYTHONPATH='src'
    python scripts\\compare_layer0_roofline_vs_measured.py ^
      examples\\graphs\\layer0\\layer0_0630.json ^
      --config examples\\configs\\performance_config.json ^
      --measured golden\\layer0\\op_cycle_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from address_remapping.graph import load_graph_file
from address_remapping.hardware import HardwareSpec
from address_remapping.performance import _estimate_compute, load_runtime_config
from address_remapping.rmsnorm_bridge import normalize_graph_spec
from performance.model_config import DEFAULT_MODEL_CONFIG_PATH, DEFAULT_SEQUENCE_MULTIPLE, load_model_config
from performance.ttft import build_model_scaled_ttft_summary


def _shape_product(shape: Mapping[str, object] | Sequence[object]) -> int:
    product = 1
    values = shape.values() if isinstance(shape, Mapping) else shape
    for value in values:
        product *= int(value)
    return product


def _natural_op_key(op_id: str) -> Tuple[int, str]:
    if op_id.startswith("op") and op_id[2:].isdigit():
        return (int(op_id[2:]), op_id)
    return (10**9, op_id)


def _tensor_bytes(port_data: Mapping[str, object], dtype_bits_fn) -> int:
    layout = port_data.get("layout")
    dtype = layout.dtype if layout is not None else str(port_data["dtype"])
    resolved_shape = port_data.get("resolved_shape", port_data.get("shape"))
    if resolved_shape is None:
        raise ValueError(f"Port data is missing shape/resolved_shape: {port_data}")
    elements = _shape_product(resolved_shape)
    return elements * dtype_bits_fn(dtype) // 8


def _load_measured_cycles(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    measured: Dict[str, float] = {}
    for item in payload.get("records", []):
        op_id = str(item.get("op_id", "")).strip()
        if not op_id:
            continue
        if item.get("cycle") is not None:
            measured[op_id] = max(0.0, float(item["cycle"]))
            continue
        start_time = float(item.get("start_time_ns", 0))
        completed_time = float(item.get("completed_time_ns", 0))
        measured[op_id] = max(0.0, completed_time - start_time)
    for item in payload.get("operators", []):
        op_id = str(item.get("op_id", "")).strip()
        if not op_id:
            continue
        start_cycle = float(item.get("start_cycle", 0))
        completed_cycle = float(item.get("completed_cycle", 0))
        measured[op_id] = max(0.0, completed_cycle - start_cycle)
    return measured


def _load_inline_measured_cycles(normalized_graph: Mapping[str, object]) -> Dict[str, float]:
    measured: Dict[str, float] = {}
    for op_name, op_data in dict(normalized_graph["ops"]).items():
        raw = dict(op_data).get("hardware_measured")
        if raw is None:
            continue
        measured[str(op_name)] = float(raw)
    return measured


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _shape_list(port_data: Mapping[str, object]) -> List[int]:
    shape = port_data.get("resolved_shape", port_data.get("shape"))
    if shape is None:
        raise ValueError(f"Port data is missing shape/resolved_shape: {port_data}")
    if isinstance(shape, Mapping):
        return [int(value) for value in shape.values()]
    return [int(value) for value in shape]


def _format_port_shapes(op_data: Mapping[str, object], direction: str) -> str:
    ports = dict(op_data[direction])
    formatted: List[str] = []
    for port_name, port_data in ports.items():
        debug = dict(port_data.get("external_shape_debug", {}))
        # External dimensions retain fan-in axes that are intentionally absent
        # from the logical layout of remote-reduction input ports.
        shape = debug.get("external_dims") or _shape_list(port_data)
        if len(shape) == 3:
            shape = shape[1:]
        formatted.append(f"{port_name}=[{' x '.join(str(value) for value in shape)}]")
    return "; ".join(formatted)


def _format_remote_sum_geometry(geometry: Mapping[str, object]) -> str:
    fan_in = int(geometry["remote_fan_in"])
    return f"fan_in={fan_in}"


def _count_used_slices(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("0b"):
        return text.count("1")
    try:
        return int(text)
    except ValueError:
        return None


def _estimate_gemv_ring_compute(
    op_data: Mapping[str, object],
    hardware: HardwareSpec,
) -> Tuple[float, float, int, Dict[str, object]]:
    geometry = _gemv_ring_geometry(op_data)
    work_ops = float(
        hardware.compute.gemm_core.mac_ops
        * int(geometry["gemv_m_dim"])
        * int(geometry["gemv_global_k_dim"])
        * int(geometry["gemv_local_n_dim"])
    )
    peak = hardware.gemm_peak_ops_per_cycle
    return (
        work_ops,
        work_ops / max(1, peak),
        peak,
        {
            "work_scope": "per_slice_full_k_local_n",
            **geometry,
        },
    )


def _gemv_ring_geometry(op_data: Mapping[str, object]) -> Dict[str, int]:
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    output_port = next(iter(outputs))
    a_shape = _shape_list(inputs["A"])
    b_shape = _shape_list(inputs["B"])
    output_shape = _shape_list(outputs[output_port])
    used_slices = _count_used_slices(op_data.get("used_slices")) or 1

    local_k_dim = int(a_shape[-1])
    if len(b_shape) >= 2:
        local_k_dim = min(local_k_dim, int(b_shape[-2]))
    global_k_dim = local_k_dim * used_slices
    global_n_dim = int(b_shape[-1]) if b_shape else 1
    local_n_dim = int(output_shape[-1]) if output_shape else max(1, global_n_dim // used_slices)
    m_dim = _shape_product(output_shape[:-1]) if len(output_shape) > 1 else 1
    return {
        "used_slices": int(used_slices),
        "gemv_m_dim": int(m_dim),
        "gemv_global_k_dim": int(global_k_dim),
        "gemv_global_n_dim": int(global_n_dim),
        "gemv_local_k_dim": int(local_k_dim),
        "gemv_local_n_dim": int(local_n_dim),
    }


def _gemv_ring_tensor_bytes(
    op_data: Mapping[str, object],
    dtype_bits_fn,
) -> Tuple[int, int, Dict[str, object]]:
    geometry = _gemv_ring_geometry(op_data)
    m_dim = int(geometry["gemv_m_dim"])
    global_k_dim = int(geometry["gemv_global_k_dim"])
    local_n_dim = int(geometry["gemv_local_n_dim"])
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    a = dict(inputs["A"])
    a_dtype = str(a.get("dtype", "fp16"))
    bytes_read = m_dim * global_k_dim * dtype_bits_fn(a_dtype) // 8
    for port_name in ("B", "B'"):
        if port_name not in inputs:
            continue
        b = dict(inputs[port_name])
        b_dtype = str(b.get("dtype", "fp16"))
        bytes_read += global_k_dim * local_n_dim * dtype_bits_fn(b_dtype) // 8
    output_port = next(iter(outputs))
    output = dict(outputs[output_port])
    output_dtype = str(output.get("dtype", "fp16"))
    bytes_written = m_dim * local_n_dim * dtype_bits_fn(output_dtype) // 8
    return bytes_read, bytes_written, {"byte_scope": "per_slice_full_k_local_n"}


REMOTE_SUM_TRANSPORT_AXI_PULL = "axi_pull"
REMOTE_SUM_TRANSPORT_RING2RING_N2N = "ring2ring_n2n"


def _remote_axi_bandwidth_bytes_per_cycle() -> float:
    return (128.0 / 8.0) * 0.5 / 28.0


def _remote_ring2ring_bandwidth_bytes_per_cycle() -> float:
    return 256.0 / 8.0


def _global_axi_total_bandwidth_bytes_per_cycle() -> float:
    return (128.0 / 8.0) * 0.5


def _remote_sum_geometry(op_data: Mapping[str, object]) -> Dict[str, int]:
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    input_port = next(iter(inputs))
    output_port = next(iter(outputs))
    input_debug = dict(inputs[input_port].get("external_shape_debug", {}))
    external_dims = [int(value) for value in input_debug.get("external_dims", [])]
    fan_in = int(external_dims[1]) if len(external_dims) > 1 else int(op_data.get("used_slices", 1))
    output_shape = _shape_list(outputs[output_port])
    output_elements = _shape_product(output_shape)
    active_slices = _count_used_slices(op_data.get("used_slices")) or fan_in
    return {
        "remote_fan_in": max(1, fan_in),
        "remote_active_slices": max(1, active_slices),
        "remote_group_count": max(1, (active_slices + fan_in - 1) // fan_in),
        "remote_output_elements": int(output_elements),
    }


def _remote_sum_tensor_bytes(
    op_data: Mapping[str, object],
    dtype_bits_fn,
    remote_sum_transport: str = REMOTE_SUM_TRANSPORT_AXI_PULL,
) -> Tuple[int, int, Dict[str, object]]:
    geometry = _remote_sum_geometry(op_data)
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    input_port = next(iter(inputs))
    output_port = next(iter(outputs))
    input_layout = inputs[input_port].get("layout")
    output_layout = outputs[output_port].get("layout")
    input_dtype = str(inputs[input_port].get("dtype", getattr(input_layout, "dtype", "fp32")))
    output_dtype = str(outputs[output_port].get("dtype", getattr(output_layout, "dtype", "fp32")))
    fan_in = int(geometry["remote_fan_in"])
    active_slices = int(geometry["remote_active_slices"])
    group_count = int(geometry["remote_group_count"])
    output_elements = int(geometry["remote_output_elements"])
    input_bytes_per_element = dtype_bits_fn(input_dtype) // 8
    local_write_bytes = output_elements * dtype_bits_fn(output_dtype) // 8
    if remote_sum_transport == REMOTE_SUM_TRANSPORT_RING2RING_N2N:
        local_read_bytes = output_elements * input_bytes_per_element
        ring_transfer_bytes = max(0, fan_in - 1) * output_elements * input_bytes_per_element
        return (
            local_read_bytes + ring_transfer_bytes,
            local_write_bytes,
            {
                "byte_scope": "remote_ring2ring_local_read_n2n_local_write",
                "local_read_bytes": local_read_bytes,
                "ring_transfer_bytes": ring_transfer_bytes,
                "local_write_bytes": local_write_bytes,
                "ring_bandwidth_bytes_per_cycle": _remote_ring2ring_bandwidth_bytes_per_cycle(),
                **geometry,
            },
        )
    if remote_sum_transport != REMOTE_SUM_TRANSPORT_AXI_PULL:
        raise ValueError(f"Unsupported remote_sum transport: {remote_sum_transport}")

    # Each active slice reads every partial in its fan-in group through the
    # shared global AXI interface, including its local partial.
    global_read_bytes = active_slices * fan_in * output_elements * input_bytes_per_element
    total_local_write_bytes = active_slices * local_write_bytes
    centralized_global_read_bytes = group_count * fan_in * output_elements * input_bytes_per_element
    centralized_global_return_bytes = group_count * max(0, fan_in - 1) * local_write_bytes
    return (
        global_read_bytes,
        total_local_write_bytes,
        {
            "byte_scope": "remote_pull_global_read_total_local_write",
            "global_read_bytes": global_read_bytes,
            "local_write_bytes": local_write_bytes,
            "total_local_write_bytes": total_local_write_bytes,
            "centralized_global_read_bytes": centralized_global_read_bytes,
            "centralized_global_return_bytes": centralized_global_return_bytes,
            "global_bandwidth_bytes_per_cycle": _global_axi_total_bandwidth_bytes_per_cycle(),
            **geometry,
        },
    )


def _estimate_remote_sum_compute(
    op_data: Mapping[str, object],
    hardware: HardwareSpec,
) -> Tuple[float, float, int, Dict[str, object]]:
    geometry = _remote_sum_geometry(op_data)
    fan_in = int(geometry["remote_fan_in"])
    output_elements = int(geometry["remote_output_elements"])
    work_ops = float(max(0, fan_in - 1) * output_elements)
    peak = hardware.general_peak_ops_per_cycle
    return (
        work_ops,
        work_ops / max(1, peak),
        peak,
        {
            "work_scope": "remote_reduction_fan_in_minus_one",
            **geometry,
        },
    )


def _percent_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value) * 100.0


def _estimate_roofline_bytes(
    op_type: str,
    op_data: Mapping[str, object],
    dtype_bits_fn,
    remote_sum_transport: str = REMOTE_SUM_TRANSPORT_AXI_PULL,
) -> Tuple[int, int, Dict[str, object]]:
    if "remote_sum" in op_type:
        return _remote_sum_tensor_bytes(op_data, dtype_bits_fn, remote_sum_transport)
    if op_type == "prefill_gemv_ring":
        return _gemv_ring_tensor_bytes(op_data, dtype_bits_fn)
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    bytes_read = sum(_tensor_bytes(port_data, dtype_bits_fn) for port_data in inputs.values())
    bytes_written = sum(_tensor_bytes(port_data, dtype_bits_fn) for port_data in outputs.values())
    return bytes_read, bytes_written, {}


def _estimate_roofline_compute(
    op_type: str,
    op_data: Mapping[str, object],
    hardware: HardwareSpec,
) -> Tuple[float, float, int, Dict[str, object]]:
    if "remote_sum" in op_type:
        return _estimate_remote_sum_compute(op_data, hardware)
    if op_type == "prefill_gemv_ring":
        return _estimate_gemv_ring_compute(op_data, hardware)

    if op_type == "prefill_silu_fp16MN_fp32MN":
        outputs = dict(op_data["outputs"])
        output_port = next(iter(outputs))
        output_shape = {str(k): int(v) for k, v in dict(outputs[output_port]["resolved_shape"]).items()}
        work_ops = float(_shape_product(output_shape))
        peak = 8
        return work_ops, work_ops / max(1, peak), peak, {}

    if op_type not in {
        "ring_gemm_fp16_fp16_fp16",
        "gemm_local_fp16_fp16_fp16",
        "gemm_local_qkt_fp16_fp16_fp32",
    }:
        work_ops, compute_cycles, peak = _estimate_compute(op_type, op_data, hardware)
        return work_ops, compute_cycles, peak, {}

    inputs = dict(op_data["inputs"])
    in_a_shape = {str(k): int(v) for k, v in dict(inputs["inA"]["resolved_shape"]).items()}
    in_b_shape = {str(k): int(v) for k, v in dict(inputs["inB"]["resolved_shape"]).items()}
    m_dim = int(in_a_shape["M"])
    kb_dim = int(in_b_shape["K"])
    nb_dim = int(in_b_shape["N"])
    work_ops = float(hardware.compute.gemm_core.mac_ops * m_dim * kb_dim * nb_dim)
    peak = hardware.gemm_peak_ops_per_cycle
    return work_ops, work_ops / max(1, peak), peak, {}


def _raw_external_graph_for_roofline(payload: Mapping[str, object]) -> Dict[str, object]:
    operators = payload.get("operators")
    if not isinstance(operators, list):
        raise ValueError("Graph payload must define an 'operators' list.")
    graph_used_slices = payload.get("used_slices")
    ops: Dict[str, object] = {}
    for operator in operators:
        if not isinstance(operator, Mapping):
            raise ValueError("Each operator entry must be a mapping.")
        op_id = str(operator.get("id", "")).strip()
        if not op_id:
            raise ValueError("Each operator must define an id.")
        output = operator.get("output")
        if not isinstance(output, Mapping):
            raise ValueError(f"Operator '{op_id}' must define an output mapping.")
        op_used_slices = operator.get("used_slices", graph_used_slices)
        ops[op_id] = {
            "op_type": str(operator.get("type", "")),
            "inputs": dict(operator.get("inputs", {})),
            "outputs": {"out": dict(output)},
            "hardware_measured": operator.get("hardware_measured"),
            "used_slices": _count_used_slices(op_used_slices),
        }
    return {"ops": ops}


def _load_roofline_graph(graph_path: Path) -> Dict[str, object]:
    payload = load_graph_file(str(graph_path))
    if (
        isinstance(payload, Mapping)
        and isinstance(payload.get("operators"), list)
        and any(
            isinstance(operator, Mapping) and str(operator.get("type", "")).strip() == "prefill_gemv_ring"
            for operator in payload["operators"]
        )
    ):
        return _raw_external_graph_for_roofline(payload)
    try:
        return normalize_graph_spec(payload, require_base_addrs=False)
    except Exception:
        if isinstance(payload, Mapping) and isinstance(payload.get("operators"), list):
            return _raw_external_graph_for_roofline(payload)
        raise


def _build_rows(
    normalized_graph: Mapping[str, object],
    measured_cycles: Mapping[str, float],
    hardware: HardwareSpec,
    peak_memory_bandwidth_bytes_per_cycle: float,
    dtype_bits_fn,
    gemm_peak_ops_per_cycle: int,
    remote_sum_transport: str = REMOTE_SUM_TRANSPORT_AXI_PULL,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for op_name in sorted(dict(normalized_graph["ops"]).keys(), key=_natural_op_key):
        op_data = normalized_graph["ops"][op_name]
        op_type = str(op_data["op_type"])
        work_ops, compute_bound_cycles, peak_compute_ops, compute_metadata = _estimate_roofline_compute(
            op_type,
            op_data,
            hardware,
        )
        bytes_read, bytes_written, byte_metadata = _estimate_roofline_bytes(
            op_type,
            op_data,
            dtype_bits_fn,
            remote_sum_transport,
        )
        total_bytes = bytes_read + bytes_written
        is_remote_sum = "remote_sum" in op_type
        if is_remote_sum:
            local_write_bytes = int(byte_metadata["local_write_bytes"])
            local_write_bound_cycles = local_write_bytes / peak_memory_bandwidth_bytes_per_cycle if local_write_bytes else 0.0
            if "global_read_bytes" in byte_metadata:
                global_read_bytes = int(byte_metadata["global_read_bytes"])
                global_bw = float(byte_metadata["global_bandwidth_bytes_per_cycle"])
                global_read_bound_cycles = global_read_bytes / global_bw if global_bw and global_read_bytes else 0.0
                local_read_bound_cycles = None
                ring_transfer_bound_cycles = None
                bandwidth_bound_cycles = max(global_read_bound_cycles, local_write_bound_cycles)
            else:
                global_read_bound_cycles = None
                local_read_bytes = int(byte_metadata["local_read_bytes"])
                ring_transfer_bytes = int(byte_metadata["ring_transfer_bytes"])
                ring_bw = float(byte_metadata["ring_bandwidth_bytes_per_cycle"])
                local_read_bound_cycles = (
                    local_read_bytes / peak_memory_bandwidth_bytes_per_cycle if local_read_bytes else 0.0
                )
                ring_transfer_bound_cycles = ring_transfer_bytes / ring_bw if ring_bw and ring_transfer_bytes else 0.0
                # The local partial is read before remote partials arrive, so
                # these two stages form one serial input path.
                ring_read_path_cycles = local_read_bound_cycles + ring_transfer_bound_cycles
                bandwidth_bound_cycles = max(
                    ring_read_path_cycles,
                    local_write_bound_cycles,
                )
            roofline_cycles = max(compute_bound_cycles, bandwidth_bound_cycles)
        else:
            global_read_bound_cycles = None
            local_read_bound_cycles = None
            ring_transfer_bound_cycles = None
            local_write_bound_cycles = None
            bandwidth_bound_cycles = (
                total_bytes / peak_memory_bandwidth_bytes_per_cycle if total_bytes else 0.0
            )
            roofline_cycles = max(compute_bound_cycles, bandwidth_bound_cycles)
        measured = measured_cycles.get(op_name)
        roofline_bound = "compute" if compute_bound_cycles >= bandwidth_bound_cycles else "bandwidth"
        roofline_ops_per_cycle = _safe_div(work_ops, roofline_cycles)
        measured_ops_per_cycle = _safe_div(work_ops, measured) if measured is not None else None
        if is_remote_sum and "global_read_bytes" in byte_metadata:
            global_read_bytes = float(byte_metadata["global_read_bytes"])
            local_write_bytes = float(byte_metadata["local_write_bytes"])
            remote_global_bw = float(byte_metadata["global_bandwidth_bytes_per_cycle"])
            roofline_global_bw_bytes_per_cycle = _safe_div(global_read_bytes, roofline_cycles) if global_read_bytes else None
            measured_global_bw_bytes_per_cycle = (
                _safe_div(global_read_bytes, measured) if measured is not None and global_read_bytes else None
            )
            roofline_local_bw_bytes_per_cycle = _safe_div(local_write_bytes, roofline_cycles) if local_write_bytes else None
            measured_local_bw_bytes_per_cycle = (
                _safe_div(local_write_bytes, measured) if measured is not None and local_write_bytes else None
            )
            roofline_bandwidth_bytes_per_cycle = roofline_global_bw_bytes_per_cycle
            measured_bandwidth_bytes_per_cycle = measured_global_bw_bytes_per_cycle
            roofline_global_bandwidth_utilization = (
                _safe_div(roofline_global_bw_bytes_per_cycle, remote_global_bw)
                if roofline_global_bw_bytes_per_cycle is not None
                else None
            )
            measured_global_bandwidth_utilization = (
                _safe_div(measured_global_bw_bytes_per_cycle, remote_global_bw)
                if measured_global_bw_bytes_per_cycle is not None
                else None
            )
            roofline_local_bandwidth_utilization = (
                _safe_div(roofline_local_bw_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                if roofline_local_bw_bytes_per_cycle is not None
                else None
            )
            measured_local_bandwidth_utilization = (
                _safe_div(measured_local_bw_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                if measured_local_bw_bytes_per_cycle is not None
                else None
            )
            roofline_bandwidth_utilization = max(
                value for value in [roofline_global_bandwidth_utilization, roofline_local_bandwidth_utilization]
                if value is not None
            )
            measured_bandwidth_utilization = max(
                value for value in [measured_global_bandwidth_utilization, measured_local_bandwidth_utilization]
                if value is not None
            )
            roofline_ring_bandwidth_utilization = None
            measured_ring_bandwidth_utilization = None
        elif is_remote_sum:
            local_read_bytes = float(byte_metadata["local_read_bytes"])
            ring_transfer_bytes = float(byte_metadata["ring_transfer_bytes"])
            local_write_bytes = float(byte_metadata["local_write_bytes"])
            ring_bw = float(byte_metadata["ring_bandwidth_bytes_per_cycle"])
            roofline_ring_bw_bytes_per_cycle = (
                _safe_div(ring_transfer_bytes, roofline_cycles) if ring_transfer_bytes else None
            )
            measured_ring_bw_bytes_per_cycle = (
                _safe_div(ring_transfer_bytes, measured) if measured is not None and ring_transfer_bytes else None
            )
            roofline_local_read_bw_bytes_per_cycle = (
                _safe_div(local_read_bytes, roofline_cycles) if local_read_bytes else None
            )
            measured_local_read_bw_bytes_per_cycle = (
                _safe_div(local_read_bytes, measured) if measured is not None and local_read_bytes else None
            )
            roofline_local_write_bw_bytes_per_cycle = (
                _safe_div(local_write_bytes, roofline_cycles) if local_write_bytes else None
            )
            measured_local_write_bw_bytes_per_cycle = (
                _safe_div(local_write_bytes, measured) if measured is not None and local_write_bytes else None
            )
            roofline_bandwidth_bytes_per_cycle = roofline_ring_bw_bytes_per_cycle
            measured_bandwidth_bytes_per_cycle = measured_ring_bw_bytes_per_cycle
            roofline_global_bandwidth_utilization = None
            measured_global_bandwidth_utilization = None
            roofline_ring_bandwidth_utilization = (
                _safe_div(roofline_ring_bw_bytes_per_cycle, ring_bw)
                if roofline_ring_bw_bytes_per_cycle is not None
                else None
            )
            measured_ring_bandwidth_utilization = (
                _safe_div(measured_ring_bw_bytes_per_cycle, ring_bw)
                if measured_ring_bw_bytes_per_cycle is not None
                else None
            )
            roofline_local_read_bandwidth_utilization = (
                _safe_div(roofline_local_read_bw_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                if roofline_local_read_bw_bytes_per_cycle is not None
                else None
            )
            measured_local_read_bandwidth_utilization = (
                _safe_div(measured_local_read_bw_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                if measured_local_read_bw_bytes_per_cycle is not None
                else None
            )
            roofline_local_bandwidth_utilization = max(
                value
                for value in [roofline_local_read_bandwidth_utilization, _safe_div(roofline_local_write_bw_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle) if roofline_local_write_bw_bytes_per_cycle is not None else None]
                if value is not None
            )
            measured_local_bandwidth_utilization = max(
                value
                for value in [measured_local_read_bandwidth_utilization, _safe_div(measured_local_write_bw_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle) if measured_local_write_bw_bytes_per_cycle is not None else None]
                if value is not None
            )
            roofline_bandwidth_utilization = max(
                value for value in [roofline_ring_bandwidth_utilization, roofline_local_bandwidth_utilization]
                if value is not None
            )
            measured_bandwidth_utilization = max(
                value for value in [measured_ring_bandwidth_utilization, measured_local_bandwidth_utilization]
                if value is not None
            )
        else:
            roofline_bandwidth_bytes_per_cycle = _safe_div(float(total_bytes), roofline_cycles) if total_bytes else None
            measured_bandwidth_bytes_per_cycle = (
                _safe_div(float(total_bytes), measured) if measured is not None and total_bytes else None
            )
            roofline_global_bandwidth_utilization = None
            measured_global_bandwidth_utilization = None
            roofline_local_bandwidth_utilization = None
            measured_local_bandwidth_utilization = None
            roofline_ring_bandwidth_utilization = None
            measured_ring_bandwidth_utilization = None
            roofline_bandwidth_utilization = (
                _safe_div(roofline_bandwidth_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                if roofline_bandwidth_bytes_per_cycle is not None
                else None
            )
            measured_bandwidth_utilization = (
                _safe_div(measured_bandwidth_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                if measured_bandwidth_bytes_per_cycle is not None
                else None
            )
        arithmetic_intensity = _safe_div(work_ops, float(total_bytes)) if total_bytes else None
        peak_for_util = float(peak_compute_ops)

        rows.append(
            {
                "op_id": op_name,
                "op_type": op_type,
                "input_shapes": _format_port_shapes(op_data, "inputs"),
                "output_shapes": _format_port_shapes(op_data, "outputs"),
                "remote_sum_geometry": (
                    _format_remote_sum_geometry(byte_metadata) if is_remote_sum else "-"
                ),
                "peak_compute_ops_per_cycle": peak_for_util,
                "peak_memory_bandwidth_bytes_per_cycle": peak_memory_bandwidth_bytes_per_cycle,
                "work_ops": work_ops,
                "bytes_read": bytes_read,
                "bytes_written": bytes_written,
                "total_bytes": total_bytes,
                "arithmetic_intensity_ops_per_byte": arithmetic_intensity,
                "roofline_bound": roofline_bound,
                "roofline_compute_bound_cycles": compute_bound_cycles,
                "roofline_bandwidth_bound_cycles": bandwidth_bound_cycles,
                "roofline_global_read_bound_cycles": global_read_bound_cycles,
                "roofline_local_read_bound_cycles": local_read_bound_cycles,
                "roofline_ring_transfer_bound_cycles": ring_transfer_bound_cycles,
                "roofline_local_write_bound_cycles": local_write_bound_cycles,
                "roofline_cycles": roofline_cycles,
                "roofline_ops_per_cycle": roofline_ops_per_cycle,
                "roofline_compute_utilization": _safe_div(roofline_ops_per_cycle, peak_for_util)
                if roofline_ops_per_cycle is not None
                else None,
                "roofline_compute_utilization_percent": (
                    _safe_div(roofline_ops_per_cycle, peak_for_util) * 100.0
                    if roofline_ops_per_cycle is not None
                    else None
                ),
                "roofline_bandwidth_bytes_per_cycle": roofline_bandwidth_bytes_per_cycle,
                "roofline_bandwidth_utilization": roofline_bandwidth_utilization,
                "roofline_global_bandwidth_utilization": roofline_global_bandwidth_utilization,
                "measured_global_bandwidth_utilization": measured_global_bandwidth_utilization,
                "roofline_ring_bandwidth_utilization": roofline_ring_bandwidth_utilization,
                "measured_ring_bandwidth_utilization": measured_ring_bandwidth_utilization,
                "roofline_local_bandwidth_utilization": roofline_local_bandwidth_utilization,
                "measured_local_bandwidth_utilization": measured_local_bandwidth_utilization,
                "measured_cycles": measured,
                "measured_ops_per_cycle": measured_ops_per_cycle,
                "measured_compute_utilization": _safe_div(measured_ops_per_cycle, peak_for_util)
                if measured_ops_per_cycle is not None
                else None,
                "measured_compute_utilization_percent": (
                    _safe_div(measured_ops_per_cycle, peak_for_util) * 100.0
                    if measured_ops_per_cycle is not None
                    else None
                ),
                "measured_bandwidth_bytes_per_cycle": measured_bandwidth_bytes_per_cycle,
                "measured_bandwidth_utilization": measured_bandwidth_utilization,
                "measured_vs_roofline_ratio": _safe_div(measured, roofline_cycles)
                if measured is not None
                else None,
                "roofline_gap_percent": (
                    ((measured - roofline_cycles) / roofline_cycles) * 100.0
                    if measured is not None and roofline_cycles
                    else None
                ),
                "compute_domain": "gemm" if peak_compute_ops == gemm_peak_ops_per_cycle else "general",
                **compute_metadata,
                **byte_metadata,
            }
        )
    return rows


def _build_layer_summary(
    rows: Sequence[Mapping[str, object]],
    hardware: HardwareSpec,
    *,
    compute_peak_ops_per_cycle: Optional[float] = None,
) -> Dict[str, object]:
    total_work_ops = sum(float(row["work_ops"]) for row in rows)
    total_bytes = sum(int(row["total_bytes"]) for row in rows)
    total_roofline_cycles = sum(float(row["roofline_cycles"]) for row in rows)
    total_measured_cycles = sum(float(row["measured_cycles"]) for row in rows if row.get("measured_cycles") is not None)
    measured_count = sum(1 for row in rows if row.get("measured_cycles") is not None)

    roofline_ops_per_cycle = _safe_div(total_work_ops, total_roofline_cycles)
    measured_ops_per_cycle = _safe_div(total_work_ops, total_measured_cycles) if measured_count == len(rows) else None
    roofline_bw_bytes_per_cycle = _safe_div(float(total_bytes), total_roofline_cycles) if total_bytes else None
    measured_bw_bytes_per_cycle = (
        _safe_div(float(total_bytes), total_measured_cycles)
        if measured_count == len(rows) and total_bytes
        else None
    )
    compute_peak = (
        float(compute_peak_ops_per_cycle)
        if compute_peak_ops_per_cycle is not None
        else float(hardware.gemm_peak_ops_per_cycle)
    )

    return {
        "op_count": len(rows),
        "measured_op_count": measured_count,
        "total_work_ops": total_work_ops,
        "total_bytes": total_bytes,
        "total_roofline_cycles": total_roofline_cycles,
        "total_measured_cycles": total_measured_cycles if measured_count == len(rows) else None,
        "roofline_ops_per_cycle": roofline_ops_per_cycle,
        "measured_ops_per_cycle": measured_ops_per_cycle,
        "compute_peak_ops_per_cycle": compute_peak,
        "roofline_compute_utilization": (
            _safe_div(roofline_ops_per_cycle, compute_peak)
            if roofline_ops_per_cycle is not None
            else None
        ),
        "roofline_compute_utilization_percent": (
            _safe_div(roofline_ops_per_cycle, compute_peak) * 100.0
            if roofline_ops_per_cycle is not None
            else None
        ),
        "measured_compute_utilization": (
            _safe_div(measured_ops_per_cycle, compute_peak)
            if measured_ops_per_cycle is not None
            else None
        ),
        "measured_compute_utilization_percent": (
            _safe_div(measured_ops_per_cycle, compute_peak) * 100.0
            if measured_ops_per_cycle is not None
            else None
        ),
        "roofline_bandwidth_bytes_per_cycle": roofline_bw_bytes_per_cycle,
        "measured_bandwidth_bytes_per_cycle": measured_bw_bytes_per_cycle,
        "roofline_bandwidth_utilization": (
            _safe_div(roofline_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle)
            if roofline_bw_bytes_per_cycle is not None
            else None
        ),
        "roofline_bandwidth_utilization_percent": (
            _safe_div(roofline_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle) * 100.0
            if roofline_bw_bytes_per_cycle is not None
            else None
        ),
        "measured_bandwidth_utilization": (
            _safe_div(measured_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle)
            if measured_bw_bytes_per_cycle is not None
            else None
        ),
        "measured_bandwidth_utilization_percent": (
            _safe_div(measured_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle) * 100.0
            if measured_bw_bytes_per_cycle is not None
            else None
        ),
        "measured_vs_roofline_ratio": (
            _safe_div(total_measured_cycles, total_roofline_cycles)
            if measured_count == len(rows)
            else None
        ),
    }


def _build_domain_summaries(
    rows: Sequence[Mapping[str, object]],
    hardware: HardwareSpec,
) -> Dict[str, Dict[str, object]]:
    gemm_rows = [row for row in rows if row.get("compute_domain") == "gemm"]
    general_rows = [row for row in rows if row.get("compute_domain") != "gemm"]
    gemm_summary = _build_layer_summary(
        gemm_rows,
        hardware,
        compute_peak_ops_per_cycle=float(hardware.gemm_peak_ops_per_cycle),
    )
    non_gemm_summary = _build_layer_summary(
        general_rows,
        hardware,
        compute_peak_ops_per_cycle=float(hardware.general_peak_ops_per_cycle),
    )
    gemm_summary["operators"] = _summarize_domain_operators(gemm_rows)
    non_gemm_summary["operators"] = _summarize_domain_operators(general_rows)
    return {
        "gemm": gemm_summary,
        "non_gemm": non_gemm_summary,
    }


_COMPACT_SUMMARY_DROP_KEYS = {
    "roofline_ops_per_cycle",
    "measured_ops_per_cycle",
    "compute_peak_ops_per_cycle",
    "roofline_compute_utilization",
    "measured_compute_utilization",
    "roofline_bandwidth_bytes_per_cycle",
    "measured_bandwidth_bytes_per_cycle",
    "roofline_bandwidth_utilization",
    "measured_bandwidth_utilization",
    "measured_vs_roofline_ratio",
}


def _compact_summary(summary: Mapping[str, object]) -> Dict[str, object]:
    return {
        str(key): value
        for key, value in dict(summary).items()
        if str(key) not in _COMPACT_SUMMARY_DROP_KEYS and not str(key).startswith("layer_window_")
    }


def _compact_domain_summaries(summaries: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(domain): _compact_summary(summary) for domain, summary in dict(summaries).items()}


_LAYER0_OP_LABELS = {
    "op0": "q_norm_summac",
    "op1": "q_norm_remote_sum",
    "op2": "q_norm_mac_sfu",
    "op3": "q_norm_scale",
    "op4": "q_norm_apply",
    "op5": "q_gen",
    "op6": "q_bias_add",
    "op7": "q_rope_mul_a",
    "op8": "q_rope_mul_b",
    "op9": "q_rope_out",
    "op10": "k_norm_summac",
    "op11": "k_norm_remote_sum",
    "op12": "k_norm_mac_sfu",
    "op13": "k_norm_scale",
    "op14": "k_norm_apply",
    "op15": "k_gen",
    "op16": "k_bias_add",
    "op17": "k_rope_mul_a",
    "op18": "k_rope_mul_b",
    "op19": "k_rope_out",
    "op20": "v_gen",
    "op21": "v_bias_add",
    "op22": "local_gemm_qkt",
    "op23": "qkt_remote_sum",
    "op24": "qkt_score_add",
    "op25": "softmax_max",
    "op26": "softmax_sub",
    "op27": "softmax_sum_rec",
    "op28": "softmax_scale",
    "op29": "local_gemm_sv",
    "op30": "atten_out",
    "op31": "atten_residual_add",
    "op32": "ffn_norm_summac",
    "op33": "ffn_norm_remote_sum",
    "op34": "ffn_norm_mac_sfu",
    "op35": "ffn_norm_scale",
    "op36": "ffn_norm_apply",
    "op37": "ffn_gate",
    "op38": "ffn_up",
    "op39": "ffn_silu",
    "op40": "ffn_gate_mul",
    "op41": "ffn_down",
    "op42": "ffn_residual_add",
}


def _operator_label(row: Mapping[str, object]) -> str:
    op_id = str(row.get("op_id", ""))
    return _LAYER0_OP_LABELS.get(op_id, op_id)


def _summarize_domain_operators(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "name": _operator_label(row),
            "op_id": row["op_id"],
            "op_type": row["op_type"],
            "input_shapes": row.get("input_shapes"),
            "output_shapes": row.get("output_shapes"),
            "remote_sum_geometry": row.get("remote_sum_geometry"),
            "compute_domain": row.get("compute_domain"),
            "work_ops": row["work_ops"],
            "total_bytes": row["total_bytes"],
            "roofline_cycles": row["roofline_cycles"],
            "measured_cycles": row.get("measured_cycles"),
            "roofline_compute_utilization_percent": row.get("roofline_compute_utilization_percent"),
            "measured_compute_utilization_percent": row.get("measured_compute_utilization_percent"),
            "roofline_bandwidth_utilization_percent": _percent_or_none(row.get("roofline_bandwidth_utilization")),
            "measured_bandwidth_utilization_percent": _percent_or_none(row.get("measured_bandwidth_utilization")),
            "global_read_bytes": row.get("global_read_bytes"),
            "local_write_bytes": row.get("local_write_bytes"),
            "roofline_global_bandwidth_utilization_percent": _percent_or_none(
                row.get("roofline_global_bandwidth_utilization")
            ),
            "measured_global_bandwidth_utilization_percent": _percent_or_none(
                row.get("measured_global_bandwidth_utilization")
            ),
            "roofline_local_bandwidth_utilization_percent": _percent_or_none(
                row.get("roofline_local_bandwidth_utilization")
            ),
            "measured_local_bandwidth_utilization_percent": _percent_or_none(
                row.get("measured_local_bandwidth_utilization")
            ),
        }
        for row in rows
    ]


def _domain_window_utilization(
    rows: Sequence[Mapping[str, object]],
    hardware: HardwareSpec,
    *,
    total_roofline_cycles: float,
    total_measured_cycles: Optional[float],
    compute_peak_ops_per_cycle: float,
) -> Dict[str, object]:
    total_work_ops = sum(float(row["work_ops"]) for row in rows)
    total_bytes = sum(int(row["total_bytes"]) for row in rows)
    roofline_ops_per_cycle = _safe_div(total_work_ops, total_roofline_cycles)
    measured_ops_per_cycle = (
        _safe_div(total_work_ops, total_measured_cycles)
        if total_measured_cycles is not None
        else None
    )
    roofline_bw_bytes_per_cycle = _safe_div(float(total_bytes), total_roofline_cycles) if total_bytes else None
    measured_bw_bytes_per_cycle = (
        _safe_div(float(total_bytes), total_measured_cycles)
        if total_measured_cycles is not None and total_bytes
        else None
    )
    return {
        "total_work_ops": total_work_ops,
        "total_bytes": total_bytes,
        "compute_peak_ops_per_cycle": float(compute_peak_ops_per_cycle),
        "roofline_ops_per_cycle": roofline_ops_per_cycle,
        "measured_ops_per_cycle": measured_ops_per_cycle,
        "roofline_compute_utilization": (
            _safe_div(roofline_ops_per_cycle, compute_peak_ops_per_cycle)
            if roofline_ops_per_cycle is not None
            else None
        ),
        "roofline_compute_utilization_percent": (
            _safe_div(roofline_ops_per_cycle, compute_peak_ops_per_cycle) * 100.0
            if roofline_ops_per_cycle is not None
            else None
        ),
        "measured_compute_utilization": (
            _safe_div(measured_ops_per_cycle, compute_peak_ops_per_cycle)
            if measured_ops_per_cycle is not None
            else None
        ),
        "measured_compute_utilization_percent": (
            _safe_div(measured_ops_per_cycle, compute_peak_ops_per_cycle) * 100.0
            if measured_ops_per_cycle is not None
            else None
        ),
        "roofline_bandwidth_bytes_per_cycle": roofline_bw_bytes_per_cycle,
        "measured_bandwidth_bytes_per_cycle": measured_bw_bytes_per_cycle,
        "roofline_bandwidth_utilization": (
            _safe_div(roofline_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle)
            if roofline_bw_bytes_per_cycle is not None
            else None
        ),
        "roofline_bandwidth_utilization_percent": (
            _safe_div(roofline_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle) * 100.0
            if roofline_bw_bytes_per_cycle is not None
            else None
        ),
        "measured_bandwidth_utilization": (
            _safe_div(measured_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle)
            if measured_bw_bytes_per_cycle is not None
            else None
        ),
        "measured_bandwidth_utilization_percent": (
            _safe_div(measured_bw_bytes_per_cycle, hardware.peak_memory_bandwidth_bytes_per_cycle) * 100.0
            if measured_bw_bytes_per_cycle is not None
            else None
        ),
    }


def _build_layer_window_domain_summary(
    rows: Sequence[Mapping[str, object]],
    layer_summary: Mapping[str, object],
    hardware: HardwareSpec,
) -> Dict[str, Dict[str, object]]:
    total_roofline_cycles = float(layer_summary["total_roofline_cycles"])
    total_measured_cycles = layer_summary.get("total_measured_cycles")
    measured_cycles = float(total_measured_cycles) if total_measured_cycles is not None else None
    gemm_rows = [row for row in rows if row.get("compute_domain") == "gemm"]
    general_rows = [row for row in rows if row.get("compute_domain") != "gemm"]
    return {
        "gemm": _domain_window_utilization(
            gemm_rows,
            hardware,
            total_roofline_cycles=total_roofline_cycles,
            total_measured_cycles=measured_cycles,
            compute_peak_ops_per_cycle=float(hardware.gemm_peak_ops_per_cycle),
        ),
        "non_gemm": _domain_window_utilization(
            general_rows,
            hardware,
            total_roofline_cycles=total_roofline_cycles,
            total_measured_cycles=measured_cycles,
            compute_peak_ops_per_cycle=float(hardware.general_peak_ops_per_cycle),
        ),
    }


def _format_float(value: object) -> object:
    if value is None:
        return None
    return f"{float(value):.4f}"


def _format_percent(value: object) -> object:
    if value is None:
        return None
    return f"{float(value) * 100.0:.2f}"


def _measured_cycles_desc_key(row: Mapping[str, object]) -> Tuple[int, float, Tuple[int, str]]:
    measured = row.get("measured_cycles")
    if measured is None:
        return (1, 0.0, _natural_op_key(str(row.get("op_id", ""))))
    return (0, -float(measured), _natural_op_key(str(row.get("op_id", ""))))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    formatted_rows: List[Dict[str, object]] = []
    for row in sorted(rows, key=_measured_cycles_desc_key):
        formatted_rows.append(
            {
                **row,
                "arithmetic_intensity_ops_per_byte": _format_float(row.get("arithmetic_intensity_ops_per_byte")),
                "roofline_compute_bound_cycles": _format_float(row.get("roofline_compute_bound_cycles")),
                "roofline_bandwidth_bound_cycles": _format_float(row.get("roofline_bandwidth_bound_cycles")),
                "roofline_cycles": _format_float(row.get("roofline_cycles")),
                "roofline_ops_per_cycle": _format_float(row.get("roofline_ops_per_cycle")),
                "roofline_compute_utilization": _format_percent(row.get("roofline_compute_utilization")),
                "roofline_compute_utilization_percent": _format_float(row.get("roofline_compute_utilization_percent")),
                "roofline_bandwidth_bytes_per_cycle": _format_float(row.get("roofline_bandwidth_bytes_per_cycle")),
                "roofline_bandwidth_utilization": _format_percent(row.get("roofline_bandwidth_utilization")),
                "roofline_global_bandwidth_utilization": _format_percent(
                    row.get("roofline_global_bandwidth_utilization")
                ),
                "roofline_local_bandwidth_utilization": _format_percent(
                    row.get("roofline_local_bandwidth_utilization")
                ),
                "measured_cycles": _format_float(row.get("measured_cycles")),
                "measured_ops_per_cycle": _format_float(row.get("measured_ops_per_cycle")),
                "measured_compute_utilization": _format_percent(row.get("measured_compute_utilization")),
                "measured_compute_utilization_percent": _format_float(row.get("measured_compute_utilization_percent")),
                "measured_bandwidth_bytes_per_cycle": _format_float(row.get("measured_bandwidth_bytes_per_cycle")),
                "measured_bandwidth_utilization": _format_percent(row.get("measured_bandwidth_utilization")),
                "measured_global_bandwidth_utilization": _format_percent(
                    row.get("measured_global_bandwidth_utilization")
                ),
                "measured_local_bandwidth_utilization": _format_percent(
                    row.get("measured_local_bandwidth_utilization")
                ),
                "measured_vs_roofline_ratio": _format_float(row.get("measured_vs_roofline_ratio")),
                "roofline_gap_percent": _format_float(row.get("roofline_gap_percent")),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({str(key) for row in formatted_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_rows)


def _fmt_number(value: object, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def _fmt_percent_value(value: object, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}%"


def _fmt_ms(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f} ms"


def _ttft_ms(per_layer_cycles: object, num_hidden_layers: int, frequency_hz: float) -> Optional[float]:
    if per_layer_cycles is None or frequency_hz <= 0:
        return None
    return float(per_layer_cycles) * float(num_hidden_layers) / frequency_hz * 1000.0


def _load_model_context(
    graph_path: Path,
    *,
    model_config_path: Path = DEFAULT_MODEL_CONFIG_PATH,
    model_name: Optional[str] = None,
    sequence_length: Optional[int] = None,
    sequence_multiple: int = DEFAULT_SEQUENCE_MULTIPLE,
    frequency_hz: float = 800_000_000.0,
) -> Dict[str, object]:
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_params = dict(graph_payload.get("params", {})) if isinstance(graph_payload, Mapping) else {}
    model_params: Dict[str, object] = {}
    model_execution = None
    if model_config_path.exists():
        model_execution = load_model_config(
            model_config_path,
            sequence_length=sequence_length,
            sequence_multiple=sequence_multiple,
        )
        model_params = model_execution.summary()
    resolved_model_name = model_name or (model_execution.model_name if model_execution is not None else None)
    ttft_params = model_params or graph_params
    layers = int(ttft_params.get("num_hidden_layers", graph_params.get("num_hidden_layers", 28) or 28))
    if model_execution is None and layers <= 1:
        layers = 28
    return {
        "graph_params": graph_params,
        "model_params": model_params,
        "model_execution": model_execution,
        "model_name": resolved_model_name,
        "model_config_path": str(model_config_path) if model_params else None,
        "ttft_num_hidden_layers": layers,
        "ttft_frequency_hz": frequency_hz,
        "sequence_multiple": sequence_multiple,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _remote_sum_rows_by_op_id(rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    return {str(row["op_id"]): row for row in rows if "remote_sum" in str(row.get("op_type", ""))}


def _centralized_global_remote_sum_roofline_cycles(row: Mapping[str, object]) -> float:
    global_read_bytes = float(row.get("centralized_global_read_bytes", 0.0))
    global_return_bytes = float(row.get("centralized_global_return_bytes", 0.0))
    local_write_bytes = float(row.get("local_write_bytes", 0.0))
    global_bound_cycles = (global_read_bytes + global_return_bytes) / _global_axi_total_bandwidth_bytes_per_cycle()
    local_write_bound_cycles = (
        local_write_bytes / float(row["peak_memory_bandwidth_bytes_per_cycle"]) if local_write_bytes else 0.0
    )
    compute_bound_cycles = float(row.get("roofline_compute_bound_cycles", 0.0))
    return max(global_bound_cycles, local_write_bound_cycles, compute_bound_cycles)


def _append_remote_sum_transport_comparison(
    lines: List[str],
    *,
    axi_rows: Sequence[Mapping[str, object]],
    ring_rows: Sequence[Mapping[str, object]],
    axi_layer_summary: Mapping[str, object],
    ring_layer_summary: Mapping[str, object],
    axi_domain_summaries: Mapping[str, Mapping[str, object]],
    ring_domain_summaries: Mapping[str, Mapping[str, object]],
    axi_layer_window_domain_summary: Mapping[str, Mapping[str, object]],
    ring_layer_window_domain_summary: Mapping[str, Mapping[str, object]],
    model_context: Mapping[str, object],
    model_scaled_ttft_summary: Optional[Mapping[str, object]] = None,
) -> None:
    axi_remote_rows = _remote_sum_rows_by_op_id(axi_rows)
    ring_remote_rows = _remote_sum_rows_by_op_id(ring_rows)
    remote_op_ids = sorted(axi_remote_rows.keys(), key=_natural_op_key)
    axi_remote_cycles = sum(float(axi_remote_rows[op_id]["roofline_cycles"]) for op_id in remote_op_ids)
    ring_remote_cycles = sum(float(ring_remote_rows[op_id]["roofline_cycles"]) for op_id in remote_op_ids)
    centralized_global_remote_cycles = sum(
        _centralized_global_remote_sum_roofline_cycles(axi_remote_rows[op_id]) for op_id in remote_op_ids
    )
    axi_total_cycles = float(axi_layer_summary["total_roofline_cycles"])
    ring_total_cycles = float(ring_layer_summary["total_roofline_cycles"])
    axi_gemm_cycles = float(axi_domain_summaries["gemm"]["total_roofline_cycles"])
    ring_gemm_cycles = float(ring_domain_summaries["gemm"]["total_roofline_cycles"])
    axi_non_gemm_cycles = float(axi_domain_summaries["non_gemm"]["total_roofline_cycles"])
    ring_non_gemm_cycles = float(ring_domain_summaries["non_gemm"]["total_roofline_cycles"])
    centralized_global_total_cycles = axi_total_cycles - axi_remote_cycles + centralized_global_remote_cycles
    centralized_global_gemm_cycles = axi_gemm_cycles
    centralized_global_non_gemm_cycles = axi_non_gemm_cycles - axi_remote_cycles + centralized_global_remote_cycles
    measured_total_cycles = float(axi_layer_summary["total_measured_cycles"])
    measured_remote_cycles = sum(float(axi_remote_rows[op_id].get("measured_cycles", 0.0)) for op_id in remote_op_ids)
    measured_gemm_cycles = float(axi_domain_summaries["gemm"]["total_measured_cycles"])
    measured_non_gemm_cycles = float(axi_domain_summaries["non_gemm"]["total_measured_cycles"])
    projected_remote_cycles = sum(
        float(axi_remote_rows[op_id].get("measured_cycles", 0.0))
        * _safe_div(float(ring_remote_rows[op_id]["roofline_cycles"]), float(axi_remote_rows[op_id]["roofline_cycles"]))
        for op_id in remote_op_ids
    )
    centralized_projected_remote_cycles = sum(
        float(axi_remote_rows[op_id].get("measured_cycles", 0.0))
        * _safe_div(
            _centralized_global_remote_sum_roofline_cycles(axi_remote_rows[op_id]),
            float(axi_remote_rows[op_id]["roofline_cycles"]),
        )
        for op_id in remote_op_ids
    )
    projected_total_cycles = measured_total_cycles - measured_remote_cycles + projected_remote_cycles
    centralized_projected_total_cycles = (
        measured_total_cycles - measured_remote_cycles + centralized_projected_remote_cycles
    )
    projected_non_gemm_cycles = measured_non_gemm_cycles - measured_remote_cycles + projected_remote_cycles
    centralized_projected_non_gemm_cycles = (
        measured_non_gemm_cycles - measured_remote_cycles + centralized_projected_remote_cycles
    )
    gemm_work_ops = float(axi_domain_summaries["gemm"]["total_work_ops"])
    gemm_total_bytes = float(axi_domain_summaries["gemm"]["total_bytes"])
    non_gemm_total_bytes = float(axi_domain_summaries["non_gemm"]["total_bytes"])
    layer_total_bytes = float(axi_layer_summary["total_bytes"])
    gemm_peak_ops_per_cycle = float(axi_domain_summaries["gemm"]["compute_peak_ops_per_cycle"])
    memory_peak_bytes_per_cycle = float(axi_rows[0]["peak_memory_bandwidth_bytes_per_cycle"]) if axi_rows else 0.0
    projected_gemm_full_compute_util = _safe_div(
        _safe_div(gemm_work_ops, projected_total_cycles),
        gemm_peak_ops_per_cycle,
    )
    centralized_projected_gemm_full_compute_util = _safe_div(
        _safe_div(gemm_work_ops, centralized_projected_total_cycles),
        gemm_peak_ops_per_cycle,
    )
    projected_gemm_full_bandwidth_util = _safe_div(
        _safe_div(gemm_total_bytes, projected_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_projected_gemm_full_bandwidth_util = _safe_div(
        _safe_div(gemm_total_bytes, centralized_projected_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    projected_layer_bandwidth_util = _safe_div(
        _safe_div(layer_total_bytes, projected_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_projected_layer_bandwidth_util = _safe_div(
        _safe_div(layer_total_bytes, centralized_projected_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_projected_non_gemm_bandwidth_util = _safe_div(
        _safe_div(non_gemm_total_bytes, centralized_projected_non_gemm_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_projected_non_gemm_full_bandwidth_util = _safe_div(
        _safe_div(non_gemm_total_bytes, centralized_projected_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    projected_non_gemm_bandwidth_util = _safe_div(
        _safe_div(non_gemm_total_bytes, projected_non_gemm_cycles),
        memory_peak_bytes_per_cycle,
    )
    projected_non_gemm_full_bandwidth_util = _safe_div(
        _safe_div(non_gemm_total_bytes, projected_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_global_gemm_full_compute_util = _safe_div(
        _safe_div(gemm_work_ops, centralized_global_total_cycles),
        gemm_peak_ops_per_cycle,
    )
    centralized_global_gemm_full_bandwidth_util = _safe_div(
        _safe_div(gemm_total_bytes, centralized_global_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_global_layer_bandwidth_util = _safe_div(
        _safe_div(layer_total_bytes, centralized_global_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_global_non_gemm_bandwidth_util = _safe_div(
        _safe_div(non_gemm_total_bytes, centralized_global_non_gemm_cycles),
        memory_peak_bytes_per_cycle,
    )
    centralized_global_non_gemm_full_bandwidth_util = _safe_div(
        _safe_div(non_gemm_total_bytes, centralized_global_total_cycles),
        memory_peak_bytes_per_cycle,
    )
    ttft_layers = int(model_context["ttft_num_hidden_layers"])
    ttft_frequency_hz = float(model_context["ttft_frequency_hz"])
    model_ttft_scenarios = dict(model_scaled_ttft_summary.get("scenarios", {})) if model_scaled_ttft_summary else {}

    lines.extend(
        [
            "",
            "## Remote-Sum Transport Comparison",
            "",
            "Ring2Ring remote-sum model: each slice first reads its local partial result, then receives the other "
            "slice partial results through the slice-to-slice n2n datapath. The ring datapath bandwidth is modeled "
            "as 256 bit/cycle = 32 B/cycle per slice.",
            "",
            "AXI-pull model: all active slices read every `fan_in` partial in their group through AXI, including "
            "their local partial. The global read traffic is `active_slices * fan_in * output_elements * dtype_bytes` "
            "and is divided by the total 8 B/cycle global AXI bandwidth.",
            "",
            "For each remote-sum op: `local_read_bytes = output_elements * dtype_bytes`, "
            "`ring_transfer_bytes = (fan_in - 1) * output_elements * dtype_bytes`, "
            "`local_write_bytes = output_elements * output_dtype_bytes`, and "
            "`ring2ring_roofline_cycles = max(local_read_bytes / local_bw + "
            "ring_transfer_bytes / 32, local_write_bytes / local_bw, reduction_ops / general_peak)`.",
            "",
            "Centralized-global remote-sum projection: each group has one central slice read all `fan_in` partials, "
            "perform the reduction, then return the result to the other `(fan_in - 1)` slices. Read and return "
            "traffic are both accumulated across `active_slices / fan_in` groups and use the total 8 B/cycle global AXI bandwidth. "
            "This projection keeps all non-remote-sum measured cycles unchanged and scales each remote-sum measured "
            "cycle count by `centralized_global_roofline_cycles / axi_pull_roofline_cycles`.",
            "",
            "### Layer Roofline Summary",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "Metric",
                "Scope",
                "Measured",
                "Projected measured with centralized global remote-sum",
                "Projected measured with Ring2Ring remote-sum",
                "AXI pull roofline",
                "Centralized global roofline",
                "Ring2Ring n2n roofline",
            ],
            [
                [
                    "GEMM compute utilization",
                    "GEMM-only cycles",
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("measured_compute_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("measured_compute_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("measured_compute_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("roofline_compute_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("roofline_compute_utilization_percent")),
                    _fmt_percent_value(ring_domain_summaries["gemm"].get("roofline_compute_utilization_percent")),
                ],
                [
                    "GEMM bandwidth utilization",
                    "GEMM-only cycles",
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("measured_bandwidth_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("measured_bandwidth_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("measured_bandwidth_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(axi_domain_summaries["gemm"].get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(ring_domain_summaries["gemm"].get("roofline_bandwidth_utilization_percent")),
                ],
                [
                    "GEMM compute utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(
                        axi_layer_window_domain_summary["gemm"].get("measured_compute_utilization_percent")
                    ),
                    _fmt_percent_value(_percent_or_none(centralized_projected_gemm_full_compute_util)),
                    _fmt_percent_value(_percent_or_none(projected_gemm_full_compute_util)),
                    _fmt_percent_value(
                        axi_layer_window_domain_summary["gemm"].get("roofline_compute_utilization_percent")
                    ),
                    _fmt_percent_value(_percent_or_none(centralized_global_gemm_full_compute_util)),
                    _fmt_percent_value(
                        ring_layer_window_domain_summary["gemm"].get("roofline_compute_utilization_percent")
                    ),
                ],
                [
                    "GEMM bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(
                        axi_layer_window_domain_summary["gemm"].get("measured_bandwidth_utilization_percent")
                    ),
                    _fmt_percent_value(_percent_or_none(centralized_projected_gemm_full_bandwidth_util)),
                    _fmt_percent_value(_percent_or_none(projected_gemm_full_bandwidth_util)),
                    _fmt_percent_value(
                        axi_layer_window_domain_summary["gemm"].get("roofline_bandwidth_utilization_percent")
                    ),
                    _fmt_percent_value(_percent_or_none(centralized_global_gemm_full_bandwidth_util)),
                    _fmt_percent_value(
                        ring_layer_window_domain_summary["gemm"].get("roofline_bandwidth_utilization_percent")
                    ),
                ],
                [
                    "non-GEMM bandwidth utilization",
                    "non-GEMM-only cycles",
                    _fmt_percent_value(axi_domain_summaries["non_gemm"].get("measured_bandwidth_utilization_percent")),
                    _fmt_percent_value(_percent_or_none(centralized_projected_non_gemm_bandwidth_util)),
                    _fmt_percent_value(_percent_or_none(projected_non_gemm_bandwidth_util)),
                    _fmt_percent_value(axi_domain_summaries["non_gemm"].get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(_percent_or_none(centralized_global_non_gemm_bandwidth_util)),
                    _fmt_percent_value(ring_domain_summaries["non_gemm"].get("roofline_bandwidth_utilization_percent")),
                ],
                [
                    "non-GEMM bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(
                        axi_layer_window_domain_summary["non_gemm"].get("measured_bandwidth_utilization_percent")
                    ),
                    _fmt_percent_value(_percent_or_none(centralized_projected_non_gemm_full_bandwidth_util)),
                    _fmt_percent_value(_percent_or_none(projected_non_gemm_full_bandwidth_util)),
                    _fmt_percent_value(
                        axi_layer_window_domain_summary["non_gemm"].get("roofline_bandwidth_utilization_percent")
                    ),
                    _fmt_percent_value(_percent_or_none(centralized_global_non_gemm_full_bandwidth_util)),
                    _fmt_percent_value(
                        ring_layer_window_domain_summary["non_gemm"].get("roofline_bandwidth_utilization_percent")
                    ),
                ],
                [
                    "Whole-layer bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(axi_layer_summary.get("measured_bandwidth_utilization_percent")),
                    _fmt_percent_value(_percent_or_none(centralized_projected_layer_bandwidth_util)),
                    _fmt_percent_value(_percent_or_none(projected_layer_bandwidth_util)),
                    _fmt_percent_value(axi_layer_summary.get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(_percent_or_none(centralized_global_layer_bandwidth_util)),
                    _fmt_percent_value(ring_layer_summary.get("roofline_bandwidth_utilization_percent")),
                ],
                [
                    "GEMM time share",
                    "Cycles",
                    _fmt_percent_value(
                        _safe_div(measured_gemm_cycles, measured_total_cycles) * 100.0 if measured_total_cycles else None
                    ),
                    _fmt_percent_value(
                        _safe_div(measured_gemm_cycles, centralized_projected_total_cycles) * 100.0
                        if centralized_projected_total_cycles
                        else None
                    ),
                    _fmt_percent_value(
                        _safe_div(measured_gemm_cycles, projected_total_cycles) * 100.0
                        if projected_total_cycles
                        else None
                    ),
                    _fmt_percent_value(_safe_div(axi_gemm_cycles, axi_total_cycles) * 100.0 if axi_total_cycles else None),
                    _fmt_percent_value(
                        _safe_div(centralized_global_gemm_cycles, centralized_global_total_cycles) * 100.0
                        if centralized_global_total_cycles
                        else None
                    ),
                    _fmt_percent_value(
                        _safe_div(ring_gemm_cycles, ring_total_cycles) * 100.0 if ring_total_cycles else None
                    ),
                ],
                [
                    "non-GEMM time share",
                    "Cycles",
                    _fmt_percent_value(
                        _safe_div(measured_non_gemm_cycles, measured_total_cycles) * 100.0
                        if measured_total_cycles
                        else None
                    ),
                    _fmt_percent_value(
                        _safe_div(centralized_projected_non_gemm_cycles, centralized_projected_total_cycles) * 100.0
                        if centralized_projected_total_cycles
                        else None
                    ),
                    _fmt_percent_value(
                        _safe_div(projected_non_gemm_cycles, projected_total_cycles) * 100.0
                        if projected_total_cycles
                        else None
                    ),
                    _fmt_percent_value(
                        _safe_div(axi_non_gemm_cycles, axi_total_cycles) * 100.0 if axi_total_cycles else None
                    ),
                    _fmt_percent_value(
                        _safe_div(centralized_global_non_gemm_cycles, centralized_global_total_cycles) * 100.0
                        if centralized_global_total_cycles
                        else None
                    ),
                    _fmt_percent_value(
                        _safe_div(ring_non_gemm_cycles, ring_total_cycles) * 100.0 if ring_total_cycles else None
                    ),
                ],
                [
                    "Total cycles",
                    "Cycles",
                    _fmt_number(measured_total_cycles, digits=0),
                    _fmt_number(centralized_projected_total_cycles, digits=0),
                    _fmt_number(projected_total_cycles, digits=0),
                    _fmt_number(axi_total_cycles, digits=0),
                    _fmt_number(centralized_global_total_cycles, digits=0),
                    _fmt_number(ring_total_cycles, digits=0),
                ],
                [
                    "Template TTFT",
                    f"{ttft_layers} layers @ {ttft_frequency_hz / 1_000_000:.0f} MHz",
                    _fmt_ms(_ttft_ms(measured_total_cycles, ttft_layers, ttft_frequency_hz)),
                    _fmt_ms(_ttft_ms(centralized_projected_total_cycles, ttft_layers, ttft_frequency_hz)),
                    _fmt_ms(_ttft_ms(projected_total_cycles, ttft_layers, ttft_frequency_hz)),
                    _fmt_ms(_ttft_ms(axi_total_cycles, ttft_layers, ttft_frequency_hz)),
                    _fmt_ms(_ttft_ms(centralized_global_total_cycles, ttft_layers, ttft_frequency_hz)),
                    _fmt_ms(_ttft_ms(ring_total_cycles, ttft_layers, ttft_frequency_hz)),
                ],
                [
                    "Model-scaled TTFT",
                    f"{ttft_layers} layers @ {ttft_frequency_hz / 1_000_000:.0f} MHz",
                    _fmt_ms(dict(model_ttft_scenarios.get("measured", {})).get("ttft_ms")),
                    _fmt_ms(
                        dict(
                            model_ttft_scenarios.get(
                                "projected_measured_centralized_global_remote_sum",
                                {},
                            )
                        ).get("ttft_ms")
                    ),
                    _fmt_ms(
                        dict(model_ttft_scenarios.get("projected_measured_ring2ring_remote_sum", {})).get("ttft_ms")
                    ),
                    _fmt_ms(dict(model_ttft_scenarios.get("axi_pull_roofline", {})).get("ttft_ms")),
                    _fmt_ms(dict(model_ttft_scenarios.get("centralized_global_roofline", {})).get("ttft_ms")),
                    _fmt_ms(dict(model_ttft_scenarios.get("ring2ring_n2n_roofline", {})).get("ttft_ms")),
                ],
                [
                    "Remote-sum cycles",
                    "Cycles",
                    _fmt_number(measured_remote_cycles, digits=0),
                    _fmt_number(centralized_projected_remote_cycles, digits=0),
                    _fmt_number(projected_remote_cycles, digits=0),
                    _fmt_number(axi_remote_cycles, digits=0),
                    _fmt_number(centralized_global_remote_cycles, digits=0),
                    _fmt_number(ring_remote_cycles, digits=0),
                ],
                [
                    "Speedup vs measured",
                    "Measured total / scenario total",
                    "1.00x",
                    f"{_safe_div(measured_total_cycles, centralized_projected_total_cycles):.2f}x"
                    if centralized_projected_total_cycles
                    else "N/A",
                    f"{_safe_div(measured_total_cycles, projected_total_cycles):.2f}x"
                    if projected_total_cycles
                    else "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                ],
            ],
        )
    )

    lines.extend(["", "### Remote-Sum Operators", ""])
    lines.extend(
        _markdown_table(
            [
                "Operator",
                "Type",
                "Input shape",
                "Output shape",
                "Reduction geometry",
                "Fan-in",
                "AXI roofline cycles",
                "Centralized global roofline cycles",
                "Ring2Ring roofline cycles",
                "Speedup",
                "Measured cycles",
                "Projected measured cycles (centralized global)",
                "Projected measured cycles (Ring2Ring)",
                "AXI roofline layer share",
                "Ring2Ring roofline layer share",
                "Ring transfer bytes",
            ],
            [
                [
                    _operator_label(axi_remote_rows[op_id]),
                    str(axi_remote_rows[op_id]["op_type"]),
                    str(axi_remote_rows[op_id].get("input_shapes", "N/A")),
                    str(axi_remote_rows[op_id].get("output_shapes", "N/A")),
                    str(axi_remote_rows[op_id].get("remote_sum_geometry", "N/A")),
                    str(axi_remote_rows[op_id].get("remote_fan_in", "N/A")),
                    _fmt_number(axi_remote_rows[op_id].get("roofline_cycles"), digits=0),
                    _fmt_number(
                        _centralized_global_remote_sum_roofline_cycles(axi_remote_rows[op_id]),
                        digits=0,
                    ),
                    _fmt_number(ring_remote_rows[op_id].get("roofline_cycles"), digits=0),
                    f"{_safe_div(float(axi_remote_rows[op_id]['roofline_cycles']), float(ring_remote_rows[op_id]['roofline_cycles'])):.2f}x"
                    if float(ring_remote_rows[op_id]["roofline_cycles"])
                    else "N/A",
                    _fmt_number(axi_remote_rows[op_id].get("measured_cycles"), digits=0),
                    _fmt_number(
                        float(axi_remote_rows[op_id].get("measured_cycles", 0.0))
                        * _safe_div(
                            _centralized_global_remote_sum_roofline_cycles(axi_remote_rows[op_id]),
                            float(axi_remote_rows[op_id]["roofline_cycles"]),
                        ),
                        digits=0,
                    ),
                    _fmt_number(
                        float(axi_remote_rows[op_id].get("measured_cycles", 0.0))
                        * _safe_div(
                            float(ring_remote_rows[op_id]["roofline_cycles"]),
                            float(axi_remote_rows[op_id]["roofline_cycles"]),
                        ),
                        digits=0,
                    ),
                    _fmt_percent_value(
                        _safe_div(float(axi_remote_rows[op_id]["roofline_cycles"]), axi_total_cycles) * 100.0
                        if axi_total_cycles
                        else None
                    ),
                    _fmt_percent_value(
                        _safe_div(float(ring_remote_rows[op_id]["roofline_cycles"]), ring_total_cycles) * 100.0
                        if ring_total_cycles
                        else None
                    ),
                    _fmt_number(ring_remote_rows[op_id].get("ring_transfer_bytes"), digits=0),
                ]
                for op_id in sorted(remote_op_ids, key=lambda op_id: -float(axi_remote_rows[op_id]["roofline_cycles"]))
            ],
        )
    )


def _write_summary_tables(
    path: Path,
    layer_summary: Mapping[str, object],
    domain_summaries: Mapping[str, Mapping[str, object]],
    layer_window_domain_summary: Mapping[str, Mapping[str, object]],
    *,
    model_context: Mapping[str, object],
    rows: Optional[Sequence[Mapping[str, object]]] = None,
    remote_sum_comparison: Optional[Mapping[str, object]] = None,
    model_scaled_ttft_summary: Optional[Mapping[str, object]] = None,
) -> None:
    gemm = domain_summaries["gemm"]
    non_gemm = domain_summaries["non_gemm"]
    gemm_operators = sorted(gemm["operators"], key=_measured_cycles_desc_key)
    non_gemm_operators = sorted(non_gemm["operators"], key=_measured_cycles_desc_key)
    total_cycles = float(layer_summary["total_measured_cycles"])
    gemm_cycles = float(gemm["total_measured_cycles"])
    non_gemm_cycles = float(non_gemm["total_measured_cycles"])
    ttft_layers = int(model_context["ttft_num_hidden_layers"])
    ttft_frequency_hz = float(model_context["ttft_frequency_hz"])
    model_params = dict(model_context.get("model_params", {}))
    graph_params = dict(model_context.get("graph_params", {}))
    display_params = model_params or graph_params
    model_ttft_scenarios = dict(model_scaled_ttft_summary.get("scenarios", {})) if model_scaled_ttft_summary else {}

    lines: List[str] = [
        "# Layer0 Roofline vs Measured Summary Tables",
        "",
        "## Model Parameters",
        "",
    ]
    if model_context.get("model_config_path"):
        lines.append(f"- Model config: `{model_context['model_config_path']}`")
    if model_context.get("model_name"):
        lines.append(f"- Model name: `{model_context['model_name']}`")
    lines.extend(
        [
            f"- TTFT layers: `{ttft_layers}`",
            f"- TTFT frequency: `{ttft_frequency_hz / 1_000_000:.0f} MHz`",
            "- Layer graph: `layer0`; template TTFT multiplies layer0 cycles, model-scaled TTFT recomputes op sizes "
            "from the model config.",
        ]
    )
    for key in [
        "hidden_size",
        "execution_hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "padded_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_hidden_layers",
        "requested_sequence_length",
        "sequence_length",
        "slice_per_head",
        "used_slices",
        "kv_padding",
        "kv_padding_a",
        "kv_padding_b",
        "clusters",
        "attention_waves",
        "kv_heads_per_cluster",
    ]:
        if key in display_params:
            lines.append(f"- `{key}` = `{display_params[key]}`")
    if model_params and graph_params:
        lines.extend(
            [
                f"- Layer graph params: `hidden_size={graph_params.get('hidden_size')}`, "
                f"`intermediate_size={graph_params.get('intermediate_size')}`, "
                f"`sequence_length={graph_params.get('sequence_length')}`, "
                f"`num_hidden_layers={graph_params.get('num_hidden_layers')}`",
            ]
        )
    lines.extend(
        [
            "",
        "## Summary Metrics",
        "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Metric", "Scope", "Roofline", "Measured"],
            [
                [
                    "GEMM compute utilization",
                    "GEMM-only cycles",
                    _fmt_percent_value(gemm.get("roofline_compute_utilization_percent")),
                    _fmt_percent_value(gemm.get("measured_compute_utilization_percent")),
                ],
                [
                    "GEMM bandwidth utilization",
                    "GEMM-only cycles",
                    _fmt_percent_value(gemm.get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(gemm.get("measured_bandwidth_utilization_percent")),
                ],
                [
                    "GEMM compute utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(layer_window_domain_summary["gemm"].get("roofline_compute_utilization_percent")),
                    _fmt_percent_value(layer_window_domain_summary["gemm"].get("measured_compute_utilization_percent")),
                ],
                [
                    "GEMM bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(layer_window_domain_summary["gemm"].get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(layer_window_domain_summary["gemm"].get("measured_bandwidth_utilization_percent")),
                ],
                [
                    "non-GEMM bandwidth utilization",
                    "non-GEMM-only cycles",
                    _fmt_percent_value(non_gemm.get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(non_gemm.get("measured_bandwidth_utilization_percent")),
                ],
                [
                    "non-GEMM bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(
                        layer_window_domain_summary["non_gemm"].get("roofline_bandwidth_utilization_percent")
                    ),
                    _fmt_percent_value(
                        layer_window_domain_summary["non_gemm"].get("measured_bandwidth_utilization_percent")
                    ),
                ],
                [
                    "Whole-layer bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(layer_summary.get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(layer_summary.get("measured_bandwidth_utilization_percent")),
                ],
                [
                    "Template TTFT",
                    f"{ttft_layers} layers @ {ttft_frequency_hz / 1_000_000:.0f} MHz",
                    _fmt_ms(_ttft_ms(layer_summary.get("total_roofline_cycles"), ttft_layers, ttft_frequency_hz)),
                    _fmt_ms(_ttft_ms(layer_summary.get("total_measured_cycles"), ttft_layers, ttft_frequency_hz)),
                ],
                [
                    "Model-scaled TTFT",
                    f"{ttft_layers} layers @ {ttft_frequency_hz / 1_000_000:.0f} MHz",
                    _fmt_ms(dict(model_ttft_scenarios.get("axi_pull_roofline", {})).get("ttft_ms")),
                    _fmt_ms(dict(model_ttft_scenarios.get("measured", {})).get("ttft_ms")),
                ],
                [
                    "Model-scaled per-layer cycles",
                    "Target model operator sizes",
                    _fmt_number(dict(model_ttft_scenarios.get("axi_pull_roofline", {})).get("per_layer_cycles"), digits=0),
                    _fmt_number(dict(model_ttft_scenarios.get("measured", {})).get("per_layer_cycles"), digits=0),
                ],
                [
                    "GEMM time share",
                    "Cycles",
                    _fmt_percent_value(
                        (float(gemm["total_roofline_cycles"]) / float(layer_summary["total_roofline_cycles"])) * 100.0
                        if float(layer_summary["total_roofline_cycles"])
                        else None
                    ),
                    _fmt_percent_value((gemm_cycles / total_cycles) * 100.0 if total_cycles else None),
                ],
                [
                    "non-GEMM time share",
                    "Cycles",
                    _fmt_percent_value(
                        (float(non_gemm["total_roofline_cycles"]) / float(layer_summary["total_roofline_cycles"])) * 100.0
                        if float(layer_summary["total_roofline_cycles"])
                        else None
                    ),
                    _fmt_percent_value((non_gemm_cycles / total_cycles) * 100.0 if total_cycles else None),
                ],
            ],
        )
    )

    lines.extend(["", "## GEMM Operators", ""])
    lines.extend(
        _markdown_table(
            [
                "Op ID",
                "Kernel",
                "Type",
                "Work ops",
                "Roofline cycles",
                "Measured cycles",
                "Roofline layer time share",
                "Layer time share",
                "Roofline compute util",
                "Measured compute util",
                "Roofline bandwidth util",
                "Measured bandwidth util",
            ],
            [
                [
                    str(row["op_id"]),
                    str(row["name"]),
                    str(row["op_type"]),
                    _fmt_number(row["work_ops"], digits=0),
                    _fmt_number(row.get("roofline_cycles"), digits=0),
                    _fmt_number(row.get("measured_cycles"), digits=0),
                    _fmt_percent_value(
                        (float(row["roofline_cycles"]) / float(layer_summary["total_roofline_cycles"])) * 100.0
                        if float(layer_summary["total_roofline_cycles"]) and row.get("roofline_cycles") is not None
                        else None
                    ),
                    _fmt_percent_value(
                        (float(row["measured_cycles"]) / total_cycles) * 100.0
                        if total_cycles and row.get("measured_cycles") is not None
                        else None
                    ),
                    _fmt_percent_value(row.get("roofline_compute_utilization_percent")),
                    _fmt_percent_value(row.get("measured_compute_utilization_percent")),
                    _fmt_percent_value(row.get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(row.get("measured_bandwidth_utilization_percent")),
                ]
                for row in gemm_operators
            ],
        )
    )

    lines.extend(["", "## non-GEMM Operators", ""])
    lines.extend(
        _markdown_table(
            [
                "Op ID",
                "Operator",
                "Type",
                "Input shape",
                "Output shape",
                "Remote-sum geometry",
                "Total bytes",
                "Roofline cycles",
                "Measured cycles",
                "Roofline layer time share",
                "Layer time share",
                "Roofline bandwidth util",
                "Measured bandwidth util",
                "Roofline global BW util",
                "Measured global BW util",
            ],
            [
                [
                    str(row["op_id"]),
                    str(row["name"]),
                    str(row["op_type"]),
                    str(row.get("input_shapes", "N/A")),
                    str(row.get("output_shapes", "N/A")),
                    str(row.get("remote_sum_geometry", "-")),
                    _fmt_number(row["total_bytes"], digits=0),
                    _fmt_number(row.get("roofline_cycles"), digits=0),
                    _fmt_number(row.get("measured_cycles"), digits=0),
                    _fmt_percent_value(
                        (float(row["roofline_cycles"]) / float(layer_summary["total_roofline_cycles"])) * 100.0
                        if float(layer_summary["total_roofline_cycles"]) and row.get("roofline_cycles") is not None
                        else None
                    ),
                    _fmt_percent_value(
                        (float(row["measured_cycles"]) / total_cycles) * 100.0
                        if total_cycles and row.get("measured_cycles") is not None
                        else None
                    ),
                    _fmt_percent_value(row.get("roofline_bandwidth_utilization_percent")),
                    _fmt_percent_value(row.get("measured_bandwidth_utilization_percent")),
                    _fmt_percent_value(row.get("roofline_global_bandwidth_utilization_percent")),
                    _fmt_percent_value(row.get("measured_global_bandwidth_utilization_percent")),
                ]
                for row in non_gemm_operators
            ],
        )
    )
    if model_scaled_ttft_summary is not None:
        _append_model_scaled_operator_projection(lines, model_scaled_ttft_summary)
    if remote_sum_comparison is not None and rows is not None:
        _append_remote_sum_transport_comparison(
            lines,
            axi_rows=rows,
            ring_rows=remote_sum_comparison["rows"],
            axi_layer_summary=layer_summary,
            ring_layer_summary=remote_sum_comparison["layer_summary"],
            axi_domain_summaries=domain_summaries,
            ring_domain_summaries=remote_sum_comparison["domain_summaries"],
            axi_layer_window_domain_summary=remote_sum_comparison["axi_layer_window_domain_summary"],
            ring_layer_window_domain_summary=remote_sum_comparison["ring_layer_window_domain_summary"],
            model_context=model_context,
            model_scaled_ttft_summary=model_scaled_ttft_summary,
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_model_scaled_summary(
    lines: List[str],
    model_scaled_ttft_summary: Mapping[str, object],
) -> None:
    scenarios = dict(model_scaled_ttft_summary.get("scenarios", {}))
    operators = [dict(row) for row in model_scaled_ttft_summary.get("operators", [])]
    measured_per_layer = float(dict(scenarios.get("measured", {})).get("per_layer_cycles", 0.0) or 0.0)
    roofline_per_layer = float(dict(scenarios.get("axi_pull_roofline", {})).get("per_layer_cycles", 0.0) or 0.0)
    gemm_ops = [row for row in operators if row.get("compute_domain") == "gemm"]
    non_gemm_ops = [row for row in operators if row.get("compute_domain") != "gemm"]
    gemm_work = sum(float(row.get("work_ops", 0.0) or 0.0) for row in gemm_ops)
    gemm_bytes = sum(float(row.get("total_bytes", 0.0) or 0.0) for row in gemm_ops)
    non_gemm_bytes = sum(float(row.get("total_bytes", 0.0) or 0.0) for row in non_gemm_ops)
    total_bytes = gemm_bytes + non_gemm_bytes
    gemm_measured_cycles = sum(float(row.get("measured_cycles", 0.0) or 0.0) for row in gemm_ops)
    gemm_roofline_cycles = sum(float(row.get("axi_pull_roofline_cycles", 0.0) or 0.0) for row in gemm_ops)
    non_gemm_measured_cycles = sum(float(row.get("measured_cycles", 0.0) or 0.0) for row in non_gemm_ops)
    non_gemm_roofline_cycles = sum(float(row.get("axi_pull_roofline_cycles", 0.0) or 0.0) for row in non_gemm_ops)
    gemm_peak = _first_positive(gemm_ops, "peak_compute_ops_per_cycle")
    memory_peak = _first_positive(operators, "peak_memory_bandwidth_bytes_per_cycle")

    lines.extend(["", "## Summary Metrics", ""])
    lines.extend(
        _markdown_table(
            ["Metric", "Scope", "AXI pull roofline", "Projected measured"],
            [
                [
                    "GEMM compute utilization",
                    "GEMM-only cycles",
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(gemm_work, gemm_roofline_cycles), gemm_peak))),
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(gemm_work, gemm_measured_cycles), gemm_peak))),
                ],
                [
                    "GEMM compute utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(gemm_work, roofline_per_layer), gemm_peak))),
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(gemm_work, measured_per_layer), gemm_peak))),
                ],
                [
                    "non-GEMM bandwidth utilization",
                    "non-GEMM-only cycles",
                    _fmt_percent_value(
                        _percent_or_none(_safe_div(_safe_div(non_gemm_bytes, non_gemm_roofline_cycles), memory_peak))
                    ),
                    _fmt_percent_value(
                        _percent_or_none(_safe_div(_safe_div(non_gemm_bytes, non_gemm_measured_cycles), memory_peak))
                    ),
                ],
                [
                    "non-GEMM bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(non_gemm_bytes, roofline_per_layer), memory_peak))),
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(non_gemm_bytes, measured_per_layer), memory_peak))),
                ],
                [
                    "Whole-layer bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(total_bytes, roofline_per_layer), memory_peak))),
                    _fmt_percent_value(_percent_or_none(_safe_div(_safe_div(total_bytes, measured_per_layer), memory_peak))),
                ],
                [
                    "GEMM time share",
                    "Cycles",
                    _fmt_percent_value(_safe_div(gemm_roofline_cycles, roofline_per_layer) * 100.0 if roofline_per_layer else None),
                    _fmt_percent_value(_safe_div(gemm_measured_cycles, measured_per_layer) * 100.0 if measured_per_layer else None),
                ],
                [
                    "non-GEMM time share",
                    "Cycles",
                    _fmt_percent_value(
                        _safe_div(non_gemm_roofline_cycles, roofline_per_layer) * 100.0 if roofline_per_layer else None
                    ),
                    _fmt_percent_value(
                        _safe_div(non_gemm_measured_cycles, measured_per_layer) * 100.0 if measured_per_layer else None
                    ),
                ],
            ],
        )
    )

    scenario_labels = [
        ("measured", "Projected measured"),
        ("projected_measured_centralized_global_remote_sum", "Projected measured with centralized global remote-sum"),
        ("projected_measured_ring2ring_remote_sum", "Projected measured with Ring2Ring remote-sum"),
        ("axi_pull_roofline", "AXI pull roofline"),
        ("centralized_global_roofline", "Centralized global roofline"),
        ("ring2ring_n2n_roofline", "Ring2Ring n2n roofline"),
    ]
    lines.extend(["", "## Model-Scaled Summary", ""])
    lines.extend(
        _markdown_table(
            ["Scenario", "Per-layer cycles", "Total cycles", "TTFT"],
            [
                [
                    label,
                    _fmt_number(dict(scenarios.get(key, {})).get("per_layer_cycles"), digits=0),
                    _fmt_number(dict(scenarios.get(key, {})).get("total_cycles"), digits=0),
                    _fmt_ms(dict(scenarios.get(key, {})).get("ttft_ms")),
                ]
                for key, label in scenario_labels
            ],
        )
    )


def _first_positive(rows: Sequence[Mapping[str, object]], key: str) -> Optional[float]:
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        number = float(value)
        if number > 0:
            return number
    return None


def _append_model_scaled_operator_projection(
    lines: List[str],
    model_scaled_ttft_summary: Mapping[str, object],
) -> None:
    model = dict(model_scaled_ttft_summary.get("model", {}))
    scenarios = dict(model_scaled_ttft_summary.get("scenarios", {}))
    measured_scenario = dict(scenarios.get("measured", {}))
    measured_per_layer = float(measured_scenario.get("per_layer_cycles", 0.0) or 0.0)
    operators = sorted(
        [dict(row) for row in model_scaled_ttft_summary.get("operators", [])],
        key=_measured_cycles_desc_key,
    )
    if not operators:
        return

    lines.extend(
        [
            "",
            "## Model-Scaled Operator Projection",
            "",
            "This table recomputes each operator's work or bytes from the target model config, then projects cycles "
            "using the calibration layer's measured GEMM throughput or measured non-GEMM effective bandwidth.",
            "",
            f"- Target model: `{model.get('model_name', 'model')}`",
            f"- Target layers: `{model.get('num_hidden_layers', 'N/A')}`",
            f"- Target sequence length: `{model.get('sequence_length', 'N/A')}`",
            f"- Target execution hidden size: `{model.get('execution_hidden_size', 'N/A')}`",
            "",
        ]
    )
    gemm_rows = [row for row in operators if row.get("compute_domain") == "gemm"]
    non_gemm_rows = [row for row in operators if row.get("compute_domain") != "gemm"]

    lines.extend(["", "### Model-Scaled GEMM Operators", ""])
    lines.extend(
        _markdown_table(
            [
                "Op ID",
                "Operator",
                "Type",
                "Model work ops",
                "Projected measured cycles",
                "Layer share",
                "Projected compute util",
                "AXI roofline cycles",
            ],
            [
                [
                    str(row["op_id"]),
                    _operator_label(row),
                    str(row["op_type"]),
                    _fmt_number(row.get("work_ops"), digits=0),
                    _fmt_number(row.get("measured_cycles"), digits=0),
                    _fmt_percent_value(
                        _safe_div(float(row.get("measured_cycles", 0.0)), measured_per_layer) * 100.0
                        if measured_per_layer
                        else None
                    ),
                    _fmt_percent_value(
                        _percent_or_none(
                            _safe_div(
                                _safe_div(float(row.get("work_ops", 0.0)), float(row.get("measured_cycles", 0.0))),
                                float(row.get("peak_compute_ops_per_cycle", 0.0) or 0.0),
                            )
                        )
                    ),
                    _fmt_number(row.get("axi_pull_roofline_cycles"), digits=0),
                ]
                for row in gemm_rows
            ],
        )
    )

    lines.extend(["", "### Model-Scaled non-GEMM Operators", ""])
    lines.extend(
        _markdown_table(
            [
                "Op ID",
                "Operator",
                "Type",
                "Input shape",
                "Output shape",
                "Remote-sum geometry",
                "Model bytes",
                "Projected measured cycles",
                "Layer share",
                "Projected BW util",
                "AXI roofline cycles",
                "Centralized roofline cycles",
                "Ring2Ring roofline cycles",
            ],
            [
                [
                    str(row["op_id"]),
                    _operator_label(row),
                    str(row["op_type"]),
                    str(row.get("input_shapes", "N/A")),
                    str(row.get("output_shapes", "N/A")),
                    str(row.get("remote_sum_geometry", "-")),
                    _fmt_number(row.get("total_bytes"), digits=0),
                    _fmt_number(row.get("measured_cycles"), digits=0),
                    _fmt_percent_value(
                        _safe_div(float(row.get("measured_cycles", 0.0)), measured_per_layer) * 100.0
                        if measured_per_layer
                        else None
                    ),
                    _fmt_percent_value(
                        _percent_or_none(
                            _safe_div(
                                _safe_div(float(row.get("total_bytes", 0.0)), float(row.get("measured_cycles", 0.0))),
                                float(row.get("peak_memory_bandwidth_bytes_per_cycle", 0.0) or 0.0),
                            )
                        )
                    ),
                    _fmt_number(row.get("axi_pull_roofline_cycles"), digits=0),
                    _fmt_number(row.get("centralized_global_roofline_cycles"), digits=0),
                    _fmt_number(row.get("ring2ring_n2n_roofline_cycles"), digits=0),
                ]
                for row in non_gemm_rows
            ],
        )
    )


def _safe_filename_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "model"


def _default_output_prefix(graph_path: Path, model_name: Optional[str] = None) -> Path:
    model_dir = _safe_filename_part(model_name) if model_name else "model"
    return Path("outputs") / "roofline_only" / model_dir / graph_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-op roofline-only estimates against measured cycles.")
    parser.add_argument("graph", help="External graph JSON path, e.g. examples/graphs/layer0/layer0_0630.json")
    parser.add_argument(
        "--measured",
        default="golden/layer0/op_cycle_summary.json",
        help="Measured per-op cycle summary JSON.",
    )
    parser.add_argument(
        "--config",
        default="examples/configs/performance_config.json",
        help="Hardware/performance config JSON.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output path prefix. Defaults to outputs/roofline_only/<model_name>/<graph_stem>.",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG_PATH),
        help=f"Model config JSON used for model-scaled TTFT. Default: {DEFAULT_MODEL_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--model-name",
        help="Model name used in report metadata and default output filenames. Defaults to model_name in --model-config.",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        default=800.0,
        help="Frequency used for TTFT conversion. Default: 800.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Override sequence_length from --model-config before model-scaled TTFT estimation.",
    )
    parser.add_argument(
        "--sequence-multiple",
        type=int,
        default=DEFAULT_SEQUENCE_MULTIPLE,
        help=f"Round sequence_length up to this multiple. Default: {DEFAULT_SEQUENCE_MULTIPLE}.",
    )
    parser.add_argument(
        "--append-remote-sum-comparison",
        dest="append_remote_sum_comparison",
        action="store_true",
        default=True,
        help="Append AXI-pull vs ring2ring-n2n remote-sum roofline comparison to the summary markdown.",
    )
    parser.add_argument(
        "--no-append-remote-sum-comparison",
        dest="append_remote_sum_comparison",
        action="store_false",
        help="Do not append the remote-sum transport comparison section.",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph)
    measured_path = Path(args.measured)

    hardware, _perf, _solver = load_runtime_config(args.config)
    normalized_graph = _load_roofline_graph(graph_path)
    frequency_hz = float(args.frequency_mhz) * 1_000_000.0
    model_context = _load_model_context(
        graph_path,
        model_config_path=Path(args.model_config),
        model_name=args.model_name,
        sequence_length=args.sequence_length,
        sequence_multiple=args.sequence_multiple,
        frequency_hz=frequency_hz,
    )
    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else _default_output_prefix(graph_path, model_context.get("model_name"))
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    measured_cycles = {
        **_load_measured_cycles(measured_path),
        **_load_inline_measured_cycles(normalized_graph),
    }
    rows = _build_rows(
        normalized_graph=normalized_graph,
        measured_cycles=measured_cycles,
        hardware=hardware,
        peak_memory_bandwidth_bytes_per_cycle=hardware.peak_memory_bandwidth_bytes_per_cycle,
        dtype_bits_fn=hardware.dtype_bits,
        gemm_peak_ops_per_cycle=hardware.gemm_peak_ops_per_cycle,
        remote_sum_transport=REMOTE_SUM_TRANSPORT_AXI_PULL,
    )
    layer_summary = _build_layer_summary(rows, hardware)
    domain_summaries = _build_domain_summaries(rows, hardware)
    layer_window_domain_summary = _build_layer_window_domain_summary(rows, layer_summary, hardware)
    remote_sum_comparison = None
    if args.append_remote_sum_comparison:
        ring2ring_rows = _build_rows(
            normalized_graph=normalized_graph,
            measured_cycles=measured_cycles,
            hardware=hardware,
            peak_memory_bandwidth_bytes_per_cycle=hardware.peak_memory_bandwidth_bytes_per_cycle,
            dtype_bits_fn=hardware.dtype_bits,
            gemm_peak_ops_per_cycle=hardware.gemm_peak_ops_per_cycle,
            remote_sum_transport=REMOTE_SUM_TRANSPORT_RING2RING_N2N,
        )
        ring2ring_layer_summary = _build_layer_summary(ring2ring_rows, hardware)
        ring2ring_domain_summaries = _build_domain_summaries(ring2ring_rows, hardware)
        remote_sum_comparison = {
            "rows": ring2ring_rows,
            "layer_summary": ring2ring_layer_summary,
            "domain_summaries": ring2ring_domain_summaries,
            "axi_layer_window_domain_summary": layer_window_domain_summary,
            "ring_layer_window_domain_summary": _build_layer_window_domain_summary(
                ring2ring_rows,
                ring2ring_layer_summary,
                hardware,
            ),
        }
    model_scaled_ttft_summary = None
    model_execution = model_context.get("model_execution")
    if model_execution is not None:
        model_scaled_ttft_summary = build_model_scaled_ttft_summary(
            graph_path=graph_path,
            model=model_execution,
            baseline_rows=rows,
            ring_rows=remote_sum_comparison["rows"] if remote_sum_comparison is not None else None,
            frequency_hz=frequency_hz,
        )

    summary = {
        "graph_path": str(graph_path),
        "measured_path": str(measured_path),
        "model_name": model_context.get("model_name"),
        "op_count": len(rows),
        "hardware": {
            "peak_memory_bandwidth_bytes_per_cycle": hardware.peak_memory_bandwidth_bytes_per_cycle,
            "gemm_peak_ops_per_cycle": hardware.gemm_peak_ops_per_cycle,
            "general_peak_ops_per_cycle": hardware.general_peak_ops_per_cycle,
        },
        "layer_summary": layer_summary,
        "layer_gemm_summary": _compact_summary(domain_summaries["gemm"]),
        "layer_non_gemm_summary": _compact_summary(domain_summaries["non_gemm"]),
        "layer_window_domain_summary": _compact_domain_summaries(layer_window_domain_summary),
        "model_scaled_ttft_summary": model_scaled_ttft_summary,
        "operators": rows,
    }

    json_path = output_prefix.with_name(output_prefix.name + "_roofline_vs_measured.json")
    csv_path = output_prefix.with_name(output_prefix.name + "_roofline_vs_measured.csv")
    tables_path = output_prefix.with_name(output_prefix.name + "_summary_tables.md")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_summary_tables(
        tables_path,
        layer_summary=layer_summary,
        domain_summaries=domain_summaries,
        layer_window_domain_summary=layer_window_domain_summary,
        model_context=model_context,
        rows=rows,
        remote_sum_comparison=remote_sum_comparison,
        model_scaled_ttft_summary=model_scaled_ttft_summary,
    )

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {tables_path}")
    print(f"operators={len(rows)}")
    print(
        "layer roofline/measured compute util(%)="
        f"{_format_float(layer_summary.get('roofline_compute_utilization_percent'))}/"
        f"{_format_float(layer_summary.get('measured_compute_utilization_percent'))}"
    )
    print(
        "layer roofline/measured bandwidth util(%)="
        f"{_format_float(layer_summary.get('roofline_bandwidth_utilization_percent'))}/"
        f"{_format_float(layer_summary.get('measured_bandwidth_utilization_percent'))}"
    )
    print(
        "gemm roofline/measured compute util(%)="
        f"{_format_float(domain_summaries['gemm'].get('roofline_compute_utilization_percent'))}/"
        f"{_format_float(domain_summaries['gemm'].get('measured_compute_utilization_percent'))}"
    )
    print(
        "gemm layer-window roofline/measured compute util(%)="
        f"{_format_float(layer_window_domain_summary['gemm'].get('roofline_compute_utilization_percent'))}/"
        f"{_format_float(layer_window_domain_summary['gemm'].get('measured_compute_utilization_percent'))}"
    )
    print(
        "gemm layer-window roofline/measured bandwidth util(%)="
        f"{_format_float(layer_window_domain_summary['gemm'].get('roofline_bandwidth_utilization_percent'))}/"
        f"{_format_float(layer_window_domain_summary['gemm'].get('measured_bandwidth_utilization_percent'))}"
    )
    print(
        "non-gemm roofline/measured compute util(%)="
        f"{_format_float(domain_summaries['non_gemm'].get('roofline_compute_utilization_percent'))}/"
        f"{_format_float(domain_summaries['non_gemm'].get('measured_compute_utilization_percent'))}"
    )
    print(
        "non-gemm layer-window roofline/measured compute util(%)="
        f"{_format_float(layer_window_domain_summary['non_gemm'].get('roofline_compute_utilization_percent'))}/"
        f"{_format_float(layer_window_domain_summary['non_gemm'].get('measured_compute_utilization_percent'))}"
    )
    print(
        "non-gemm layer-window roofline/measured bandwidth util(%)="
        f"{_format_float(layer_window_domain_summary['non_gemm'].get('roofline_bandwidth_utilization_percent'))}/"
        f"{_format_float(layer_window_domain_summary['non_gemm'].get('measured_bandwidth_utilization_percent'))}"
    )


if __name__ == "__main__":
    main()
