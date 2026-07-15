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
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from address_remapping.graph import load_graph_file
from address_remapping.hardware import HardwareSpec
from address_remapping.performance import _estimate_compute, load_runtime_config
from address_remapping.rmsnorm_bridge import normalize_graph_spec


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
    if denominator == 0:
        return None
    return numerator / denominator


def _shape_list(port_data: Mapping[str, object]) -> List[int]:
    shape = port_data.get("resolved_shape", port_data.get("shape"))
    if shape is None:
        raise ValueError(f"Port data is missing shape/resolved_shape: {port_data}")
    if isinstance(shape, Mapping):
        return [int(value) for value in shape.values()]
    return [int(value) for value in shape]


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


def _estimate_roofline_bytes(
    op_type: str,
    op_data: Mapping[str, object],
    dtype_bits_fn,
) -> Tuple[int, int, Dict[str, object]]:
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
        )
        total_bytes = bytes_read + bytes_written
        bandwidth_bound_cycles = (
            total_bytes / peak_memory_bandwidth_bytes_per_cycle if total_bytes else 0.0
        )
        roofline_cycles = max(compute_bound_cycles, bandwidth_bound_cycles)
        measured = measured_cycles.get(op_name)
        roofline_bound = "compute" if compute_bound_cycles >= bandwidth_bound_cycles else "bandwidth"
        roofline_ops_per_cycle = _safe_div(work_ops, roofline_cycles)
        measured_ops_per_cycle = _safe_div(work_ops, measured) if measured is not None else None
        roofline_bandwidth_bytes_per_cycle = _safe_div(float(total_bytes), roofline_cycles) if total_bytes else None
        measured_bandwidth_bytes_per_cycle = (
            _safe_div(float(total_bytes), measured) if measured is not None and total_bytes else None
        )
        arithmetic_intensity = _safe_div(work_ops, float(total_bytes)) if total_bytes else None
        peak_for_util = float(peak_compute_ops)

        rows.append(
            {
                "op_id": op_name,
                "op_type": op_type,
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
                "roofline_bandwidth_utilization": (
                    _safe_div(roofline_bandwidth_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                    if roofline_bandwidth_bytes_per_cycle is not None
                    else None
                ),
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
                "measured_bandwidth_utilization": (
                    _safe_div(measured_bandwidth_bytes_per_cycle, peak_memory_bandwidth_bytes_per_cycle)
                    if measured_bandwidth_bytes_per_cycle is not None
                    else None
                ),
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
            "compute_domain": row.get("compute_domain"),
            "work_ops": row["work_ops"],
            "total_bytes": row["total_bytes"],
            "roofline_cycles": row["roofline_cycles"],
            "measured_cycles": row.get("measured_cycles"),
            "roofline_compute_utilization_percent": row.get("roofline_compute_utilization_percent"),
            "measured_compute_utilization_percent": row.get("measured_compute_utilization_percent"),
            "roofline_bandwidth_utilization_percent": (
                float(row["roofline_bandwidth_utilization"]) * 100.0
                if row.get("roofline_bandwidth_utilization") is not None
                else None
            ),
            "measured_bandwidth_utilization_percent": (
                float(row["measured_bandwidth_utilization"]) * 100.0
                if row.get("measured_bandwidth_utilization") is not None
                else None
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    formatted_rows: List[Dict[str, object]] = []
    for row in rows:
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
                "measured_cycles": _format_float(row.get("measured_cycles")),
                "measured_ops_per_cycle": _format_float(row.get("measured_ops_per_cycle")),
                "measured_compute_utilization": _format_percent(row.get("measured_compute_utilization")),
                "measured_compute_utilization_percent": _format_float(row.get("measured_compute_utilization_percent")),
                "measured_bandwidth_bytes_per_cycle": _format_float(row.get("measured_bandwidth_bytes_per_cycle")),
                "measured_bandwidth_utilization": _format_percent(row.get("measured_bandwidth_utilization")),
                "measured_vs_roofline_ratio": _format_float(row.get("measured_vs_roofline_ratio")),
                "roofline_gap_percent": _format_float(row.get("roofline_gap_percent")),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(formatted_rows[0].keys()))
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


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _write_summary_tables(
    path: Path,
    layer_summary: Mapping[str, object],
    domain_summaries: Mapping[str, Mapping[str, object]],
    layer_window_domain_summary: Mapping[str, Mapping[str, object]],
) -> None:
    gemm = domain_summaries["gemm"]
    non_gemm = domain_summaries["non_gemm"]
    total_cycles = float(layer_summary["total_measured_cycles"])
    gemm_cycles = float(gemm["total_measured_cycles"])
    non_gemm_cycles = float(non_gemm["total_measured_cycles"])

    lines: List[str] = [
        "# Layer0 Roofline vs Measured Summary Tables",
        "",
        "## Summary Metrics",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Metric", "Scope", "Value"],
            [
                ["GEMM compute utilization", "GEMM-only cycles", _fmt_percent_value(gemm.get("measured_compute_utilization_percent"))],
                ["GEMM bandwidth utilization", "GEMM-only cycles", _fmt_percent_value(gemm.get("measured_bandwidth_utilization_percent"))],
                [
                    "GEMM compute utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(layer_window_domain_summary["gemm"].get("measured_compute_utilization_percent")),
                ],
                [
                    "GEMM bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(layer_window_domain_summary["gemm"].get("measured_bandwidth_utilization_percent")),
                ],
                [
                    "Whole-layer bandwidth utilization",
                    "Full-layer cycles",
                    _fmt_percent_value(layer_summary.get("measured_bandwidth_utilization_percent")),
                ],
                ["GEMM time share", "Measured cycles", _fmt_percent_value((gemm_cycles / total_cycles) * 100.0 if total_cycles else None)],
                [
                    "non-GEMM time share",
                    "Measured cycles",
                    _fmt_percent_value((non_gemm_cycles / total_cycles) * 100.0 if total_cycles else None),
                ],
            ],
        )
    )

    lines.extend(["", "## GEMM Operators", ""])
    lines.extend(
        _markdown_table(
            ["Kernel", "Type", "Work ops", "Measured cycles", "Compute util", "Bandwidth util"],
            [
                [
                    str(row["name"]),
                    str(row["op_type"]),
                    _fmt_number(row["work_ops"], digits=0),
                    _fmt_number(row.get("measured_cycles"), digits=0),
                    _fmt_percent_value(row.get("measured_compute_utilization_percent")),
                    _fmt_percent_value(row.get("measured_bandwidth_utilization_percent")),
                ]
                for row in gemm["operators"]
            ],
        )
    )

    lines.extend(["", "## non-GEMM Operators", ""])
    lines.extend(
        _markdown_table(
            ["Operator", "Type", "Total bytes", "Measured cycles", "Bandwidth util"],
            [
                [
                    str(row["name"]),
                    str(row["op_type"]),
                    _fmt_number(row["total_bytes"], digits=0),
                    _fmt_number(row.get("measured_cycles"), digits=0),
                    _fmt_percent_value(row.get("measured_bandwidth_utilization_percent")),
                ]
                for row in non_gemm["operators"]
            ],
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_output_prefix(graph_path: Path) -> Path:
    return Path("outputs") / "roofline_only" / graph_path.stem


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
        help="Output path prefix. Defaults to outputs/roofline_only/<graph_stem>.",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph)
    measured_path = Path(args.measured)
    output_prefix = Path(args.output_prefix) if args.output_prefix else _default_output_prefix(graph_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    hardware, _perf, _solver = load_runtime_config(args.config)
    normalized_graph = _load_roofline_graph(graph_path)
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
    )
    layer_summary = _build_layer_summary(rows, hardware)
    domain_summaries = _build_domain_summaries(rows, hardware)
    layer_window_domain_summary = _build_layer_window_domain_summary(rows, layer_summary, hardware)

    summary = {
        "graph_path": str(graph_path),
        "measured_path": str(measured_path),
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
