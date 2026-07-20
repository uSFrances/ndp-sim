from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .model_config import ModelExecutionConfig


GEMM_TYPES = {
    "prefill_gemm_ring_4slice",
    "prefill_gemm_local_qkt",
    "prefill_gemm_local",
    "ring_gemm_fp16_fp16_fp16",
    "gemm_local_fp16_fp16_fp16",
    "gemm_local_qkt_fp16_fp16_fp32",
}

ATTENTION_WAVE_MULTIPLIER_IDS = {
    "op6",
    "op7",
    "op8",
    "op9",
    "op23",
    "op24",
    "op25",
    "op26",
    "op27",
    "op28",
}

REMOTE_SUM_TRANSPORT_AXI_PULL = "axi_pull"
REMOTE_SUM_TRANSPORT_CENTRALIZED_GLOBAL = "centralized_global"
REMOTE_SUM_TRANSPORT_RING2RING_N2N = "ring2ring_n2n"


def build_model_scaled_ttft_summary(
    *,
    graph_path: Path,
    model: ModelExecutionConfig,
    baseline_rows: Sequence[Mapping[str, object]],
    ring_rows: Optional[Sequence[Mapping[str, object]]],
    frequency_hz: float,
) -> Dict[str, object]:
    operators = _load_graph_operators(graph_path)
    values = model.values()
    row_by_id = {str(row["op_id"]): row for row in baseline_rows}
    ring_by_id = {str(row["op_id"]): row for row in ring_rows or []}

    scenarios = {
        "measured": {"per_op": {}, "per_layer_cycles": 0.0},
        "projected_measured_centralized_global_remote_sum": {"per_op": {}, "per_layer_cycles": 0.0},
        "projected_measured_ring2ring_remote_sum": {"per_op": {}, "per_layer_cycles": 0.0},
        "axi_pull_roofline": {"per_op": {}, "per_layer_cycles": 0.0},
        "centralized_global_roofline": {"per_op": {}, "per_layer_cycles": 0.0},
        "ring2ring_n2n_roofline": {"per_op": {}, "per_layer_cycles": 0.0},
    }
    operator_projections: List[Dict[str, object]] = []

    for operator in operators:
        op_id = str(operator["id"])
        op_type = str(operator["type"])
        if op_id not in row_by_id:
            continue
        row = row_by_id[op_id]
        work_ops, total_bytes = estimate_operator_work_bytes(operator, model)
        measured_cycles = _project_measured_cycles(row, work_ops, total_bytes)
        axi_roofline_cycles = _roofline_cycles_from_row(row, work_ops, total_bytes)
        centralized_roofline_cycles = (
            _centralized_global_remote_sum_cycles(row, total_bytes)
            if "remote_sum" in op_type
            else axi_roofline_cycles
        )
        ring_roofline_cycles = (
            _roofline_cycles_from_row(ring_by_id.get(op_id, row), work_ops, total_bytes)
            if "remote_sum" in op_type
            else axi_roofline_cycles
        )

        centralized_measured_cycles = measured_cycles
        ring_measured_cycles = measured_cycles
        if "remote_sum" in op_type:
            centralized_measured_cycles = measured_cycles * _safe_ratio(centralized_roofline_cycles, axi_roofline_cycles)
            ring_measured_cycles = measured_cycles * _safe_ratio(ring_roofline_cycles, axi_roofline_cycles)

        _add_scenario_op(scenarios["measured"], op_id, measured_cycles)
        _add_scenario_op(
            scenarios["projected_measured_centralized_global_remote_sum"],
            op_id,
            centralized_measured_cycles,
        )
        _add_scenario_op(scenarios["projected_measured_ring2ring_remote_sum"], op_id, ring_measured_cycles)
        _add_scenario_op(scenarios["axi_pull_roofline"], op_id, axi_roofline_cycles)
        _add_scenario_op(scenarios["centralized_global_roofline"], op_id, centralized_roofline_cycles)
        _add_scenario_op(scenarios["ring2ring_n2n_roofline"], op_id, ring_roofline_cycles)
        operator_projections.append(
            {
                "op_id": op_id,
                "op_type": op_type,
                "input_shapes": _format_operator_port_shapes(operator, "inputs", values),
                "output_shapes": _format_operator_port_shapes(operator, "output", values),
                "remote_sum_geometry": _model_remote_sum_geometry(operator, values)
                if "remote_sum" in op_type
                else "-",
                "compute_domain": row.get("compute_domain"),
                "peak_compute_ops_per_cycle": row.get("peak_compute_ops_per_cycle"),
                "peak_memory_bandwidth_bytes_per_cycle": row.get("peak_memory_bandwidth_bytes_per_cycle"),
                "work_ops": work_ops,
                "total_bytes": total_bytes,
                "calibration_measured_cycles": row.get("measured_cycles"),
                "calibration_measured_ops_per_cycle": row.get("measured_ops_per_cycle"),
                "calibration_measured_bandwidth_bytes_per_cycle": row.get("measured_bandwidth_bytes_per_cycle"),
                "measured_cycles": measured_cycles,
                "projected_measured_centralized_global_remote_sum_cycles": centralized_measured_cycles,
                "projected_measured_ring2ring_remote_sum_cycles": ring_measured_cycles,
                "axi_pull_roofline_cycles": axi_roofline_cycles,
                "centralized_global_roofline_cycles": centralized_roofline_cycles,
                "ring2ring_n2n_roofline_cycles": ring_roofline_cycles,
            }
        )

    for scenario in scenarios.values():
        per_layer = float(scenario["per_layer_cycles"])
        scenario["ttft_ms"] = ttft_ms(per_layer, model.num_hidden_layers, frequency_hz)
        scenario["total_cycles"] = per_layer * float(model.num_hidden_layers)

    return {
        "model": model.summary(),
        "frequency_hz": frequency_hz,
        "graph_template": str(graph_path),
        "operators": operator_projections,
        "scenarios": scenarios,
    }


def _format_operator_port_shapes(
    operator: Mapping[str, object],
    direction: str,
    values: Mapping[str, int],
) -> str:
    raw_ports = operator[direction]
    ports = {"out": raw_ports} if direction == "output" else dict(raw_ports)
    return "; ".join(
        f"{port_name}=[{' x '.join(str(value) for value in _display_shape(dict(port_spec)['shape'], values))}]"
        for port_name, port_spec in ports.items()
    )


def _display_shape(shape: Sequence[object], values: Mapping[str, int]) -> List[int]:
    resolved = [_eval_shape_expr(value, values) for value in shape]
    return resolved[1:] if len(resolved) == 3 else resolved


def _model_remote_sum_geometry(operator: Mapping[str, object], values: Mapping[str, int]) -> str:
    inputs = dict(operator["inputs"])
    input_port = next(iter(inputs.values()))
    input_shape = [_eval_shape_expr(value, values) for value in dict(input_port)["shape"]]
    fan_in = input_shape[1] if len(input_shape) > 1 else int(values["used_slices"])
    return f"fan_in={fan_in}"


def estimate_operator_work_bytes(
    operator: Mapping[str, object],
    model: ModelExecutionConfig,
) -> Tuple[float, int]:
    op_id = str(operator["id"])
    op_type = str(operator["type"])
    values = model.values()
    if op_type in GEMM_TYPES:
        return _gemm_work_for_operator(op_id, op_type, values), 0
    return 0.0, _non_gemm_bytes_for_operator(operator, values)


def ttft_ms(per_layer_cycles: object, num_hidden_layers: int, frequency_hz: float) -> Optional[float]:
    if per_layer_cycles is None or frequency_hz <= 0:
        return None
    return float(per_layer_cycles) * float(num_hidden_layers) / frequency_hz * 1000.0


def _load_graph_operators(graph_path: Path) -> List[Mapping[str, object]]:
    payload = json.loads(graph_path.read_text(encoding="utf-8-sig"))
    operators = payload.get("operators", [])
    if not isinstance(operators, list):
        raise ValueError("Graph template must define an operators list for model TTFT scaling.")
    return [operator for operator in operators if isinstance(operator, Mapping)]


def _gemm_work(m: int, k: int, n_local: int) -> float:
    return 2.0 * float(m) * float(k) * float(n_local)


def _gemm_work_for_operator(op_id: str, op_type: str, values: Mapping[str, int]) -> float:
    seq_len = int(values["sequence_length"])
    hidden_exec = int(values["hidden_size"])
    intermediate_size = int(values["intermediate_size"])
    used_slices = int(values["used_slices"])
    head_dim = int(values["head_dim"])
    slice_per_head = int(values["slice_per_head"])
    attention_waves = int(values["attention_waves"])
    kv_heads_per_cluster = int(values["kv_heads_per_cluster"])

    q_local_width = head_dim // slice_per_head
    kv_local_width = (kv_heads_per_cluster * head_dim) // slice_per_head
    hidden_local_width = hidden_exec // used_slices
    ffn_local_width = intermediate_size // used_slices

    if op_type in {"prefill_gemm_local_qkt", "gemm_local_qkt_fp16_fp16_fp32"}:
        return _gemm_work(seq_len, q_local_width, seq_len) * attention_waves
    if op_type in {"prefill_gemm_local", "gemm_local_fp16_fp16_fp16"}:
        return _gemm_work(seq_len, seq_len, q_local_width) * attention_waves
    if op_id == "op5":
        return _gemm_work(seq_len, hidden_exec, q_local_width) * attention_waves
    if op_id in {"op15", "op20"}:
        return _gemm_work(seq_len, hidden_exec, kv_local_width)
    if op_id == "op30":
        return _gemm_work(seq_len, hidden_exec, q_local_width) * attention_waves
    if op_id in {"op37", "op38"}:
        return _gemm_work(seq_len, hidden_exec, ffn_local_width)
    if op_id == "op41":
        return _gemm_work(seq_len, intermediate_size, hidden_local_width)
    raise ValueError(f"Unhandled GEMM operator for model TTFT scaling: {op_id} ({op_type})")


def _non_gemm_bytes_for_operator(operator: Mapping[str, object], values: Mapping[str, int]) -> int:
    op_id = str(operator["id"])
    op_type = str(operator["type"])
    input_bytes_map, output_bytes = _dtype_bytes_by_role(op_type)
    input_total_bytes = 0
    for port_name, port_spec in dict(operator.get("inputs", {})).items():
        if port_name not in input_bytes_map:
            raise ValueError(f"Missing dtype-byte mapping for {op_type} input port {port_name}")
        elements = _shape_elements(list(dict(port_spec)["shape"]), values)
        input_total_bytes += elements * input_bytes_map[port_name]
    output_elements = _shape_elements(list(dict(operator["output"])["shape"]), values)
    total_bytes = input_total_bytes + output_elements * output_bytes
    multiplier = int(values["attention_waves"]) if op_id in ATTENTION_WAVE_MULTIPLIER_IDS else 1
    return total_bytes * multiplier


def _dtype_bytes_by_role(op_type: str) -> Tuple[Dict[str, int], int]:
    mapping = {
        "prefill_summac_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_remote_sum_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_remote_sum_4slice_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_mac_SFU_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_mul_fp32MN_fp32M_fp32MN": ({"A": 4, "B": 4}, 4),
        "prefill_mul_fp32MN_fp32N_fp16MN": ({"A": 4, "B": 4}, 2),
        "prefill_add_fp16MN_fp32N_fp32MN": ({"A": 2, "B": 4}, 4),
        "prefill_mul_fp32MN_fp32MN_fp32MN": ({"A": 4, "B": 4}, 4),
        "prefill_add_fp32MN_fp32MN_fp16MN": ({"A": 4, "B": 4}, 2),
        "prefill_add_V_fp16MN_fp32N_fp16MN": ({"A": 2, "B": 4}, 2),
        "prefill_add_fp32MN_fp32MN_fp32MN": ({"A": 4, "B": 4}, 4),
        "prefill_max_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_sub_SFU_fp32MN_fp32M_fp32MN": ({"A": 4, "B": 4}, 4),
        "prefill_sum_rec_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_mul_fp32MN_fp32M_fp16MN": ({"A": 4, "B": 4}, 2),
        "prefill_add_fp32MN_fp16MN_fp32MN": ({"A": 4, "B": 2}, 4),
        "prefill_silu_fp16MN_fp32MN": ({"A": 2}, 4),
        "prefill_mul_fp32MN_fp16MN_fp16MN": ({"A": 4, "B": 2}, 2),
    }
    if op_type not in mapping:
        raise ValueError(f"No dtype-byte mapping registered for op_type={op_type}")
    return mapping[op_type]


def _project_measured_cycles(row: Mapping[str, object], work_ops: float, total_bytes: int) -> float:
    if work_ops > 0:
        measured_ops_per_cycle = row.get("measured_ops_per_cycle")
        if measured_ops_per_cycle:
            return work_ops / float(measured_ops_per_cycle)
        util = row.get("measured_compute_utilization")
        peak = float(row.get("peak_compute_ops_per_cycle") or 0.0)
        if util and peak:
            return work_ops / (float(util) * peak)
    measured_bw = row.get("measured_bandwidth_bytes_per_cycle")
    if total_bytes and measured_bw:
        return float(total_bytes) / float(measured_bw)
    return _roofline_cycles_from_row(row, work_ops, total_bytes)


def _roofline_cycles_from_row(row: Mapping[str, object], work_ops: float, total_bytes: int) -> float:
    peak = float(row.get("peak_compute_ops_per_cycle") or 0.0)
    memory_bw = float(row.get("peak_memory_bandwidth_bytes_per_cycle") or 0.0)
    compute_cycles = work_ops / peak if peak and work_ops else 0.0
    bandwidth_cycles = float(total_bytes) / memory_bw if memory_bw and total_bytes else 0.0
    return max(compute_cycles, bandwidth_cycles)


def _centralized_global_remote_sum_cycles(row: Mapping[str, object], total_bytes: float) -> float:
    fan_in = int(row.get("remote_fan_in", 1))
    local_write_bytes = float(row.get("local_write_bytes", 0.0))
    if local_write_bytes:
        global_bytes = max(0.0, total_bytes - local_write_bytes) + max(0, fan_in - 1) * local_write_bytes
    else:
        global_bytes = total_bytes
    global_bw = (128.0 / 8.0) * 0.5
    global_cycles = global_bytes / global_bw if global_bw else 0.0
    local_bw = float(row.get("peak_memory_bandwidth_bytes_per_cycle") or 0.0)
    local_cycles = local_write_bytes / local_bw if local_bw and local_write_bytes else 0.0
    compute_cycles = float(row.get("roofline_compute_bound_cycles") or 0.0)
    return max(global_cycles, local_cycles, compute_cycles)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _add_scenario_op(scenario: Dict[str, object], op_id: str, cycles: float) -> None:
    scenario["per_op"][op_id] = cycles
    scenario["per_layer_cycles"] = float(scenario["per_layer_cycles"]) + cycles


def _shape_elements(shape_spec: List[object], values: Mapping[str, int]) -> int:
    total = 1
    for dim in shape_spec:
        total *= _eval_shape_expr(dim, values)
    return total


def _eval_shape_expr(expr: object, values: Mapping[str, int]) -> int:
    if isinstance(expr, int):
        return expr
    if not isinstance(expr, str):
        raise ValueError(f"Unsupported shape expression type: {type(expr)!r}")
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body, values)


def _eval_node(node: ast.AST, values: Mapping[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(f"Unknown identifier in shape expression: {node.id}")
        return int(values[node.id])
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Div):
            return left // right
    raise ValueError(f"Unsupported shape expression: {ast.unparse(node)}")
