import argparse
import ast
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple


PEAK_GEMM_OPS_PER_CYCLE = 256.0
SLICE_FREQUENCY_HZ = 1_000_000_000.0
DEFAULT_NON_GEMM_BANDWIDTH_BYTES_PER_CYCLE = 16.0
DEFAULT_GRAPH_PATH = "examples/graphs/layer0/layer0_padding_0529.json"
DEFAULT_SEQUENCE_MULTIPLE = 32

GEMM_TYPES = {
    "prefill_gemm_ring_4slice",
    "prefill_gemm_local_qkt",
    "prefill_gemm_local",
}

# These non-GEMM ops are executed per attention wave because they sit on top of
# per-head attention activations rather than on the full hidden-state tensor.
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


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _gemm_cycles(m: int, k: int, n_local: int, utilization: float) -> float:
    return (2.0 * float(m) * float(k) * float(n_local)) / (utilization * PEAK_GEMM_OPS_PER_CYCLE)


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


def _shape_elements(shape_spec: List[object], values: Mapping[str, int]) -> int:
    total = 1
    for dim in shape_spec:
        total *= _eval_shape_expr(dim, values)
    return total


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        raise ValueError("sequence_length must be positive.")
    if multiple <= 0:
        raise ValueError("sequence_length rounding multiple must be positive.")
    return int(math.ceil(value / float(multiple)) * multiple)


def _build_execution_values(
    config: Mapping[str, object],
    sequence_length: Optional[int],
    sequence_multiple: int,
) -> Dict[str, int]:
    logical_hidden = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    slice_per_head = int(config["slice_per_head"])
    used_slices = int(config["used_slices"])
    kv_padding = int(config["kv_padding"])
    requested_seq_len = int(sequence_length if sequence_length is not None else config["sequence_length"])
    seq_len = _round_up_to_multiple(requested_seq_len, sequence_multiple)

    if used_slices % slice_per_head != 0:
        raise ValueError("used_slices must be divisible by slice_per_head.")
    if head_dim % slice_per_head != 0:
        raise ValueError("head_dim must be divisible by slice_per_head.")
    if intermediate_size % used_slices != 0:
        raise ValueError("intermediate_size must be divisible by used_slices.")
    if (num_key_value_heads * head_dim) % slice_per_head != 0:
        raise ValueError("num_key_value_heads * head_dim must be divisible by slice_per_head.")

    clusters = used_slices // slice_per_head
    padded_attention_heads = int(math.ceil(num_attention_heads / float(clusters)) * clusters)
    attention_waves = padded_attention_heads // clusters
    hidden_exec = padded_attention_heads * head_dim
    if hidden_exec % used_slices != 0:
        raise ValueError("execution hidden size must be divisible by used_slices.")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads for grouped KV mapping.")

    q_heads_per_cluster = padded_attention_heads // clusters
    q_heads_per_kv_head = num_attention_heads // num_key_value_heads
    kv_heads_per_cluster = int(math.ceil(q_heads_per_cluster / float(q_heads_per_kv_head)))

    values = {
        "logical_hidden_size": logical_hidden,
        "requested_sequence_length": requested_seq_len,
        "hidden_size": hidden_exec,
        "intermediate_size": intermediate_size,
        "num_attention_heads": padded_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "sequence_length": seq_len,
        "slice_per_head": slice_per_head,
        "used_slices": used_slices,
        "kv_padding": kv_padding,
        "clusters": clusters,
        "padded_attention_heads": padded_attention_heads,
        "attention_waves": attention_waves,
        "q_heads_per_cluster": q_heads_per_cluster,
        "q_heads_per_kv_head": q_heads_per_kv_head,
        "kv_heads_per_cluster": kv_heads_per_cluster,
    }
    return values


def _dtype_bytes_by_role(op_type: str) -> Tuple[Dict[str, int], int]:
    mapping = {
        "prefill_summac_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_remote_sum_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_remote_sum_4slice_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_mac_SFU_fp32MN_fp32MN": ({"A": 4}, 4),
        "prefill_mul_fp32MN_fp32M_fp32MN": ({"A": 4, "B": 4}, 4),
        "prefill_mul_fp32MN_fp32N_fp16MN": ({"A": 4, "B": 4}, 2),
        "prefill_add_fp16MN_fp32N_fp32MN": ({"A": 4, "B": 2}, 4),
        "prefill_mul_fp32MN_fp32MN_fp32MN": ({"A": 4, "B": 4}, 4),
        "prefill_add_fp32MN_fp32MN_fp16MN": ({"A": 4, "B": 4}, 2),
        "prefill_add_V_fp16MN_fp32N_fp16MN": ({"A": 4, "B": 2}, 2),
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


def _gemm_cycles_for_operator(op_id: str, op_type: str, values: Mapping[str, int], utilization: float) -> float:
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

    if op_type == "prefill_gemm_local_qkt":
        return _gemm_cycles(seq_len, q_local_width, seq_len, utilization) * attention_waves
    if op_type == "prefill_gemm_local":
        return _gemm_cycles(seq_len, seq_len, q_local_width, utilization) * attention_waves
    if op_id == "op5":
        return _gemm_cycles(seq_len, hidden_exec, q_local_width, utilization) * attention_waves
    if op_id in {"op15", "op20"}:
        return _gemm_cycles(seq_len, hidden_exec, kv_local_width, utilization)
    if op_id == "op30":
        return _gemm_cycles(seq_len, hidden_exec, q_local_width, utilization) * attention_waves
    if op_id in {"op37", "op38"}:
        return _gemm_cycles(seq_len, hidden_exec, ffn_local_width, utilization)
    if op_id == "op41":
        return _gemm_cycles(seq_len, intermediate_size, hidden_local_width, utilization)
    raise ValueError(f"Unhandled GEMM operator: {op_id} ({op_type})")


def _non_gemm_cycles_for_operator(
    operator: Mapping[str, object],
    values: Mapping[str, int],
    effective_bandwidth_bytes_per_cycle: float,
) -> Tuple[float, int]:
    op_id = str(operator["id"])
    op_type = str(operator["type"])
    input_bytes_map, output_bytes = _dtype_bytes_by_role(op_type)

    input_total_bytes = 0
    inputs = dict(operator.get("inputs", {}))
    for port_name, port_spec in inputs.items():
        if port_name not in input_bytes_map:
            raise ValueError(f"Missing dtype-byte mapping for {op_type} input port {port_name}")
        elements = _shape_elements(list(dict(port_spec)["shape"]), values)
        input_total_bytes += elements * input_bytes_map[port_name]

    output_spec = dict(operator["output"])
    output_elements = _shape_elements(list(output_spec["shape"]), values)
    total_bytes = input_total_bytes + output_elements * output_bytes

    multiplier = int(values["attention_waves"]) if op_id in ATTENTION_WAVE_MULTIPLIER_IDS else 1
    total_bytes *= multiplier
    cycles = total_bytes / effective_bandwidth_bytes_per_cycle
    return cycles, total_bytes


def build_estimate(
    config: Dict[str, object],
    sequence_length: Optional[int] = None,
    gemm_utilization: float = 0.92,
    non_gemm_effective_bandwidth_bytes_per_cycle: float = DEFAULT_NON_GEMM_BANDWIDTH_BYTES_PER_CYCLE,
    graph_path: str = DEFAULT_GRAPH_PATH,
    sequence_multiple: int = DEFAULT_SEQUENCE_MULTIPLE,
) -> Dict[str, object]:
    if gemm_utilization <= 0.0 or gemm_utilization > 1.0:
        raise ValueError("gemm_utilization must be in (0, 1].")
    if non_gemm_effective_bandwidth_bytes_per_cycle <= 0.0:
        raise ValueError("non_gemm_effective_bandwidth_bytes_per_cycle must be positive.")

    values = _build_execution_values(config, sequence_length, sequence_multiple)
    graph = _load_json(Path(graph_path))
    operators = list(graph["operators"])

    ring_gemm_breakdown: Dict[str, float] = {}
    non_gemm_breakdown_cycles: Dict[str, float] = {}
    non_gemm_breakdown_bytes: Dict[str, int] = {}

    for operator in operators:
        op_id = str(operator["id"])
        op_type = str(operator["type"])
        if op_type in GEMM_TYPES:
            ring_gemm_breakdown[op_id] = _gemm_cycles_for_operator(op_id, op_type, values, gemm_utilization)
        else:
            cycles, total_bytes = _non_gemm_cycles_for_operator(
                operator,
                values,
                non_gemm_effective_bandwidth_bytes_per_cycle,
            )
            non_gemm_breakdown_cycles[op_id] = cycles
            non_gemm_breakdown_bytes[op_id] = total_bytes

    ring_gemm_per_layer = sum(ring_gemm_breakdown.values())
    non_gemm_per_layer = sum(non_gemm_breakdown_cycles.values())
    total_per_layer = ring_gemm_per_layer + non_gemm_per_layer
    num_hidden_layers = int(config["num_hidden_layers"])
    total_cycles = total_per_layer * float(num_hidden_layers)
    total_ms = (total_cycles / SLICE_FREQUENCY_HZ) * 1_000.0

    return {
        "inputs": {
            "logical_hidden_size": int(config["hidden_size"]),
            "intermediate_size": int(config["intermediate_size"]),
            "num_attention_heads": int(config["num_attention_heads"]),
            "num_key_value_heads": int(config["num_key_value_heads"]),
            "head_dim": int(config["head_dim"]),
            "slice_per_head": int(config["slice_per_head"]),
            "used_slices": int(config["used_slices"]),
            "num_hidden_layers": int(config["num_hidden_layers"]),
            "requested_sequence_length": int(values["requested_sequence_length"]),
            "sequence_length": int(values["sequence_length"]),
            "kv_padding": int(config["kv_padding"]),
        },
        "execution_assumptions": {
            "clusters": int(values["clusters"]),
            "padded_attention_heads": int(values["padded_attention_heads"]),
            "attention_waves": int(values["attention_waves"]),
            "q_heads_per_cluster": int(values["q_heads_per_cluster"]),
            "q_heads_per_kv_head": int(values["q_heads_per_kv_head"]),
            "kv_heads_per_cluster": int(values["kv_heads_per_cluster"]),
            "execution_hidden_size": int(values["hidden_size"]),
            "gemm_utilization": gemm_utilization,
            "non_gemm_effective_bandwidth_bytes_per_cycle": non_gemm_effective_bandwidth_bytes_per_cycle,
            "sequence_multiple": sequence_multiple,
            "slice_frequency_hz": SLICE_FREQUENCY_HZ,
            "graph_template": graph_path,
        },
        "ring_gemm_breakdown_per_layer_cycles": {
            "op5_q_projection": ring_gemm_breakdown.get("op5", 0.0),
            "op15_k_projection": ring_gemm_breakdown.get("op15", 0.0),
            "op20_v_projection": ring_gemm_breakdown.get("op20", 0.0),
            "op22_qkt_local_gemm": ring_gemm_breakdown.get("op22", 0.0),
            "op29_sv_local_gemm": ring_gemm_breakdown.get("op29", 0.0),
            "op30_attention_output_projection": ring_gemm_breakdown.get("op30", 0.0),
            "op37_ffn_gate_projection": ring_gemm_breakdown.get("op37", 0.0),
            "op38_ffn_up_projection": ring_gemm_breakdown.get("op38", 0.0),
            "op41_ffn_down_projection": ring_gemm_breakdown.get("op41", 0.0),
            "subtotal": ring_gemm_per_layer,
        },
        "non_gemm_breakdown_per_layer_cycles": non_gemm_breakdown_cycles,
        "non_gemm_breakdown_per_layer_total_bytes": non_gemm_breakdown_bytes,
        "non_gemm_per_layer_cycles": non_gemm_per_layer,
        "total_per_layer_cycles": total_per_layer,
        "total_cycles": total_cycles,
        "total_ms": total_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone TTFT estimate from config.json using GEMM utilization and explicit non-GEMM byte accounting."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="examples/configs/config.json",
        help="Path to the model config JSON. Defaults to examples/configs/config.json.",
    )
    parser.add_argument(
        "--graph",
        default=DEFAULT_GRAPH_PATH,
        help=f"Layer graph template used for explicit non-GEMM accounting. Default: {DEFAULT_GRAPH_PATH}",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Override sequence_length from the config without editing the file. The script rounds it up to the configured multiple.",
    )
    parser.add_argument(
        "--sequence-multiple",
        type=int,
        default=DEFAULT_SEQUENCE_MULTIPLE,
        help=f"Round sequence_length up to this multiple before estimating TTFT. Default: {DEFAULT_SEQUENCE_MULTIPLE}.",
    )
    parser.add_argument(
        "--gemm-utilization",
        type=float,
        default=0.92,
        help="Effective utilization for GEMM operators. Default: 0.92.",
    )
    parser.add_argument(
        "--non-gemm-bandwidth",
        type=float,
        default=DEFAULT_NON_GEMM_BANDWIDTH_BYTES_PER_CYCLE,
        help="Effective non-GEMM bandwidth in bytes/cycle. Default: 16.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format. Default: text.",
    )
    args = parser.parse_args()

    estimate = build_estimate(
        config=_load_json(Path(args.config)),
        sequence_length=args.sequence_length,
        gemm_utilization=args.gemm_utilization,
        non_gemm_effective_bandwidth_bytes_per_cycle=args.non_gemm_bandwidth,
        graph_path=args.graph,
        sequence_multiple=args.sequence_multiple,
    )

    if args.format == "json":
        print(json.dumps(estimate, indent=2))
        return

    inputs = estimate["inputs"]
    assumptions = estimate["execution_assumptions"]
    ring = estimate["ring_gemm_breakdown_per_layer_cycles"]
    print("TTFT Estimate")
    print(f"config: {Path(args.config)}")
    print(f"graph: {Path(args.graph)}")
    print(
        f"sequence_length: requested={inputs['requested_sequence_length']}, "
        f"effective={inputs['sequence_length']}"
    )
    print(
        "execution: "
        f"clusters={assumptions['clusters']}, "
        f"padded_heads={assumptions['padded_attention_heads']}, "
        f"waves={assumptions['attention_waves']}, "
        f"hidden_exec={assumptions['execution_hidden_size']}"
    )
    print(
        "assumptions: "
        f"gemm_util={assumptions['gemm_utilization']:.2f}, "
        f"non_gemm_bw={assumptions['non_gemm_effective_bandwidth_bytes_per_cycle']:.2f} B/cycle"
    )
    print("gemm_per_layer_cycles:")
    for name, value in ring.items():
        if name == "subtotal":
            continue
        print(f"  {name}: {value:.2f}")
    print(f"  subtotal: {ring['subtotal']:.2f}")
    print(f"non_gemm_per_layer_cycles: {estimate['non_gemm_per_layer_cycles']:.2f}")
    print(f"total_per_layer_cycles: {estimate['total_per_layer_cycles']:.2f}")
    print(f"total_cycles: {estimate['total_cycles']:.2f}")
    print(f"total_ms: {estimate['total_ms']:.6f}")


if __name__ == "__main__":
    main()
