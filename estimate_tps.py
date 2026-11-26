#!/usr/bin/env python3
"""Estimate decode TPS from RTL per-operator cycle summaries.

The script consumes the per-op cycle summary produced by the RTL simulation
(``records`` with ``op_id`` and ``cycle``), the full-size Decode layer graph
that describes the 58-operator two-pass layer, and a model config describing
the target network.  It projects:

* per-layer cycles at an arbitrary KV/attention length,
* per-token latency (TPOT) and steady-state decode TPS,
* TTFT from an optional prefill roofline projection,
* end-to-end TPS for a matrix of input/output lengths.

Only Python's standard library is used so the estimate can be reproduced on
any machine that can run the project.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CYCLE_SUMMARY = BASE_DIR / "doc" / "local_mse0_wdata_op_cycle_summary.json"
DEFAULT_GRAPH = (
    BASE_DIR
    / "generate_python_golden"
    / "model_execplan"
    / "examples"
    / "layer0_decode_optimized_full.json"
)
DEFAULT_MODEL_CONFIG = BASE_DIR / "generate_python_golden" / "config_full.json"
DEFAULT_PREFILL_SCRIPT = (
    BASE_DIR / "address_remapping" / "scripts" / "estimate_ttft_from_config.py"
)
DEFAULT_PREFILL_CONFIG = (
    BASE_DIR / "address_remapping" / "examples" / "configs" / "config.json"
)
DEFAULT_PREFILL_GRAPH = (
    BASE_DIR
    / "address_remapping"
    / "examples"
    / "graphs"
    / "layer0"
    / "layer0_padding_0529.json"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "tps"
DEFAULT_FREQ_MHZ = 800.0
DEFAULT_NUM_LAYERS = 28
DEFAULT_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_ATTN_ROUND = 32
DEFAULT_MAX_CONTEXT = 2048


# ---------------------------------------------------------------------------
# Shape / work helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("length list must not be empty")
    return values


def eval_shape_expr(expr: object, values: Mapping[str, int]) -> int:
    if isinstance(expr, bool):
        raise ValueError(f"Unsupported boolean shape expression: {expr!r}")
    if isinstance(expr, int):
        return int(expr)
    if not isinstance(expr, str):
        raise ValueError(f"Unsupported shape expression type: {type(expr)!r}")
    tree = ast.parse(expr, mode="eval")
    return eval_node(tree.body, values)


def eval_node(node: ast.AST, values: Mapping[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(f"Unknown identifier in shape expression: {node.id}")
        return int(values[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_node(node.operand, values)
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left, values)
        right = eval_node(node.right, values)
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


def shape_elements(shape: Sequence[object], values: Mapping[str, int]) -> int:
    total = 1
    for dim in shape:
        total *= eval_shape_expr(dim, values)
    return total


def operator_work(operator: Mapping[str, object], values: Mapping[str, int]) -> float:
    """Return a shape-derived work proxy (element count) for one operator.

    Summing input and output elements is enough to detect which operators are
    attention-length dependent: constant-shape operators keep the same value,
    while QK^T / softmax / SV / remote-sum operators grow with the KV length.
    """

    total = 0
    inputs = dict(operator.get("inputs", {}))
    for port_name, port_data in inputs.items():
        if not isinstance(port_data, Mapping):
            raise ValueError(f"Operator {operator.get('id')} input {port_name} is not a mapping")
        shape = port_data.get("shape")
        if shape is None:
            raise ValueError(f"Operator {operator.get('id')} input {port_name} has no shape")
        total += shape_elements(list(shape), values)
    output = operator.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"Operator {operator.get('id')} has no output mapping")
    output_shape = output.get("shape")
    if output_shape is None:
        raise ValueError(f"Operator {operator.get('id')} output has no shape")
    total += shape_elements(list(output_shape), values)
    return float(total)


def build_values(params: Mapping[str, object], attention_length: int) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for key, value in params.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            values[str(key)] = int(value)
    values["decode_attention_length"] = int(attention_length)
    values["attention_length"] = int(attention_length)
    return values


def effective_context_len(
    context_len: int,
    base_attention_len: int,
    attention_round: int,
) -> int:
    if context_len <= base_attention_len:
        return base_attention_len
    rounded = int(math.ceil(context_len / float(attention_round)) * attention_round)
    return max(base_attention_len, rounded)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_graph(path: Path) -> Tuple[Dict[str, object], List[Mapping[str, object]]]:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Graph must be a JSON object: {path}")
    operators = payload.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError(f"Graph must define a non-empty 'operators' list: {path}")
    ops = [op for op in operators if isinstance(op, Mapping)]
    if len(ops) != len(operators):
        raise ValueError(f"Graph contains non-mapping operator entries: {path}")
    return dict(payload), ops


def optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_measured_cycles(
    path: Path,
    frequency_hz: float,
) -> Tuple[Dict[str, Dict[str, object]], List[str], Dict[str, object], List[str]]:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Cycle summary must be a JSON object: {path}")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raw_records = payload.get("operators")
    if not isinstance(raw_records, list):
        raise ValueError("Cycle summary must define 'records' or 'operators'")

    parsed: List[Dict[str, object]] = []
    skipped_status: List[str] = []
    for item in raw_records:
        if not isinstance(item, Mapping):
            continue
        op_id = str(item.get("op_id", "")).strip()
        if not op_id:
            continue
        status = item.get("status")
        if status is not None and str(status).lower() not in {"completed", "complete"}:
            skipped_status.append(f"{op_id}({status})")
            continue
        parsed.append(
            {
                "op_id": op_id,
                "cycle": optional_float(item.get("cycle")),
                "start_time": optional_float(item.get("start_time_ns")),
                "completed_time": optional_float(item.get("completed_time_ns")),
                "start_cycle": optional_float(item.get("start_cycle")),
                "completed_cycle": optional_float(item.get("completed_cycle")),
                "raw": dict(item),
            }
        )

    # The summary's cycle value is authoritative.  When timestamps are present
    # too, infer the timestamp unit from records that contain both, instead of
    # assuming the field name's unit is correct.
    factors: List[float] = []
    for record in parsed:
        cycle = record["cycle"]
        start = record["start_time"]
        completed = record["completed_time"]
        if cycle is not None and start is not None and completed is not None:
            delta = float(completed) - float(start)
            if delta > 0:
                factors.append(float(cycle) / delta)

    warnings: List[str] = []
    cycles_per_time_unit: Optional[float] = (
        float(statistics.median(factors)) if factors else None
    )
    if factors and min(factors) > 0:
        spread = max(factors) / min(factors)
        if spread > 1.2:
            warnings.append(
                "Timestamp-derived cycle rates vary by "
                f"{spread:.2f}x across records; timestamps may mix clock domains "
                "or separate runs. The per-op 'cycle' fields are used as-is."
            )

    measured: Dict[str, Dict[str, object]] = {}
    duplicates: List[str] = []
    missing_cycle_fallback = False

    for record in parsed:
        op_id = str(record["op_id"])
        cycle = record["cycle"]
        if cycle is None:
            start_cycle = record["start_cycle"]
            completed_cycle = record["completed_cycle"]
            if start_cycle is not None and completed_cycle is not None:
                cycle = float(completed_cycle) - float(start_cycle)
        if cycle is None:
            start = record["start_time"]
            completed = record["completed_time"]
            if start is not None and completed is not None:
                delta = float(completed) - float(start)
                if cycles_per_time_unit is not None:
                    cycle = delta * cycles_per_time_unit
                else:
                    cycle = delta * frequency_hz / 1e9
                    missing_cycle_fallback = True
        if cycle is None or cycle <= 0:
            raise ValueError(
                f"Operator {op_id} does not have a positive measured cycle "
                "and cannot be derived from start/completed fields"
            )
        if op_id in measured:
            duplicates.append(op_id)
        measured[op_id] = {"cycle": float(cycle), "raw": dict(record["raw"])}

    if skipped_status:
        warnings.append(
            "Skipped non-completed records: " + ", ".join(sorted(set(skipped_status)))
        )
    if missing_cycle_fallback:
        warnings.append(
            "Some records had no cycle field and no timestamp-derived cycle rate; "
            "their cycles were derived using --freq-mhz and the *_time_ns fields."
        )

    starts = [
        float(record["start_time"])
        for record in parsed
        if record["start_time"] is not None
    ]
    ends = [
        float(record["completed_time"])
        for record in parsed
        if record["completed_time"] is not None
    ]
    span_info: Dict[str, object] = {}
    if starts and ends:
        span_time = max(ends) - min(starts)
        if cycles_per_time_unit is not None:
            span_cycles = span_time * cycles_per_time_unit
        else:
            span_cycles = span_time * frequency_hz / 1e9
        span_info = {
            "start_time_min": min(starts),
            "completed_time_max": max(ends),
            "span_time_units": span_time,
            "cycles_per_time_unit": cycles_per_time_unit,
            "span_cycles": span_cycles,
            "note": (
                "Span is diagnostic only and assumes all records come from one "
                "single continuous simulation log. The default layer latency is "
                "the sum of per-op cycles. cycles_per_time_unit is inferred from "
                "records that contain both 'cycle' and start/completed times."
            ),
        }

    return measured, duplicates, {"payload": dict(payload), "span": span_info}, warnings


def validate_operator_set(
    graph_ops: Sequence[Mapping[str, object]],
    measured: Mapping[str, Dict[str, object]],
) -> Tuple[List[str], List[str]]:
    graph_ids = [str(op["id"]) for op in graph_ops]
    expected = set(graph_ids)
    missing = [op_id for op_id in graph_ids if op_id not in measured]
    extra = sorted(set(measured.keys()) - expected)
    return missing, extra


# ---------------------------------------------------------------------------
# TPS projection
# ---------------------------------------------------------------------------


def build_projection(
    *,
    graph_ops: Sequence[Mapping[str, object]],
    params: Mapping[str, object],
    measured: Mapping[str, Dict[str, object]],
    num_layers: int,
    frequency_hz: float,
    base_attention_len: int,
    attention_round: int,
    input_lengths: Sequence[int],
    output_lengths: Sequence[int],
    ttft_provider: Callable[[int], Optional[float]],
    overhead_ms: float,
    max_context: int,
    warnings: List[str],
    latency_mode: str = "sum",
    timestamp_span_cycles: Optional[float] = None,
) -> Dict[str, object]:
    base_values = build_values(params, base_attention_len)
    base_work = {
        str(op["id"]): operator_work(op, base_values) for op in graph_ops
    }
    op_by_id = {str(op["id"]): op for op in graph_ops}
    measured_ids = [op_id for op_id in measured if op_id in op_by_id]

    base_sum_cycles = sum(float(measured[op_id]["cycle"]) for op_id in measured_ids)
    use_span = False
    gap_cycles = 0.0
    if timestamp_span_cycles is not None and base_sum_cycles > 0:
        if latency_mode == "span":
            use_span = True
        elif latency_mode == "auto":
            ratio = timestamp_span_cycles / base_sum_cycles
            use_span = 0.5 <= ratio <= 2.0
        if use_span:
            gap_cycles = max(0.0, float(timestamp_span_cycles) - base_sum_cycles)
    if latency_mode == "span" and not use_span:
        warnings.append(
            "Span latency mode was requested but no usable timestamp span was "
            "available; falling back to the sum of per-op cycles."
        )

    scale_cache: Dict[int, Dict[str, float]] = {}
    tpot_cache: Dict[int, Dict[str, float]] = {}

    def scales_for(effective_len: int) -> Dict[str, float]:
        cached = scale_cache.get(effective_len)
        if cached is not None:
            return cached
        target_values = build_values(params, effective_len)
        scales: Dict[str, float] = {}
        for op_id, base_units in base_work.items():
            if base_units <= 0:
                scales[op_id] = 1.0
                continue
            target_units = operator_work(op_by_id[op_id], target_values)
            scales[op_id] = target_units / base_units
        scale_cache[effective_len] = scales
        return scales

    def layer_cycles_for(effective_len: int) -> float:
        scales = scales_for(effective_len)
        total = 0.0
        for op_id in measured_ids:
            total += float(measured[op_id]["cycle"]) * scales.get(op_id, 1.0)
        if use_span:
            total += gap_cycles
        return total

    def tpot_for(effective_len: int) -> Dict[str, float]:
        cached = tpot_cache.get(effective_len)
        if cached is not None:
            return cached
        layer_cycles = layer_cycles_for(effective_len)
        overhead_cycles = (overhead_ms / 1000.0) * frequency_hz
        token_cycles = layer_cycles * float(num_layers) + overhead_cycles
        tpot_ms = token_cycles / frequency_hz * 1000.0
        result = {
            "layer_cycles": layer_cycles,
            "token_cycles": token_cycles,
            "tpot_ms": tpot_ms,
            "decode_tps": (1000.0 / tpot_ms) if tpot_ms > 0 else 0.0,
        }
        tpot_cache[effective_len] = result
        return result

    max_decode_steps = max(0, max(output_lengths) - 1)
    requested_contexts = sorted(
        {
            int(input_len) + step
            for input_len in input_lengths
            for step in range(1, max_decode_steps + 1)
        }
    )
    effective_contexts = sorted(
        {
            effective_context_len(ctx, base_attention_len, attention_round)
            for ctx in requested_contexts
        }
    )

    over_limit = [ctx for ctx in requested_contexts if ctx > max_context]
    if over_limit:
        warnings.append(
            f"{len(over_limit)} requested context lengths exceed --max-context="
            f"{max_context}; the estimate is still computed but may exceed the "
            f"current KV-cache/padding configuration (max requested="
            f"{max(over_limit)})."
        )

    context_curve: List[Dict[str, object]] = []
    for effective_len in effective_contexts:
        tpot = tpot_for(effective_len)
        context_curve.append(
            {
                "effective_context_len": effective_len,
                **tpot,
            }
        )

    matrix: List[Dict[str, object]] = []
    for input_len in input_lengths:
        ttft_ms = ttft_provider(int(input_len))
        for output_len in output_lengths:
            decode_steps = max(0, int(output_len) - 1)
            decode_time_ms = 0.0
            for step in range(1, decode_steps + 1):
                effective_len = effective_context_len(
                    int(input_len) + step, base_attention_len, attention_round
                )
                decode_time_ms += tpot_for(effective_len)["tpot_ms"]

            avg_tpot_ms = decode_time_ms / decode_steps if decode_steps else None
            decode_tps = (
                decode_steps / (decode_time_ms / 1000.0)
                if decode_steps and decode_time_ms > 0
                else None
            )
            total_ms = (float(ttft_ms) if ttft_ms is not None else 0.0) + decode_time_ms
            e2e_tps = (
                float(output_len) / (total_ms / 1000.0)
                if ttft_ms is not None and total_ms > 0
                else None
            )
            matrix.append(
                {
                    "input_len": int(input_len),
                    "output_len": int(output_len),
                    "ttft_ms": ttft_ms,
                    "decode_time_ms": decode_time_ms,
                    "avg_tpot_ms": avg_tpot_ms,
                    "decode_tps": decode_tps,
                    "total_ms": total_ms,
                    "e2e_tps": e2e_tps,
                }
            )

    base_curve = tpot_for(base_attention_len)
    operators: List[Dict[str, object]] = []
    for op in graph_ops:
        op_id = str(op["id"])
        record = measured.get(op_id)
        operators.append(
            {
                "op_id": op_id,
                "op_type": str(op.get("type", "")),
                "measured_cycles": float(record["cycle"]) if record else None,
                "base_work_units": base_work[op_id],
                "scaled_cycles_by_effective_context": {
                    str(effective_len): (
                        float(record["cycle"]) * scale_cache[effective_len][op_id]
                        if record
                        else None
                    )
                    for effective_len in effective_contexts
                },
            }
        )

    return {
        "context_curve": context_curve,
        "matrix": matrix,
        "operators": operators,
        "base": {
            "attention_len": base_attention_len,
            "latency_mode_requested": latency_mode,
            "latency_mode_used": "span" if use_span else "sum",
            "base_sum_cycles": base_sum_cycles,
            "timestamp_span_cycles": timestamp_span_cycles,
            "timestamp_gap_cycles": (
                float(timestamp_span_cycles) - base_sum_cycles
                if timestamp_span_cycles is not None
                else None
            ),
            "gap_cycles": gap_cycles,
            **base_curve,
        },
        "effective_contexts": effective_contexts,
    }


# ---------------------------------------------------------------------------
# TTFT providers
# ---------------------------------------------------------------------------


def make_prefill_config(
    prefill_config_path: Path,
    num_layers: int,
    temp_dir: Path,
) -> Path:
    config = load_json(prefill_config_path)
    if not isinstance(config, Mapping):
        raise ValueError(f"Prefill config must be a JSON object: {prefill_config_path}")
    patched = dict(config)
    patched["num_hidden_layers"] = int(num_layers)
    target = temp_dir / "prefill_config.json"
    target.write_text(json.dumps(patched, indent=2), encoding="utf-8")
    return target


def make_ttft_provider(
    args: argparse.Namespace,
    temp_dir: Optional[Path],
) -> Callable[[int], Optional[float]]:
    if args.ttft_mode == "none":
        return lambda _input_len: None

    if args.ttft_mode == "fixed":
        if args.ttft_ms is None:
            raise SystemExit("--ttft-mode fixed requires --ttft-ms")
        fixed_ms = float(args.ttft_ms)
        return lambda _input_len: fixed_ms

    if temp_dir is None:
        raise SystemExit("internal error: roofline TTFT requires a temporary directory")
    if not args.prefill_script.exists():
        raise SystemExit(f"Prefill estimator not found: {args.prefill_script}")
    if not args.prefill_graph.exists():
        raise SystemExit(f"Prefill graph not found: {args.prefill_graph}")

    prefill_config = make_prefill_config(
        args.prefill_config, args.num_layers, temp_dir
    )
    cache: Dict[int, float] = {}
    scale_to_frequency = 1000.0 / float(args.freq_mhz)

    def provider(input_len: int) -> Optional[float]:
        if input_len in cache:
            return cache[input_len]
        cmd = [
            sys.executable,
            str(args.prefill_script),
            str(prefill_config),
            "--graph",
            str(args.prefill_graph),
            "--sequence-length",
            str(input_len),
            "--format",
            "json",
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Prefill roofline estimator failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        try:
            payload = json.loads(completed.stdout)
            total_ms_at_1ghz = float(payload["total_ms"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Could not parse prefill estimator output.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ) from exc
        ttft_ms = total_ms_at_1ghz * scale_to_frequency
        cache[input_len] = ttft_ms
        return ttft_ms

    return provider


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    model_name: str,
) -> None:
    fieldnames = [
        "model_name",
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "decode_time_ms",
        "avg_tpot_ms",
        "decode_tps_tok_s",
        "total_ms",
        "e2e_tps_tok_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": model_name,
                    "input_tokens": row.get("input_len"),
                    "output_tokens": row.get("output_len"),
                    "ttft_ms": row.get("ttft_ms"),
                    "decode_time_ms": row.get("decode_time_ms"),
                    "avg_tpot_ms": row.get("avg_tpot_ms"),
                    "decode_tps_tok_s": row.get("decode_tps"),
                    "total_ms": row.get("total_ms"),
                    "e2e_tps_tok_s": row.get("e2e_tps"),
                }
            )


def write_insufficient_data_report(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    graph_ops: Sequence[Mapping[str, object]],
    measured: Mapping[str, Dict[str, object]],
    missing: Sequence[str],
    extra: Sequence[str],
    duplicates: Sequence[str],
    payload: Mapping[str, object],
    base_attention_len: int,
    warnings: Sequence[str],
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tps_estimate.json"
    csv_path = output_dir / "tps_estimate.csv"
    md_path = output_dir / "tps_estimate.md"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "insufficient_data",
        "model_name": args.model_name,
        "message": (
            "No TPS was calculated. The cycle summary does not cover every "
            "operator in the full-size Decode layer."
        ),
        "assumptions": {
            "frequency_mhz": args.freq_mhz,
            "num_layers": args.num_layers,
            "base_attention_length": base_attention_len,
            "attention_round": args.attn_round,
            "latency_mode": args.latency_mode,
            "ttft_mode": args.ttft_mode,
        },
        "inputs": {
            "cycle_summary": str(args.cycle_summary),
            "graph": str(args.graph),
            "input_lengths": parse_int_list(args.input_lengths),
            "output_lengths": parse_int_list(args.output_lengths),
        },
        "cycle_summary": {
            "expected_ops": len(graph_ops),
            "measured_ops": len(measured),
            "missing_ops": list(missing),
            "extra_ops": list(extra),
            "duplicate_ops": sorted(set(duplicates)),
            "total_ops": payload.get("total_ops"),
            "completed_ops": payload.get("completed_ops"),
            "incomplete_ops": payload.get("incomplete_ops"),
            "input_log": payload.get("input_log"),
        },
        "next_steps": [
            "Run or merge the RTL simulation summaries until all expected op ids are present.",
            "Pass the combined summary with --cycle-summary.",
            "Use --allow-partial only for debugging, never for a reported TPS.",
        ],
        "warnings": list(warnings),
        "tps": None,
    }
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(csv_path, [], args.model_name)

    lines: List[str] = []
    lines.append("# Decode TPS Estimate - Insufficient Data")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Model: `{args.model_name}`")
    lines.append(f"- Cycle summary: `{args.cycle_summary}`")
    lines.append(f"- Decode graph: `{args.graph}`")
    lines.append(f"- Expected operators: {len(graph_ops)}")
    lines.append(f"- Measured operators: {len(measured)}")
    lines.append("")
    lines.append("**No TPS numbers were generated for this run.**")
    lines.append("")
    lines.append("## Missing operators")
    lines.append("")
    lines.append(", ".join(missing) if missing else "None")
    lines.append("")
    if extra:
        lines.append("## Unexpected operators")
        lines.append("")
        lines.append(", ".join(extra))
        lines.append("")
    if duplicates:
        lines.append("## Duplicate records")
        lines.append("")
        lines.append(", ".join(sorted(set(duplicates))))
        lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("1. Run or merge the RTL simulation summary until all expected op ids are present.")
    lines.append("2. Re-run with the combined summary: `python estimate_tps.py --cycle-summary <summary>.json`.")
    lines.append("3. Use `--allow-partial` only for debugging; its TPS is not a valid full-layer result.")
    lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, md_path


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    graph_ops: Sequence[Mapping[str, object]],
    measured: Mapping[str, Dict[str, object]],
    missing: Sequence[str],
    extra: Sequence[str],
    duplicates: Sequence[str],
    payload: Mapping[str, object],
    projection: Mapping[str, object],
    warnings: Sequence[str],
    partial: bool,
) -> None:
    base = projection["base"]  # type: ignore[index]
    lines: List[str] = []
    lines.append(f"# Decode TPS Estimate - {args.model_name}")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Model: `{args.model_name}`")
    lines.append(f"- Cycle summary: `{args.cycle_summary}`")
    lines.append(f"- Decode graph: `{args.graph}`")
    lines.append(f"- Frequency: `{args.freq_mhz:g} MHz`")
    lines.append(f"- Model layers: `{args.num_layers}`")
    lines.append(f"- Base attention length: `{base['attention_len']}`")
    lines.append(
        f"- Latency model: `{base.get('latency_mode_used', 'sum')}` "
        f"(requested `{base.get('latency_mode_requested', args.latency_mode)}`)"
    )
    if base.get("timestamp_span_cycles") is not None:
        lines.append(
            f"- Timestamp span: {fmt_number(base['timestamp_span_cycles'], 1)} cycles; "
            f"span-op gap: {fmt_number(base.get('timestamp_gap_cycles', 0.0), 1)} cycles; "
            f"gap used in latency: {fmt_number(base.get('gap_cycles', 0.0), 1)} cycles"
        )
    lines.append(f"- TTFT mode: `{args.ttft_mode}`")
    lines.append(
        "- Units: cycles = hardware cycles; latency = ms; "
        "throughput = tokens/s (tok/s); context/input/output = tokens."
    )
    if partial:
        lines.append("- **PARTIAL RESULT: the cycle summary does not contain every layer op.**")
    lines.append("")

    lines.append("## Cycle summary")
    lines.append("")
    lines.append(f"- Expected operators: {len(graph_ops)}")
    lines.append(f"- Measured operators: {len(measured)}")
    lines.append(f"- Missing operators: {', '.join(missing) if missing else 'none'}")
    lines.append(f"- Unexpected operators: {', '.join(extra) if extra else 'none'}")
    lines.append(
        f"- Duplicate records (last wins): {', '.join(sorted(set(duplicates))) if duplicates else 'none'}"
    )
    if payload.get("total_ops") is not None:
        lines.append(
            f"- Summary counters: total_ops={payload.get('total_ops')}, "
            f"completed_ops={payload.get('completed_ops')}, "
            f"incomplete_ops={payload.get('incomplete_ops')}"
        )
    lines.append("")

    lines.append("## Base projection")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Layer cycles (cycles) | {fmt_number(base['layer_cycles'], 1)} |")
    lines.append(f"| Token cycles (cycles) | {fmt_number(base['token_cycles'], 1)} |")
    lines.append(f"| TPOT (ms) | {fmt_number(base['tpot_ms'])} |")
    lines.append(f"| Decode TPS (tok/s) | {fmt_number(base['decode_tps'], 2)} |")
    lines.append("")

    lines.append("## Context curve")
    lines.append("")
    lines.append(
        "Context lengths are rounded up to the hardware attention granularity "
        f"(`--attn-round {args.attn_round}`)."
    )
    lines.append("")
    lines.append(
        "| Effective context len (tokens) | Layer cycles (cycles) | TPOT (ms) | Decode TPS (tok/s) |"
    )
    lines.append("|---:|---:|---:|---:|")
    for row in projection["context_curve"]:  # type: ignore[index]
        lines.append(
            f"| {row['effective_context_len']} | "
            f"{fmt_number(row['layer_cycles'], 1)} | "
            f"{fmt_number(row['tpot_ms'])} | "
            f"{fmt_number(row['decode_tps'], 2)} |"
        )
    if not projection["context_curve"]:  # type: ignore[index]
        lines.append("| - | - | - | - |")
    lines.append("")

    lines.append(f"## Input / output length matrix - {args.model_name}")
    lines.append("")
    lines.append(
        "TTFT is a prefill roofline projection, not an RTL measurement. "
        "Decode time covers the output tokens produced after the first token."
    )
    lines.append("")
    lines.append(
        "| Input (tokens) | Output (tokens) | TTFT (ms) | Decode time (ms) | "
        "Avg TPOT (ms) | Decode TPS (tok/s) | E2E TPS (tok/s) | Total (ms) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in projection["matrix"]:  # type: ignore[index]
        lines.append(
            f"| {row['input_len']} | {row['output_len']} | "
            f"{fmt_number(row['ttft_ms'])} | "
            f"{fmt_number(row['decode_time_ms'])} | "
            f"{fmt_number(row['avg_tpot_ms'])} | "
            f"{fmt_number(row['decode_tps'], 2)} | "
            f"{fmt_number(row['e2e_tps'], 2)} | "
            f"{fmt_number(row['total_ms'])} |"
        )
    lines.append("")

    measured_ops = [
        op
        for op in projection["operators"]  # type: ignore[index]
        if op.get("measured_cycles") is not None
    ]
    measured_ops.sort(key=lambda op: float(op["measured_cycles"]), reverse=True)
    if measured_ops:
        base_layer_cycles = float(base["layer_cycles"]) or 1.0
        lines.append("## Top operators by measured cycles")
        lines.append("")
        lines.append(
            "| Op | Type | Measured cycles (cycles) | Share of base layer (%) |"
        )
        lines.append("|---|---|---:|---:|")
        for op in measured_ops[:20]:
            share = float(op["measured_cycles"]) / base_layer_cycles * 100.0
            lines.append(
                f"| {op['op_id']} | {op['op_type']} | "
                f"{fmt_number(op['measured_cycles'], 1)} | {share:.2f}% |"
            )
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_table(rows: Sequence[Mapping[str, object]]) -> None:
    header = (
        f"{'in(tok)':>8} {'out(tok)':>9} {'TTFT(ms)':>10} {'Decode(ms)':>11} "
        f"{'TPOT(ms)':>9} {'DecTPS(t/s)':>12} {'E2E(t/s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['input_len']:>6} {row['output_len']:>6} "
            f"{fmt_number(row['ttft_ms']):>10} "
            f"{fmt_number(row['decode_time_ms']):>10} "
            f"{fmt_number(row['avg_tpot_ms']):>9} "
            f"{fmt_number(row['decode_tps'], 2):>9} "
            f"{fmt_number(row['e2e_tps'], 2):>9}"
        )


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def run_self_test(args: argparse.Namespace) -> int:
    _, graph_ops = load_graph(args.graph)
    graph_ids = [str(op["id"]) for op in graph_ops]
    measured = {op_id: {"cycle": 1000.0, "raw": {}} for op_id in graph_ids}
    params = dict(load_json(args.graph).get("params", {}))  # type: ignore[union-attr]
    base_len = 32
    warnings: List[str] = []

    projection = build_projection(
        graph_ops=graph_ops,
        params=params,
        measured=measured,
        num_layers=1,
        frequency_hz=800e6,
        base_attention_len=base_len,
        attention_round=32,
        input_lengths=[32, 64],
        output_lengths=[1, 33],
        ttft_provider=lambda _input_len: 10.0,
        overhead_ms=0.0,
        max_context=2048,
        warnings=warnings,
    )

    base_values = build_values(params, base_len)
    target_values = build_values(params, 64)
    scaled = [
        str(op["id"])
        for op in graph_ops
        if operator_work(op, target_values) != operator_work(op, base_values)
    ]
    if len(scaled) != 16:
        raise AssertionError(f"expected 16 attention-scaled ops, got {len(scaled)}: {scaled}")

    base = projection["base"]
    if abs(float(base["layer_cycles"]) - 1000.0 * len(graph_ids)) > 1e-6:
        raise AssertionError("base layer cycle sum is incorrect")
    if float(base["tpot_ms"]) <= 0:
        raise AssertionError("base TPOT must be positive")

    matrix = projection["matrix"]
    first = next(row for row in matrix if row["input_len"] == 32 and row["output_len"] == 1)
    if abs(float(first["e2e_tps"]) - 100.0) > 1e-6:
        raise AssertionError("single-token E2E TPS must equal 1/TTFT")
    second = next(row for row in matrix if row["input_len"] == 32 and row["output_len"] == 33)
    if float(second["decode_tps"]) <= 0 or float(second["e2e_tps"]) <= 0:
        raise AssertionError("multi-token TPS must be positive")

    span_warnings: List[str] = []
    span_projection = build_projection(
        graph_ops=graph_ops,
        params=params,
        measured=measured,
        num_layers=1,
        frequency_hz=800e6,
        base_attention_len=base_len,
        attention_round=32,
        input_lengths=[32],
        output_lengths=[1],
        ttft_provider=lambda _input_len: 10.0,
        overhead_ms=0.0,
        max_context=2048,
        warnings=span_warnings,
        latency_mode="span",
        timestamp_span_cycles=60000.0,
    )
    span_base = span_projection["base"]
    if span_base["latency_mode_used"] != "span":
        raise AssertionError("span latency mode was not selected")
    if abs(float(span_base["layer_cycles"]) - 60000.0) > 1e-6:
        raise AssertionError("span layer cycles must include the timestamp gap")

    print("SELF-TEST PASSED")
    print(f"  operators: {len(graph_ids)}")
    print(f"  attention-scaled operators: {len(scaled)}")
    print(f"  base layer cycles: {base['layer_cycles']:.1f}")
    print(f"  base TPOT: {base['tpot_ms']:.6f} ms")
    print(f"  span layer cycles: {span_base['layer_cycles']:.1f}")
    print(f"  output_len=1 e2e TPS @10ms TTFT: {first['e2e_tps']:.3f}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate decode TPS from an RTL per-operator cycle summary. "
            "The default graph is the full-size 58-op two-pass Decode layer."
        )
    )
    parser.add_argument(
        "--cycle-summary",
        type=Path,
        default=DEFAULT_CYCLE_SUMMARY,
        help=f"Per-op RTL cycle summary JSON. Default: {DEFAULT_CYCLE_SUMMARY}",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Full-size Decode layer graph JSON. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name shown in reports. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=f"Number of model layers. Default: {DEFAULT_NUM_LAYERS}",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=DEFAULT_FREQ_MHZ,
        help=f"Chip frequency in MHz. Default: {DEFAULT_FREQ_MHZ:g}",
    )
    parser.add_argument(
        "--base-attn-len",
        type=int,
        default=None,
        help=(
            "Attention/KV length used by the RTL cycle summary. "
            "Defaults to decode_attention_length from the graph params."
        ),
    )
    parser.add_argument(
        "--attn-round",
        type=int,
        default=DEFAULT_ATTN_ROUND,
        help=f"Round context length up to this multiple. Default: {DEFAULT_ATTN_ROUND}",
    )
    parser.add_argument(
        "--input-lengths",
        default="32,128,512,2048",
        help="Comma-separated prefill/input lengths. Default: 32,128,512,2048",
    )
    parser.add_argument(
        "--output-lengths",
        default="1,32,128,512",
        help="Comma-separated output lengths. Default: 1,32,128,512",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help=(
            "Warn when input+output exceeds this context length. "
            f"Default: {DEFAULT_MAX_CONTEXT}"
        ),
    )
    parser.add_argument(
        "--overhead-ms",
        type=float,
        default=0.0,
        help="Optional per-token host/runtime overhead in ms. Default: 0",
    )
    parser.add_argument(
        "--latency-mode",
        choices=["sum", "span", "auto"],
        default="sum",
        help=(
            "Layer latency source: sum of per-op cycles, timestamp span, or "
            "auto-detect. Default: sum"
        ),
    )
    parser.add_argument(
        "--ttft-mode",
        choices=["roofline", "fixed", "none"],
        default="roofline",
        help=(
            "TTFT source: prefill roofline projection, a fixed value, or none. "
            "Default: roofline"
        ),
    )
    parser.add_argument(
        "--ttft-ms",
        type=float,
        default=None,
        help="Fixed TTFT in ms (used with --ttft-mode fixed).",
    )
    parser.add_argument(
        "--prefill-script",
        type=Path,
        default=DEFAULT_PREFILL_SCRIPT,
        help=f"Prefill roofline estimator script. Default: {DEFAULT_PREFILL_SCRIPT}",
    )
    parser.add_argument(
        "--prefill-config",
        type=Path,
        default=DEFAULT_PREFILL_CONFIG,
        help=f"Prefill model config. Default: {DEFAULT_PREFILL_CONFIG}",
    )
    parser.add_argument(
        "--prefill-graph",
        type=Path,
        default=DEFAULT_PREFILL_GRAPH,
        help=f"Prefill layer graph. Default: {DEFAULT_PREFILL_GRAPH}",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow a summary that does not cover every graph operator. "
            "The result is marked partial and is not a valid full-layer TPS."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal formula checks on synthetic cycles and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final output paths.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.num_layers <= 0:
        raise SystemExit("--num-layers must be positive")
    if args.freq_mhz <= 0:
        raise SystemExit("--freq-mhz must be positive")
    if args.attn_round <= 0:
        raise SystemExit("--attn-round must be positive")
    if args.overhead_ms < 0:
        raise SystemExit("--overhead-ms must not be negative")

    if args.self_test:
        return run_self_test(args)

    graph_payload, graph_ops = load_graph(args.graph)
    params = dict(graph_payload.get("params", {}))
    if not isinstance(params, Mapping):
        raise SystemExit("Graph params must be a JSON object")

    base_attention_len = args.base_attn_len
    if base_attention_len is None:
        base_attention_len = int(params.get("decode_attention_length", 32))
    if base_attention_len <= 0:
        raise SystemExit("--base-attn-len must be positive")

    input_lengths = parse_int_list(args.input_lengths)
    output_lengths = parse_int_list(args.output_lengths)
    if any(value <= 0 for value in input_lengths):
        raise SystemExit("--input-lengths must be positive")
    if any(value <= 0 for value in output_lengths):
        raise SystemExit("--output-lengths must be positive")

    frequency_hz = float(args.freq_mhz) * 1e6
    warnings: List[str] = []

    measured, duplicates, summary_meta, load_warnings = load_measured_cycles(
        args.cycle_summary, frequency_hz
    )
    warnings.extend(load_warnings)

    payload = summary_meta["payload"]  # type: ignore[index]
    strict = not args.allow_partial
    missing, extra = validate_operator_set(graph_ops, measured)

    if strict and (missing or extra):
        report_warnings = list(warnings)
        paths = write_insufficient_data_report(
            args.output_dir,
            args=args,
            graph_ops=graph_ops,
            measured=measured,
            missing=missing,
            extra=extra,
            duplicates=duplicates,
            payload=payload,
            base_attention_len=base_attention_len,
            warnings=report_warnings,
        )
        lines = [
            "INSUFFICIENT DATA: no TPS was calculated.",
            f"Expected {len(graph_ops)} operators, measured {len(measured)}.",
        ]
        if missing:
            lines.append("Missing operators: " + ", ".join(missing))
        if extra:
            lines.append("Unexpected operators: " + ", ".join(extra))
        lines.append("Wrote diagnostic report:")
        for path in paths:
            lines.append(f"  {path}")
        lines.append(
            "Run the full RTL simulation and pass the complete summary, or use "
            "--allow-partial only for debugging."
        )
        print("\n".join(lines), file=sys.stderr)
        return 2

    if missing:
        warnings.append(
            "Missing operators are excluded from the sum; this is a partial estimate: "
            + ", ".join(missing)
        )
    if extra:
        warnings.append("Unexpected operators are ignored: " + ", ".join(extra))
    if duplicates:
        warnings.append(
            "Duplicate records found for: "
            + ", ".join(sorted(set(duplicates)))
            + " (last record wins)"
        )

    if payload.get("total_ops") is not None and payload.get("completed_ops") is not None:
        if int(payload["completed_ops"]) < int(payload["total_ops"]):
            warnings.append(
                "The summary reports incomplete_ops="
                f"{payload.get('incomplete_ops')} out of total_ops={payload['total_ops']}."
            )

    is_partial = bool(missing or extra)
    timestamp_span_cycles: Optional[float] = None
    span = summary_meta.get("span") or {}  # type: ignore[union-attr]
    if span:
        sum_cycles = sum(float(record["cycle"]) for record in measured.values())
        span_cycles = float(span.get("span_cycles", 0.0))
        if sum_cycles > 0 and span_cycles > 0:
            timestamp_span_cycles = span_cycles
            ratio = span_cycles / sum_cycles
            if ratio < 0.9 or ratio > 1.1:
                warnings.append(
                    "Timestamp span differs from the sum of per-op cycles by "
                    f"{abs(ratio - 1.0) * 100.0:.1f}% (span={span_cycles:.1f}, "
                    f"sum={sum_cycles:.1f}). The default uses the sum; select "
                    "--latency-mode span to include the timestamp gap."
                )
    if is_partial and timestamp_span_cycles is not None:
        warnings.append(
            "Timestamp span latency is disabled for a partial summary because "
            "the span may include operators that are missing from the cycle sum."
        )
        timestamp_span_cycles = None

    if args.ttft_mode == "roofline":
        warnings.append(
            "TTFT is a prefill roofline projection from the existing performance "
            "model, not an RTL measurement. Decode TPOT/TPS uses the RTL cycle "
            "summary."
        )

    temp_dir_path: Optional[Path] = None
    try:
        if args.ttft_mode == "roofline":
            temp_dir_path = Path(tempfile.mkdtemp(prefix="ndp_tps_"))
        ttft_provider = make_ttft_provider(args, temp_dir_path)

        projection = build_projection(
            graph_ops=graph_ops,
            params=params,
            measured=measured,
            num_layers=args.num_layers,
            frequency_hz=frequency_hz,
            base_attention_len=base_attention_len,
            attention_round=args.attn_round,
            input_lengths=input_lengths,
            output_lengths=output_lengths,
            ttft_provider=ttft_provider,
            overhead_ms=args.overhead_ms,
            max_context=args.max_context,
            warnings=warnings,
            latency_mode=args.latency_mode,
            timestamp_span_cycles=timestamp_span_cycles,
        )
    finally:
        if temp_dir_path is not None:
            shutil.rmtree(temp_dir_path, ignore_errors=True)

    partial = is_partial

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tps_estimate.json"
    csv_path = output_dir / "tps_estimate.csv"
    md_path = output_dir / "tps_estimate.md"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "units": {
            "cycles": "hardware cycles",
            "latency": "ms",
            "throughput": "tokens/s (tok/s)",
            "length": "tokens",
        },
        "assumptions": {
            "model_name": args.model_name,
            "frequency_mhz": args.freq_mhz,
            "num_layers": args.num_layers,
            "base_attention_length": base_attention_len,
            "attention_round": args.attn_round,
            "overhead_ms": args.overhead_ms,
            "max_context": args.max_context,
            "latency_mode": args.latency_mode,
            "latency_model": (
                "timestamp_span_with_scaled_attention_gap"
                if projection["base"]["latency_mode_used"] == "span"  # type: ignore[index]
                else "sum_of_per_op_cycles"
            ),
            "ttft_mode": args.ttft_mode,
            "ttft_note": (
                "roofline prefill projection, not RTL"
                if args.ttft_mode == "roofline"
                else "not estimated"
                if args.ttft_mode == "none"
                else "fixed value"
            ),
        },
        "inputs": {
            "cycle_summary": str(args.cycle_summary),
            "graph": str(args.graph),
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
        },
        "cycle_summary": {
            "expected_ops": len(graph_ops),
            "measured_ops": len(measured),
            "missing_ops": missing,
            "extra_ops": extra,
            "duplicate_ops": sorted(set(duplicates)),
            "total_ops": payload.get("total_ops"),
            "completed_ops": payload.get("completed_ops"),
            "incomplete_ops": payload.get("incomplete_ops"),
            "input_log": payload.get("input_log"),
            "sum_measured_cycles": sum(
                float(record["cycle"]) for record in measured.values()
            ),
            "timestamp_span": span,
        },
        "base": projection["base"],
        "operators": projection["operators"],
        "context_curve": projection["context_curve"],
        "length_matrix": projection["matrix"],
        "warnings": warnings,
        "partial": partial,
    }

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(csv_path, projection["matrix"], args.model_name)  # type: ignore[arg-type]
    write_markdown(
        md_path,
        args=args,
        graph_ops=graph_ops,
        measured=measured,
        missing=missing,
        extra=extra,
        duplicates=duplicates,
        payload=payload,
        projection=projection,
        warnings=warnings,
        partial=partial,
    )

    if not args.quiet:
        base = projection["base"]  # type: ignore[index]
        print(f"Model         : {args.model_name}")
        print(f"Cycle summary : {args.cycle_summary}")
        print(f"Decode graph  : {args.graph}")
        print(
            f"Operators     : expected={len(graph_ops)} measured={len(measured)}"
            + (" (PARTIAL)" if partial else "")
        )
        print(f"Layers / freq : {args.num_layers} / {args.freq_mhz:g} MHz")
        print(f"Base attn len : {base_attention_len} tokens")
        print(
            f"Latency mode  : requested={args.latency_mode}, "
            f"used={base['latency_mode_used']}"
        )
        if base.get("timestamp_span_cycles") is not None:
            print(
                f"Span / gap    : {fmt_number(base['timestamp_span_cycles'], 1)} / "
                f"{fmt_number(base.get('timestamp_gap_cycles', 0.0), 1)} cycles "
                f"(used: {fmt_number(base.get('gap_cycles', 0.0), 1)})"
            )
        print(f"Layer cycles  : {fmt_number(base['layer_cycles'], 1)} cycles")
        print(f"Token cycles  : {fmt_number(base['token_cycles'], 1)} cycles")
        print(f"TPOT          : {fmt_number(base['tpot_ms'], 4)} ms")
        print(f"Decode TPS    : {fmt_number(base['decode_tps'], 2)} tok/s")
        print("")
        print_console_table(projection["matrix"])  # type: ignore[arg-type]
        print("")
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            print("")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
