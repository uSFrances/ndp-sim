import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .graph import load_graph_file
from .performance import analyze_graph_performance, load_performance_config


def generate_roofline_artifacts(
    graph_path: str,
    config_path: Optional[str] = None,
    mode: str = "remap",
    explicit_output: Optional[str] = None,
) -> Dict[str, object]:
    hardware, perf_cfg = load_performance_config(config_path)
    payload = analyze_graph_performance(
        load_graph_file(graph_path),
        hardware,
        perf_cfg,
        include_request_traces=False,
    )
    summary = build_roofline_summary(payload, mode=mode, graph_name=Path(graph_path).stem)
    svg = render_roofline_svg(summary)

    output_path, summary_path = _resolve_roofline_output_paths(
        graph_path=graph_path,
        mode=mode,
        explicit_output=explicit_output,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "mode": mode,
        "graph_name": summary["graph_name"],
        "svg_path": str(output_path),
        "summary_path": str(summary_path),
        "hardware": summary["hardware"],
        "operators": summary["operators"],
    }


def _resolve_roofline_output_paths(
    graph_path: str,
    mode: str,
    explicit_output: Optional[str],
) -> Tuple[Path, Path]:
    if not explicit_output:
        svg_path = _default_roofline_output_path(graph_path, mode)
        return svg_path, svg_path.with_suffix(".json")

    out = Path(explicit_output)
    suffix = out.suffix.lower()
    if suffix == ".svg":
        return out, out.with_suffix(".json")
    if suffix == ".json":
        return out.with_suffix(".svg"), out
    return out, out.with_suffix(".json")


def build_roofline_summary(
    performance_payload: Mapping[str, object],
    mode: str,
    graph_name: str,
) -> Dict[str, object]:
    hardware = dict(performance_payload["hardware"])
    derived = dict(hardware.get("derived", {}))
    bandwidth = float(derived["peak_memory_bandwidth_bytes_per_cycle"])
    general_peak = float(derived["general_peak_ops_per_cycle"])
    gemm_peak = float(derived["gemm_peak_ops_per_cycle"])

    op_breakdown = list(dict(performance_payload["modes"])[mode]["op_breakdown"])
    operators: List[Dict[str, object]] = []
    for op in op_breakdown:
        if str(op.get("kind")) != "op":
            continue
        intensity = op.get("arithmetic_intensity")
        latency = float(op["latency_cycles"])
        hardware_measured_cycles = op.get("hardware_measured_cycles")
        work_ops = float(op["work_ops"])
        if intensity is None or latency <= 0.0 or work_ops <= 0.0:
            continue
        intensity_value = float(intensity)
        peak_compute = float(dict(op["true_roofline"])["peak_compute_ops_per_cycle"])
        roofline_limit = min(peak_compute, bandwidth * intensity_value)
        achieved_perf = work_ops / latency
        analytical_bandwidth = float(op["total_bytes"]) / latency if latency > 0.0 else None
        measured_perf = (
            work_ops / float(hardware_measured_cycles)
            if hardware_measured_cycles is not None and float(hardware_measured_cycles) > 0.0
            else None
        )
        measured_bandwidth = (
            float(op["total_bytes"]) / float(hardware_measured_cycles)
            if hardware_measured_cycles is not None and float(hardware_measured_cycles) > 0.0
            else None
        )
        analytical_bandwidth_utilization = (analytical_bandwidth / bandwidth) if bandwidth > 0.0 else None
        measured_bandwidth_utilization = (
            measured_bandwidth / bandwidth if bandwidth > 0.0 and measured_bandwidth is not None else None
        )
        analytical_efficiency = (achieved_perf / roofline_limit) if roofline_limit > 0.0 else None
        measured_efficiency = (
            measured_perf / roofline_limit if roofline_limit > 0.0 and measured_perf is not None else None
        )
        operators.append(
            {
                "op_name": str(op["op_name"]),
                "op_type": str(op["op_type"]),
                "arithmetic_intensity_ops_per_byte": intensity_value,
                "roofline_limit_ops_per_cycle": roofline_limit,
                "achieved_ops_per_cycle": achieved_perf,
                "hardware_measured_cycles": float(hardware_measured_cycles) if hardware_measured_cycles is not None else None,
                "measured_ops_per_cycle": measured_perf,
                "peak_compute_ops_per_cycle": peak_compute,
                "latency_cycles": latency,
                "work_ops": work_ops,
                "total_bytes": int(op["total_bytes"]),
                "analytical_bandwidth_bytes_per_cycle": analytical_bandwidth,
                "measured_bandwidth_bytes_per_cycle": measured_bandwidth,
                "analytical_bandwidth_utilization": analytical_bandwidth_utilization,
                "measured_bandwidth_utilization": measured_bandwidth_utilization,
                "analytical_efficiency": analytical_efficiency,
                "measured_efficiency": measured_efficiency,
                "measured_vs_analytical_ratio": (measured_perf / achieved_perf)
                if measured_perf is not None and achieved_perf > 0.0
                else None,
                "utilization": {
                    "analytical_compute": _format_percent(analytical_efficiency),
                    "measured_compute": _format_percent(measured_efficiency),
                    "analytical_bandwidth": _format_percent(analytical_bandwidth_utilization),
                    "measured_bandwidth": _format_percent(measured_bandwidth_utilization),
                },
            }
        )

    operators.sort(key=lambda item: float(item["achieved_ops_per_cycle"]), reverse=True)
    return {
        "graph_name": graph_name,
        "mode": mode,
        "hardware": {
            "peak_memory_bandwidth_bytes_per_cycle": bandwidth,
            "general_peak_ops_per_cycle": general_peak,
            "gemm_peak_ops_per_cycle": gemm_peak,
            "general_ridge_point_ops_per_byte": (general_peak / bandwidth) if bandwidth else None,
            "gemm_ridge_point_ops_per_byte": (gemm_peak / bandwidth) if bandwidth else None,
        },
        "operators": operators,
    }


def render_roofline_svg(summary: Mapping[str, object]) -> str:
    operators = list(summary["operators"])
    hardware = dict(summary["hardware"])
    bandwidth = float(hardware["peak_memory_bandwidth_bytes_per_cycle"])
    general_peak = float(hardware["general_peak_ops_per_cycle"])
    gemm_peak = float(hardware["gemm_peak_ops_per_cycle"])
    general_ridge = float(hardware["general_ridge_point_ops_per_byte"])
    gemm_ridge = float(hardware["gemm_ridge_point_ops_per_byte"])

    positive_x = [float(op["arithmetic_intensity_ops_per_byte"]) for op in operators if float(op["arithmetic_intensity_ops_per_byte"]) > 0.0]
    positive_y = []
    for op in operators:
        roof = float(op["roofline_limit_ops_per_cycle"])
        actual = float(op["achieved_ops_per_cycle"])
        measured = op.get("measured_ops_per_cycle")
        if roof > 0.0:
            positive_y.append(roof)
        if actual > 0.0:
            positive_y.append(actual)
        if measured is not None and float(measured) > 0.0:
            positive_y.append(float(measured))
    positive_y.extend([general_peak, gemm_peak])

    x_min = _nice_lower_bound(min(positive_x + [general_ridge, gemm_ridge]))
    x_max = _nice_upper_bound(max(positive_x + [general_ridge, gemm_ridge]))
    y_min = _nice_lower_bound(min(positive_y))
    y_max = _nice_upper_bound(max(positive_y))

    width = 1200
    height = 820
    margin_left = 110
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_to_svg(value: float) -> float:
        return margin_left + _log_interp(value, x_min, x_max) * plot_width

    def y_to_svg(value: float) -> float:
        return margin_top + (1.0 - _log_interp(value, y_min, y_max)) * plot_height

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf7"/>',
        f'<text x="{margin_left}" y="34" font-size="26" font-family="monospace" fill="#1f2937">'
        f'Roofline: {summary["graph_name"]} [{summary["mode"]}]</text>',
        f'<text x="{margin_left}" y="58" font-size="14" font-family="monospace" fill="#4b5563">'
        f'BW={bandwidth:.2f} B/cycle, general_peak={general_peak:.2f} ops/cycle, special_peak={gemm_peak:.2f} ops/cycle</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
    ]

    for tick in _log_ticks(x_min, x_max):
        x = x_to_svg(tick)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="monospace" fill="#374151">{_fmt_tick(tick)}</text>'
        )
    for tick in _log_ticks(y_min, y_max):
        y = y_to_svg(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="monospace" fill="#374151">{_fmt_tick(tick)}</text>'
        )

    svg_parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="15" font-family="monospace" fill="#111827">Arithmetic intensity (ops/byte)</text>'
    )
    svg_parts.append(
        f'<text x="26" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="15" font-family="monospace" fill="#111827" transform="rotate(-90 26 {margin_top + plot_height / 2:.2f})">Performance (ops/cycle)</text>'
    )

    _append_piecewise_roofline(
        svg_parts,
        x_to_svg,
        y_to_svg,
        x_min,
        x_max,
        ridge=general_ridge,
        peak=general_peak,
        bandwidth=bandwidth,
        color="#2563eb",
        label="general roofline",
    )
    _append_piecewise_roofline(
        svg_parts,
        x_to_svg,
        y_to_svg,
        x_min,
        x_max,
        ridge=gemm_ridge,
        peak=gemm_peak,
        bandwidth=bandwidth,
        color="#dc2626",
        label="special roofline",
    )

    placed_labels: List[Tuple[float, float, float, float]] = []
    for op in operators:
        intensity = float(op["arithmetic_intensity_ops_per_byte"])
        roof = float(op["roofline_limit_ops_per_cycle"])
        actual = float(op["achieved_ops_per_cycle"])
        measured = op.get("measured_ops_per_cycle")
        is_gemm = float(op["peak_compute_ops_per_cycle"]) > general_peak
        color = "#dc2626" if is_gemm else "#2563eb"
        x = x_to_svg(intensity)
        roof_y = y_to_svg(roof)
        actual_y = y_to_svg(actual)
        svg_parts.append(f'<line x1="{x:.2f}" y1="{roof_y:.2f}" x2="{x:.2f}" y2="{actual_y:.2f}" stroke="{color}" stroke-opacity="0.35" stroke-width="1.5"/>')
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{roof_y:.2f}" r="5.5" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{actual_y:.2f}" r="4.5" fill="{color}" fill-opacity="0.85"/>')
        label_anchor_y = actual_y
        if measured is not None and float(measured) > 0.0:
            measured_y = y_to_svg(float(measured))
            svg_parts.append(
                f'<line x1="{x:.2f}" y1="{actual_y:.2f}" x2="{x:.2f}" y2="{measured_y:.2f}" stroke="#047857" stroke-opacity="0.45" stroke-width="1.5" stroke-dasharray="4 3"/>'
            )
            svg_parts.append(
                f'<rect x="{x - 4.5:.2f}" y="{measured_y - 4.5:.2f}" width="9" height="9" fill="#ecfdf5" stroke="#047857" stroke-width="2"/>'
            )
            # Measured utilization labels should attach to the measured marker.
            label_anchor_y = measured_y
        label_text = _short_label(str(op["op_type"]), str(op["op_name"]))
        utilization_suffix = _operator_utilization_suffix(op)
        if utilization_suffix:
            label_text = f"{label_text} {utilization_suffix}"
        approx_width = max(42.0, float(len(label_text) * 6.8))
        label_x, label_y, anchor, placed_labels = _place_label(
            x=x,
            actual_y=label_anchor_y,
            roof_y=roof_y,
            margin_left=margin_left,
            margin_right=margin_left + plot_width,
            margin_top=margin_top,
            margin_bottom=margin_top + plot_height,
            approx_width=approx_width,
            placed_labels=placed_labels,
        )
        label_attach_x = label_x - 3.0 if anchor == "start" else label_x + 3.0
        label_attach_y = label_y - 4.0
        if abs(label_attach_x - x) > 10.0 or abs(label_attach_y - label_anchor_y) > 10.0:
            svg_parts.append(
                f'<line x1="{x:.2f}" y1="{label_anchor_y:.2f}" x2="{label_attach_x:.2f}" y2="{label_attach_y:.2f}" stroke="#6b7280" stroke-opacity="0.65" stroke-width="1"/>'
            )
        svg_parts.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" font-size="11" font-family="monospace" fill="#111827">{_escape_xml(label_text)}</text>'
        )

    legend_x = margin_left + plot_width - 290
    legend_y = margin_top + plot_height - 166
    svg_parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="268" height="146" rx="8" fill="#ffffff" stroke="#d1d5db"/>')
    svg_parts.append(f'<line x1="{legend_x + 16}" y1="{legend_y + 24}" x2="{legend_x + 44}" y2="{legend_y + 24}" stroke="#2563eb" stroke-width="3"/>')
    svg_parts.append(f'<text x="{legend_x + 54}" y="{legend_y + 28}" font-size="12" font-family="monospace" fill="#111827">general roofline</text>')
    svg_parts.append(f'<line x1="{legend_x + 16}" y1="{legend_y + 46}" x2="{legend_x + 44}" y2="{legend_y + 46}" stroke="#dc2626" stroke-width="3"/>')
    svg_parts.append(f'<text x="{legend_x + 54}" y="{legend_y + 50}" font-size="12" font-family="monospace" fill="#111827">special roofline</text>')
    svg_parts.append(f'<circle cx="{legend_x + 30}" cy="{legend_y + 69}" r="5.5" fill="#ffffff" stroke="#111827" stroke-width="2"/>')
    svg_parts.append(f'<text x="{legend_x + 54}" y="{legend_y + 73}" font-size="12" font-family="monospace" fill="#111827">theoretical limit at this AI</text>')
    svg_parts.append(f'<circle cx="{legend_x + 30}" cy="{legend_y + 91}" r="4.5" fill="#111827"/>')
    svg_parts.append(f'<text x="{legend_x + 54}" y="{legend_y + 95}" font-size="12" font-family="monospace" fill="#111827">analytical achieved performance</text>')
    svg_parts.append(f'<rect x="{legend_x + 25.5}" y="{legend_y + 107.5}" width="9" height="9" fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
    svg_parts.append(f'<text x="{legend_x + 54}" y="{legend_y + 117}" font-size="12" font-family="monospace" fill="#111827">hardware measured performance</text>')
    svg_parts.append(
        f'<text x="{legend_x + 16}" y="{legend_y + 138}" font-size="11" font-family="monospace" fill="#4b5563">U=utilization, M=measured, A=analytical</text>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts) + "\n"


def _format_percent(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value * 100.0:.2f}%"


def _operator_utilization_suffix(op: Mapping[str, object]) -> Optional[str]:
    roofline_limit = float(op["roofline_limit_ops_per_cycle"])
    if roofline_limit <= 0.0:
        return None

    measured = op.get("measured_ops_per_cycle")
    if measured is not None and float(measured) > 0.0:
        utilization = float(measured) / roofline_limit
        return f"[U={utilization * 100.0:.1f}%(M)]"

    achieved = float(op["achieved_ops_per_cycle"])
    if achieved <= 0.0:
        return None
    utilization = achieved / roofline_limit
    return f"[U={utilization * 100.0:.1f}%(A)]"


def _append_piecewise_roofline(
    svg_parts: List[str],
    x_to_svg,
    y_to_svg,
    x_min: float,
    x_max: float,
    ridge: float,
    peak: float,
    bandwidth: float,
    color: str,
    label: str,
) -> None:
    left_x = x_min
    right_x = x_max
    slope_end = min(max(ridge, x_min), x_max)
    if left_x < slope_end:
        x1 = x_to_svg(left_x)
        y1 = y_to_svg(bandwidth * left_x)
        x2 = x_to_svg(slope_end)
        y2 = y_to_svg(min(peak, bandwidth * slope_end))
        svg_parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="3"/>')
    if ridge < right_x:
        x1 = x_to_svg(max(ridge, x_min))
        y1 = y_to_svg(peak)
        x2 = x_to_svg(right_x)
        y2 = y_to_svg(peak)
        svg_parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="3"/>')
        svg_parts.append(
            f'<text x="{x2 - 6:.2f}" y="{y2 - 8:.2f}" text-anchor="end" font-size="12" font-family="monospace" fill="{color}">{label}</text>'
        )


def _default_roofline_output_path(graph_path: str, mode: str) -> Path:
    source = Path(graph_path)
    return Path.cwd() / "outputs" / "performance" / source.stem / f"{source.stem}_roofline_{mode}.svg"


def _log_interp(value: float, lower: float, upper: float) -> float:
    return (math.log10(value) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))


def _log_ticks(lower: float, upper: float) -> Sequence[float]:
    ticks: List[float] = []
    start = int(math.floor(math.log10(lower)))
    end = int(math.ceil(math.log10(upper)))
    for exp in range(start, end + 1):
        for base in (1, 2, 5):
            value = base * (10 ** exp)
            if lower <= value <= upper:
                ticks.append(float(value))
    return ticks


def _nice_lower_bound(value: float) -> float:
    exp = math.floor(math.log10(max(value, 1e-12)))
    base = value / (10 ** exp)
    if base <= 1:
        nice = 0.5
    elif base <= 2:
        nice = 1.0
    elif base <= 5:
        nice = 2.0
    else:
        nice = 5.0
    return nice * (10 ** exp)


def _nice_upper_bound(value: float) -> float:
    exp = math.floor(math.log10(max(value, 1e-12)))
    base = value / (10 ** exp)
    if base <= 1:
        nice = 1.0
    elif base <= 2:
        nice = 2.0
    elif base <= 5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10 ** exp)


def _fmt_tick(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}" if abs(value - round(value)) > 1e-9 else f"{value:.0f}"
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _short_label(op_type: str, op_name: str) -> str:
    if op_type.startswith("ring_gemm"):
        return "ring_gemm"
    if op_type.startswith("prefill_"):
        body = op_type[len("prefill_"):]
        for suffix in ("_fp16_fp16_fp16", "_fp16_fp32_fp32", "_fp32MN_fp32M_fp32MN", "_Mfp32_Mfp32"):
            if body.endswith(suffix):
                body = body[: -len(suffix)]
        return body
    return op_name


def _place_label(
    x: float,
    actual_y: float,
    roof_y: float,
    margin_left: float,
    margin_right: float,
    margin_top: float,
    margin_bottom: float,
    approx_width: float,
    placed_labels: Sequence[Tuple[float, float, float, float]],
) -> Tuple[float, float, str, List[Tuple[float, float, float, float]]]:
    preferred_dy = -8.0 if actual_y > roof_y + 18.0 else 16.0
    y_offsets = [preferred_dy, preferred_dy + 12.0, preferred_dy - 12.0, preferred_dy + 24.0, preferred_dy - 24.0, 28.0, -20.0, 40.0, -32.0]
    x_offsets = [8.0, 14.0, 20.0, 28.0]
    candidates: List[Tuple[str, float, float]] = []
    for dx in x_offsets:
        for dy in y_offsets:
            candidates.append(("start", x + dx, actual_y + dy))
            candidates.append(("end", x - dx, actual_y + dy))

    # Favor the side that keeps labels closer to the chart center before trying the opposite side.
    center_x = (margin_left + margin_right) / 2.0
    if x > center_x:
        candidates = [c for c in candidates if c[0] == "end"] + [c for c in candidates if c[0] == "start"]
    else:
        candidates = [c for c in candidates if c[0] == "start"] + [c for c in candidates if c[0] == "end"]

    inflate = 2.0
    for anchor, label_x, label_y in candidates:
        left = label_x if anchor == "start" else label_x - approx_width
        right = label_x + approx_width if anchor == "start" else label_x
        top = label_y - 10.0
        bottom = label_y + 2.0
        if left < margin_left or right > margin_right or top < margin_top or bottom > margin_bottom:
            continue
        overlap = False
        for other_left, other_top, other_right, other_bottom in placed_labels:
            if not (
                right + inflate < other_left
                or left - inflate > other_right
                or bottom + inflate < other_top
                or top - inflate > other_bottom
            ):
                overlap = True
                break
        if not overlap:
            updated = list(placed_labels)
            updated.append((left, top, right, bottom))
            return label_x, label_y, anchor, updated

    # Last resort: stack vertically along the preferred side until a non-overlapping slot is found.
    fallback_anchor = "end" if x > center_x else "start"
    fallback_x = x - 18.0 if fallback_anchor == "end" else x + 18.0
    for step in range(0, 24):
        direction = -1.0 if step % 2 == 0 else 1.0
        magnitude = (step // 2) * 12.0
        fallback_y = min(max(actual_y + direction * magnitude + 10.0, margin_top + 12.0), margin_bottom - 4.0)
        left = fallback_x if fallback_anchor == "start" else fallback_x - approx_width
        right = fallback_x + approx_width if fallback_anchor == "start" else fallback_x
        top = fallback_y - 10.0
        bottom = fallback_y + 2.0
        if left < margin_left or right > margin_right:
            continue
        overlap = False
        for other_left, other_top, other_right, other_bottom in placed_labels:
            if not (
                right + inflate < other_left
                or left - inflate > other_right
                or bottom + inflate < other_top
                or top - inflate > other_bottom
            ):
                overlap = True
                break
        if not overlap:
            updated = list(placed_labels)
            updated.append((left, top, right, bottom))
            return fallback_x, fallback_y, fallback_anchor, updated

    fallback_y = min(max(actual_y + 16.0, margin_top + 12.0), margin_bottom - 4.0)
    left = fallback_x if fallback_anchor == "start" else fallback_x - approx_width
    right = fallback_x + approx_width if fallback_anchor == "start" else fallback_x
    updated = list(placed_labels)
    updated.append((left, fallback_y - 10.0, right, fallback_y + 2.0))
    return fallback_x, fallback_y, fallback_anchor, updated


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
