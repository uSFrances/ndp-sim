import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .addressing import AddressTransform, compose_physical_address, decode_physical_address, encode_physical_address
from .graph import load_graph_file, solve_graph
from .hardware import HardwareSpec, PerformanceConfig
from .layout import LayoutError, LayoutSpec
from .model_parser import expand_model_spec
from .rmsnorm_bridge import normalize_graph_spec
from .solver import (
    EdgeSolveResult,
    _layout_bit_labels,
    _serialize_bound_layout,
    _visible_subword_layout_bits,
)


MODE_BASELINE = "baseline"
MODE_REMAP = "remap"
MODE_REMAP_INTERLEAVE = "remap_interleave"
ALL_MODES = (MODE_BASELINE, MODE_REMAP, MODE_REMAP_INTERLEAVE)


@dataclass(frozen=True)
class PhysicalRequest:
    request_id: int
    tensor_name: str
    edge_name: str
    ag_id: str
    role: str
    logical_addr: int
    base_addr: int
    address_transform: Dict[str, object]
    physical_addr: int
    slice_id: int
    bank_id: int
    row_id: int
    col_id: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "tensor_name": self.tensor_name,
            "edge_name": self.edge_name,
            "ag_id": self.ag_id,
            "role": self.role,
            "logical_addr": self.logical_addr,
            "base_addr": self.base_addr,
            "address_transform": self.address_transform,
            "physical_addr": self.physical_addr,
            "slice_id": self.slice_id,
            "bank_id": self.bank_id,
            "row_id": self.row_id,
            "col_id": self.col_id,
        }


@dataclass
class _BankTimelineState:
    cycles: float = 0.0
    phase: Optional[str] = None
    open_row: Optional[int] = None
    row_switch_count: int = 0
    phase_switch_count: int = 0
    read_request_count: int = 0
    write_request_count: int = 0


@dataclass
class _ClosedLoopControllerState:
    now: float = 0.0
    write_queue_occupancy: int = 0
    q_w_full_cycles: float = 0.0
    slice_blocked_cycles: float = 0.0
    forced_drain_count: int = 0
    rw_switch_count: int = 0
    row_switch_count: int = 0
    arbiter1_wins: int = 0
    arbiter2_wins: int = 0


@dataclass(frozen=True)
class _TimedPhysicalRequest:
    request: PhysicalRequest
    release_cycle: float
    group_key: str


@dataclass(frozen=True)
class _RingGemmExecutionGeometry:
    ring_participants: int
    m_dim: int
    n_dim: int
    local_k_dim: int
    total_k_dim: int
    bytes_per_elem: int
    pe_m: int
    pe_n: int
    pe_k: int
    a_buffer_m: int
    a_buffer_k: int
    b_buffer_k: int
    b_buffer_n: int
    output_tile_m: int
    output_tile_n: int
    output_tile_count_m: int
    output_tile_count_n: int
    output_tile_count: int
    local_a_k_tiles_per_output_tile: int
    a_ring_hops_per_local_a_tile: int
    total_k_tiles_per_output_tile: int
    a_buffer_tile_count: int
    b_buffer_tile_count: int
    output_writeback_tile_count: int
    a_buffer_bytes: int
    b_buffer_bytes: int
    output_tile_bytes: int
    a_reuse_factor: int
    b_reuse_factor: int
    pe_micro_ops_per_output_tile: int
    total_coarse_compute_events: int
    total_pe_micro_ops: int


def load_performance_config(path: Optional[str]) -> Tuple[HardwareSpec, PerformanceConfig]:
    if not path:
        return HardwareSpec(), PerformanceConfig()

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "hardware" not in payload:
        raise ValueError("config file must define a top-level 'hardware' section.")
    if "performance" not in payload:
        raise ValueError("config file must define a top-level 'performance' section.")
    if not isinstance(payload.get("hardware"), Mapping):
        raise ValueError("config.hardware must be a mapping.")
    if not isinstance(payload.get("performance"), Mapping):
        raise ValueError("config.performance must be a mapping.")
    hardware_values = dict(payload["hardware"])
    perf_values = dict(payload["performance"])
    hardware = HardwareSpec.from_dict(hardware_values)
    perf = PerformanceConfig.from_dict(perf_values)
    return hardware, perf


def analyze_graph_performance(
    graph_spec: Mapping[str, object],
    hardware: Optional[HardwareSpec] = None,
    perf_cfg: Optional[PerformanceConfig] = None,
    include_request_traces: bool = False,
) -> Dict[str, object]:
    hw = hardware or HardwareSpec()
    perf = perf_cfg or PerformanceConfig()
    _validate_perf_config(hw, perf)

    normalized_graph = normalize_graph_spec(graph_spec, require_base_addrs=True)
    expanded = expand_model_spec(normalized_graph) if "model" in normalized_graph else dict(normalized_graph)
    solve_results = solve_graph(dict(expanded), hw)
    edges = list(expanded["edges"])
    edge_results_by_consumer_port = {
        (str(edge["consumer"]), str(edge["consumer_port"])): result
        for edge, result in zip(edges, solve_results)
    }

    modes: Dict[str, Dict[str, object]] = {}
    baseline_latency: Optional[float] = None
    for mode in ALL_MODES:
        mode_report = _analyze_mode(
            expanded,
            edge_results_by_consumer_port,
            mode,
            hw,
            perf,
            tensors=dict(expanded.get("tensors", {})),
            include_request_traces=include_request_traces,
        )
        modes[mode] = mode_report
        if mode == MODE_BASELINE:
            baseline_latency = float(mode_report["total_latency_cycles"])

    assert baseline_latency is not None
    for mode_name, mode_report in modes.items():
        total_latency = float(mode_report["total_latency_cycles"])
        mode_report["speedup_vs_baseline"] = baseline_latency / total_latency if total_latency else None

    true_roofline = _summarize_true_roofline(modes[MODE_BASELINE]["op_breakdown"], hw)
    mode_summaries = {
        mode_name: _build_mode_summary(mode_report, true_roofline)
        for mode_name, mode_report in modes.items()
    }
    overview = _build_overview(
        graph_name=str(graph_spec.get("name", "graph")),
        graph_summary={
            "op_count": len(expanded["ops"]),
            "edge_count": len(expanded["edges"]),
            "tensor_count": len(expanded.get("tensors", {})),
        },
        true_roofline=true_roofline,
        mode_summaries=mode_summaries,
    )
    summary_markdown = _render_summary_markdown(
        graph_name=str(graph_spec.get("name", "graph")),
        modes=modes,
        true_roofline=true_roofline,
    )
    return {
        "overview": overview,
        "mode_summaries": mode_summaries,
        "hardware": _serialize_hardware(hw),
        "performance_config": perf.to_dict(),
        "graph_summary": {
            "op_count": len(expanded["ops"]),
            "edge_count": len(expanded["edges"]),
            "tensor_count": len(expanded.get("tensors", {})),
        },
        "true_roofline": true_roofline,
        "modes": modes,
        "summary_markdown": summary_markdown,
    }


def write_performance_outputs(
    input_path: str,
    payload: Mapping[str, object],
    explicit_output: Optional[str] = None,
) -> Tuple[Path, Path]:
    output_path = Path(explicit_output) if explicit_output else _default_performance_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_with_units = _add_display_units(payload)
    output_path.write_text(json.dumps(payload_with_units, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(str(payload["summary_markdown"]).strip() + "\n", encoding="utf-8")
    return output_path, markdown_path


def _add_display_units(value: object) -> object:
    if isinstance(value, list):
        return [_add_display_units(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    result: Dict[str, object] = {}
    for key, item in value.items():
        result[key] = _add_display_units(item)
        display = _format_metric_display(str(key), item)
        if display is not None:
            result[f"{key}_display"] = display
    return result


def _format_metric_display(key: str, value: object) -> Optional[str]:
    if value is None or isinstance(value, (Mapping, list, bool)):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        numeric = float(value)
    elif isinstance(value, float):
        numeric = value
    else:
        return None

    if key.endswith("_cycles"):
        return f"{numeric:.2f} cycles"
    if key.endswith("_bytes"):
        return f"{numeric:.2f} B"
    if key in {
        "peak_memory_bandwidth_bytes_per_cycle",
    }:
        return f"{numeric:.2f} B/cycle"
    if key in {
        "general_peak_ops_per_cycle",
        "gemm_peak_ops_per_cycle",
        "peak_compute_ops_per_cycle",
        "roofline_limit_ops_per_cycle",
        "achieved_ops_per_cycle",
        "measured_ops_per_cycle",
    }:
        return f"{numeric:.2f} ops/cycle"
    if key in {
        "arithmetic_intensity",
        "arithmetic_intensity_ops_per_byte",
        "general_ridge_point_ops_per_byte",
        "gemm_ridge_point_ops_per_byte",
    }:
        return f"{numeric:.2f} ops/byte"
    if key in {
        "latency_share",
        "analytical_efficiency",
        "measured_efficiency",
    }:
        return f"{numeric * 100.0:.2f}%"
    if key == "speedup_vs_baseline":
        return f"{numeric:.2f}x"
    return None


def pack_physical_address(
    request: Mapping[str, object],
    hw: HardwareSpec,
) -> int:
    if "physical_addr" in request:
        return int(request["physical_addr"])
    return encode_physical_address(
        slice_id=int(request["slice_id"]),
        bank_id=int(request["bank_id"]),
        row_id=int(request["row_id"]),
        col_id=int(request["col_id"]),
        hw=hw,
    )


def render_ramulator_trace_lines(
    requests: Sequence[Mapping[str, object]],
    hw: HardwareSpec,
) -> List[str]:
    lines: List[str] = []
    for request in requests:
        op = "ST" if str(request["role"]) == "writeback" else "LD"
        addr = pack_physical_address(request, hw)
        lines.append(f"{op} 0x{addr:x}")
    return lines


def _analyze_mode(
    expanded: Mapping[str, object],
    edge_results_by_consumer_port: Mapping[Tuple[str, str], EdgeSolveResult],
    mode: str,
    hw: HardwareSpec,
    perf: PerformanceConfig,
    tensors: Mapping[str, Mapping[str, object]],
    include_request_traces: bool = False,
) -> Dict[str, object]:
    op_reports: List[Dict[str, object]] = []
    edge_reports: List[Dict[str, object]] = []
    request_trace: List[PhysicalRequest] = []
    total_latency = 0.0
    total_lower_bound = 0.0

    for op_name, op_data in dict(expanded["ops"]).items():
        op_report = _analyze_op(
            op_name=str(op_name),
            op_data=dict(op_data),
            edge_results_by_consumer_port=edge_results_by_consumer_port,
            mode=mode,
            hw=hw,
            perf=perf,
            tensors=tensors,
        )
        pre_stages = list(op_report.pop("pre_stages", []))
        for stage_report in [*pre_stages, op_report]:
            op_reports.append(stage_report)
            request_trace.extend(stage_report.pop("request_trace"))
            total_latency += float(stage_report["latency_cycles"])
            total_lower_bound += float(stage_report["analytical_model"]["lower_bound_cycles"])
            edge_reports.extend(stage_report["edge_breakdown"])
    report = {
        "mode": mode,
        "cycle_domain": "slice-cycle",
        "memory_timing_domain": "bank-cycle",
        "total_latency_cycles": total_latency,
        "analytical_model": {
            "estimated_total_cycles": total_latency,
            "lower_bound_cycles": total_lower_bound,
            "latency_to_lower_bound_ratio": (total_latency / total_lower_bound) if total_lower_bound else None,
            "compute_bound_cycles": sum(float(op["analytical_model"]["compute_bound_cycles"]) for op in op_reports),
            "memory_access_bound_cycles": sum(
                float(op["analytical_model"]["memory_access_bound_cycles"]) for op in op_reports
            ),
            "ag_issue_bound_cycles": sum(float(op["analytical_model"]["ag_issue_bound_cycles"]) for op in op_reports),
            "ring_transfer_bound_cycles": sum(
                float(op["analytical_model"].get("ring_transfer_bound_cycles", 0.0)) for op in op_reports
            ),
        },
        "op_breakdown": op_reports,
        "edge_breakdown": edge_reports,
        "address_transforms": _collect_unique_transforms(op_reports),
    }
    if include_request_traces:
        report["request_trace"] = [request.to_dict() for request in request_trace]
    return report


def _analyze_op(
    op_name: str,
    op_data: Mapping[str, object],
    edge_results_by_consumer_port: Mapping[Tuple[str, str], EdgeSolveResult],
    mode: str,
    hw: HardwareSpec,
    perf: PerformanceConfig,
    tensors: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    op_type = str(op_data["op_type"])
    hardware_measured_cycles = _coerce_optional_positive_float(op_data.get("hardware_measured"))
    streams: List[Dict[str, object]] = []
    edge_breakdown: List[Dict[str, object]] = []
    per_stream_requests: List[List[PhysicalRequest]] = []
    pre_stages: List[Dict[str, object]] = []
    all_read_requests: List[PhysicalRequest] = []

    for port_name, port_data in dict(op_data["inputs"]).items():
        role, ag_ids = _classify_input_stream(op_type, str(port_name))
        tensor_name = str(port_data.get("source_tensor", port_name))
        base_addr = _require_tensor_base_addr(tensor_name, tensors)
        edge_result = edge_results_by_consumer_port.get((op_name, str(port_name)))
        if edge_result is not None and mode == MODE_BASELINE and _needs_software_relayout(edge_result):
            relayout_stage = _analyze_relayout_stage(
                edge_result=edge_result,
                op_name=op_name,
                port_name=str(port_name),
                tensor_name=tensor_name,
                base_addr=base_addr,
                hw=hw,
                perf=perf,
            )
            pre_stages.append(relayout_stage)
        requests = (
            _requests_from_edge_result(edge_result, mode, hw, perf, tensor_name, role, ag_ids, base_addr)
            if edge_result is not None
            else _requests_from_source_tensor(
                tensor_name=tensor_name,
                port_data=dict(port_data),
                mode=mode,
                hw=hw,
                perf=perf,
                role=role,
                ag_ids=ag_ids,
                base_addr=base_addr,
            )
        )
        stream_reports = _build_stream_reports(
            requests=requests,
            ag_ids=ag_ids,
            tensor_name=tensor_name,
            edge_name=f"{tensor_name}->{op_name}:{port_name}",
            role=role,
            mode=mode,
            hw=hw,
            perf=perf,
        )
        streams.extend(stream_reports)
        per_stream_requests.append(list(requests))
        all_read_requests.extend(requests)
        edge_breakdown.extend(_edge_reports_from_streams(stream_reports))

    output_port_name = next(iter(op_data["outputs"]))
    output_data = dict(op_data["outputs"][output_port_name])
    output_tensor_name = _find_output_tensor_name(op_name, output_data)
    output_base_addr = _require_tensor_base_addr(output_tensor_name, tensors)
    write_requests = _requests_from_output_tensor(
        tensor_name=output_tensor_name,
        output_data=output_data,
        mode=mode,
        hw=hw,
        perf=perf,
        base_addr=output_base_addr,
    )
    writeback_reports = _build_stream_reports(
        requests=write_requests,
        ag_ids=("ag4",),
        tensor_name=output_tensor_name,
        edge_name=f"{op_name}->{output_tensor_name}",
        role="writeback",
        mode=mode,
        hw=hw,
        perf=perf,
    )
    streams.extend(writeback_reports)
    per_stream_requests.append(list(write_requests))

    read_streams = [stream for stream in streams if stream["role"] != "writeback"]
    write_stream = writeback_reports[0] if writeback_reports else None

    if op_type == "ring_gemm_fp16_fp16_fp16":
        report = _analyze_ring_gemm_op(
            op_name=op_name,
            op_type=op_type,
            op_data=op_data,
            streams=streams,
            edge_breakdown=edge_breakdown,
            per_stream_requests=per_stream_requests,
            pre_stages=pre_stages,
            write_stream=write_stream,
            hw=hw,
            perf=perf,
        )
        report["hardware_measured_cycles"] = hardware_measured_cycles
        return report

    work_ops, compute_cycles, peak_compute_ops = _estimate_compute(op_type, op_data, hw)
    read_bank_cycles = defaultdict(float)
    for stream in read_streams:
        for bank_id, value in dict(stream["bank_cycles"]).items():
            read_bank_cycles[int(bank_id)] += float(value)
    ag_issue_bound_cycles = max((float(stream["issue_cycles"]) for stream in streams), default=0.0)

    read_cycles = max((float(stream["adjusted_stream_cycles"]) for stream in read_streams), default=0.0)
    conflict_penalty = max(
        0.0,
        max(read_bank_cycles.values(), default=0.0)
        - max((float(stream["raw_bank_max_cycles"]) for stream in read_streams), default=0.0),
    )
    write_cycles = float(write_stream["adjusted_stream_cycles"]) if write_stream is not None else 0.0
    bank_timeline = _simulate_per_bank_timeline(
        read_requests=all_read_requests,
        write_requests=write_requests,
        hw=hw,
        perf=perf,
    )
    memory_timeline_cycles = float(bank_timeline["memory_timeline_cycles"])
    memory_access_bound_cycles = memory_timeline_cycles
    lower_bound = max(compute_cycles, memory_access_bound_cycles, ag_issue_bound_cycles)
    if op_type == "ring_gemm_fp16_fp16_fp16":
        read_nonoverlap = read_cycles * (1.0 - (_read_overlap_ratio(op_type, perf)))
        latency = max(compute_cycles, read_cycles, write_cycles) + read_nonoverlap + write_cycles + conflict_penalty
    else:
        read_nonoverlap = 0.0
        latency = max(compute_cycles, memory_timeline_cycles, ag_issue_bound_cycles)

    bytes_read = sum(int(stream["request_count"]) * (hw.block_bits // 8) for stream in read_streams)
    bytes_written = int(write_stream["request_count"]) * (hw.block_bits // 8) if write_stream else 0
    total_bytes = bytes_read + bytes_written
    arithmetic_intensity = (
        float(work_ops) / total_bytes
        if total_bytes
        else None
    )
    true_roofline = _true_roofline_from_totals(
        work_ops=work_ops,
        total_bytes=total_bytes,
        peak_compute_ops_per_cycle=peak_compute_ops,
        peak_memory_bandwidth_bytes_per_cycle=hw.peak_memory_bandwidth_bytes_per_cycle,
    )

    return {
        "kind": "op",
        "stage_name": op_name,
        "op_name": op_name,
        "op_type": op_type,
        "hardware_measured_cycles": hardware_measured_cycles,
        "latency_cycles": latency,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "total_bytes": total_bytes,
        "work_ops": work_ops,
        "arithmetic_intensity": arithmetic_intensity,
        "true_roofline": true_roofline,
        "analytical_model": {
            "estimated_latency_cycles": latency,
            "compute_bound_cycles": compute_cycles,
            "memory_access_bound_cycles": memory_access_bound_cycles,
            "ag_issue_bound_cycles": ag_issue_bound_cycles,
            "lower_bound_cycles": lower_bound,
            "latency_to_lower_bound_ratio": (latency / lower_bound) if lower_bound else None,
        },
        "streams": streams,
        "edge_breakdown": edge_breakdown,
        "bank_timeline": bank_timeline,
        "address_transforms": _collect_unique_transforms_from_requests(
            [request for stream_requests in per_stream_requests for request in stream_requests]
        ),
        "pre_stages": pre_stages,
        "request_trace": [
            request
            for stream_requests in per_stream_requests
            for request in stream_requests
        ],
    }


def _analyze_ring_gemm_op(
    op_name: str,
    op_type: str,
    op_data: Mapping[str, object],
    streams: Sequence[Mapping[str, object]],
    edge_breakdown: Sequence[Mapping[str, object]],
    per_stream_requests: Sequence[Sequence[PhysicalRequest]],
    pre_stages: Sequence[Mapping[str, object]],
    write_stream: Optional[Mapping[str, object]],
    hw: HardwareSpec,
    perf: PerformanceConfig,
) -> Dict[str, object]:
    a_streams = [stream for stream in streams if stream["role"] == "A"]
    b_streams = [stream for stream in streams if stream["role"] == "B"]
    write_cycles = float(write_stream["adjusted_stream_cycles"]) if write_stream is not None else 0.0
    request_trace = [
        request
        for stream_requests in per_stream_requests
        for request in stream_requests
    ]
    local_a_requests = [request for request in request_trace if request.role == "A"]
    b_requests = [request for request in request_trace if request.role == "B"]
    write_requests = [request for request in request_trace if request.role == "writeback"]

    geometry = _ring_gemm_execution_geometry(op_data, hw)
    ring_participants = geometry.ring_participants
    m_dim = geometry.m_dim
    n_dim = geometry.n_dim
    local_k_dim = geometry.local_k_dim
    peak_compute_ops = hw.gemm_peak_ops_per_cycle
    work_ops = float(hw.compute.gemm_core.mac_ops * m_dim * n_dim * geometry.total_k_dim)
    per_output_tile_work_ops = float(
        hw.compute.gemm_core.mac_ops
        * geometry.output_tile_m
        * geometry.output_tile_n
        * geometry.total_k_dim
    )
    per_pe_micro_op_work_ops = float(
        hw.compute.gemm_core.mac_ops
        * geometry.pe_m
        * geometry.pe_n
        * geometry.pe_k
    )
    per_pe_micro_op_compute_cycles = math.ceil(per_pe_micro_op_work_ops / max(1, peak_compute_ops))
    compute_cycles = work_ops / max(1, peak_compute_ops)

    local_a_bytes = _ring_local_a_bytes(op_data, hw)
    ring_a_total_bytes = local_a_bytes * max(0, ring_participants - 1)
    ring_bandwidth_bytes_per_cycle = _ring_bandwidth_bytes_per_cycle()
    per_a_buffer_ring_transfer_cycles = (
        math.ceil(geometry.a_buffer_bytes / ring_bandwidth_bytes_per_cycle)
        if ring_participants > 1 and geometry.a_buffer_bytes
        else 0.0
    )
    ring_a_transfer_cycles = (
        per_a_buffer_ring_transfer_cycles
        * max(0, ring_participants - 1)
        * geometry.a_buffer_tile_count
    )
    local_a_read_cycles = max((float(stream["adjusted_stream_cycles"]) for stream in a_streams), default=0.0)
    a_buffer_tile_requests = _split_requests_exact(
        local_a_requests,
        geometry.a_buffer_tile_count,
        "ring_gemm local A buffer tiles",
    )
    b_buffer_tile_requests = _split_requests_exact(
        b_requests,
        geometry.b_buffer_tile_count,
        "ring_gemm B buffer tiles",
    )
    output_writeback_tile_requests = _split_requests_exact(
        write_requests,
        geometry.output_writeback_tile_count,
        "ring_gemm output writeback tiles",
    )
    ring_bank_timeline, ring_microtile_timeline, ring_timeline_cycles, compute_pipeline_completion = _simulate_ring_microtile_timeline(
        a_buffer_tile_requests=a_buffer_tile_requests,
        b_buffer_tile_requests=b_buffer_tile_requests,
        output_writeback_tile_requests=output_writeback_tile_requests,
        geometry=geometry,
        per_pe_micro_op_compute_cycles=per_pe_micro_op_compute_cycles,
        per_a_buffer_ring_transfer_cycles=per_a_buffer_ring_transfer_cycles,
        hw=hw,
    )
    memory_access_bound_cycles = float(ring_bank_timeline["memory_timeline_cycles"])
    ag_issue_bound_cycles = max((float(stream["issue_cycles"]) for stream in streams), default=0.0)
    ping_pong_startup_cycles = local_a_read_cycles
    ping_pong_steady_cycles = 0.0
    if geometry.a_buffer_tile_count > 0:
        ping_pong_steady_cycles = compute_pipeline_completion - ping_pong_startup_cycles
    ping_pong_pipeline_cycles = ping_pong_startup_cycles + ping_pong_steady_cycles
    lower_bound = max(
        compute_cycles,
        memory_access_bound_cycles,
        ring_timeline_cycles,
        write_cycles,
        ag_issue_bound_cycles,
    )
    latency = max(
        compute_pipeline_completion,
        memory_access_bound_cycles,
        ag_issue_bound_cycles,
        ring_timeline_cycles,
    )

    read_streams = [stream for stream in streams if stream["role"] != "writeback"]
    bytes_read = sum(int(stream["request_count"]) * (hw.block_bits // 8) for stream in read_streams)
    bytes_written = int(write_stream["request_count"]) * (hw.block_bits // 8) if write_stream else 0
    total_bytes = bytes_read + bytes_written
    arithmetic_intensity = (float(work_ops) / total_bytes) if total_bytes else None
    true_roofline = _true_roofline_from_totals(
        work_ops=work_ops,
        total_bytes=total_bytes,
        peak_compute_ops_per_cycle=peak_compute_ops,
        peak_memory_bandwidth_bytes_per_cycle=hw.peak_memory_bandwidth_bytes_per_cycle,
    )

    return {
        "kind": "op",
        "stage_name": op_name,
        "op_name": op_name,
        "op_type": op_type,
        "hardware_measured_cycles": _coerce_optional_positive_float(op_data.get("hardware_measured")),
        "latency_cycles": latency,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "total_bytes": total_bytes,
        "work_ops": work_ops,
        "arithmetic_intensity": arithmetic_intensity,
        "ring_participants": ring_participants,
        "local_a_bytes": local_a_bytes,
        "ring_a_total_bytes": ring_a_total_bytes,
        "microtile_bytes": geometry.a_buffer_bytes,
        "microtile_k": geometry.a_buffer_k,
        "a_buffer_bytes": geometry.a_buffer_bytes,
        "b_buffer_bytes": geometry.b_buffer_bytes,
        "a_reuse_factor": geometry.a_reuse_factor,
        "b_reuse_factor": geometry.b_reuse_factor,
        "output_tile_m": geometry.output_tile_m,
        "output_tile_n": geometry.output_tile_n,
        "pe_micro_ops_per_output_tile": geometry.pe_micro_ops_per_output_tile,
        "local_a_read_cycles": local_a_read_cycles,
        "per_tile_compute_cycles": math.ceil(per_output_tile_work_ops / max(1, peak_compute_ops)),
        "per_tile_ring_a_transfer_cycles": math.ceil(geometry.a_buffer_bytes / ring_bandwidth_bytes_per_cycle) if ring_participants > 1 and geometry.a_buffer_bytes else 0.0,
        "ping_pong_startup_cycles": ping_pong_startup_cycles,
        "ping_pong_steady_cycles": ping_pong_steady_cycles,
        "ping_pong_pipeline_cycles": ping_pong_pipeline_cycles,
        "ring_a_transfer_cycles": ring_a_transfer_cycles,
        "ring_bandwidth_bytes_per_cycle": ring_bandwidth_bytes_per_cycle,
        "true_roofline": true_roofline,
        "analytical_model": {
            "estimated_latency_cycles": latency,
            "compute_bound_cycles": compute_cycles,
            "memory_access_bound_cycles": memory_access_bound_cycles,
            "ag_issue_bound_cycles": ag_issue_bound_cycles,
            "ring_transfer_bound_cycles": ring_timeline_cycles,
            "lower_bound_cycles": lower_bound,
            "latency_to_lower_bound_ratio": (latency / lower_bound) if lower_bound else None,
        },
        "ring_tile_timeline": _summarize_ring_participant_timeline(
            ring_microtile_timeline=ring_microtile_timeline,
            ring_participants=ring_participants,
            microtile_count_per_participant=geometry.a_buffer_tile_count,
        ),
        "ring_microtile_timeline": ring_microtile_timeline,
        "a_buffer_timeline": ring_microtile_timeline["a_buffer_timeline"],
        "b_buffer_timeline": ring_microtile_timeline["b_buffer_timeline"],
        "pe_compute_timeline": ring_microtile_timeline["pe_compute_timeline"],
        "psum_timeline": ring_microtile_timeline["psum_timeline"],
        "output_writeback_timeline": ring_microtile_timeline["output_writeback_timeline"],
        "ring_bank_timeline": ring_bank_timeline,
        "streams": list(streams),
        "edge_breakdown": list(edge_breakdown),
        "address_transforms": _collect_unique_transforms_from_requests(
            request_trace
        ),
        "pre_stages": list(pre_stages),
        "request_trace": request_trace,
    }


def _requests_from_edge_result(
    result: EdgeSolveResult,
    mode: str,
    hw: HardwareSpec,
    perf: PerformanceConfig,
    tensor_name: str,
    role: str,
    ag_ids: Sequence[str],
    base_addr: int,
) -> List[PhysicalRequest]:
    consumer_bits = list(result.consumer_visible_outer_bits or [])
    if not consumer_bits:
        return []
    address_transform = AddressTransform.from_edge_result(result.to_dict(), mode, hw)
    return _materialize_requests(
        tensor_name=tensor_name,
        edge_name=f"{result.producer}->{result.consumer}:{tensor_name}",
        logical_labels=consumer_bits,
        address_transform=address_transform,
        base_addr=base_addr,
        hw=hw,
        perf=perf,
        role=role,
        ag_ids=ag_ids,
    )


def _requests_from_source_tensor(
    tensor_name: str,
    port_data: Mapping[str, object],
    mode: str,
    hw: HardwareSpec,
    perf: PerformanceConfig,
    role: str,
    ag_ids: Sequence[str],
    base_addr: int,
) -> List[PhysicalRequest]:
    dtype = str(port_data["layout"].dtype)
    resolved_shape = {str(k): int(v) for k, v in dict(port_data["resolved_shape"]).items()}
    num_requests = _num_requests_from_shape(dtype, resolved_shape, hw)
    return _materialize_sequential_requests(
        tensor_name=tensor_name,
        edge_name=f"source->{tensor_name}",
        num_requests=num_requests,
        address_transform=AddressTransform.identity(
            _default_logical_bit_labels(min(hw.remap_bits, max(1, num_requests.bit_length())))
        ),
        base_addr=base_addr,
        hw=hw,
        perf=perf,
        role=role,
        ag_ids=ag_ids,
    )


def _requests_from_output_tensor(
    tensor_name: str,
    output_data: Mapping[str, object],
    mode: str,
    hw: HardwareSpec,
    perf: PerformanceConfig,
    base_addr: int,
) -> List[PhysicalRequest]:
    dtype = str(output_data["layout"].dtype)
    resolved_shape = {str(k): int(v) for k, v in dict(output_data["resolved_shape"]).items()}
    num_requests = _num_requests_from_shape(dtype, resolved_shape, hw)
    return _materialize_sequential_requests(
        tensor_name=tensor_name,
        edge_name=f"{tensor_name}:writeback",
        num_requests=num_requests,
        address_transform=AddressTransform.identity(
            _default_logical_bit_labels(min(hw.remap_bits, max(1, num_requests.bit_length())))
        ),
        base_addr=base_addr,
        hw=hw,
        perf=perf,
        role="writeback",
        ag_ids=("ag4",),
    )


def _materialize_requests(
    tensor_name: str,
    edge_name: str,
    logical_labels: Sequence[str],
    address_transform: AddressTransform,
    base_addr: int,
    hw: HardwareSpec,
    perf: PerformanceConfig,
    role: str,
    ag_ids: Sequence[str],
) -> List[PhysicalRequest]:
    num_requests = 1 << len(logical_labels)
    requests: List[PhysicalRequest] = []
    for req_id in range(num_requests):
        label_values = {
            logical_labels[bit_index]: (req_id >> bit_index) & 1 for bit_index in range(len(logical_labels))
        }
        logical_addr = 0
        for bit_index, label in enumerate(logical_labels):
            logical_addr |= label_values.get(label, 0) << bit_index
        ag_id = str(ag_ids[req_id % len(ag_ids)])
        physical_addr = compose_physical_address(
            base_addr=base_addr,
            logical_addr=logical_addr,
            transform=address_transform,
            hw=hw,
        )
        decoded = _decode_request_address(physical_addr=physical_addr, hw=hw)
        requests.append(
            PhysicalRequest(
                request_id=req_id,
                tensor_name=tensor_name,
                edge_name=edge_name,
                ag_id=ag_id,
                role=role,
                logical_addr=logical_addr,
                base_addr=base_addr,
                address_transform=address_transform.to_dict(),
                physical_addr=physical_addr,
                slice_id=decoded["slice_id"],
                bank_id=decoded["bank_id"],
                row_id=decoded["row_id"],
                col_id=decoded["col_id"],
            )
        )
    return requests


def _materialize_sequential_requests(
    tensor_name: str,
    edge_name: str,
    num_requests: int,
    address_transform: AddressTransform,
    base_addr: int,
    hw: HardwareSpec,
    perf: PerformanceConfig,
    role: str,
    ag_ids: Sequence[str],
) -> List[PhysicalRequest]:
    requests: List[PhysicalRequest] = []
    for req_id in range(num_requests):
        ag_id = str(ag_ids[req_id % len(ag_ids)])
        logical_addr = req_id
        physical_addr = compose_physical_address(
            base_addr=base_addr,
            logical_addr=logical_addr,
            transform=address_transform,
            hw=hw,
        )
        decoded = _decode_request_address(physical_addr=physical_addr, hw=hw)
        requests.append(
            PhysicalRequest(
                request_id=req_id,
                tensor_name=tensor_name,
                edge_name=edge_name,
                ag_id=ag_id,
                role=role,
                logical_addr=logical_addr,
                base_addr=base_addr,
                address_transform=address_transform.to_dict(),
                physical_addr=physical_addr,
                slice_id=decoded["slice_id"],
                bank_id=decoded["bank_id"],
                row_id=decoded["row_id"],
                col_id=decoded["col_id"],
            )
        )
    return requests


def _build_stream_reports(
    requests: Sequence[PhysicalRequest],
    ag_ids: Sequence[str],
    tensor_name: str,
    edge_name: str,
    role: str,
    mode: str,
    hw: HardwareSpec,
    perf: PerformanceConfig,
) -> List[Dict[str, object]]:
    by_ag: Dict[str, List[PhysicalRequest]] = defaultdict(list)
    for request in requests:
        by_ag[request.ag_id].append(request)

    reports: List[Dict[str, object]] = []
    for ag_id in ag_ids:
        ag_requests = by_ag.get(str(ag_id), [])
        reports.append(
            _analyze_request_stream(
                ag_id=str(ag_id),
                tensor_name=tensor_name,
                edge_name=edge_name,
                role=role,
                mode=mode,
                requests=ag_requests,
                hw=hw,
                perf=perf,
            )
        )
    return reports


def _request_phase(role: str) -> str:
    return "write" if role == "writeback" else "read"


def _bank_request_delta(
    state: _BankTimelineState,
    request: PhysicalRequest,
    hw: HardwareSpec,
) -> float:
    phase = _request_phase(request.role)
    if state.phase is None:
        return hw.request_latency_cycles

    if phase != state.phase:
        return hw.row_switch_penalty_cycles + hw.bank_return_interval_cycles
    if state.open_row == request.row_id:
        return hw.bank_return_interval_cycles
    return hw.row_switch_penalty_cycles + hw.bank_return_interval_cycles


def _apply_bank_request(
    state: _BankTimelineState,
    request: PhysicalRequest,
    hw: HardwareSpec,
) -> float:
    phase = _request_phase(request.role)
    if state.phase is None:
        state.phase = phase
        state.open_row = request.row_id
        if phase == "read":
            state.read_request_count += 1
        else:
            state.write_request_count += 1
        return hw.request_latency_cycles

    delta = _bank_request_delta(state, request, hw)
    if phase != state.phase:
        state.phase_switch_count += 1
        state.phase = phase
        state.open_row = request.row_id
    elif state.open_row == request.row_id:
        pass
    else:
        state.row_switch_count += 1
        state.open_row = request.row_id

    if phase == "read":
        state.read_request_count += 1
    else:
        state.write_request_count += 1
    return delta


def _release_ready_writes(
    write_release_order: Sequence[PhysicalRequest],
    write_ready_by_bank: Mapping[int, deque[PhysicalRequest]],
    released_count: int,
    target_release_count: int,
) -> Tuple[int, int]:
    occupancy_increase = 0
    while released_count < min(len(write_release_order), target_release_count):
        request = write_release_order[released_count]
        write_ready_by_bank[request.bank_id].append(request)
        released_count += 1
        occupancy_increase += 1
    return released_count, occupancy_increase


def _release_timed_requests(
    pending: List[_TimedPhysicalRequest],
    ready_by_bank: Mapping[int, deque[_TimedPhysicalRequest]],
    now: float,
) -> List[_TimedPhysicalRequest]:
    still_pending: List[_TimedPhysicalRequest] = []
    released: List[_TimedPhysicalRequest] = []
    for entry in pending:
        if entry.release_cycle <= now:
            ready_by_bank[entry.request.bank_id].append(entry)
            released.append(entry)
        else:
            still_pending.append(entry)
    pending[:] = still_pending
    return released


def _split_requests_evenly(
    requests: Sequence[PhysicalRequest],
    parts: int,
) -> List[List[PhysicalRequest]]:
    if parts <= 0:
        return []
    ordered = sorted(requests, key=lambda request: request.request_id)
    base, extra = divmod(len(ordered), parts)
    chunks: List[List[PhysicalRequest]] = []
    cursor = 0
    for part_idx in range(parts):
        chunk_size = base + (1 if part_idx < extra else 0)
        chunks.append(list(ordered[cursor:cursor + chunk_size]))
        cursor += chunk_size
    return chunks


def _split_requests_exact(
    requests: Sequence[PhysicalRequest],
    parts: int,
    label: str,
) -> List[List[PhysicalRequest]]:
    if parts <= 0:
        raise ValueError(f"{label} split count must be positive.")
    ordered = sorted(requests, key=lambda request: request.request_id)
    if len(ordered) % parts != 0:
        raise ValueError(
            f"{label} request count {len(ordered)} is not divisible by split count {parts}."
        )
    chunk_size = len(ordered) // parts
    return [
        list(ordered[idx * chunk_size:(idx + 1) * chunk_size])
        for idx in range(parts)
    ]


def _bank_timeline_summary(
    bank_states: Mapping[int, _BankTimelineState],
    *,
    forced_drain_count: int,
    write_buffer_bytes: int,
    hw: HardwareSpec,
    write_queue_depth: Optional[int] = None,
    q_w_full_cycles: float = 0.0,
    slice_blocked_cycles: float = 0.0,
    rw_switch_count: int = 0,
    row_switch_count: Optional[int] = None,
    arbiter1_wins: int = 0,
    arbiter2_wins: int = 0,
    memory_timeline_cycles: Optional[float] = None,
) -> Dict[str, object]:
    phase_switch_penalty_cycles = sum(
        state.phase_switch_count * hw.row_switch_penalty_cycles for state in bank_states.values()
    )
    row_switch_penalty_cycles = sum(
        state.row_switch_count * hw.row_switch_penalty_cycles for state in bank_states.values()
    )
    completion_cycles = {
        str(bank_id): state.cycles for bank_id, state in bank_states.items()
    }
    return {
        "write_buffer_bytes": write_buffer_bytes,
        "write_queue_depth": write_queue_depth,
        "read_priority_policy": "closed_loop_two_level_arbiter_with_write_backpressure",
        "forced_drain_count": forced_drain_count,
        "memory_timeline_cycles": (
            memory_timeline_cycles
            if memory_timeline_cycles is not None
            else max((state.cycles for state in bank_states.values()), default=0.0)
        ),
        "phase_switch_penalty_cycles": phase_switch_penalty_cycles,
        "row_switch_penalty_cycles": row_switch_penalty_cycles,
        "q_w_full_cycles": q_w_full_cycles,
        "slice_blocked_cycles": slice_blocked_cycles,
        "rw_switch_count": rw_switch_count,
        "row_switch_count": row_switch_count
        if row_switch_count is not None
        else sum(state.row_switch_count for state in bank_states.values()),
        "arbiter1_wins": arbiter1_wins,
        "arbiter2_wins": arbiter2_wins,
        "arbiter1_win_rate": (
            arbiter1_wins / max(1, arbiter1_wins + arbiter2_wins)
            if (arbiter1_wins + arbiter2_wins) > 0
            else 0.0
        ),
        "per_bank_completion_cycles": completion_cycles,
        "per_bank_breakdown": {
            str(bank_id): {
                "final_cycles": state.cycles,
                "phase": state.phase,
                "open_row": state.open_row,
                "row_switch_count": state.row_switch_count,
                "phase_switch_count": state.phase_switch_count,
                "read_request_count": state.read_request_count,
                "write_request_count": state.write_request_count,
            }
            for bank_id, state in bank_states.items()
        },
    }


def _simulate_per_bank_timeline(
    read_requests: Sequence[PhysicalRequest],
    write_requests: Sequence[PhysicalRequest],
    hw: HardwareSpec,
    perf: PerformanceConfig,
) -> Dict[str, object]:
    def _build_arrival_timeline(requests: Sequence[PhysicalRequest]) -> List[_TimedPhysicalRequest]:
        issue_slots_by_ag: Dict[str, int] = defaultdict(int)
        timed: List[_TimedPhysicalRequest] = []
        for request in sorted(requests, key=lambda item: item.request_id):
            issue_slot = issue_slots_by_ag[request.ag_id]
            issue_slots_by_ag[request.ag_id] += 1
            timed.append(
                _TimedPhysicalRequest(
                    request=request,
                    release_cycle=(issue_slot / max(1, hw.ag_issue_rate)),
                    group_key=f"{request.role}:{request.ag_id}",
                )
            )
        timed.sort(key=lambda item: (item.release_cycle, item.request.request_id))
        return timed

    def _pick_bank_candidate(
        queue: deque[PhysicalRequest],
        bank_state: _BankTimelineState,
        *,
        force_write_drain: bool,
    ) -> Optional[Tuple[int, PhysicalRequest, str]]:
        if not queue:
            return None

        phase = bank_state.phase
        open_row = bank_state.open_row
        arbiter1: List[Tuple[int, PhysicalRequest]] = []
        arbiter2: List[Tuple[int, PhysicalRequest]] = []
        for idx, request in enumerate(queue):
            req_phase = _request_phase(request.role)
            if force_write_drain and req_phase != "write":
                continue
            if phase is not None and open_row is not None and req_phase == phase and request.row_id == open_row:
                arbiter1.append((idx, request))
            else:
                arbiter2.append((idx, request))

        if arbiter1:
            idx, request = min(arbiter1, key=lambda item: item[1].request_id)
            return idx, request, "arbiter1"
        if arbiter2:
            idx, request = min(arbiter2, key=lambda item: item[1].request_id)
            return idx, request, "arbiter2"
        return None

    def _advance_time(
        controller_state: _ClosedLoopControllerState,
        *,
        next_now: float,
        queue_depth: int,
        slice_blocked: bool,
    ) -> None:
        if next_now <= controller_state.now:
            return
        delta = next_now - controller_state.now
        if controller_state.write_queue_occupancy >= queue_depth:
            controller_state.q_w_full_cycles += delta
        if slice_blocked:
            controller_state.slice_blocked_cycles += delta
        controller_state.now = next_now

    if not read_requests and not write_requests:
        return _bank_timeline_summary(
            {},
            forced_drain_count=0,
            write_buffer_bytes=hw.write_buffer_bytes,
            hw=hw,
            write_queue_depth=perf.controller_write_queue_depth,
            memory_timeline_cycles=0.0,
        )

    read_arrivals = _build_arrival_timeline(read_requests)
    write_arrivals = _build_arrival_timeline(write_requests)
    read_idx = 0
    write_idx = 0
    slice_pending_writes: deque[PhysicalRequest] = deque()
    controller_ready_by_bank: Dict[int, deque[PhysicalRequest]] = defaultdict(deque)

    all_banks = sorted({request.bank_id for request in [*read_requests, *write_requests]})
    bank_states: Dict[int, _BankTimelineState] = {bank_id: _BankTimelineState() for bank_id in all_banks}
    controller_state = _ClosedLoopControllerState()
    write_queue_depth = max(1, perf.controller_write_queue_depth)
    slice_write_buffer_depth = max(1, perf.slice_write_buffer_depth)
    slice_blocked = False

    def _release_due_requests() -> None:
        nonlocal read_idx, write_idx, slice_blocked
        now = controller_state.now

        while write_idx < len(write_arrivals) and write_arrivals[write_idx].release_cycle <= now:
            slice_pending_writes.append(write_arrivals[write_idx].request)
            write_idx += 1

        while (
            slice_pending_writes
            and controller_state.write_queue_occupancy < write_queue_depth
        ):
            request = slice_pending_writes.popleft()
            controller_ready_by_bank[request.bank_id].append(request)
            controller_state.write_queue_occupancy += 1

        next_slice_blocked = len(slice_pending_writes) >= slice_write_buffer_depth
        if next_slice_blocked and not slice_blocked:
            controller_state.forced_drain_count += 1
        slice_blocked = next_slice_blocked

        if not slice_blocked:
            while read_idx < len(read_arrivals) and read_arrivals[read_idx].release_cycle <= now:
                request = read_arrivals[read_idx].request
                controller_ready_by_bank[request.bank_id].append(request)
                read_idx += 1

    while True:
        _release_due_requests()

        has_ready_requests = any(queue for queue in controller_ready_by_bank.values())
        has_future_requests = (
            read_idx < len(read_arrivals)
            or write_idx < len(write_arrivals)
            or bool(slice_pending_writes)
        )
        if not has_ready_requests and not has_future_requests:
            break

        force_write_drain = slice_blocked
        candidate_events: List[Tuple[float, float, int, int, PhysicalRequest, str]] = []
        for bank_id in all_banks:
            queue = controller_ready_by_bank[bank_id]
            if not queue:
                continue
            selected = _pick_bank_candidate(
                queue,
                bank_states[bank_id],
                force_write_drain=force_write_drain,
            )
            if selected is None:
                continue
            queue_idx, request, arbiter = selected
            start_cycle = max(bank_states[bank_id].cycles, controller_state.now)
            delta = _bank_request_delta(bank_states[bank_id], request, hw)
            finish_cycle = start_cycle + delta
            candidate_events.append((finish_cycle, start_cycle, bank_id, queue_idx, request, arbiter))

        if not candidate_events:
            next_release_times: List[float] = []
            if write_idx < len(write_arrivals):
                next_release_times.append(write_arrivals[write_idx].release_cycle)
            if not slice_blocked and read_idx < len(read_arrivals):
                next_release_times.append(read_arrivals[read_idx].release_cycle)
            if not next_release_times:
                break
            _advance_time(
                controller_state,
                next_now=min(next_release_times),
                queue_depth=write_queue_depth,
                slice_blocked=slice_blocked,
            )
            continue

        finish_cycle, start_cycle, bank_id, queue_idx, request, arbiter = min(
            candidate_events,
            key=lambda item: (item[0], item[2], item[4].request_id),
        )
        _advance_time(
            controller_state,
            next_now=finish_cycle,
            queue_depth=write_queue_depth,
            slice_blocked=slice_blocked,
        )

        queue = controller_ready_by_bank[bank_id]
        if queue_idx == 0:
            queue.popleft()
        else:
            compact = list(queue)
            compact.pop(queue_idx)
            queue.clear()
            queue.extend(compact)

        bank_state = bank_states[bank_id]
        previous_phase_switches = bank_state.phase_switch_count
        previous_row_switches = bank_state.row_switch_count
        bank_state.cycles = start_cycle
        bank_state.cycles += _apply_bank_request(bank_state, request, hw)
        controller_state.rw_switch_count += bank_state.phase_switch_count - previous_phase_switches
        controller_state.row_switch_count += bank_state.row_switch_count - previous_row_switches
        if arbiter == "arbiter1":
            controller_state.arbiter1_wins += 1
        else:
            controller_state.arbiter2_wins += 1
        if _request_phase(request.role) == "write":
            controller_state.write_queue_occupancy = max(0, controller_state.write_queue_occupancy - 1)

    return _bank_timeline_summary(
        bank_states,
        forced_drain_count=controller_state.forced_drain_count,
        write_buffer_bytes=hw.write_buffer_bytes,
        hw=hw,
        write_queue_depth=write_queue_depth,
        q_w_full_cycles=controller_state.q_w_full_cycles,
        slice_blocked_cycles=controller_state.slice_blocked_cycles,
        rw_switch_count=controller_state.rw_switch_count,
        row_switch_count=controller_state.row_switch_count,
        arbiter1_wins=controller_state.arbiter1_wins,
        arbiter2_wins=controller_state.arbiter2_wins,
    )


def _run_ring_bank_event_loop(
    bank_states: Mapping[int, _BankTimelineState],
    future_reads: List[_TimedPhysicalRequest],
    future_writes: List[_TimedPhysicalRequest],
    completion_by_group: Mapping[str, float],
    pending_groups: Sequence[str],
    hw: HardwareSpec,
    state: Mapping[str, object],
) -> None:
    while True:
        if all(group_key in completion_by_group for group_key in pending_groups):
            return

        now = float(state["now"])
        read_ready_by_bank = state["read_ready_by_bank"]
        write_ready_by_bank = state["write_ready_by_bank"]
        released_writes = _release_timed_requests(future_writes, write_ready_by_bank, now)
        if released_writes:
            state["write_buffer_occupancy"] = int(state["write_buffer_occupancy"]) + len(released_writes)
        _release_timed_requests(future_reads, read_ready_by_bank, now)

        write_buffer_occupancy = int(state["write_buffer_occupancy"])
        in_forced_drain = bool(state["in_forced_drain"])
        if in_forced_drain and write_buffer_occupancy <= 0:
            state["in_forced_drain"] = False
            in_forced_drain = False

        released_reads_exist = any(read_ready_by_bank.values())
        released_writes_exist = any(write_ready_by_bank.values())
        if write_buffer_occupancy >= int(state["write_buffer_capacity_reqs"]) and not in_forced_drain:
            state["forced_drain_count"] = int(state["forced_drain_count"]) + 1
            state["in_forced_drain"] = True
            in_forced_drain = True

        target_ready_by_bank = (
            read_ready_by_bank
            if released_reads_exist and not in_forced_drain
            else write_ready_by_bank
            if released_writes_exist
            else None
        )
        if target_ready_by_bank is None:
            next_release = min(
                [entry.release_cycle for entry in [*future_reads, *future_writes]],
                default=None,
            )
            if next_release is None:
                return
            state["now"] = max(now, float(next_release))
            continue

        candidate_events: List[Tuple[float, int, _TimedPhysicalRequest]] = []
        for bank_id, queue in target_ready_by_bank.items():
            if not queue:
                continue
            entry = queue[0]
            bank_state = bank_states[bank_id]
            start_cycle = max(bank_state.cycles, entry.release_cycle)
            finish_cycle = start_cycle + _bank_request_delta(bank_state, entry.request, hw)
            candidate_events.append((finish_cycle, bank_id, entry))

        if not candidate_events:
            state["now"] = max(
                now,
                min([entry.release_cycle for entry in [*future_reads, *future_writes]], default=now),
            )
            continue

        finish_cycle, bank_id, entry = min(
            candidate_events,
            key=lambda item: (item[0], item[1], item[2].request.request_id),
        )
        bank_state = bank_states[bank_id]
        bank_state.cycles = max(bank_state.cycles, entry.release_cycle)
        target_ready_by_bank[bank_id].popleft()
        bank_state.cycles += _apply_bank_request(bank_state, entry.request, hw)
        completion_by_group[entry.group_key] = max(completion_by_group.get(entry.group_key, 0.0), finish_cycle)
        if _request_phase(entry.request.role) == "write":
            state["write_buffer_occupancy"] = max(0, int(state["write_buffer_occupancy"]) - 1)
        state["now"] = finish_cycle


def _simulate_ring_microtile_timeline(
    a_buffer_tile_requests: Sequence[Sequence[PhysicalRequest]],
    b_buffer_tile_requests: Sequence[Sequence[PhysicalRequest]],
    output_writeback_tile_requests: Sequence[Sequence[PhysicalRequest]],
    geometry: _RingGemmExecutionGeometry,
    per_pe_micro_op_compute_cycles: float,
    per_a_buffer_ring_transfer_cycles: float,
    hw: HardwareSpec,
) -> Tuple[Dict[str, object], Dict[str, object], float, float]:
    all_requests = [
        *[request for tile_requests in a_buffer_tile_requests for request in tile_requests],
        *[request for tile_requests in b_buffer_tile_requests for request in tile_requests],
        *[request for tile_requests in output_writeback_tile_requests for request in tile_requests],
    ]
    if not all_requests:
        empty_bank_timeline = _bank_timeline_summary(
            {},
            forced_drain_count=0,
            write_buffer_bytes=hw.write_buffer_bytes,
            hw=hw,
            memory_timeline_cycles=0.0,
        )
        empty_timeline = {
            "microtile_bytes": geometry.a_buffer_bytes,
            "microtile_count_per_participant": geometry.a_buffer_tile_count,
            "total_compute_microtiles": geometry.total_coarse_compute_events,
            "ping_pong_assignment": [],
            "local_load_ready_cycles": [],
            "ring_ready_cycles": [],
            "b_ready_cycles": [],
            "compute_start_cycles": [],
            "compute_end_cycles": [],
            "final_write_release_cycle": 0.0,
            "a_buffer_timeline": {"tile_count": 0, "tile_bytes": geometry.a_buffer_bytes, "tiles": []},
            "b_buffer_timeline": {"tile_count": 0, "tile_bytes": geometry.b_buffer_bytes, "tiles": []},
            "pe_compute_timeline": {
                "micro_op_count": 0,
                "micro_ops_per_output_tile": geometry.pe_micro_ops_per_output_tile,
                "per_micro_op_compute_cycles": per_pe_micro_op_compute_cycles,
                "micro_op_start_cycles": [],
                "micro_op_end_cycles": [],
                "output_tile_completion_cycles": [],
            },
            "psum_timeline": {"tile_count": 0, "transfer_cycles_per_tile": 0.0, "tiles": []},
            "output_writeback_timeline": {"tile_count": 0, "tiles": []},
        }
        return empty_bank_timeline, empty_timeline, 0.0, 0.0

    bank_ids = sorted({request.bank_id for request in all_requests})
    bank_states: Dict[int, _BankTimelineState] = {bank_id: _BankTimelineState() for bank_id in bank_ids}
    future_reads: List[_TimedPhysicalRequest] = []
    future_writes: List[_TimedPhysicalRequest] = []
    runtime_state: Dict[str, object] = {
        "now": 0.0,
        "read_ready_by_bank": defaultdict(deque),
        "write_ready_by_bank": defaultdict(deque),
        "write_buffer_capacity_reqs": max(1, hw.write_buffer_bytes // max(1, hw.block_bits // 8)),
        "write_buffer_occupancy": 0,
        "forced_drain_count": 0,
        "in_forced_drain": False,
    }
    completion_by_group: Dict[str, float] = {}
    ring_link_completion_cycles = 0.0
    psum_transfer_cycles = math.ceil(
        geometry.output_tile_bytes / max(1, hw.write_buffer_bytes)
    ) if geometry.output_tile_bytes else 0.0

    a_slot_free_cycles = {"ping": 0.0, "pong": 0.0}
    b_slot_free_cycles = {"ping": 0.0, "pong": 0.0}
    psum_slot_free_cycles = {"ping": 0.0, "pong": 0.0}
    a_tile_ready_cycles: Dict[int, float] = {}
    b_tile_ready_cycles: Dict[int, float] = {}
    a_tile_last_consumer_cycles: Dict[int, float] = {}
    b_tile_last_consumer_cycles: Dict[int, float] = {}
    a_ring_ready_cycles_by_hop: Dict[Tuple[int, int], float] = {}

    a_timeline_tiles: List[Dict[str, object]] = []
    b_timeline_tiles: List[Dict[str, object]] = []
    coarse_local_load_ready_cycles: List[float] = []
    coarse_ring_ready_cycles: List[float] = []
    coarse_b_ready_cycles: List[float] = []
    coarse_compute_start_cycles: List[float] = []
    coarse_compute_end_cycles: List[float] = []
    coarse_ping_pong_assignment: List[str] = []

    pe_micro_op_start_cycles: List[float] = []
    pe_micro_op_end_cycles: List[float] = []
    pe_micro_op_metadata: List[Dict[str, int]] = []
    output_tile_completion_cycles: List[float] = []
    psum_tiles: List[Dict[str, object]] = []
    output_writeback_tiles: List[Dict[str, object]] = []
    pe_available_cycle = 0.0

    def _ensure_a_tile_loaded(a_tile_idx: int) -> None:
        if a_tile_idx in a_tile_ready_cycles:
            return
        slot_name = "ping" if a_tile_idx % 2 == 0 else "pong"
        group_key = f"a:{a_tile_idx}"
        release_cycle = a_slot_free_cycles[slot_name]
        future_reads.extend(
            _TimedPhysicalRequest(request=request, release_cycle=release_cycle, group_key=group_key)
            for request in sorted(a_buffer_tile_requests[a_tile_idx], key=lambda request: request.request_id)
        )
        _run_ring_bank_event_loop(
            bank_states,
            future_reads,
            future_writes,
            completion_by_group,
            [group_key],
            hw,
            runtime_state,
        )
        ready_cycle = completion_by_group.get(group_key, release_cycle)
        a_tile_ready_cycles[a_tile_idx] = ready_cycle
        a_timeline_tiles.append(
            {
                "tile_index": a_tile_idx,
                "buffer": slot_name,
                "load_release_cycle": release_cycle,
                "load_ready_cycle": ready_cycle,
            }
        )

    def _ensure_b_tile_loaded(b_tile_idx: int) -> None:
        if b_tile_idx in b_tile_ready_cycles:
            return
        slot_name = "ping" if b_tile_idx % 2 == 0 else "pong"
        group_key = f"b:{b_tile_idx}"
        release_cycle = b_slot_free_cycles[slot_name]
        future_reads.extend(
            _TimedPhysicalRequest(request=request, release_cycle=release_cycle, group_key=group_key)
            for request in sorted(b_buffer_tile_requests[b_tile_idx], key=lambda request: request.request_id)
        )
        _run_ring_bank_event_loop(
            bank_states,
            future_reads,
            future_writes,
            completion_by_group,
            [group_key],
            hw,
            runtime_state,
        )
        ready_cycle = completion_by_group.get(group_key, release_cycle)
        b_tile_ready_cycles[b_tile_idx] = ready_cycle
        b_timeline_tiles.append(
            {
                "tile_index": b_tile_idx,
                "buffer": slot_name,
                "load_release_cycle": release_cycle,
                "load_ready_cycle": ready_cycle,
            }
        )

    for k_tile_idx in range(geometry.total_k_tiles_per_output_tile):
        local_k_tile_idx = k_tile_idx // geometry.a_ring_hops_per_local_a_tile
        ring_hop_idx = k_tile_idx % geometry.a_ring_hops_per_local_a_tile
        for output_tile_n_idx in range(geometry.output_tile_count_n):
            b_tile_idx = output_tile_n_idx * geometry.total_k_tiles_per_output_tile + k_tile_idx
            _ensure_b_tile_loaded(b_tile_idx)
            b_ready_cycle = b_tile_ready_cycles[b_tile_idx]
            for output_tile_m_idx in range(geometry.output_tile_count_m):
                output_tile_idx = output_tile_n_idx * geometry.output_tile_count_m + output_tile_m_idx
                a_tile_idx = output_tile_m_idx * geometry.local_a_k_tiles_per_output_tile + local_k_tile_idx
                _ensure_a_tile_loaded(a_tile_idx)
                a_ready_cycle = a_tile_ready_cycles[a_tile_idx]
                ring_ready_cycle = a_ready_cycle + ring_hop_idx * per_a_buffer_ring_transfer_cycles
                a_ring_ready_cycles_by_hop[(a_tile_idx, ring_hop_idx)] = ring_ready_cycle
                ring_link_completion_cycles = max(ring_link_completion_cycles, ring_ready_cycle)

                coarse_local_load_ready_cycles.append(a_ready_cycle)
                coarse_ring_ready_cycles.append(ring_ready_cycle)
                coarse_b_ready_cycles.append(b_ready_cycle)
                coarse_ping_pong_assignment.append("ping" if a_tile_idx % 2 == 0 else "pong")

                output_psum_slot = "ping" if output_tile_idx % 2 == 0 else "pong"
                coarse_start = max(
                    pe_available_cycle,
                    ring_ready_cycle,
                    b_ready_cycle,
                    psum_slot_free_cycles[output_psum_slot] if k_tile_idx == 0 else 0.0,
                )
                coarse_compute_start_cycles.append(coarse_start)
                current_cycle = coarse_start
                for pe_m_idx in range(geometry.b_reuse_factor):
                    for pe_n_idx in range(geometry.a_reuse_factor):
                        pe_micro_op_start_cycles.append(current_cycle)
                        current_cycle += per_pe_micro_op_compute_cycles
                        pe_micro_op_end_cycles.append(current_cycle)
                        pe_micro_op_metadata.append(
                            {
                                "output_tile_index": output_tile_idx,
                                "k_tile_index": k_tile_idx,
                                "ring_hop_index": ring_hop_idx,
                                "output_tile_m_index": output_tile_m_idx,
                                "output_tile_n_index": output_tile_n_idx,
                                "pe_m_index": pe_m_idx,
                                "pe_n_index": pe_n_idx,
                            }
                        )
                coarse_compute_end_cycles.append(current_cycle)
                pe_available_cycle = current_cycle
                a_tile_last_consumer_cycles[a_tile_idx] = max(a_tile_last_consumer_cycles.get(a_tile_idx, 0.0), current_cycle)
                b_tile_last_consumer_cycles[b_tile_idx] = max(b_tile_last_consumer_cycles.get(b_tile_idx, 0.0), current_cycle)

                if ring_hop_idx == geometry.a_ring_hops_per_local_a_tile - 1:
                    a_slot_name = "ping" if a_tile_idx % 2 == 0 else "pong"
                    a_slot_free_cycles[a_slot_name] = max(a_slot_free_cycles[a_slot_name], a_tile_last_consumer_cycles[a_tile_idx])
                if output_tile_m_idx == geometry.output_tile_count_m - 1:
                    b_slot_name = "ping" if b_tile_idx % 2 == 0 else "pong"
                    b_slot_free_cycles[b_slot_name] = max(b_slot_free_cycles[b_slot_name], b_tile_last_consumer_cycles[b_tile_idx])

                if k_tile_idx == geometry.total_k_tiles_per_output_tile - 1:
                    output_tile_completion_cycles.append(current_cycle)
                    psum_start = current_cycle
                    psum_end = psum_start + psum_transfer_cycles
                    psum_tiles.append(
                        {
                            "tile_index": output_tile_idx,
                            "buffer": output_psum_slot,
                            "compute_complete_cycle": current_cycle,
                            "transfer_start_cycle": psum_start,
                            "transfer_end_cycle": psum_end,
                        }
                    )
                    psum_slot_free_cycles[output_psum_slot] = psum_end
                    write_group_key = f"w:{output_tile_idx}"
                    future_writes.extend(
                        _TimedPhysicalRequest(request=request, release_cycle=psum_end, group_key=write_group_key)
                        for request in sorted(output_writeback_tile_requests[output_tile_idx], key=lambda request: request.request_id)
                    )
                    output_writeback_tiles.append(
                        {
                            "tile_index": output_tile_idx,
                            "release_cycle": psum_end,
                            "buffer": output_psum_slot,
                        }
                    )

    remaining_groups = [entry.group_key for entry in future_writes]
    if remaining_groups:
        _run_ring_bank_event_loop(
            bank_states,
            future_reads,
            future_writes,
            completion_by_group,
            remaining_groups,
            hw,
            runtime_state,
        )
    _run_ring_bank_event_loop(
        bank_states,
        future_reads,
        future_writes,
        completion_by_group,
        [],
        hw,
        runtime_state,
    )

    memory_timeline_cycles = max((state.cycles for state in bank_states.values()), default=0.0)
    bank_timeline = _bank_timeline_summary(
        bank_states,
        forced_drain_count=int(runtime_state["forced_drain_count"]),
        write_buffer_bytes=hw.write_buffer_bytes,
        hw=hw,
        memory_timeline_cycles=memory_timeline_cycles,
    )
    microtile_timeline = {
        "microtile_bytes": geometry.a_buffer_bytes,
        "microtile_count_per_participant": geometry.a_buffer_tile_count,
        "total_compute_microtiles": geometry.total_coarse_compute_events,
        "ping_pong_assignment": coarse_ping_pong_assignment,
        "local_load_ready_cycles": coarse_local_load_ready_cycles,
        "ring_ready_cycles": coarse_ring_ready_cycles,
        "b_ready_cycles": coarse_b_ready_cycles,
        "compute_start_cycles": coarse_compute_start_cycles,
        "compute_end_cycles": coarse_compute_end_cycles,
        "final_write_release_cycle": max((tile["release_cycle"] for tile in output_writeback_tiles), default=0.0),
        "a_buffer_timeline": {
            "tile_count": geometry.a_buffer_tile_count,
            "tile_bytes": geometry.a_buffer_bytes,
            "tiles": a_timeline_tiles,
        },
        "b_buffer_timeline": {
            "tile_count": geometry.b_buffer_tile_count,
            "tile_bytes": geometry.b_buffer_bytes,
            "tiles": b_timeline_tiles,
        },
        "pe_compute_timeline": {
            "micro_op_count": len(pe_micro_op_end_cycles),
            "micro_ops_per_output_tile": geometry.pe_micro_ops_per_output_tile,
            "per_micro_op_compute_cycles": per_pe_micro_op_compute_cycles,
            "micro_op_start_cycles": pe_micro_op_start_cycles,
            "micro_op_end_cycles": pe_micro_op_end_cycles,
            "micro_ops": pe_micro_op_metadata,
            "output_tile_completion_cycles": output_tile_completion_cycles,
        },
        "psum_timeline": {
            "tile_count": geometry.output_tile_count,
            "transfer_cycles_per_tile": psum_transfer_cycles,
            "tiles": psum_tiles,
        },
        "output_writeback_timeline": {
            "tile_count": geometry.output_writeback_tile_count,
            "tiles": output_writeback_tiles,
        },
    }
    return bank_timeline, microtile_timeline, ring_link_completion_cycles, pe_available_cycle


def _analyze_request_stream(
    ag_id: str,
    tensor_name: str,
    edge_name: str,
    role: str,
    mode: str,
    requests: Sequence[PhysicalRequest],
    hw: HardwareSpec,
    perf: PerformanceConfig,
) -> Dict[str, object]:
    bank_sequence = [request.bank_id for request in requests]
    bank_transitions = sum(
        1 for left, right in zip(bank_sequence, bank_sequence[1:]) if left != right
    )
    round_robin_score = (
        bank_transitions / max(1, len(bank_sequence) - 1) if len(bank_sequence) > 1 else 0.0
    )
    same_bank_runs = _run_lengths(bank_sequence)

    per_bank_rows: Dict[int, List[int]] = defaultdict(list)
    for request in requests:
        per_bank_rows[request.bank_id].append(request.row_id)

    bank_cycles: Dict[int, float] = {}
    bank_stats: Dict[int, Dict[str, int]] = {}
    row_switch_penalty = 0.0
    for bank_id, row_sequence in per_bank_rows.items():
        hits = 0
        misses = 0
        empty = 0
        last_row: Optional[int] = None
        for row_id in row_sequence:
            if last_row is None:
                empty += 1
            elif row_id == last_row:
                hits += 1
            else:
                misses += 1
                row_switch_penalty += hw.row_switch_penalty_cycles
            last_row = row_id
        request_count = len(row_sequence)
        bank_cycles[bank_id] = (
            hw.request_latency_cycles + max(0, request_count - 1) * hw.bank_return_interval_cycles
            if request_count
            else 0.0
        )
        bank_stats[bank_id] = {"hits": hits, "misses": misses, "empty": empty}

    active_banks = len(bank_cycles)
    fifo_factor = min(1.0, hw.request_fifo_depth / max(1, hw.bank_count_per_slice))
    bank_spread_factor = (
        min(1.0, (active_banks - 1) / max(1, hw.bank_count_per_slice - 1))
        if hw.bank_count_per_slice > 1
        else 0.0
    )
    hidden_fraction = round_robin_score * fifo_factor * bank_spread_factor
    raw_bank_max = max(bank_cycles.values(), default=0.0)
    issue_cycles = math.ceil(len(requests) / max(1, hw.ag_issue_rate))
    exposed_row_switch_cycles = row_switch_penalty * (1.0 - hidden_fraction)
    adjusted_cycles = max(raw_bank_max, issue_cycles) + exposed_row_switch_cycles

    return {
        "ag_id": ag_id,
        "tensor_name": tensor_name,
        "edge_name": edge_name,
        "role": role,
        "mode": mode,
        "request_count": len(requests),
        "issue_cycles": issue_cycles,
        "bank_sequence_sample": bank_sequence[:16],
        "same_bank_run_lengths": same_bank_runs[:16],
        "bank_transition_count": bank_transitions,
        "round_robin_score": round_robin_score,
        "active_banks": active_banks,
        "bank_cycles": {str(k): v for k, v in bank_cycles.items()},
        "bank_stats": {str(k): v for k, v in bank_stats.items()},
        "raw_bank_max_cycles": raw_bank_max,
        "row_switch_penalty_cycles": row_switch_penalty,
        "row_switch_hiding_gain": row_switch_penalty * hidden_fraction,
        "exposed_row_switch_cycles": exposed_row_switch_cycles,
        "adjusted_stream_cycles": adjusted_cycles,
        "sample_requests": [request.to_dict() for request in requests[:8]],
    }


def _edge_reports_from_streams(stream_reports: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    for stream in stream_reports:
        reports.append(
            {
                "edge_name": str(stream["edge_name"]),
                "tensor_name": str(stream["tensor_name"]),
                "role": str(stream["role"]),
                "ag_id": str(stream["ag_id"]),
                "request_count": int(stream["request_count"]),
                "active_banks": int(stream["active_banks"]),
                "round_robin_score": float(stream["round_robin_score"]),
                "row_switch_penalty_cycles": float(stream["row_switch_penalty_cycles"]),
                "adjusted_stream_cycles": float(stream["adjusted_stream_cycles"]),
                "bank_sequence_sample": list(stream["bank_sequence_sample"]),
            }
        )
    return reports


def _estimate_compute(op_type: str, op_data: Mapping[str, object], hw: HardwareSpec) -> Tuple[float, float, int]:
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    output_port = next(iter(outputs))
    output_shape = {str(k): int(v) for k, v in dict(outputs[output_port]["resolved_shape"]).items()}
    op_work: float
    peak: int

    if op_type == "ring_gemm_fp16_fp16_fp16":
        in_a_shape = {str(k): int(v) for k, v in dict(inputs["inA"]["resolved_shape"]).items()}
        in_b_shape = {str(k): int(v) for k, v in dict(inputs["inB"]["resolved_shape"]).items()}
        m = int(in_a_shape.get("M", output_shape.get("M", 1)))
        k = int(in_a_shape.get("K", in_b_shape.get("K", 1)))
        n = int(in_b_shape.get("N", output_shape.get("N", 1)))
        op_work = float(hw.compute.gemm_core.mac_ops * m * n * k)
        peak = hw.gemm_peak_ops_per_cycle
    elif "remote_sum" in op_type:
        # Remote reductions do not currently encode the fan-in in the local op shape,
        # so retain the per-output accumulation approximation until that metadata exists.
        op_work = float(_shape_product(output_shape))
        peak = hw.general_peak_ops_per_cycle
    elif op_type in {"prefill_summac", "prefill_sum_rec", "prefill_max"}:
        input_port = next(iter(inputs))
        input_shape = {str(k): int(v) for k, v in dict(inputs[input_port]["resolved_shape"]).items()}
        input_elements = _shape_product(input_shape)
        output_elements = _shape_product(output_shape)
        # A tree-free reduction over R inputs into one output costs R-1 adds/comparisons.
        op_work = float(max(output_elements, input_elements - output_elements))
        peak = hw.general_peak_ops_per_cycle
    else:
        op_work = float(_shape_product(output_shape))
        peak = hw.general_peak_ops_per_cycle

    return op_work, math.ceil(op_work / max(1, peak)), peak


def _classify_input_stream(op_type: str, port_name: str) -> Tuple[str, Tuple[str, ...]]:
    if op_type == "ring_gemm_fp16_fp16_fp16":
        if port_name == "inA":
            return "A", ("ag0",)
        if port_name == "inB":
            return "B", ("ag1", "ag2")
    if "add_MN_N" in op_type and port_name == "inB":
        return "bias", ("ag3",)
    if port_name == "inA":
        return "A", ("ag0",)
    if port_name == "inB":
        return "B", ("ag1",)
    return "aux", ("ag1",)


def _decode_request_address(
    physical_addr: int,
    hw: HardwareSpec,
) -> Dict[str, int]:
    decoded = decode_physical_address(physical_addr, hw)
    decoded["bank_id"] = decoded["bank_id"] % hw.bank_count_per_slice
    return decoded


def _num_requests_from_shape(dtype: str, resolved_shape: Mapping[str, int], hw: HardwareSpec) -> int:
    total_elements = _shape_product(resolved_shape)
    bytes_per_element = hw.dtype_bits(dtype) // hw.address_unit_bits
    total_bytes = total_elements * bytes_per_element
    return math.ceil(total_bytes / (hw.block_bits // hw.address_unit_bits))


def _shape_product(shape: Mapping[str, int]) -> int:
    product = 1
    for value in shape.values():
        product *= int(value)
    return product


def _run_lengths(sequence: Sequence[int]) -> List[int]:
    if not sequence:
        return []
    lengths: List[int] = []
    current = sequence[0]
    count = 1
    for value in sequence[1:]:
        if value == current:
            count += 1
            continue
        lengths.append(count)
        current = value
        count = 1
    lengths.append(count)
    return lengths


def _read_overlap_ratio(op_type: str, perf: PerformanceConfig) -> float:
    return perf.gemm_read_overlap if op_type == "ring_gemm_fp16_fp16_fp16" else perf.general_read_overlap


def _ring_participants(op_data: Mapping[str, object]) -> int:
    ring_scope = str(dict(op_data.get("call_kwargs", {})).get("ring_scope", "cluster"))
    return 28 if ring_scope == "global" else 4


def _ring_bandwidth_bytes_per_cycle() -> float:
    return 256.0 / 8.0


def _ring_local_a_bytes(op_data: Mapping[str, object], hw: HardwareSpec) -> int:
    inputs = dict(op_data["inputs"])
    in_a = dict(inputs["inA"])
    resolved_shape = {str(k): int(v) for k, v in dict(in_a["resolved_shape"]).items()}
    dtype = str(in_a["layout"].dtype)
    return _shape_product(resolved_shape) * (hw.dtype_bits(dtype) // hw.address_unit_bits)


def _factor_extent_product(
    layout: LayoutSpec,
    resolved_shape: Mapping[str, int],
    axis: str,
) -> int:
    extent = 1
    for factor in layout.factors:
        if factor.parent_axis != axis or factor.kind != "tile":
            continue
        value = factor.extent_value(dict(resolved_shape))
        if value <= 0:
            raise ValueError(f"Layout tile factor {factor.name} on axis {axis} must be positive.")
        extent *= value
    return extent


def _fit_buffer_axis(
    dim: int,
    nominal_tile: int,
    *,
    label: str,
) -> Tuple[int, int]:
    if nominal_tile <= 0:
        raise ValueError(f"{label} nominal tile must be positive.")
    if dim <= 0:
        raise ValueError(f"{label} dimension must be positive.")
    if dim <= nominal_tile:
        return dim, 1
    if dim % nominal_tile != 0:
        raise ValueError(
            f"{label} dimension {dim} requires splitting by nominal tile {nominal_tile}, but the tiling is not integral."
        )
    return nominal_tile, dim // nominal_tile


def _ring_gemm_execution_geometry(
    op_data: Mapping[str, object],
    hw: HardwareSpec,
) -> _RingGemmExecutionGeometry:
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    in_a = dict(inputs["inA"])
    in_b = dict(inputs["inB"])
    output_port = next(iter(outputs))
    out = dict(outputs[output_port])
    a_shape = {str(k): int(v) for k, v in dict(in_a["resolved_shape"]).items()}
    b_shape = {str(k): int(v) for k, v in dict(in_b["resolved_shape"]).items()}
    out_shape = {str(k): int(v) for k, v in dict(out["resolved_shape"]).items()}
    dtype = str(in_a["layout"].dtype)
    bytes_per_elem = hw.dtype_bits(dtype) // hw.address_unit_bits
    ring_participants, m_dim, n_dim, local_k_dim = _ring_gemm_tile_geometry(op_data)
    total_k_dim = int(
        b_shape.get(
            "K",
            next(
                (
                    int(extent)
                    for axis, extent in b_shape.items()
                    if str(axis) != "N" or int(extent) != int(out_shape.get("N", n_dim))
                ),
                ring_participants * local_k_dim,
            ),
        )
    )

    if hw.ring_a_buffer_bytes <= 0:
        raise ValueError("ring_a_buffer_bits must be positive.")
    if hw.ring_b_buffer_bytes <= 0:
        raise ValueError("ring_b_buffer_bits must be positive.")

    a_layout = in_a["layout"]
    b_layout = in_b["layout"]
    out_layout = out["layout"]
    a_nominal_m = _factor_extent_product(a_layout, a_shape, "M")
    a_nominal_k = _factor_extent_product(a_layout, a_shape, "K")
    b_nominal_k = _factor_extent_product(b_layout, b_shape, "K")
    b_nominal_n = _factor_extent_product(b_layout, b_shape, "N")
    output_tile_m = _factor_extent_product(out_layout, out_shape, "M")
    output_tile_n = _factor_extent_product(out_layout, out_shape, "N")

    a_buffer_m, output_tile_count_m = _fit_buffer_axis(m_dim, a_nominal_m, label="ring_gemm A/M")
    a_buffer_k, local_a_k_tiles_per_output_tile = _fit_buffer_axis(
        local_k_dim,
        a_nominal_k,
        label="ring_gemm A/K",
    )
    b_buffer_k, total_k_tiles_per_output_tile = _fit_buffer_axis(
        total_k_dim,
        b_nominal_k,
        label="ring_gemm B/K",
    )
    b_buffer_n, output_tile_count_n = _fit_buffer_axis(n_dim, b_nominal_n, label="ring_gemm B/N")
    actual_output_tile_m, output_tile_m_count_from_output = _fit_buffer_axis(
        m_dim,
        output_tile_m,
        label="ring_gemm output/M",
    )
    actual_output_tile_n, output_tile_n_count_from_output = _fit_buffer_axis(
        n_dim,
        output_tile_n,
        label="ring_gemm output/N",
    )
    if actual_output_tile_m != a_buffer_m or actual_output_tile_n != b_buffer_n:
        raise ValueError("ring_gemm registered A/B buffer geometry must match the output tile geometry.")
    if output_tile_count_m != output_tile_m_count_from_output or output_tile_count_n != output_tile_n_count_from_output:
        raise ValueError("ring_gemm A/B buffer tiling disagrees with output tile tiling.")
    if total_k_tiles_per_output_tile % local_a_k_tiles_per_output_tile != 0:
        raise ValueError(
            "ring_gemm total K tiles inferred from B must be an integer multiple of local A K tiles."
        )
    a_ring_hops_per_local_a_tile = total_k_tiles_per_output_tile // local_a_k_tiles_per_output_tile

    a_buffer_bytes = a_buffer_m * a_buffer_k * bytes_per_elem
    b_buffer_bytes = b_buffer_k * b_buffer_n * bytes_per_elem
    if a_buffer_bytes > hw.ring_a_buffer_bytes:
        raise ValueError(
            f"ring_gemm A buffer tile {a_buffer_bytes} bytes exceeds ring_a_buffer_bytes={hw.ring_a_buffer_bytes}."
        )
    if b_buffer_bytes > hw.ring_b_buffer_bytes:
        raise ValueError(
            f"ring_gemm B buffer tile {b_buffer_bytes} bytes exceeds ring_b_buffer_bytes={hw.ring_b_buffer_bytes}."
        )

    pe_m = hw.compute.gemm_core.rows
    pe_n = hw.compute.gemm_core.cols
    pe_k = hw.compute.gemm_core.k_per_cycle
    if pe_m <= 0 or pe_n <= 0 or pe_k <= 0:
        raise ValueError("ring_gemm requires a positive GEMM array shape.")
    if a_buffer_k != pe_k or b_buffer_k != pe_k:
        raise ValueError("ring_gemm buffer K tile must match the PE array K per cycle.")
    if a_buffer_m % pe_m != 0 or b_buffer_n % pe_n != 0:
        raise ValueError("ring_gemm output tile must be divisible by the PE array footprint.")

    a_reuse_factor = b_buffer_n // pe_n
    b_reuse_factor = a_buffer_m // pe_m
    pe_micro_ops_per_output_tile = a_reuse_factor * b_reuse_factor
    output_tile_count = output_tile_count_m * output_tile_count_n
    a_buffer_tile_count = output_tile_count_m * local_a_k_tiles_per_output_tile
    b_buffer_tile_count = output_tile_count_n * total_k_tiles_per_output_tile
    total_coarse_compute_events = output_tile_count * total_k_tiles_per_output_tile
    total_pe_micro_ops = total_coarse_compute_events * pe_micro_ops_per_output_tile

    return _RingGemmExecutionGeometry(
        ring_participants=ring_participants,
        m_dim=m_dim,
        n_dim=n_dim,
        local_k_dim=local_k_dim,
        total_k_dim=total_k_dim,
        bytes_per_elem=bytes_per_elem,
        pe_m=pe_m,
        pe_n=pe_n,
        pe_k=pe_k,
        a_buffer_m=a_buffer_m,
        a_buffer_k=a_buffer_k,
        b_buffer_k=b_buffer_k,
        b_buffer_n=b_buffer_n,
        output_tile_m=actual_output_tile_m,
        output_tile_n=actual_output_tile_n,
        output_tile_count_m=output_tile_count_m,
        output_tile_count_n=output_tile_count_n,
        output_tile_count=output_tile_count,
        local_a_k_tiles_per_output_tile=local_a_k_tiles_per_output_tile,
        a_ring_hops_per_local_a_tile=a_ring_hops_per_local_a_tile,
        total_k_tiles_per_output_tile=total_k_tiles_per_output_tile,
        a_buffer_tile_count=a_buffer_tile_count,
        b_buffer_tile_count=b_buffer_tile_count,
        output_writeback_tile_count=output_tile_count,
        a_buffer_bytes=a_buffer_bytes,
        b_buffer_bytes=b_buffer_bytes,
        output_tile_bytes=actual_output_tile_m * actual_output_tile_n * bytes_per_elem,
        a_reuse_factor=a_reuse_factor,
        b_reuse_factor=b_reuse_factor,
        pe_micro_ops_per_output_tile=pe_micro_ops_per_output_tile,
        total_coarse_compute_events=total_coarse_compute_events,
        total_pe_micro_ops=total_pe_micro_ops,
    )


def _ring_gemm_tile_geometry(op_data: Mapping[str, object]) -> Tuple[int, int, int, int]:
    participants = _ring_participants(op_data)
    inputs = dict(op_data["inputs"])
    outputs = dict(op_data["outputs"])
    in_a_shape = {str(k): int(v) for k, v in dict(inputs["inA"]["resolved_shape"]).items()}
    in_b_shape = {str(k): int(v) for k, v in dict(inputs["inB"]["resolved_shape"]).items()}
    output_port = next(iter(outputs))
    out_shape = {str(k): int(v) for k, v in dict(outputs[output_port]["resolved_shape"]).items()}
    output_m = int(out_shape.get("M", 1))
    output_n = int(out_shape.get("N", 1))

    def _infer_contraction_dim(
        shape: Mapping[str, int],
        *,
        preferred_axis: str,
        preserved_axis: str,
        preserved_extent: int,
    ) -> Optional[int]:
        if preferred_axis in shape:
            return int(shape[preferred_axis])
        candidates = [
            int(extent)
            for axis, extent in shape.items()
            if str(axis) != preserved_axis or int(extent) != preserved_extent
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    m_dim = int(in_a_shape.get("M", out_shape.get("M", 1)))
    n_dim = int(in_b_shape.get("N", out_shape.get("N", 1)))
    inferred_a_k = _infer_contraction_dim(
        in_a_shape,
        preferred_axis="K",
        preserved_axis="M",
        preserved_extent=output_m,
    )
    inferred_b_k = _infer_contraction_dim(
        in_b_shape,
        preferred_axis="K",
        preserved_axis="N",
        preserved_extent=output_n,
    )
    if inferred_a_k is not None:
        local_k_dim = inferred_a_k
    elif inferred_b_k is not None:
        local_k_dim = max(1, inferred_b_k // max(1, participants))
    else:
        local_k_dim = 1
    return participants, m_dim, n_dim, local_k_dim


def _summarize_ring_participant_timeline(
    ring_microtile_timeline: Mapping[str, object],
    ring_participants: int,
    microtile_count_per_participant: int,
) -> Dict[str, object]:
    compute_end = list(ring_microtile_timeline["compute_end_cycles"])
    compute_start = list(ring_microtile_timeline["compute_start_cycles"])
    ring_ready = list(ring_microtile_timeline["ring_ready_cycles"])
    b_ready = list(ring_microtile_timeline["b_ready_cycles"])
    local_ready = list(ring_microtile_timeline["local_load_ready_cycles"])
    a_buffer_tiles = sorted(
        list(dict(ring_microtile_timeline.get("a_buffer_timeline", {})).get("tiles", [])),
        key=lambda tile: int(tile.get("tile_index", 0)),
    )
    ping_pong_assignment = [str(tile.get("buffer", "ping")) for tile in a_buffer_tiles[:microtile_count_per_participant]]
    if len(ping_pong_assignment) < microtile_count_per_participant:
        ping_pong_assignment = list(ring_microtile_timeline["ping_pong_assignment"][:microtile_count_per_participant])
    per_round_completion: List[float] = []
    for round_idx in range(microtile_count_per_participant):
        start = round_idx * ring_participants
        end = start + ring_participants
        per_round_completion.append(max(compute_end[start:end], default=0.0))
    return {
        "tile_count": ring_participants,
        "tile_ready_cycles": [
            {
                "a_ready_cycle": ring_ready[idx],
                "b_ready_cycle": b_ready[idx],
                "local_load_ready_cycle": local_ready[idx],
            }
            for idx in range(len(compute_end))
        ],
        "tile_compute_start_cycles": compute_start,
        "tile_compute_end_cycles": compute_end,
        "tile_write_release_cycles": per_round_completion,
        "ping_pong_assignment": ping_pong_assignment,
    }


def _require_tensor_base_addr(tensor_name: str, tensors: Mapping[str, Mapping[str, object]]) -> int:
    tensor = dict(tensors.get(tensor_name, {}))
    if tensor.get("base_addr") is None:
        raise ValueError(
            f"Tensor '{tensor_name}' is missing required base_addr for physical request generation."
        )
    base_addr = int(tensor["base_addr"])
    if base_addr < 0:
        raise ValueError(f"Tensor '{tensor_name}' base_addr must be non-negative.")
    return base_addr


def _default_logical_bit_labels(bit_count: int) -> List[str]:
    return [f"addr_bit_{idx}" for idx in range(bit_count)]


def _collect_unique_transforms(op_reports: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    transforms: List[Dict[str, object]] = []
    for op in op_reports:
        for transform in list(op.get("address_transforms", [])):
            key = json.dumps(transform, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            transforms.append(transform)
    return transforms


def _collect_unique_transforms_from_requests(requests: Sequence[PhysicalRequest]) -> List[Dict[str, object]]:
    seen = set()
    transforms: List[Dict[str, object]] = []
    for request in requests:
        key = json.dumps(request.address_transform, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        transforms.append(dict(request.address_transform))
    return transforms


def _find_output_tensor_name(op_name: str, output_data: Mapping[str, object]) -> str:
    return str(output_data.get("source_tensor") or f"{op_name}:out")


def _serialize_hardware(hw: HardwareSpec) -> Dict[str, object]:
    return hw.to_dict()


def _validate_perf_config(hw: HardwareSpec, perf: PerformanceConfig) -> None:
    max_banks = 1 << hw.bank_bits
    if hw.bank_count_per_slice <= 0:
        raise ValueError("bank_count_per_slice must be positive.")
    if hw.bank_count_per_slice > max_banks:
        raise ValueError(
            f"bank_count_per_slice={hw.bank_count_per_slice} exceeds addressable banks {max_banks}; "
            "increase hardware.bank_bits in the config if needed."
        )
    if hw.request_fifo_depth <= 0:
        raise ValueError("request_fifo_depth must be positive.")
    if hw.slice_frequency_hz <= 0 or hw.memory_frequency_hz <= 0:
        raise ValueError("slice_frequency_hz and memory_frequency_hz must be positive.")
    if hw.bank_count_per_slice <= 0 or hw.bank_bandwidth_bits_per_cycle <= 0:
        raise ValueError("slice_bank_count and bank_bandwidth_bits_per_cycle must be positive.")
    if hw.peak_memory_bandwidth_bytes_per_cycle <= 0:
        raise ValueError("peak_memory_bandwidth_bytes_per_cycle must be positive.")
    if hw.write_buffer_bits <= 0 or hw.write_buffer_bytes <= 0:
        raise ValueError("write_buffer_bits must be positive.")
    if hw.ring_a_buffer_bits <= 0 or hw.ring_a_buffer_bytes <= 0:
        raise ValueError("ring_a_buffer_bits must be positive.")
    if hw.ring_b_buffer_bits <= 0 or hw.ring_b_buffer_bytes <= 0:
        raise ValueError("ring_b_buffer_bits must be positive.")
    for value in (perf.gemm_read_overlap, perf.general_read_overlap, perf.writeback_overlap):
        if value < 0.0 or value > 1.0:
            raise ValueError("Performance overlap ratios must be within [0, 1].")
    if perf.controller_write_queue_depth <= 0:
        raise ValueError("performance.controller.write_queue_depth must be positive.")
    if perf.slice_write_buffer_depth <= 0:
        raise ValueError("performance.controller.slice_write_buffer_depth must be positive.")
    if perf.scheduler_epoch_cycles <= 0:
        raise ValueError("performance.controller.scheduler_epoch_cycles must be positive.")


def _render_summary_markdown(
    graph_name: str,
    modes: Mapping[str, Mapping[str, object]],
    true_roofline: Mapping[str, object],
) -> str:
    baseline = modes[MODE_BASELINE]
    remap = modes[MODE_REMAP]
    interleave = modes[MODE_REMAP_INTERLEAVE]
    return (
        f"# Performance Summary: {graph_name}\n\n"
        f"- Cycle domain: `slice-cycle`\n"
        f"- Memory timing domain: `bank-cycle`\n"
        f"- True roofline cycles: {float(true_roofline['roofline_cycles']):.2f} cycles\n"
        f"- True compute-bound cycles: {float(true_roofline['compute_bound_cycles']):.2f} cycles\n"
        f"- True bandwidth-bound cycles: {float(true_roofline['bandwidth_bound_cycles']):.2f} cycles\n"
        f"- Baseline latency: {baseline['total_latency_cycles']:.2f} cycles\n"
        f"- Remap latency: {remap['total_latency_cycles']:.2f} cycles\n"
        f"- Remap + Interleave latency: {interleave['total_latency_cycles']:.2f} cycles\n"
        f"- Remap speedup vs baseline: {remap['speedup_vs_baseline']:.4f}x\n"
        f"- Remap + Interleave speedup vs baseline: {interleave['speedup_vs_baseline']:.4f}x\n"
        f"- Baseline / analytical lower bound: {baseline['analytical_model']['latency_to_lower_bound_ratio']:.4f}\n"
        f"- Remap / analytical lower bound: {remap['analytical_model']['latency_to_lower_bound_ratio']:.4f}\n"
        f"- Remap + Interleave / analytical lower bound: {interleave['analytical_model']['latency_to_lower_bound_ratio']:.4f}\n"
    )


def _default_performance_output_path(input_path: str) -> Path:
    source = Path(input_path)
    return Path.cwd() / "outputs" / "performance" / source.stem / f"{source.stem}_performance.json"


def _build_overview(
    graph_name: str,
    graph_summary: Mapping[str, int],
    true_roofline: Mapping[str, object],
    mode_summaries: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    ordered_modes = sorted(
        mode_summaries.items(),
        key=lambda item: float(item[1]["estimated_latency_cycles"]),
    )
    best_mode = ordered_modes[0][0]
    roofline_bound = (
        "compute"
        if float(true_roofline["compute_bound_cycles"]) >= float(true_roofline["bandwidth_bound_cycles"])
        else "bandwidth"
    )
    return {
        "graph_name": graph_name,
        "graph_summary": dict(graph_summary),
        "best_mode_by_estimated_latency": best_mode,
        "mode_order_by_estimated_latency": [name for name, _ in ordered_modes],
        "cycle_domain": "slice-cycle",
        "memory_timing_domain": "bank-cycle",
        "baseline_latency_cycles": mode_summaries[MODE_BASELINE]["estimated_latency_cycles"],
        "true_roofline_cycles": true_roofline["roofline_cycles"],
        "true_roofline_bound": roofline_bound,
    }


def _build_mode_summary(
    mode_report: Mapping[str, object],
    true_roofline: Mapping[str, object],
) -> Dict[str, object]:
    total_latency = float(mode_report["total_latency_cycles"])
    analytical = dict(mode_report["analytical_model"])
    op_breakdown = list(mode_report["op_breakdown"])
    top_ops = sorted(op_breakdown, key=lambda op: float(op["latency_cycles"]), reverse=True)[:3]
    return {
        "estimated_latency_cycles": total_latency,
        "speedup_vs_baseline": mode_report.get("speedup_vs_baseline"),
        "latency_vs_true_roofline": (
            total_latency / float(true_roofline["roofline_cycles"])
            if float(true_roofline["roofline_cycles"])
            else None
        ),
        "latency_vs_analytical_lower_bound": analytical["latency_to_lower_bound_ratio"],
        "compute_bound_cycles": analytical["compute_bound_cycles"],
        "memory_access_bound_cycles": analytical["memory_access_bound_cycles"],
        "ag_issue_bound_cycles": analytical["ag_issue_bound_cycles"],
        "software_relayout_stage_count": sum(1 for op in op_breakdown if op.get("kind") == "relayout"),
        "software_relayout_total_latency_cycles": sum(
            float(op["latency_cycles"]) for op in op_breakdown if op.get("kind") == "relayout"
        ),
        "software_relayout_total_bytes": sum(
            int(op["total_bytes"]) for op in op_breakdown if op.get("kind") == "relayout"
        ),
        "top_ops_by_latency": [
            {
                "stage_name": op.get("stage_name", op.get("op_name")),
                "kind": op.get("kind", "op"),
                "op_type": op["op_type"],
                "latency_cycles": op["latency_cycles"],
                "latency_share": (float(op["latency_cycles"]) / total_latency) if total_latency else None,
            }
            for op in top_ops
        ],
    }


def _true_roofline_from_totals(
    work_ops: float,
    total_bytes: int,
    peak_compute_ops_per_cycle: int,
    peak_memory_bandwidth_bytes_per_cycle: float,
) -> Dict[str, object]:
    compute_bound_cycles = work_ops / max(1, peak_compute_ops_per_cycle)
    bandwidth_bound_cycles = total_bytes / peak_memory_bandwidth_bytes_per_cycle if total_bytes else 0.0
    arithmetic_intensity = (work_ops / total_bytes) if total_bytes else None
    compute_to_bandwidth_ratio = (
        peak_compute_ops_per_cycle / peak_memory_bandwidth_bytes_per_cycle
        if peak_memory_bandwidth_bytes_per_cycle
        else None
    )
    return {
        "work_ops": work_ops,
        "total_bytes": total_bytes,
        "arithmetic_intensity_ops_per_byte": arithmetic_intensity,
        "peak_compute_ops_per_cycle": peak_compute_ops_per_cycle,
        "peak_memory_bandwidth_bytes_per_cycle": peak_memory_bandwidth_bytes_per_cycle,
        "compute_bound_cycles": compute_bound_cycles,
        "bandwidth_bound_cycles": bandwidth_bound_cycles,
        "roofline_cycles": max(compute_bound_cycles, bandwidth_bound_cycles),
        "ridge_point_ops_per_byte": compute_to_bandwidth_ratio,
    }


def _summarize_true_roofline(
    op_reports: Sequence[Mapping[str, object]],
    hw: HardwareSpec,
) -> Dict[str, object]:
    true_ops = [op for op in op_reports if op.get("kind") != "relayout"]
    total_ops = sum(float(op["work_ops"]) for op in true_ops)
    total_bytes = sum(int(op["total_bytes"]) for op in true_ops)
    compute_bound_cycles = sum(float(op["true_roofline"]["compute_bound_cycles"]) for op in true_ops)
    bandwidth_bound_cycles = total_bytes / hw.peak_memory_bandwidth_bytes_per_cycle if total_bytes else 0.0
    return {
        "work_ops": total_ops,
        "total_bytes": total_bytes,
        "arithmetic_intensity_ops_per_byte": (total_ops / total_bytes) if total_bytes else None,
        "cycle_domain": "slice-cycle",
        "memory_timing_domain": "bank-cycle",
        "peak_memory_bandwidth_bytes_per_cycle": hw.peak_memory_bandwidth_bytes_per_cycle,
        "compute_bound_cycles": compute_bound_cycles,
        "bandwidth_bound_cycles": bandwidth_bound_cycles,
        "roofline_cycles": max(compute_bound_cycles, bandwidth_bound_cycles),
    }


def _needs_software_relayout(edge_result: EdgeSolveResult) -> bool:
    return bool(edge_result.write_reg_required)


def _analyze_relayout_stage(
    edge_result: EdgeSolveResult,
    op_name: str,
    port_name: str,
    tensor_name: str,
    base_addr: int,
    hw: HardwareSpec,
    perf: PerformanceConfig,
) -> Dict[str, object]:
    producer_bits = list(edge_result.producer_visible_outer_bits or [])
    consumer_bits = list(edge_result.consumer_visible_outer_bits or [])
    read_requests = _materialize_requests(
        tensor_name=tensor_name,
        edge_name=f"{edge_result.producer}->{edge_result.consumer}:{tensor_name}:relayout_read",
        logical_labels=producer_bits or consumer_bits,
        address_transform=AddressTransform.identity(producer_bits or consumer_bits, name="software_relayout_read"),
        base_addr=base_addr,
        hw=hw,
        perf=perf,
        role="relayout_read",
        ag_ids=("ag0",),
    )
    write_requests = _materialize_requests(
        tensor_name=tensor_name,
        edge_name=f"{edge_result.producer}->{edge_result.consumer}:{tensor_name}:relayout_write",
        logical_labels=consumer_bits or producer_bits,
        address_transform=AddressTransform.identity(consumer_bits or producer_bits, name="software_relayout_write"),
        base_addr=base_addr,
        hw=hw,
        perf=perf,
        role="writeback",
        ag_ids=("ag4",),
    )
    read_reports = _build_stream_reports(
        requests=read_requests,
        ag_ids=("ag0",),
        tensor_name=tensor_name,
        edge_name=f"{edge_result.producer}->{edge_result.consumer}:{port_name}:relayout_read",
        role="relayout_read",
        mode=MODE_BASELINE,
        hw=hw,
        perf=perf,
    )
    write_reports = _build_stream_reports(
        requests=write_requests,
        ag_ids=("ag4",),
        tensor_name=tensor_name,
        edge_name=f"{edge_result.producer}->{edge_result.consumer}:{port_name}:relayout_write",
        role="writeback",
        mode=MODE_BASELINE,
        hw=hw,
        perf=perf,
    )
    streams = [*read_reports, *write_reports]
    read_stream = read_reports[0] if read_reports else None
    write_stream = write_reports[0] if write_reports else None
    bytes_read = len(read_requests) * (hw.block_bits // 8)
    bytes_written = len(write_requests) * (hw.block_bits // 8)
    total_bytes = bytes_read + bytes_written
    block_elements = hw.block_elements(edge_result.producer_bound_layout["dtype"]) if edge_result.producer_bound_layout else hw.block_elements("fp16")
    work_ops = float(max(len(read_requests), len(write_requests)) * block_elements)
    compute_bound_cycles = math.ceil(work_ops / max(1, hw.general_peak_ops_per_cycle))
    memory_access_bound_cycles = max(
        float(read_stream["raw_bank_max_cycles"]) if read_stream else 0.0,
        float(write_stream["raw_bank_max_cycles"]) if write_stream else 0.0,
    )
    ag_issue_bound_cycles = max(float(stream["issue_cycles"]) for stream in streams) if streams else 0.0
    lower_bound = max(compute_bound_cycles, memory_access_bound_cycles, ag_issue_bound_cycles)
    read_cycles = float(read_stream["adjusted_stream_cycles"]) if read_stream else 0.0
    write_cycles = float(write_stream["adjusted_stream_cycles"]) if write_stream else 0.0
    latency = (
        max(compute_bound_cycles, read_cycles, write_cycles)
        + read_cycles * (1.0 - perf.general_read_overlap)
        + write_cycles * (1.0 - perf.writeback_overlap)
    )
    true_roofline = _true_roofline_from_totals(
        work_ops=work_ops,
        total_bytes=total_bytes,
        peak_compute_ops_per_cycle=hw.general_peak_ops_per_cycle,
        peak_memory_bandwidth_bytes_per_cycle=hw.peak_memory_bandwidth_bytes_per_cycle,
    )
    return {
        "kind": "relayout",
        "stage_name": f"{edge_result.producer}->{op_name}:{port_name}:software_relayout",
        "op_name": f"{edge_result.producer}->{op_name}:{port_name}:software_relayout",
        "op_type": "software_relayout",
        "latency_cycles": latency,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "total_bytes": total_bytes,
        "work_ops": work_ops,
        "arithmetic_intensity": (work_ops / total_bytes) if total_bytes else None,
        "true_roofline": true_roofline,
        "analytical_model": {
            "estimated_latency_cycles": latency,
            "compute_bound_cycles": compute_bound_cycles,
            "memory_access_bound_cycles": memory_access_bound_cycles,
            "ag_issue_bound_cycles": ag_issue_bound_cycles,
            "lower_bound_cycles": lower_bound,
            "latency_to_lower_bound_ratio": (latency / lower_bound) if lower_bound else None,
        },
        "streams": streams,
        "edge_breakdown": _edge_reports_from_streams(streams),
        "address_transforms": _collect_unique_transforms_from_requests([*read_requests, *write_requests]),
        "request_trace": [*read_requests, *write_requests],
    }


def _coerce_optional_positive_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        measured = float(value)
    except (TypeError, ValueError):
        return None
    return measured if measured > 0.0 else None
