import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .addressing import AddressTransform, encode_physical_address
from .hardware import HardwareSpec
from .performance import (
    MODE_BASELINE,
    MODE_REMAP,
    MODE_REMAP_INTERLEAVE,
    ALL_MODES,
    PerformanceConfig,
    PhysicalRequest,
    _analyze_request_stream,
    render_ramulator_trace_lines,
)


def emit_trace_artifacts(
    payload: Mapping[str, object],
    output_json_path: str,
    hw: HardwareSpec,
    ramulator_root: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    base_path = Path(output_json_path).resolve()
    base_dir = base_path.parent
    stem = base_path.stem
    artifacts: Dict[str, Dict[str, str]] = {}
    for mode in ALL_MODES:
        mode_payload = dict(payload["modes"][mode])
        request_trace = list(mode_payload.get("request_trace", []))
        trace_json_path = base_dir / f"{stem}_trace_{mode}.json"
        trace_json_path.write_text(json.dumps(request_trace, indent=2) + "\n", encoding="utf-8")

        ramulator_trace_path = base_dir / f"{stem}_ramulator_{mode}.trace"
        ramulator_trace_path.write_text(
            "\n".join(render_ramulator_trace_lines(request_trace, hw)) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "mode": mode,
            "request_count": len(request_trace),
            "source_output": str(base_path),
        }
        metadata_path = base_dir / f"{stem}_ramulator_{mode}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        ramulator_cfg_path = base_dir / f"{stem}_ramulator_{mode}.yaml"
        ramulator_cfg_path.write_text(
            _render_ramulator_config(
                trace_path=ramulator_trace_path,
                hw=hw,
                ramulator_root=Path(ramulator_root) if ramulator_root else None,
            ),
            encoding="utf-8",
        )
        artifacts[mode] = {
            "request_trace_json": str(trace_json_path),
            "ramulator_trace": str(ramulator_trace_path),
            "ramulator_metadata": str(metadata_path),
            "ramulator_config": str(ramulator_cfg_path),
        }
    return artifacts


def run_validation(
    payload: Mapping[str, object],
    hw: HardwareSpec,
    perf: PerformanceConfig,
    output_json_path: str,
    ramulator_root: Optional[str] = None,
) -> Dict[str, object]:
    artifacts = emit_trace_artifacts(payload, output_json_path, hw, ramulator_root=ramulator_root)
    internal_cases = _run_internal_validation_cases(perf, hw)
    internal_passed = all(bool(case["passed"]) for case in internal_cases)

    reference_results = _run_reference_validation(
        payload=payload,
        artifacts=artifacts,
        hw=hw,
        ramulator_root=ramulator_root,
    )

    comparison = _compare_to_reference(payload, reference_results)
    confidence = _model_confidence(internal_passed, reference_results, comparison)
    reference_mode_summaries = _build_reference_mode_summaries(reference_results, comparison)
    validation_overview = _build_validation_overview(
        payload=payload,
        internal_cases=internal_cases,
        internal_passed=internal_passed,
        reference_results=reference_results,
        comparison=comparison,
        confidence=confidence,
        reference_mode_summaries=reference_mode_summaries,
    )

    return {
        "validation_overview": validation_overview,
        "validation_summary": {
            "internal_validation_passed": internal_passed,
            "reference_validation_status": reference_results["status"],
            "reference_executable": reference_results.get("ramulator_executable"),
        },
        "reference_mode_summaries": reference_mode_summaries,
        "validation_cases": internal_cases,
        "trace_artifacts": artifacts,
        "reference_results": reference_results,
        "comparison_to_reference": comparison,
        "calibration_notes": _calibration_notes(reference_results, comparison),
        "model_confidence": confidence,
    }


def write_validation_outputs(
    output_json_path: str,
    validation_payload: Mapping[str, object],
) -> Tuple[Path, Path]:
    base = Path(output_json_path)
    validation_json = base.with_name(f"{base.stem}_validation.json")
    validation_md = base.with_name(f"{base.stem}_validation.md")
    validation_json.write_text(json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8")
    validation_md.write_text(_render_validation_markdown(validation_payload) + "\n", encoding="utf-8")
    return validation_json, validation_md


def _synthetic_request(
    request_id: int,
    edge_name: str,
    ag_id: str,
    role: str,
    *,
    slice_id: int,
    bank_id: int,
    row_id: int,
    col_id: int,
    hw: HardwareSpec,
) -> PhysicalRequest:
    physical_addr = encode_physical_address(
        slice_id=slice_id,
        bank_id=bank_id,
        row_id=row_id,
        col_id=col_id,
        hw=hw,
    )
    return PhysicalRequest(
        request_id=request_id,
        tensor_name="synthetic",
        edge_name=edge_name,
        ag_id=ag_id,
        role=role,
        logical_addr=request_id,
        base_addr=0,
        address_transform=AddressTransform.identity(["addr_bit_0"], name="synthetic_identity").to_dict(),
        physical_addr=physical_addr,
        slice_id=slice_id,
        bank_id=bank_id,
        row_id=row_id,
        col_id=col_id,
    )


def _run_internal_validation_cases(perf: PerformanceConfig, hw: HardwareSpec) -> List[Dict[str, object]]:
    grouped = [
        _synthetic_request(
            i,
            "grouped",
            "ag0",
            "A",
            slice_id=0,
            bank_id=min(i // 4, 3),
            row_id=i % 4,
            col_id=0,
            hw=hw,
        )
        for i in range(16)
    ]
    round_robin = [
        _synthetic_request(
            i,
            "round_robin",
            "ag0",
            "A",
            slice_id=0,
            bank_id=i % 4,
            row_id=i // 4,
            col_id=0,
            hw=hw,
        )
        for i in range(16)
    ]
    grouped_report = _analyze_request_stream("ag0", "synthetic", "grouped", "A", MODE_BASELINE, grouped, hw, perf)
    round_robin_report = _analyze_request_stream(
        "ag0",
        "synthetic",
        "round_robin",
        "A",
        MODE_REMAP_INTERLEAVE,
        round_robin,
        hw,
        perf,
    )

    same_bank_conflict = [
        _synthetic_request(0, "conflict", "ag0", "A", slice_id=0, bank_id=0, row_id=0, col_id=0, hw=hw),
        _synthetic_request(1, "conflict", "ag1", "B", slice_id=0, bank_id=0, row_id=1, col_id=0, hw=hw),
    ]
    cross_bank = [
        _synthetic_request(0, "parallel", "ag0", "A", slice_id=0, bank_id=0, row_id=0, col_id=0, hw=hw),
        _synthetic_request(1, "parallel", "ag1", "B", slice_id=0, bank_id=1, row_id=1, col_id=0, hw=hw),
    ]
    conflict_penalty = _same_bank_conflict_penalty(same_bank_conflict, hw)
    parallel_penalty = _same_bank_conflict_penalty(cross_bank, hw)

    return [
        {
            "name": "round_robin_hides_more_row_switch",
            "passed": round_robin_report["exposed_row_switch_cycles"] < grouped_report["exposed_row_switch_cycles"],
            "details": {
                "grouped_exposed_row_switch_cycles": grouped_report["exposed_row_switch_cycles"],
                "round_robin_exposed_row_switch_cycles": round_robin_report["exposed_row_switch_cycles"],
            },
        },
        {
            "name": "interleave_preserves_request_count",
            "passed": grouped_report["request_count"] == round_robin_report["request_count"],
            "details": {
                "grouped_request_count": grouped_report["request_count"],
                "round_robin_request_count": round_robin_report["request_count"],
            },
        },
        {
            "name": "same_bank_conflict_worse_than_cross_bank",
            "passed": conflict_penalty > parallel_penalty,
            "details": {
                "same_bank_penalty_cycles": conflict_penalty,
                "cross_bank_penalty_cycles": parallel_penalty,
            },
        },
    ]


def _same_bank_conflict_penalty(requests: Sequence[PhysicalRequest], hw: HardwareSpec) -> float:
    banks = {}
    for request in requests:
        banks.setdefault(request.bank_id, 0)
        banks[request.bank_id] += hw.request_latency_cycles
    return max(banks.values(), default=0.0)


def _run_reference_validation(
    payload: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, str]],
    hw: HardwareSpec,
    ramulator_root: Optional[str],
) -> Dict[str, object]:
    exe = _detect_ramulator_executable(ramulator_root)
    resolved_root = _resolve_ramulator_root(ramulator_root)
    if exe is None:
        return {
            "status": "skipped",
            "reason": (
                "Ramulator executable not found. Build Linux/WSL with "
                "scripts/setup_ramulator_wsl.sh or Windows with scripts/setup_ramulator_windows.ps1."
            ),
            "per_mode": {},
        }

    per_mode: Dict[str, object] = {}
    for mode, paths in artifacts.items():
        config_path = str(Path(paths["ramulator_config"]).resolve())
        try:
            completed = subprocess.run(
                [str(exe), "-f", config_path],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(resolved_root) if resolved_root is not None else None,
            )
        except OSError as exc:
            return {
                "status": "skipped",
                "reason": f"Failed to start Ramulator executable: {exc}",
                "ramulator_executable": str(exe),
                "per_mode": {},
            }
        stdout = completed.stdout
        stderr = completed.stderr
        per_mode[mode] = {
            "returncode": completed.returncode,
            "stdout_path": _write_side_output(config_path, ".stdout.txt", stdout),
            "stderr_path": _write_side_output(config_path, ".stderr.txt", stderr),
            "memory_cycles_reference": _parse_reference_cycles(stdout),
        }

    status = "ok" if all(int(info["returncode"]) == 0 for info in per_mode.values()) else "partial"
    return {
        "status": status,
        "ramulator_executable": str(exe),
        "per_mode": per_mode,
    }


def _compare_to_reference(
    payload: Mapping[str, object],
    reference_results: Mapping[str, object],
) -> Dict[str, object]:
    per_mode_reference = dict(reference_results.get("per_mode", {}))
    memory_alignment: Dict[str, object] = {}
    for mode in ALL_MODES:
        modeled = payload["modes"][mode]["analytical_model"]["memory_access_bound_cycles"]
        reference = None
        if mode in per_mode_reference:
            reference = per_mode_reference[mode].get("memory_cycles_reference")
        ratio = (modeled / reference) if modeled and reference else None
        memory_alignment[mode] = {
            "memory_cycles_modeled": modeled,
            "memory_cycles_reference": reference,
            "modeled_to_reference_ratio": ratio,
        }

    ordering_modeled = _mode_latency_order(
        {mode: float(payload["modes"][mode]["total_latency_cycles"]) for mode in ALL_MODES}
    )
    ordering_reference = None
    ref_cycles = {
        mode: per_mode_reference.get(mode, {}).get("memory_cycles_reference")
        for mode in ALL_MODES
    }
    if all(value is not None for value in ref_cycles.values()):
        ordering_reference = _mode_latency_order({mode: float(value) for mode, value in ref_cycles.items()})

    return {
        "memory_alignment": memory_alignment,
        "ordering_modeled": ordering_modeled,
        "ordering_reference": ordering_reference,
        "ordering_matches": ordering_reference == ordering_modeled if ordering_reference else None,
    }


def _model_confidence(
    internal_passed: bool,
    reference_results: Mapping[str, object],
    comparison: Mapping[str, object],
) -> Dict[str, object]:
    if internal_passed and reference_results.get("status") == "ok" and comparison.get("ordering_matches") is True:
        level = "high"
    elif internal_passed:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "internal_validation": "passed" if internal_passed else "failed",
        "external_reference": reference_results.get("status", "skipped"),
    }


def _calibration_notes(
    reference_results: Mapping[str, object],
    comparison: Mapping[str, object],
) -> List[str]:
    notes = [
        "row_switch_scale, bank_parallel_efficiency, writeback_scale, and interleave_hiding_scale remain calibration knobs.",
        "Reference validation only covers memory-side behavior; AG/buffer/array overlap is still modeled analytically.",
    ]
    if reference_results.get("status") == "skipped":
        notes.append("Ramulator reference was skipped, so current confidence mostly comes from internal order-aware sanity checks.")
    elif comparison.get("ordering_matches") is False:
        notes.append("Reference ordering disagrees with the analytical model; revisit address packing and row-switch hiding assumptions.")
    return notes


def _render_validation_markdown(validation_payload: Mapping[str, object]) -> str:
    overview = validation_payload["validation_overview"]
    summary = validation_payload["validation_summary"]
    lines = [
        "# Validation Report",
        "",
        f"- Best modeled mode: `{overview['best_modeled_mode']}`",
        f"- Best reference mode: `{overview['best_reference_mode']}`",
        f"- Ordering matches: `{overview['ordering_matches']}`",
        f"- Cycle domain: `{overview['cycle_domain']}`",
        f"- Memory timing domain: `{overview['memory_timing_domain']}`",
        f"- Baseline includes software relayout: `{overview['baseline_includes_software_relayout']}`",
        f"- Internal cases passed: `{overview['internal_cases_passed']}/{overview['internal_case_count']}`",
        f"- Internal validation passed: `{summary['internal_validation_passed']}`",
        f"- Reference validation status: `{summary['reference_validation_status']}`",
        f"- Model confidence: `{validation_payload['model_confidence']['level']}`",
        "",
        "## Reference Mode Summaries",
    ]
    for mode, mode_summary in validation_payload["reference_mode_summaries"].items():
        lines.append(
            f"- `{mode}`: returncode=`{mode_summary['returncode']}`, "
            f"modeled=`{mode_summary['memory_cycles_modeled']}`, "
            f"reference=`{mode_summary['memory_cycles_reference']}`, "
            f"ratio=`{mode_summary['modeled_to_reference_ratio']}`"
        )
    lines.extend(
        [
            "",
        "## Internal Cases",
        ]
    )
    for case in validation_payload["validation_cases"]:
        lines.append(f"- `{case['name']}`: `{case['passed']}`")
    lines.append("")
    lines.append("## Calibration Notes")
    for note in validation_payload["calibration_notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _render_ramulator_config(
    trace_path: Path,
    hw: HardwareSpec,
    ramulator_root: Optional[Path],
) -> str:
    trace_path = trace_path.resolve()
    rel_trace = trace_path
    if ramulator_root is not None:
        ramulator_root = ramulator_root.resolve()
        try:
            rel_trace = trace_path.relative_to(ramulator_root)
        except ValueError:
            rel_trace = trace_path
    dram_impl = "DDR4"
    timing_preset = "DDR4_2400R"
    org_preset = "DDR4_8Gb_x8"
    return (
        "Frontend:\n"
        "  impl: LoadStoreTrace\n"
        f"  path: {str(rel_trace).replace(chr(92), '/')}\n"
        "  clock_ratio: 1\n"
        "\n"
        "  Translation:\n"
        "    impl: NoTranslation\n"
        f"    max_addr: {1 << max(1, hw.remap_bits + hw.subword_bits)}\n"
        "\n"
        "MemorySystem:\n"
        "  impl: GenericDRAM\n"
        "  clock_ratio: 1\n"
        "  DRAM:\n"
        f"    impl: {dram_impl}\n"
        "    org:\n"
        f"      preset: {org_preset}\n"
        "      channel: 1\n"
        "      rank: 1\n"
        "    timing:\n"
        f"      preset: {timing_preset}\n"
        "\n"
        "  Controller:\n"
        "    impl: Generic\n"
        "    Scheduler:\n"
        "      impl: FRFCFS\n"
        "    RefreshManager:\n"
        "      impl: AllBank\n"
        "    RowPolicy:\n"
        "      impl: ClosedRowPolicy\n"
        "      cap: 4\n"
        "    plugins:\n"
        "\n"
        "  AddrMapper:\n"
        "    impl: RoBaRaCoCh\n"
    )


def _detect_ramulator_executable(ramulator_root: Optional[str]) -> Optional[Path]:
    root = _resolve_ramulator_root(ramulator_root)
    if root is None:
        root = Path.cwd() / "third_party" / "ramulator2"

    candidates: List[Path] = []
    candidates.extend(
        [
            root / "build-linux" / "ramulator2",
            root / "ramulator2",
            root / "build_msvc" / "ramulator2.exe",
            root / "build" / "ramulator2.exe",
            root / "ramulator2.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _write_side_output(config_path: str, suffix: str, content: str) -> str:
    path = Path(config_path).with_suffix(Path(config_path).suffix + suffix)
    path.write_text(content or "", encoding="utf-8")
    return str(path)


def _parse_reference_cycles(stdout: str) -> Optional[int]:
    preferred_keys = (
        "memory_system_cycles",
        "cycles_recorded_core_0",
        "cycles_recorded",
        "num_cycles",
        "cycles",
    )
    for key in preferred_keys:
        match = re.search(rf"{re.escape(key)}:\s*([0-9]+)", stdout)
        if match:
            return int(match.group(1))

    matches = re.findall(r"([A-Za-z_]*cycles[A-Za-z_]*):\s*([0-9]+)", stdout)
    if matches:
        for name, value in matches:
            if "memory" in name or name.startswith("cycles_recorded"):
                return int(value)
        return int(matches[-1][1])
    generic = re.findall(r"\b([0-9]+)\b", stdout)
    return int(generic[-1]) if generic else None


def _mode_latency_order(values: Mapping[str, float]) -> List[str]:
    return [name for name, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)]


def _resolve_ramulator_root(ramulator_root: Optional[str]) -> Optional[Path]:
    if ramulator_root:
        return Path(ramulator_root).resolve()
    default_root = Path.cwd() / "third_party" / "ramulator2"
    return default_root.resolve() if default_root.exists() else None


def _build_reference_mode_summaries(
    reference_results: Mapping[str, object],
    comparison: Mapping[str, object],
) -> Dict[str, Dict[str, object]]:
    memory_alignment = dict(comparison.get("memory_alignment", {}))
    per_mode_reference = dict(reference_results.get("per_mode", {}))
    summaries: Dict[str, Dict[str, object]] = {}
    for mode in ALL_MODES:
        alignment = dict(memory_alignment.get(mode, {}))
        reference = dict(per_mode_reference.get(mode, {}))
        summaries[mode] = {
            "returncode": reference.get("returncode"),
            "memory_cycles_modeled": alignment.get("memory_cycles_modeled"),
            "memory_cycles_reference": alignment.get("memory_cycles_reference"),
            "modeled_to_reference_ratio": alignment.get("modeled_to_reference_ratio"),
        }
    return summaries


def _build_validation_overview(
    payload: Mapping[str, object],
    internal_cases: Sequence[Mapping[str, object]],
    internal_passed: bool,
    reference_results: Mapping[str, object],
    comparison: Mapping[str, object],
    confidence: Mapping[str, object],
    reference_mode_summaries: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    passed_count = sum(1 for case in internal_cases if bool(case["passed"]))
    ordered_modeled = list(comparison.get("ordering_modeled", []))
    ordered_reference = comparison.get("ordering_reference")

    best_reference_mode = None
    if ordered_reference:
        best_reference_mode = ordered_reference[-1]

    best_modeled_mode = None
    if ordered_modeled:
        best_modeled_mode = ordered_modeled[-1]

    return {
        "internal_validation_passed": internal_passed,
        "internal_cases_passed": passed_count,
        "internal_case_count": len(internal_cases),
        "reference_validation_status": reference_results.get("status"),
        "best_modeled_mode": best_modeled_mode,
        "best_reference_mode": best_reference_mode,
        "ordering_matches": comparison.get("ordering_matches"),
        "modeled_order": ordered_modeled,
        "reference_order": ordered_reference,
        "model_confidence_level": confidence.get("level"),
        "cycle_domain": payload.get("overview", {}).get("cycle_domain", "slice-cycle"),
        "memory_timing_domain": payload.get("overview", {}).get("memory_timing_domain", "bank-cycle"),
        "baseline_includes_software_relayout": True,
        "baseline_software_relayout_stage_count": payload.get("mode_summaries", {})
        .get("baseline", {})
        .get("software_relayout_stage_count", 0),
        "reference_modes_with_results": [
            mode for mode, summary in reference_mode_summaries.items() if summary.get("memory_cycles_reference") is not None
        ],
    }
