import argparse
import json
import re
from pathlib import Path

from .graph import load_graph_file, solve_graph
from .hardware import HardwareSpec
from .layout import LayoutSpec
from .performance import analyze_graph_performance, load_performance_config, write_performance_outputs
from .roofline import generate_roofline_artifacts
from .rmsnorm_bridge import (
    fill_external_remapping_file,
    fill_external_remapping_with_results,
    fill_external_rmsnorm_remapping_file,
)
from .solver import solve_edge
from .validation import emit_trace_artifacts, run_validation, write_validation_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="address-remap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    edge_parser = subparsers.add_parser("solve-edge")
    edge_parser.add_argument("path")
    edge_parser.add_argument("--format", choices=["json"], default="json")
    edge_parser.add_argument("--output")

    graph_parser = subparsers.add_parser("solve-graph")
    graph_parser.add_argument("path")
    graph_parser.add_argument("--format", choices=["json"], default="json")
    graph_parser.add_argument("--output")

    rmsnorm_parser = subparsers.add_parser("fill-rmsnorm-remapping")
    rmsnorm_parser.add_argument("path")
    rmsnorm_parser.add_argument("--format", choices=["json"], default="json")
    rmsnorm_parser.add_argument("--output")

    remapping_parser = subparsers.add_parser("fill-remapping")
    remapping_parser.add_argument("path")
    remapping_parser.add_argument("--format", choices=["json"], default="json")
    remapping_parser.add_argument("--output")
    remapping_parser.add_argument("--dump-solver-results", nargs="?", const="__default__")

    perf_parser = subparsers.add_parser("analyze-performance")
    perf_parser.add_argument("path")
    perf_parser.add_argument("--format", choices=["json"], default="json")
    perf_parser.add_argument("--output")
    perf_parser.add_argument("--config")
    perf_parser.add_argument("--emit-trace", action="store_true")
    perf_parser.add_argument("--validate", action="store_true")
    perf_parser.add_argument("--ramulator-root")

    roofline_parser = subparsers.add_parser("plot-roofline")
    roofline_parser.add_argument("path")
    roofline_parser.add_argument("--format", choices=["json"], default="json")
    roofline_parser.add_argument("--output")
    roofline_parser.add_argument("--config")
    roofline_parser.add_argument("--mode", choices=["baseline", "remap", "remap_interleave"], default="remap")

    validation_parser = subparsers.add_parser("run-validation")
    validation_parser.add_argument("path")
    validation_parser.add_argument("--format", choices=["json"], default="json")
    validation_parser.add_argument("--output")
    validation_parser.add_argument("--config")
    validation_parser.add_argument("--ramulator-root")

    args = parser.parse_args()
    hardware = HardwareSpec()

    if args.command == "solve-edge":
        payload = load_graph_file(args.path)
        result = solve_edge(
            producer_layout=LayoutSpec.from_dict(payload["producer_layout"]),
            consumer_layout=LayoutSpec.from_dict(payload["consumer_layout"]),
            shape_bindings={str(k): int(v) for k, v in dict(payload["shape_bindings"]).items()},
            memory_dtype=payload.get("memory_dtype"),
            hw_cfg=hardware,
            producer=str(payload.get("producer", "")),
            consumer=str(payload.get("consumer", "")),
            tensor_name=str(payload.get("tensor_name", "")),
        )
        rendered = _render_json(result.to_dict())
        print(rendered)
        _write_output_file(args.path, args.output, rendered)
        return

    if args.command == "analyze-performance":
        perf_hardware, perf_cfg = load_performance_config(args.config)
        payload = analyze_graph_performance(
            load_graph_file(args.path),
            perf_hardware,
            perf_cfg,
            include_request_traces=(args.emit_trace or args.validate),
        )
        output_path, _ = write_performance_outputs(args.path, payload, args.output)
        if args.emit_trace:
            emit_trace_artifacts(payload, str(output_path), perf_hardware, ramulator_root=args.ramulator_root)
        if args.validate:
            validation_payload = run_validation(
                payload,
                perf_hardware,
                perf_cfg,
                str(output_path),
                ramulator_root=args.ramulator_root,
            )
            payload["validation"] = validation_payload
            output_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
            write_validation_outputs(str(output_path), validation_payload)
        print(_render_json(payload))
        return

    if args.command == "run-validation":
        perf_hardware, perf_cfg = load_performance_config(args.config)
        payload = analyze_graph_performance(
            load_graph_file(args.path),
            perf_hardware,
            perf_cfg,
            include_request_traces=True,
        )
        output_path, _ = write_performance_outputs(args.path, payload, args.output)
        validation_payload = run_validation(
            payload,
            perf_hardware,
            perf_cfg,
            str(output_path),
            ramulator_root=args.ramulator_root,
        )
        print(_render_json(validation_payload))
        write_validation_outputs(str(output_path), validation_payload)
        return

    if args.command == "plot-roofline":
        payload = generate_roofline_artifacts(
            graph_path=args.path,
            config_path=args.config,
            mode=args.mode,
            explicit_output=args.output,
        )
        print(_render_json(payload))
        return

    if args.command == "fill-rmsnorm-remapping":
        output_path = fill_external_rmsnorm_remapping_file(args.path, output_path=args.output, hw_cfg=hardware)
        print(_render_json(load_graph_file(str(output_path))))
        return

    if args.command == "fill-remapping":
        if args.dump_solver_results is not None:
            source_payload = load_graph_file(args.path)
            filled, solver_results = fill_external_remapping_with_results(
                source_payload,
                hw_cfg=hardware,
            )
            output_path = Path(args.output) if args.output else Path(args.path).with_name(
                f"{Path(args.path).stem}_remapped{Path(args.path).suffix}"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(_render_json(filled) + "\n", encoding="utf-8")

            solver_output_path = (
                _default_fill_remapping_solver_output_path(args.path)
                if args.dump_solver_results == "__default__"
                else Path(args.dump_solver_results)
            )
            solver_output_path.parent.mkdir(parents=True, exist_ok=True)
            solver_output_path.write_text(
                _render_json([result.to_dict() for result in solver_results]) + "\n",
                encoding="utf-8",
            )
        else:
            output_path = fill_external_remapping_file(args.path, output_path=args.output, hw_cfg=hardware)
        print(_render_json(load_graph_file(str(output_path))))
        return

    results = solve_graph(load_graph_file(args.path), hardware)
    rendered = _render_json([result.to_dict() for result in results])
    print(rendered)
    _write_output_file(args.path, args.output, rendered)


def _write_output_file(input_path: str, explicit_output: str, rendered: str) -> Path:
    output_path = Path(explicit_output) if explicit_output else _default_solver_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    return output_path


def _render_json(payload: object) -> str:
    rendered = json.dumps(payload, indent=2)
    return re.sub(
        r'"permutation": \[\n((?:\s+\d+,?\n)+)\s+\]',
        _collapse_permutation_block,
        rendered,
    )


def _collapse_permutation_block(match: re.Match[str]) -> str:
    numbers = re.findall(r"\d+", match.group(1))
    return f'"permutation": [{", ".join(numbers)}]'


def _default_solver_output_path(input_path: str) -> Path:
    source = Path(input_path)
    return Path.cwd() / "outputs" / "solver" / f"{source.stem}_result.json"


def _default_fill_remapping_solver_output_path(input_path: str) -> Path:
    source = Path(input_path)
    return PROJECT_ROOT / "outputs" / "solver" / source.stem / f"{source.stem}_solver_results.json"


if __name__ == "__main__":
    main()
