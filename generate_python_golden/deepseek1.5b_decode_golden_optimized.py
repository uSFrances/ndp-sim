"""Generate deterministic Golden tensors for the optimized Decode layer.

This is the optimized counterpart of ``deepseek1.5b_decode_golden.py``.
It uses :func:`build_decode_golden_cases_optimized` which produces golden
data matching ``layer0_decode_optimized.json``:

- ``prefill_remote_summac`` + ``decode_sum`` instead of ``summac`` + ``remote_sum``
- single ``decode_gemv_local_qkt`` + ``prefill_remote_sum_qkt``
- Per-head half-swap RoPE merge instead of ``rope_slice_xor2`` routing

Full size is supported through config padding: the real model has 12 Q heads
and ``hidden=1536``; the 28-slice hardware layout pads them to 14 heads and
``hidden_padded=1792`` so that each head owns 2 slices and every four-slice
group covers two consecutive Q heads.  The two padding heads and the extra
hidden columns/rows are zero.

When the real 1.5B ``model_weights_full/`` folder is absent, deterministic
synthetic real-size weights are used.  They are shared with the KV-cache
mini-prefill so the generated layer and cache stay self-consistent.

The original ``deepseek1.5b_decode_golden.py`` and its output directory
``python_golden_decode/`` are untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from decode_ops import (
    DECODE_OP_REGISTRY,
    SUPPORTED_DECODE_OPERATORS,
    build_decode_golden_cases_optimized,
    decode_padded_dims,
    load_decode_config,
    resolve_target_names,
)
from decode_data_loader import (
    LayerWeights,
    KVCache,
    generate_kv_cache,
    load_kv_cache,
    load_layer_weights,
    save_kv_cache,
    synthesize_layer_weights,
)
from tensor_io import dtype_tag, save_golden_tensor


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optimized Decode layer Golden trace."
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to python_golden_decode_optimized for the small-size config "
            "and python_golden_decode_optimized_full for a padded full-size config."
        ),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the output directory before generation.",
    )
    return parser.parse_args()


def prepare_output_directory(output_dir: Path, keep_existing: bool) -> None:
    resolved = output_dir.resolve()
    if not keep_existing and resolved.exists():
        if resolved == BASE_DIR.resolve() or BASE_DIR.resolve() not in resolved.parents:
            raise ValueError(f"refusing to clear output outside the project: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "sub_ops").mkdir(exist_ok=True)


def _default_output_dir(config: dict[str, object]) -> Path:
    dims = decode_padded_dims(config)
    padded = (
        dims["heads_padded"] != dims["heads"]
        or dims["hidden_padded"] != dims["hidden"]
    )
    suffix = "_full" if padded else ""
    return BASE_DIR / f"python_golden_decode_optimized{suffix}"


def generate_decode_golden_optimized(
    config_path: Path,
    output_dir: Path | None = None,
    keep_existing: bool = False,
    use_real_weights: bool = True,
) -> Path:
    config = load_decode_config(config_path)
    if output_dir is None:
        output_dir = _default_output_dir(config)
    prepare_output_directory(output_dir, keep_existing=keep_existing)

    # Load real model weights and KV cache
    weights = None
    kv_cache = None
    if use_real_weights:
        try:
            weights = load_layer_weights(0, config)
            print("[decode-golden-opt] loaded real weights for configured size")
        except (FileNotFoundError, ValueError) as exc:
            print(
                "[decode-golden-opt] WARNING: real weights for the configured size "
                f"not available ({exc}); using deterministic synthetic weights"
            )
            weights = synthesize_layer_weights(config)

        try:
            kv_cache = load_kv_cache(config=config)
            print("[decode-golden-opt] loaded KV cache from kv_cache/")
        except (FileNotFoundError, ValueError):
            print("[decode-golden-opt] generating KV cache (one-time prefill pass) ...")
            kv_cache = generate_kv_cache(
                weights=weights,
                config=config,
            )
            print("[decode-golden-opt] KV cache generated and saved")

    cases = build_decode_golden_cases_optimized(config, weights=weights, kv_cache=kv_cache)

    manifest: dict[str, object] = {
        "config": config,
        "decode_step": 0,
        "variant": "optimized",
        "operator_types": [spec.name for spec in SUPPORTED_DECODE_OPERATORS],
        "instances": [],
    }
    manifest_instances: list[dict[str, object]] = []

    for idx, case in enumerate(cases):
        input_entries: list[dict[str, object]] = []
        for index, tensor in enumerate(case.inputs):
            tensor_name = f"{case.instance_id}_in{index}"
            path = save_golden_tensor(output_dir, tensor_name, tensor)
            input_entries.append(
                {
                    "index": index,
                    "port": case.spec.input_ports[index],
                    "path": path.name,
                    "shape": list(tensor.shape),
                    "dtype": dtype_tag(tensor.dtype),
                }
            )
        output_name = f"{case.instance_id}_out"
        output_path = save_golden_tensor(output_dir, output_name, case.output)
        manifest_instances.append(
            {
                "instance_id": case.instance_id,
                "layer_idx": idx,
                "op_name": case.spec.name,
                "op_id": case.spec.op_id,
                "slice_policy": case.spec.slice_policy,
                "hardware_json": case.spec.hardware_json,
                "inputs": input_entries,
                "output": {
                    "port": "D",
                    "path": output_path.name,
                    "shape": list(case.output.shape),
                    "dtype": dtype_tag(case.output.dtype),
                },
            }
        )
        print(f"[decode-golden-opt] {case.instance_id}  ({case.spec.name})")

    manifest["instances"] = manifest_instances
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[decode-golden-opt] wrote {len(manifest_instances)} layer instances "
          f"({len(set(c.spec.name for c in cases))} operator types) to {output_dir}")
    return manifest_path


def main() -> int:
    args = parse_args()
    generate_decode_golden_optimized(
        config_path=args.config,
        output_dir=args.output_dir,
        keep_existing=args.keep_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
