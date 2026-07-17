"""Generate deterministic Golden tensors for a complete Decode layer.

Follows the same strategy as the Prefill golden script: one shared input
token flows through the entire layer and every intermediate tensor is saved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from decode_ops import (
    DECODE_OP_REGISTRY,
    SUPPORTED_DECODE_OPERATORS,
    build_decode_golden_cases,
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
)
from tensor_io import dtype_tag, save_golden_tensor


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one deterministic Decode layer Golden trace."
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "python_golden_decode"
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


def generate_decode_golden(
    config_path: Path,
    output_dir: Path,
    keep_existing: bool = False,
    use_real_weights: bool = True,
) -> Path:
    config = load_decode_config(config_path)
    prepare_output_directory(output_dir, keep_existing=keep_existing)

    # Load real model weights and KV cache
    weights = None
    kv_cache = None
    if use_real_weights:
        try:
            weights = load_layer_weights(0)
            print("[decode-golden] loaded real weights from model_weights_32/")
        except FileNotFoundError:
            print("[decode-golden] WARNING: model_weights_32 not found, using random weights")

        try:
            kv_cache = load_kv_cache()
            print("[decode-golden] loaded KV cache from kv_cache/")
        except FileNotFoundError:
            print("[decode-golden] generating KV cache (one-time prefill pass) ...")
            kv_cache = generate_kv_cache(
                hidden=int(config["hidden_size"]),
                head_dim=int(config["head_dim"]),
                num_heads=int(config["num_attention_heads"]),
                num_kv_heads=int(config["num_key_value_heads"]),
                attention_length=int(config["decode_attention_length"]),
                random_seed=int(config.get("random_seed", 0)),
            )
            print("[decode-golden] KV cache generated and saved")

    cases = build_decode_golden_cases(config, weights=weights, kv_cache=kv_cache)

    manifest: dict[str, object] = {
        "config": config,
        "decode_step": 0,
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
        print(f"[decode-golden] {case.instance_id}  ({case.spec.name})")

    manifest["instances"] = manifest_instances
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[decode-golden] wrote {len(manifest_instances)} layer instances "
          f"({len(set(c.spec.name for c in cases))} operator types) to {output_dir}")
    return manifest_path


def main() -> int:
    args = parse_args()
    generate_decode_golden(
        config_path=args.config,
        output_dir=args.output_dir,
        keep_existing=args.keep_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
