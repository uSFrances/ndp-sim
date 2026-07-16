"""Dispatch Decode vector slicing and GEMV weight relayout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from decode_ops import DECODE_OP_REGISTRY, load_decode_config, resolve_target_names
from single_op_data.decode_passthrough import write_passthrough_case
from single_op_data.relayout_gemv import write_gemv_case


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sliced Decode install data.")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "python_golden_decode" / "manifest.json",
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=BASE_DIR / "single_op_data" / "install_decode",
    )
    parser.add_argument(
        "--target-op",
        default=None,
        help="Operator name or 'all'; defaults to config.target_op_decode.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep install data for operators not selected by this run.",
    )
    return parser.parse_args()


def _check_manifest_config(
    current: dict[str, object], generated: dict[str, object]
) -> None:
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "head_dim",
        "slice_per_head",
        "used_slices",
        "decode_attention_length",
        "random_seed",
    )
    mismatches = [
        key for key in keys if str(current.get(key)) != str(generated.get(key))
    ]
    if mismatches:
        raise ValueError(
            "Decode Golden manifest was generated with a different config; "
            f"regenerate it first. Mismatched keys: {', '.join(mismatches)}"
        )


def _prepare_install_dir(
    install_dir: Path,
    selected_entries: list[dict[str, object]],
    clear_all: bool,
) -> None:
    resolved = install_dir.resolve()
    allowed_parent = (BASE_DIR / "single_op_data").resolve()
    if allowed_parent not in resolved.parents:
        raise ValueError(f"install directory must be inside single_op_data: {resolved}")
    if clear_all and resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    if not clear_all:
        for entry in selected_entries:
            layer_idx = int(entry.get("layer_idx", 0))
            op_dir = resolved / f"op{layer_idx}"
            if op_dir.exists():
                shutil.rmtree(op_dir)


def generate_decode_install(
    config_path: Path,
    manifest_path: Path,
    install_dir: Path,
    target: str | None = None,
    keep_existing: bool = False,
) -> Path:
    config = load_decode_config(config_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Decode Golden manifest not found: {manifest_path}. "
            "Run deepseek1.5b_decode_golden.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_manifest_config(config, manifest["config"])

    instances = manifest.get("instances", manifest.get("operators", []))
    if not instances:
        raise ValueError("Golden manifest contains no operator instances")
    entries_by_name = {
        entry.get("instance_id", entry.get("name")): entry for entry in instances
    }
    names = resolve_target_names(target or str(config.get("target_op_decode", "all")))

    # In layer mode, "all" means every instance; filter by op_name otherwise
    if names == [spec.name for spec in DECODE_OP_REGISTRY.values()]:
        selected_entries = list(instances)
    else:
        # Filter by operator type name
        selected_entries = [
            entry for entry in instances
            if entry.get("op_name", entry.get("name")) in names
        ]
        missing = [n for n in names if n not in {e.get("op_name", e.get("name")) for e in instances}]
        if missing:
            raise ValueError("selected operator types missing from manifest: " + ", ".join(missing))

    _prepare_install_dir(
        install_dir,
        selected_entries,
        clear_all=(not keep_existing and len(selected_entries) == len(instances)),
    )

    result_entries: list[dict[str, object]] = []
    for entry in selected_entries:
        policy = str(entry["slice_policy"])
        layer_idx = int(entry.get("layer_idx", 0))
        op_label = f"op{layer_idx}"
        if policy.startswith("gemv_"):
            write_gemv_case(entry, manifest_path.parent, install_dir, config, op_label)
        else:
            write_passthrough_case(entry, manifest_path.parent, install_dir, config, op_label)
        result_entries.append(
            {
                "id": op_label,
                "name": entry.get("op_name", entry.get("name", "")),
                "slice_policy": policy,
                "slice_count": int(config["used_slices"]),
            }
        )
        print(f"[decode-layout] {op_label}  ({entry.get('op_name', entry.get('name', ''))})")

    result_path = install_dir / "manifest.json"
    result_path.write_text(
        json.dumps(
            {
                "golden_manifest": str(manifest_path.resolve()),
                "operators": result_entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[decode-layout] wrote install data to {install_dir}")
    return result_path


def main() -> int:
    args = parse_args()
    generate_decode_install(
        config_path=args.config,
        manifest_path=args.manifest,
        install_dir=args.install_dir,
        target=args.target_op,
        keep_existing=args.keep_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
