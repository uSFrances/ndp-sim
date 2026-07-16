"""Generate model_execplan operator graphs for the supported Decode kernels.

The shapes in these graphs use the same string-macro convention as the Prefill
templates (e.g. ``"hidden_size//used_slices"``) so that ``gen_layer0_oplist.py``
can merge them into a ``layer0_decode.json`` with the identical structure as
``layer0_0610_remapped.json`` – only the operator type names differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decode_ops import SUPPORTED_DECODE_OPERATORS, load_decode_config


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "model_execplan" / "op_json"

# ---------------------------------------------------------------------------
# Shared helpers that produce the same JSON shape as the Prefill templates.
# ---------------------------------------------------------------------------

def _identity_remap() -> list[int]:
    """Return ``[0, 1, …, 25]``, matching the 26-entry remapping used in Prefill."""
    return list(range(26))


def _input_tensor(
    shape: list[Any],
    dtype: str,
    *,
    source: Any = {"type": "external"},
    special_type: str | None = None,
    bank_interleave: int = 1,
    remapping: Any = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "bank_interleave": bank_interleave,
        "shape": shape,
    }
    if remapping is not None:
        value["remapping"] = remapping
    if special_type is not None:
        value["type"] = special_type
    value["source"] = source
    if dtype:
        value["dtype"] = dtype
    return value


def _output_tensor(
    shape: list[Any],
    dtype: str = "",
    *,
    bank_interleave: int | None = None,
    remapping: Any = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"shape": shape}
    if remapping is not None:
        value["remapping"] = remapping
    if bank_interleave is not None:
        value["bank_interleave"] = bank_interleave
    if dtype:
        value["dtype"] = dtype
    return value


# ---------------------------------------------------------------------------
# Per-operator I/O shapes expressed with the SAME macro strings that the
# Prefill rmsnorm.json / softmax.json / gemm_ring.json templates use.
# ---------------------------------------------------------------------------

def build_execplan_operators(config: dict[str, int | str]) -> list[dict[str, Any]]:
    used_slices = int(config["used_slices"])
    mask = "0b" + "1" * used_slices
    remap = _identity_remap()

    # Short-hands for the macro strings that the existing register-update
    # handler knows how to evaluate.
    HID = "hidden_size//used_slices"           # 896//28 = 32
    HID_FULL = "hidden_size"                    # 896
    SEQ = "sequence_length"                     # Prefill used this; decode reuses it.
    USL = "used_slices"                         # 28
    HEAD_DIM_SLICE = "head_dim//slice_per_head" # 128//4 = 32
    ATTN_SLICE = "decode_attention_length//slice_per_head"  # 32//4 = 8
    S_PER_HEAD = "slice_per_head"              # 4

    # Each entry:  ({port: input_dict}, output_dict)
    io: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        # -- summac ----------------------------------------------------------
        "decode_summac_fp32N_fp32N": (
            {"A": _input_tensor([1, 1, HID], "fp32", remapping=None)},
            _output_tensor([1, 1, 1], remapping=None),
        ),
        "decode_summac_fp16N_fp32N": (
            {"A": _input_tensor([1, 1, HID], "fp16", remapping=None)},
            _output_tensor([1, 1, 1], remapping=None),
        ),
        # -- remote_sum (chained from preceding summac via op-1) -------------
        "decode_remote_sum_fp32N_fp32N": (
            {"A": _input_tensor(
                [1, USL, 1], "fp32",
                special_type="slice0",
                source="op-1",
                remapping=None,
            )},
            _output_tensor([1, 1, 1]),
        ),
        # -- mac_SFU (chained from preceding remote_sum via op-1) -----------
        "decode_mac_SFU_fp32N_fp32N": (
            {"A": _input_tensor(
                [HID_FULL, 1, 1], "fp32",
                source="op-1",
                remapping=remap,
            )},
            _output_tensor([1, 1, 1]),
        ),
        # -- max (per-head attention reduction) ------------------------------
        "decode_max_fp32N_fp32N": (
            {"A": _input_tensor([1, 1, ATTN_SLICE], "fp32", remapping=remap)},
            _output_tensor([1, 1, 1], remapping=None),
        ),
        # -- sum_rec (per-head attention reduction) --------------------------
        "decode_sum_rec_fp32N_fp32N": (
            {"A": _input_tensor([1, 1, ATTN_SLICE], "fp32", remapping=remap)},
            _output_tensor([1, 1, 1], remapping=None),
        ),
        # -- mul (vector × vector, cast to fp16) -----------------------------
        "decode_mul_fp32N_fp32N_fp16N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], "fp16", bank_interleave=1, remapping=None),
        ),
        # -- add (fp16 + fp32 residual) --------------------------------------
        "decode_add_fp16N_fp32N_fp32N": (
            {
                "A": _input_tensor([1, 1, HID], "fp16", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], remapping=None),
        ),
        # -- gemv_ring -------------------------------------------------------
        "decode_gemv_ring": (
            {
                "A": _input_tensor(
                    [1, 1, HID], "fp16",
                    remapping=remap,
                ),
                "B": _input_tensor(
                    [1, HID, HID_FULL], "fp16",
                    source={"type": "external"},
                    remapping=remap,
                ),
                "B'": _input_tensor(
                    [1, HID, HID_FULL], "fp16",
                    source={"type": "external", "dtype": "fp16"},
                    remapping=remap,
                ),
            },
            _output_tensor([1, 1, HID], "fp16", bank_interleave=1, remapping=None),
        ),
        # -- gemv_local ------------------------------------------------------
        "decode_gemv_local": (
            {
                "A": _input_tensor([1, 1, HEAD_DIM_SLICE], "fp16", remapping=remap),
                "B": _input_tensor(
                    [1, SEQ, SEQ], "fp16",
                    source={"type": "external"},
                    remapping=remap,
                ),
                "B'": _input_tensor(
                    [1, SEQ, SEQ], "fp16",
                    source={"type": "external", "dtype": "fp16"},
                    remapping=remap,
                ),
            },
            _output_tensor([1, 1, SEQ], "fp16", bank_interleave=1, remapping=None),
        ),
        # =====================================================================
        #  Inferred missing operators  (MN→N naming convention)
        # =====================================================================
        # -- mul (vector × scalar → vector, fp32) -----------------------------
        "decode_mul_fp32N_fp32_fp32N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, 1], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], remapping=None),
        ),
        # -- mul (vector × vector → vector, fp32) -----------------------------
        "decode_mul_fp32N_fp32N_fp32N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], remapping=None),
        ),
        # -- mul (vector × scalar → vector, cast to fp16) ---------------------
        "decode_mul_fp32N_fp32_fp16N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, 1], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], "fp16", bank_interleave=1, remapping=None),
        ),
        # -- mul (fp32 vector × fp16 vector → vector, fp16) -------------------
        "decode_mul_fp32N_fp16N_fp16N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp16", remapping=remap),
            },
            _output_tensor([1, 1, HID], "fp16", bank_interleave=1, remapping=None),
        ),
        # -- add (fp32 + fp32 → fp16) -----------------------------------------
        "decode_add_fp32N_fp32N_fp16N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], "fp16", bank_interleave=1, remapping=None),
        ),
        # -- add (fp32 + fp32 → fp32) -----------------------------------------
        "decode_add_fp32N_fp32N_fp32N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], remapping=None),
        ),
        # -- add (fp32 + fp16 → fp32) -----------------------------------------
        "decode_add_fp32N_fp16N_fp32N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp16", remapping=remap),
            },
            _output_tensor([1, 1, HID], remapping=None),
        ),
        # -- add (fp16 + fp32 → fp16, "V" variant) ----------------------------
        "decode_add_fp16N_fp32N_fp16N": (
            {
                "A": _input_tensor([1, 1, HID], "fp16", remapping=remap),
                "B": _input_tensor([1, 1, HID], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], "fp16", bank_interleave=1, remapping=None),
        ),
        # -- silu (fp16 → fp32) -----------------------------------------------
        "decode_silu_fp16N_fp32N": (
            {"A": _input_tensor([1, 1, HID], "fp16", remapping=remap)},
            _output_tensor([1, 1, HID], remapping=None),
        ),
        # -- sub_SFU (vector − scalar → vector) -------------------------------
        "decode_sub_SFU_fp32N_fp32_fp32N": (
            {
                "A": _input_tensor([1, 1, HID], "fp32", remapping=remap),
                "B": _input_tensor([1, 1, 1], "fp32", remapping=remap),
            },
            _output_tensor([1, 1, HID], remapping=None),
        ),
    }

    operators: list[dict[str, Any]] = []
    for spec in SUPPORTED_DECODE_OPERATORS:
        inputs, output = io[spec.name]
        operators.append(
            {
                "id": spec.op_id,
                "type": spec.name,
                "used_slices": mask,
                "inputs": inputs,
                "output": output,
            }
        )
    return operators


def generate_execplan_inputs(config_path: Path, output_dir: Path) -> list[Path]:
    config = load_decode_config(config_path)
    operators = build_execplan_operators(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_base: dict[str, Any] = {
        "params": config,
        "used_slices": int(config["used_slices"]),
    }

    written: list[Path] = []
    combined = {**payload_base, "operators": operators}
    combined_path = output_dir / "decode_all.json"
    combined_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written.append(combined_path)

    for operator in operators:
        single_path = output_dir / f"{operator['type']}_graph.json"
        single_path.write_text(
            json.dumps({**payload_base, "operators": [operator]}, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(single_path)

    print(f"[decode-execplan] wrote {len(written)} graph file(s) to {output_dir}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Decode execution-plan JSON inputs.")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_execplan_inputs(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
