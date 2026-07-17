"""Generate the Decode program JSON from the Golden manifest.

Reads ``python_golden_decode/manifest.json`` and produces a
``layer0_decode.json`` whose structure mirrors the Prefill
``layer0_0610_remapped.json``: each Golden instance becomes an
operator entry with correct source references (branches/merges)
and prefill-compatible shape macros.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from decode_ops import load_decode_config


BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Hardcoded dataflow: for each of the 43 decode layer operators,
# (op_index, port) → source operator id or "external".
# Built from the prefill ``layer0_0610_remapped.json`` dataflow.
# ---------------------------------------------------------------------------
_HID = "hidden_size//used_slices"
_HID_FULL = "hidden_size"
_INT = "intermediate_size//used_slices"
_INT_FULL = "intermediate_size"
_USL = "used_slices"
_SEQ = "sequence_length"
_HD_SLICE = "head_dim//slice_per_head"
_HD = "head_dim"
_NH = "num_attention_heads"
_SPH = "slice_per_head"
_ATN_SLICE = "decode_attention_length//slice_per_head"
_ATN = "decode_attention_length"

# KV padding macros (matching prefill: 4-slice K/V with kv_padding)
_KV_PAD_A = "kv_padding_a"   # 256
_KV_PAD_B = "kv_padding_b"   # 1024
_KV_N = "num_key_value_heads*head_dim//slice_per_head"  # 32

# GEMV B/B' port helper — prefill convention: [K, 1, N_slice], bank_interleave=1
def _gemv_b(shape: list[Any], source: str = "ext") -> dict[str, Any]:
    return {"shape": shape, "dtype": "fp16", "bank_interleave": 1, "source": source}

def _gemv_bp(shape: list[Any], source: str = "ext") -> dict[str, Any]:
    return {"shape": shape, "dtype": "fp16", "bank_interleave": 1, "source": source}

# ---------------------------------------------------------------------------
# 43-op decode layer: shape definitions and source references
# ---------------------------------------------------------------------------
_DECODE_LAYOUT: list[dict[str, Any]] = [
    # op0  summac (RMS norm part 1) — external hidden input
    {
        "id": "op0", "type": "decode_summac_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op1  remote_sum (RMS norm part 2) — only slice 0
    {
        "id": "op1", "type": "decode_remote_sum_fp32N_fp32N",
        "used_slices_mask": "0b1",
        "inputs": {"A": {"shape": [1, _USL, 1], "dtype": "f32", "type": "slice0", "source": "op0"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op2  mac_SFU (RMS norm part 3)
    {
        "id": "op2", "type": "decode_mac_SFU_fp32N_fp32N",
        "inputs": {"A": {"shape": [_HID_FULL, 1, 1], "dtype": "f32", "source": "op1"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op3  mul_scale (RMS norm apply — A=external hidden, B=op2 scale)
    {
        "id": "op3", "type": "decode_mul_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
            "B": {"shape": [1, 1, 1], "dtype": "f32", "source": "op2"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op4  mul_cast (RMS norm → fp16 — A=external scale, B=op3)
    {
        "id": "op4", "type": "decode_mul_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op3"},
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op5  gemv_ring (Q projection) — prefill: A=[K_slice,1,1], B=[K,1,N_slice]
    {
        "id": "op5", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_HID, 1, 1], "dtype": "fp16", "source": "op4"},
            "B": _gemv_b([_HID_FULL, 1, _HID]),
            "B'": _gemv_bp([_HID_FULL, 1, _HID]),
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op6  add residual (Q + residual) — A=fp16 from op5, B=fp32 external
    {
        "id": "op6", "type": "decode_add_fp16N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "fp16", "source": "op5"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op7  RoPE cos mul (A←op6, B←external cos)
    {
        "id": "op7", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op6"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op8  RoPE sin mul (A←op6, B←external sin_rot — table rearranged, no routing)
    {
        "id": "op8", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op6"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op9  RoPE merge add (A←op7, B←op8)
    {
        "id": "op9", "type": "decode_add_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op7"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op8"},
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op10 summac (KV RMS norm part 1) — external kv input
    {
        "id": "op10", "type": "decode_summac_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op11 remote_sum (KV RMS norm part 2) — only slice 0
    {
        "id": "op11", "type": "decode_remote_sum_fp32N_fp32N",
        "used_slices_mask": "0b1",
        "inputs": {"A": {"shape": [1, _USL, 1], "dtype": "f32", "type": "slice0", "source": "op10"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op12 mac_SFU (KV RMS norm part 3)
    {
        "id": "op12", "type": "decode_mac_SFU_fp32N_fp32N",
        "inputs": {"A": {"shape": [_HID_FULL, 1, 1], "dtype": "f32", "source": "op11"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op13 mul_scale (KV RMS norm apply — A=external kv, B=op12)
    {
        "id": "op13", "type": "decode_mul_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
            "B": {"shape": [1, 1, 1], "dtype": "f32", "source": "op12"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op14 mul_cast (KV → fp16 — A=external scale, B=op13)
    {
        "id": "op14", "type": "decode_mul_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op13"},
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op15 gemv_ring (K projection) — A=[HID,1,1], B=[HID_FULL,1,HID] (replicated 7×)
    {
        "id": "op15", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_HID, 1, 1], "dtype": "fp16", "source": "op14"},
            "B": _gemv_b([_HID_FULL, 1, _HID]),
            "B'": _gemv_bp([_HID_FULL, 1, _HID]),
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op16 add residual (K + residual) — A=fp16 from op15, B=fp32 external
    {
        "id": "op16", "type": "decode_add_fp16N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "fp16", "source": "op15"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op17 RoPE cos mul (K)
    {
        "id": "op17", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op16"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op18 RoPE sin mul (K) — table rearranged, no routing
    {
        "id": "op18", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op16"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op19 RoPE merge add (K)
    {
        "id": "op19", "type": "decode_add_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op17"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op18"},
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op20 gemv_ring (V projection) — same as K
    {
        "id": "op20", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_HID, 1, 1], "dtype": "fp16", "source": "op14"},
            "B": _gemv_b([_HID_FULL, 1, _HID]),
            "B'": _gemv_bp([_HID_FULL, 1, _HID]),
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op21 add residual (V) — A=fp16 from op20, B=fp32 external
    {
        "id": "op21", "type": "decode_add_fp16N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "fp16", "source": "op20"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op22 gemv_local (QK^T — A=Q from op9, B/B'=K **cache** from external)
    # Decode: Q is current token (vector), K cache has attention_length positions.
    # op19 only produces K for the *current* token → stored to cache externally.
    # op22 reads the full K cache (external), NOT op19.
    {
        "id": "op22", "type": "decode_gemv_local",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [1, 1, _HD_SLICE], "dtype": "fp16", "source": "op9"},
            "B": {"shape": [1, _ATN, _HD_SLICE], "dtype": "fp16", "bank_interleave": 1, "source": "ext"},
            "B'": {"shape": [1, _ATN, _HD_SLICE], "dtype": "fp16", "bank_interleave": 1, "source": "ext"},
        },
        "output": {"shape": [1, 1, _ATN], "dtype": "fp16"},
    },
    # op23 remote_sum (QK^T aggregation) — only slice 0
    {
        "id": "op23", "type": "decode_remote_sum_fp32N_fp32N",
        "used_slices_mask": "0b1",
        "inputs": {"A": {"shape": [1, 1, _ATN], "dtype": "f32", "type": "slice0", "source": "op22"}},
        "output": {"shape": [1, 1, _ATN]},
    },
    # op24 add_mask (scores + mask — A←op23 broadcast, B←external mask)
    {
        "id": "op24", "type": "decode_add_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _ATN], "dtype": "f32", "source": "op23"},
            "B": {"shape": [1, 1, _ATN], "dtype": "f32", "source": "ext"},
        },
        "output": {"shape": [1, 1, _ATN]},
    },
    # op25 max (per-head local maxima — A←op24, reduces attention_length dim)
    {
        "id": "op25", "type": "decode_max_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _ATN], "dtype": "f32", "source": "op24"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op26 sub_SFU (scores - max — A←op24, B←op25 broadcast scalar)
    {
        "id": "op26", "type": "decode_sub_SFU_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _ATN], "dtype": "f32", "source": "op24"},
            "B": {"shape": [1, 1, 1], "dtype": "f32", "source": "op25"},
        },
        "output": {"shape": [1, 1, _ATN]},
    },
    # op27 sum_rec (per-head sum reciprocals — A←op26)
    {
        "id": "op27", "type": "decode_sum_rec_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _ATN], "dtype": "f32", "source": "op26"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op28 mul_softmax (exp(x-max) * 1/sum — A←op26, B←op27 broadcast scalar)
    {
        "id": "op28", "type": "decode_mul_fp32N_fp32_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _ATN], "dtype": "f32", "source": "op26"},
            "B": {"shape": [1, 1, 1], "dtype": "f32", "source": "op27"},
        },
        "output": {"shape": [1, 1, _ATN], "dtype": "fp16"},
    },
    # op29 gemv_local (SV — A=softmax from op28, B/B'=V **cache** from external)
    # Decode: op21 only produces V for the *current* token → stored to cache.
    # op29 reads the full V cache (external), NOT op21.
    {
        "id": "op29", "type": "decode_gemv_local",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [1, 1, _ATN], "dtype": "fp16", "source": "op28"},
            "B": {"shape": [1, _HD_SLICE, _ATN], "dtype": "fp16", "bank_interleave": 1, "source": "ext"},
            "B'": {"shape": [1, _HD_SLICE, _ATN], "dtype": "fp16", "bank_interleave": 1, "source": "ext"},
        },
        "output": {"shape": [1, 1, _HD_SLICE], "dtype": "fp16"},
    },
    # op30 gemv_ring (output projection) — 28-slice, A per-slice = hidden_size//used_slices
    {
        "id": "op30", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_HID, 1, 1], "dtype": "fp16", "source": "op29"},
            "B": _gemv_b([_HID_FULL, 1, _HID]),
            "B'": _gemv_bp([_HID_FULL, 1, _HID]),
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op31 add residual (output + residual — A=external residual, B=op30)
    {
        "id": "op31", "type": "decode_add_fp32N_fp16N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
            "B": {"shape": [1, 1, _HID], "dtype": "fp16", "source": "op30"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op32 summac (FFN RMS norm part 1 — A←op31)
    {
        "id": "op32", "type": "decode_summac_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op31"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op33 remote_sum (FFN RMS norm part 2) — only slice 0
    {
        "id": "op33", "type": "decode_remote_sum_fp32N_fp32N",
        "used_slices_mask": "0b1",
        "inputs": {"A": {"shape": [1, _USL, 1], "dtype": "f32", "type": "slice0", "source": "op32"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op34 mac_SFU (FFN RMS norm part 3)
    {
        "id": "op34", "type": "decode_mac_SFU_fp32N_fp32N",
        "inputs": {"A": {"shape": [_HID_FULL, 1, 1], "dtype": "f32", "source": "op33"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op35 mul_scale (FFN RMS apply — A←op31, B←op34)
    {
        "id": "op35", "type": "decode_mul_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op31"},
            "B": {"shape": [1, 1, 1], "dtype": "f32", "source": "op34"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op36 mul_cast (FFN → fp16 — A=external scale, B=op35)
    {
        "id": "op36", "type": "decode_mul_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "ext"},
            "B": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op35"},
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op37 gemv_ring (FFN gate) — prefill: B=[hidden_size,1,intermediate//slices]
    {
        "id": "op37", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_HID, 1, 1], "dtype": "fp16", "source": "op36"},
            "B": _gemv_b([_HID_FULL, 1, _INT]),
            "B'": _gemv_bp([_HID_FULL, 1, _INT]),
        },
        "output": {"shape": [1, 1, _INT], "dtype": "fp16"},
    },
    # op38 gemv_ring (FFN up — same input as op37)
    {
        "id": "op38", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_HID, 1, 1], "dtype": "fp16", "source": "op36"},
            "B": _gemv_b([_HID_FULL, 1, _INT]),
            "B'": _gemv_bp([_HID_FULL, 1, _INT]),
        },
        "output": {"shape": [1, 1, _INT], "dtype": "fp16"},
    },
    # op39 silu (gate activation)
    {
        "id": "op39", "type": "decode_silu_fp16N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _INT], "dtype": "fp16", "source": "op37"}},
        "output": {"shape": [1, 1, _INT]},
    },
    # op40 mul (silu(gate) * up — A←op39, B←op38)
    {
        "id": "op40", "type": "decode_mul_fp32N_fp16N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _INT], "dtype": "f32", "source": "op39"},
            "B": {"shape": [1, 1, _INT], "dtype": "fp16", "source": "op38"},
        },
        "output": {"shape": [1, 1, _INT], "dtype": "fp16"},
    },
    # op41 gemv_ring (FFN output) — prefill: B=[intermediate,1,hidden//slices]
    {
        "id": "op41", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_INT, 1, 1], "dtype": "fp16", "source": "op40"},
            "B": _gemv_b([_INT_FULL, 1, _HID]),
            "B'": _gemv_bp([_INT_FULL, 1, _HID]),
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op42 add residual (FFN output + residual — A←op31, B←op41)
    {
        "id": "op42", "type": "decode_add_fp32N_fp16N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "dtype": "f32", "source": "op31"},
            "B": {"shape": [1, 1, _HID], "dtype": "fp16", "source": "op41"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
]


def _compact_inline_arrays(json_text: str) -> str:
    """Collapse small integer-only arrays onto a single line."""
    def _collapse(m: re.Match) -> str:
        content = m.group(1).strip()
        if re.fullmatch(r"[0-9,\s]+", content):
            return "[" + re.sub(r"\s+", " ", content).strip() + "]"
        return m.group(0)
    return re.sub(r"\[([^\]]*)\]", _collapse, json_text)


def _resolve_source(src: str) -> Any:
    """Convert internal source string to JSON source value."""
    if src == "ext":
        return {"type": "external"}
    return src


def generate_program_json(
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> Path:
    config = load_decode_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    used_slices = int(config["used_slices"])
    mask_all = "0b" + "1" * used_slices

    operators: list[dict[str, Any]] = []
    for layout_op in _DECODE_LAYOUT:
        # Resolve used_slices mask
        us_mask_spec = layout_op.get("used_slices_mask", None)
        if us_mask_spec is None or us_mask_spec == _USL:
            us_mask = mask_all
        elif us_mask_spec == "1":
            us_mask = "0b1"
        else:
            us_mask = us_mask_spec

        op_entry: dict[str, Any] = {
            "id": layout_op["id"],
            "type": layout_op["type"],
            "used_slices": us_mask,
            "inputs": {},
            "output": {},
        }

        # ---- Inputs --------------------------------------------------------
        for port, inp_spec in layout_op.get("inputs", {}).items():
            inp: dict[str, Any] = {
                "shape": inp_spec["shape"],
                "bank_interleave": inp_spec.get("bank_interleave", 1),
            }
            dtype = inp_spec.get("dtype", "")
            if dtype == "fp16":
                inp["dtype"] = "fp16"
            elif dtype == "fp32" or dtype == "f32":
                pass  # fp32 is implicit, no dtype field
            elif dtype:
                inp["dtype"] = dtype
            if inp_spec.get("remapping") is not None:
                inp["remapping"] = inp_spec["remapping"]
            inp["source"] = _resolve_source(inp_spec["source"])
            op_entry["inputs"][port] = inp

        # ---- Output --------------------------------------------------------
        out_spec = layout_op.get("output", {})
        op_entry["output"]["shape"] = out_spec.get("shape", [])
        if out_spec.get("dtype") == "fp16":
            op_entry["output"]["bank_interleave"] = 1
            op_entry["output"]["dtype"] = "fp16"
        if out_spec.get("type"):
            op_entry["output"]["type"] = out_spec["type"]

        operators.append(op_entry)

    payload: dict[str, Any] = {
        "params": {
            **config,
            "kv_padding_a": int(config.get("kv_padding", 256)),
            "kv_padding_b": int(config.get("kv_padding", 256)) * int(config.get("slice_per_head", 4)),
        },
        "used_slices": used_slices,
        "operators": operators,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, ensure_ascii=False)
    output_path.write_text(
        _compact_inline_arrays(raw) + "\n",
        encoding="utf-8",
    )
    print(f"[decode-program] wrote {len(operators)} operators to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate decode layer program JSON from Golden manifest."
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "python_golden_decode" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "model_execplan" / "examples" / "layer0_decode.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_program_json(args.config, args.manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
