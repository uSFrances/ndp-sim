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

# GEMV B/B' port helper — prefill convention: [K, 1, N_slice], bank_interleave=2
_GEMV_REMAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 20, 21, 22, 23, 24, 25]
def _gemv_b(shape: list[Any], source: str = "ext") -> dict[str, Any]:
    return {"shape": shape, "dtype": "fp16", "bank_interleave": 2, "remapping": list(_GEMV_REMAP), "source": source}

def _gemv_bp(shape: list[Any], source: str = "ext") -> dict[str, Any]:
    return {"shape": shape, "dtype": "fp16", "bank_interleave": 2, "remapping": list(_GEMV_REMAP), "source": source}

def _gemv_local_b(shape: list[Any], source: str = "ext") -> dict[str, Any]:
    return {"shape": shape, "dtype": "fp16", "bank_interleave": 2, "remapping": list(_GEMV_REMAP), "source": source}

# ---------------------------------------------------------------------------
# 43-op decode layer: shape definitions and source references
# ---------------------------------------------------------------------------
_DECODE_LAYOUT: list[dict[str, Any]] = [
    # op0  summac (RMS norm part 1) — external hidden input
    {
        "id": "op0", "type": "decode_summac_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _HID], "source": "ext"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op1  remote_sum (RMS norm part 2) — only slice 0
    {
        "id": "op1", "type": "decode_remote_sum_fp32N_fp32N",
        "used_slices_mask": "0b1",
        "inputs": {"A": {"shape": [1, _USL, 1], "type": "slice0", "source": "op0"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op2  mac_SFU (RMS norm part 3)
    {
        "id": "op2", "type": "decode_mac_SFU_fp32N_fp32N",
        "inputs": {"A": {"shape": [_HID_FULL, 1, 1], "type": "slice0", "source": "op1"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op3  mul_scale (RMS norm apply — A=external hidden, B=op2 scale)
    {
        "id": "op3", "type": "decode_mul_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "source": "ext"},
            "B": {"shape": [1, 1, 1], "source": "op2"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op4  mul_cast (RMS norm → fp16 — A=external scale, B=op3)
    {
        "id": "op4", "type": "decode_mul_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "source": "ext"},
            "B": {"shape": [1, 1, _HID], "source": "op3"},
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
            "B": {"shape": [1, 1, _HID], "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op7  RoPE cos mul (A←op6, B←external cos)
    {
        "id": "op7", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "source": "op6"},
            "B": {"shape": [1, 1, _HID], "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op8  RoPE sin mul (A←op6, B←external sin_rot — prefill style,
    #       output routed via rope_slice_xor2 for half-swap at op9 ADD)
    {
        "id": "op8", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "source": "op6"},
            "B": {"shape": [1, 1, _HID], "source": "ext"},
        },
        "output": {"shape": [1, 1, _HID], "type": "rope_slice_xor2"},
    },
    # op9  RoPE merge add (A←op7, B←op8)
    {
        "id": "op9", "type": "decode_add_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "source": "op7"},
            "B": {"shape": [1, 1, _HID], "source": "op8"},
        },
        "output": {"shape": [1, 1, _HID], "dtype": "fp16"},
    },
    # op10 summac (KV RMS norm part 1) — external kv input, kv_padding_a
    {
        "id": "op10", "type": "decode_summac_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _KV_PAD_A], "source": "ext"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op11 remote_sum (KV RMS norm part 2) — only slice 0, KV uses slice_per_head
    {
        "id": "op11", "type": "decode_remote_sum_fp32N_fp32N",
        "used_slices_mask": "0b1",
        "inputs": {"A": {"shape": [1, _SPH, 1], "type": "slice0", "source": "op10"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op12 mac_SFU (KV RMS norm part 3)
    {
        "id": "op12", "type": "decode_mac_SFU_fp32N_fp32N",
        "inputs": {"A": {"shape": [_HID_FULL, 1, 1], "type": "slice0", "source": "op11"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op13 mul_scale (KV RMS norm apply — A=external kv, B=op12) kv_padding_a
    {
        "id": "op13", "type": "decode_mul_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _KV_PAD_A], "source": "ext"},
            "B": {"shape": [1, 1, 1], "source": "op12"},
        },
        "output": {"shape": [1, 1, _KV_PAD_A]},
    },
    # op14 mul_cast (KV → fp16 — A=external scale, B=op13) kv_padding_a
    {
        "id": "op14", "type": "decode_mul_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _KV_PAD_A], "source": "ext"},
            "B": {"shape": [1, 1, _KV_PAD_A], "source": "op13"},
        },
        "output": {"shape": [1, 1, _KV_PAD_A], "dtype": "fp16"},
    },
    # op15 gemv_ring (K projection) — kv_padding: A=[kv_padding_a,1,1], B=[kv_padding_b,1,kv_n]
    {
        "id": "op15", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_KV_PAD_A, 1, 1], "dtype": "fp16", "source": "op14"},
            "B": _gemv_b([_KV_PAD_B, 1, _KV_N]),
            "B'": _gemv_bp([_KV_PAD_B, 1, _KV_N]),
        },
        "output": {"shape": [1, 1, _KV_N], "dtype": "fp16"},
    },
    # op16 add residual (K + residual) — A=fp16 from op15, B=fp32 external
    {
        "id": "op16", "type": "decode_add_fp16N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _KV_N], "dtype": "fp16", "source": "op15"},
            "B": {"shape": [1, 1, _KV_N], "source": "ext"},
        },
        "output": {"shape": [1, 1, _KV_N]},
    },
    # op17 RoPE cos mul (K) — KV_N
    {
        "id": "op17", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _KV_N], "source": "op16"},
            "B": {"shape": [1, 1, _KV_N], "source": "ext"},
        },
        "output": {"shape": [1, 1, _KV_N]},
    },
    # op18 RoPE sin mul (K) — KV_N, prefill style, output routed via rope_slice_xor2
    {
        "id": "op18", "type": "decode_mul_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _KV_N], "source": "op16"},
            "B": {"shape": [1, 1, _KV_N], "source": "ext"},
        },
        "output": {"shape": [1, 1, _KV_N], "type": "rope_slice_xor2"},
    },
    # op19 RoPE merge add (K) — KV_N
    {
        "id": "op19", "type": "decode_add_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _KV_N], "source": "op17"},
            "B": {"shape": [1, 1, _KV_N], "source": "op18"},
        },
        "output": {"shape": [1, 1, _KV_N], "dtype": "fp16"},
    },
    # op20 gemv_ring (V projection) — kv_padding same as K
    {
        "id": "op20", "type": "decode_gemv_ring",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [_KV_PAD_A, 1, 1], "dtype": "fp16", "source": "op14"},
            "B": _gemv_b([_KV_PAD_B, 1, _KV_N]),
            "B'": _gemv_bp([_KV_PAD_B, 1, _KV_N]),
        },
        "output": {"shape": [1, 1, _KV_N], "dtype": "fp16"},
    },
    # op21 add residual (V) — A=external residual, B=fp16 from op20
    {
        "id": "op21", "type": "decode_add_fp32N_fp16N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _KV_N], "source": "ext"},
            "B": {"shape": [1, 1, _KV_N], "dtype": "fp16", "source": "op20"},
        },
        "output": {"shape": [1, 1, _KV_N], "dtype": "fp16"},
    },
    # op22 gemv_local (QK^T — A=Q from op9, B/B'=K **cache** from external)
    # Decode: Q is current token (vector), K cache has attention_length positions.
    # op19 only produces K for the *current* token → stored to cache externally.
    # op22 reads the full K cache (external), NOT op19.
    {
        "id": "op22", "type": "decode_gemv_local_qkt",
        "used_slices_mask": _USL,
        "inputs": {
            "A": {"shape": [1, 1, _HD_SLICE], "dtype": "fp16", "source": "op9"},
            "B": _gemv_local_b([1, _ATN, _HD_SLICE]),
            "B'": _gemv_local_b([1, _ATN, _HD_SLICE]),
        },
        "output": {"shape": [1, 1, _ATN]},
    },
    # op23 remote_sum (QK^T aggregation) — one slice per 4-slice group, type=slice4
    {
        "id": "op23", "type": "prefill_remote_sum_4slice_fp32MN_fp32MN",
        "used_slices_mask": "0b0001000100010001000100010001",
        "inputs": {"A": {"shape": [1, _SPH, _ATN], "source": "op22"}},
        "output": {"shape": [1, 1, _ATN]},
    },
    # op24 mac (scores * scale + mask) — A←op23 type=slice4, C←external mask.
    # Decode: only 1 query token → attention scores are (1, attn_len) = 1D,
    # NOT prefill's (query_len, key_len) = (attn_len, attn_len) 2D matrix.
    {
        "id": "op24", "type": "decode_mac_fp32N_fp32N_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _ATN], "type": "slice4",
                  "remapping": list(range(26)), "source": "op23"},
            "C": {"shape": [1, 1, _ATN],
                  "remapping": list(range(26)), "source": "ext"},
        },
        "output": {"shape": [1, 1, _ATN]},
    },
    # op25 max (per-head local maxima — A←op24, reduces attention_length dim)
    {
        "id": "op25", "type": "decode_max_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _ATN], "source": "op24"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op26 sub_SFU (scores - max — A←op24, B←op25 broadcast scalar)
    {
        "id": "op26", "type": "decode_sub_SFU_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _ATN], "source": "op24"},
            "B": {"shape": [1, 1, 1], "source": "op25"},
        },
        "output": {"shape": [1, 1, _ATN]},
    },
    # op27 sum_rec (per-head sum reciprocals — A←op26)
    {
        "id": "op27", "type": "decode_sum_rec_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _ATN], "source": "op26"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op28 mul_softmax (exp(x-max) * 1/sum — A←op26, B←op27 broadcast scalar)
    {
        "id": "op28", "type": "decode_mul_fp32N_fp32_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _ATN], "source": "op26"},
            "B": {"shape": [1, 1, 1], "source": "op27"},
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
            "B": _gemv_local_b([1, _HD_SLICE, _ATN]),
            "B'": _gemv_local_b([1, _HD_SLICE, _ATN]),
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
            "A": {"shape": [1, 1, _HID], "source": "ext"},
            "B": {"shape": [1, 1, _HID], "dtype": "fp16", "source": "op30"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op32 summac (FFN RMS norm part 1 — A←op31)
    {
        "id": "op32", "type": "decode_summac_fp32N_fp32N",
        "inputs": {"A": {"shape": [1, 1, _HID], "source": "op31"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op33 remote_sum (FFN RMS norm part 2) — only slice 0
    {
        "id": "op33", "type": "decode_remote_sum_fp32N_fp32N",
        "used_slices_mask": "0b1",
        "inputs": {"A": {"shape": [1, _USL, 1], "type": "slice0", "source": "op32"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op34 mac_SFU (FFN RMS norm part 3)
    {
        "id": "op34", "type": "decode_mac_SFU_fp32N_fp32N",
        "inputs": {"A": {"shape": [_HID_FULL, 1, 1], "type": "slice0", "source": "op33"}},
        "output": {"shape": [1, 1, 1]},
    },
    # op35 mul_scale (FFN RMS apply — A←op31, B←op34)
    {
        "id": "op35", "type": "decode_mul_fp32N_fp32_fp32N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "source": "op31"},
            "B": {"shape": [1, 1, 1], "source": "op34"},
        },
        "output": {"shape": [1, 1, _HID]},
    },
    # op36 mul_cast (FFN → fp16 — A=external scale, B=op35)
    {
        "id": "op36", "type": "decode_mul_fp32N_fp32N_fp16N",
        "inputs": {
            "A": {"shape": [1, 1, _HID], "source": "ext"},
            "B": {"shape": [1, 1, _HID], "source": "op35"},
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
            "A": {"shape": [1, 1, _INT], "source": "op39"},
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
            "A": {"shape": [1, 1, _HID], "source": "op31"},
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


# ---------------------------------------------------------------------------
# Optimized JSON generation: fuse summac+remote_sum and add bank_interleave=2
# ---------------------------------------------------------------------------

# Remapping arrays for bank_interleave=2
_REMAP_ILV2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 20, 21, 22, 23, 24, 25]
_REMAP_ILV4 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 1, 21, 22, 23, 24, 25]


def _get_remap(bank_interleave: int) -> list[int]:
    """Return remapping array based on bank_interleave, not dtype."""
    if bank_interleave == 4:
        return list(_REMAP_ILV4)
    return list(_REMAP_ILV2)

# Pairs of (summac_op_id, remote_sum_op_id) to fuse.
# remote_sum type determines fusion variant:
#   decode_remote_sum_fp32N_fp32N (used_slices=0b1, M=used_slices) → 28-slice → decode_remote_summac_fp32N_fp32N
#   decode_remote_sum_fp32N_fp32N (used_slices=0b1, M=slice_per_head) → 4-slice → decode_remote_summac_4slice_fp32N_fp32N
_FUSE_PAIRS = [
    ("op0", "op1"),    # RMS norm: 28-slice reduction
    ("op10", "op11"),  # KV RMS norm: 4-slice reduction (M=slice_per_head=4)
    ("op32", "op33"),  # FFN RMS norm: 28-slice reduction
]


def _estimate_size(shape: list[Any], dtype: str = "fp32", config: dict | None = None) -> int:
    """Estimate tensor size in bytes from shape and dtype.

    Symbolic shape strings (e.g. ``"hidden_size//used_slices"``) are resolved
    using *config* when provided.
    """
    bytes_per_elem = 2 if dtype == "fp16" else 4
    total = 1
    for dim in shape:
        if isinstance(dim, str):
            if config is None:
                return 0  # symbolic — can't estimate without config
            val = _resolve_symbolic(dim, config)
            if val is None:
                return 0
            total *= val
        else:
            total *= int(dim)
    return total * bytes_per_elem


def _resolve_symbolic(expr: str, config: dict) -> int | None:
    """Resolve a symbolic expression like 'hidden_size//used_slices' using config."""
    try:
        return eval(expr, {"__builtins__": {}}, dict(config))
    except Exception:
        return None


def _get_remap_old(dtype: str) -> list[int]:
    return _get_remap(2)


def generate_optimized_program_json(
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> Path:
    """Generate an optimized decode program JSON with fused operators and bank_interleave=2.

    Optimizations:
    1. Fuse decode_summac + decode_remote_sum into decode_remote_summac (28/4-slice).
    2. Add bank_interleave=2 to the largest tensor in each operator, with consistency
       propagation: if an output is bank_interleave=2, all downstream inputs reading
       it must also be bank_interleave=2.
    3. Add remapping arrays: fp32 → identity, fp16 → shifted.
    """
    config = load_decode_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    used_slices = int(config["used_slices"])
    mask_all = "0b" + "1" * used_slices

    # Step 1: Build the base operator list (same as non-optimized, no fusion)
    operators_raw: list[dict[str, Any]] = []
    for layout_op in _DECODE_LAYOUT:
        us_mask_spec = layout_op.get("used_slices_mask", None)
        if us_mask_spec is None or us_mask_spec == _USL:
            us_mask = mask_all
        elif us_mask_spec == "1":
            us_mask = "0b1"
        else:
            us_mask = us_mask_spec

        op_entry = {
            "id": layout_op["id"],
            "type": layout_op["type"],
            "used_slices": us_mask,
            "inputs": {},
            "output": {},
        }

        for port, inp_spec in layout_op.get("inputs", {}).items():
            inp = {
                "shape": inp_spec["shape"],
                "bank_interleave": inp_spec.get("bank_interleave", 1),
            }
            dtype = inp_spec.get("dtype", "")
            if dtype == "fp16":
                inp["dtype"] = "fp16"
            elif dtype == "fp32" or dtype == "f32":
                pass
            elif dtype:
                inp["dtype"] = dtype
            if inp_spec.get("remapping") is not None:
                inp["remapping"] = inp_spec["remapping"]
            if inp_spec.get("type"):
                inp["type"] = inp_spec["type"]
            inp["source"] = _resolve_source(inp_spec["source"])
            op_entry["inputs"][port] = inp

        out_spec = layout_op.get("output", {})
        op_entry["output"]["shape"] = out_spec.get("shape", [])
        if out_spec.get("dtype") == "fp16":
            op_entry["output"]["bank_interleave"] = 1
            op_entry["output"]["dtype"] = "fp16"
        if out_spec.get("type") and out_spec["type"] != "rope_slice_xor2":
            op_entry["output"]["type"] = out_spec["type"]

        operators_raw.append(op_entry)

    # Step 1a: Replace summac+remote_sum with prefill_remote_summac + decode_sum.
    #          prefill_remote_summac: 8 PEs local reduction + cross-slice sum (fused).
    #            Input A: [1, 8, N//8] → Output: [1, 1, 8]  (8 partials)
    #          decode_sum_fp32N_fp32N: sum the 8 partials.
    #            Input A: [1, 1, 8] → Output: [1, 1, 1]
    _SUMMAC_OLD = "decode_summac_fp32N_fp32N"
    _RSUM_TYPE = "decode_remote_sum_fp32N_fp32N"
    _REMOTE_SUMMAC_28 = "prefill_remote_summac_fp32MN_fp32MN"
    _REMOTE_SUMMAC_4S = "prefill_remote_summac_4slice_fp32MN_fp32MN"
    _DECODE_SUM = "decode_sum_fp32N_fp32N"

    layout_by_id = {op["id"]: op for op in _DECODE_LAYOUT}

    # Determine which remote_sum ops are 4-slice (M=slice_per_head) vs 28-slice
    rsum_is_4slice: dict[str, bool] = {}
    for layout_op in _DECODE_LAYOUT:
        if layout_op["type"] == _RSUM_TYPE:
            rsum_a_shape = layout_op.get("inputs", {}).get("A", {}).get("shape", [])
            rsum_is_4slice[layout_op["id"]] = (
                len(rsum_a_shape) >= 2 and rsum_a_shape[1] == _SPH
            )

    # Build source map: remote_sum op_id -> its source summac op_id
    rsum_source: dict[str, str] = {}
    for layout_op in _DECODE_LAYOUT:
        if layout_op["type"] == _RSUM_TYPE:
            src = layout_op.get("inputs", {}).get("A", {}).get("source", "")
            rsum_source[layout_op["id"]] = src

    for op in operators_raw:
        # --- Replace decode_summac with prefill_remote_summac (fused, 8 PEs + cross-slice) ---
        # Reshape [1,1,N] into [1, 8, N//8]: 8 PEs each reduce N//8 elements,
        # plus cross-slice reduction built into the op.
        #   Input A:  [1, 8, N//8]   (M=8 PEs, N//8 = data per PE)
        #   Output:   [1, 1, 8]      (reduce N//8→1, keep M=8 partials)
        if op["type"] == _SUMMAC_OLD:
            # Determine N from input A's original shape
            orig_n = None
            for port, inp in op.get("inputs", {}).items():
                shp = list(inp.get("shape", []))
                if len(shp) >= 3:
                    orig_n = shp[2]
            if orig_n == "hidden_size//used_slices":
                n_per_pe = "hidden_size//used_slices//8"
            elif orig_n == "kv_padding_a":
                n_per_pe = "kv_padding_a//8"
            else:
                n_per_pe = 1

            # Check if the corresponding remote_sum is 4-slice
            my_rsum_id = None
            for rsum_id, src_smac in rsum_source.items():
                if src_smac == op["id"]:
                    my_rsum_id = rsum_id
                    break
            is_4slice = rsum_is_4slice.get(my_rsum_id, False) if my_rsum_id else False

            op["type"] = _REMOTE_SUMMAC_4S if is_4slice else _REMOTE_SUMMAC_28
            op["used_slices"] = mask_all
            for port, inp in op.get("inputs", {}).items():
                inp["shape"] = [1, 8, n_per_pe]
                inp["bank_interleave"] = 2
                inp["remapping"] = list(_REMAP_ILV2)
            # Add input B: [1,1,1] scalar, external
            op["inputs"]["B"] = {
                "bank_interleave": 2,
                "shape": [1, 1, 1],
                "remapping": None,
                "source": {"type": "external"},
            }
            op["output"]["shape"] = [1, 1, 8]
            op["output"]["bank_interleave"] = 2
            op["output"]["remapping"] = list(_REMAP_ILV2)

        # --- Replace decode_remote_sum with decode_sum (sum 8 partials) ---
        #   Input A:  [1, 1, 8]   (M=1, N=8 local reduction dim)
        #   Output:   [1, 1, 1]   (reduce N=8→1)
        if op["type"] == _RSUM_TYPE:
            src_smac = rsum_source.get(op["id"], "")
            op["type"] = _DECODE_SUM
            op["used_slices"] = mask_all
            op["inputs"] = {
                "A": {
                    "bank_interleave": 2,
                    "shape": [1, 1, 8],
                    "remapping": list(_REMAP_ILV2),
                    "source": src_smac,
                },
            }
            op["output"] = {
                "shape": [1, 1, 1],
                "bank_interleave": 2,
                "remapping": list(_REMAP_ILV2),
            }

    # Step 1c: Force op13 (decode_mul_fp32N_fp32_fp32N for KV norm scale)
    #          output to bank_interleave=2, and propagate to downstream
    #          op14 input B which reads op13's output.
    _MUL_SCALE_TYPE = "decode_mul_fp32N_fp32_fp32N"
    for op in operators_raw:
        if op["type"] == _MUL_SCALE_TYPE:
            out = op.get("output", {})
            out_shp = out.get("shape", [])
            out_size = _estimate_size(out_shp, out.get("dtype", "fp32"), config)
            if out_size > 4:  # only for non-scalar outputs (KV norm, not attn norm)
                out["bank_interleave"] = 2
                op_id = op["id"]
                # Propagate to downstream inputs reading this output
                for other in operators_raw:
                    for port, inp in other.get("inputs", {}).items():
                        if inp.get("source") == op_id:
                            inp["bank_interleave"] = 2

    # Step 1b: Set GEMV A and D to bank_interleave=2 with fp16 remapping
    _GEMV_TYPES = {
        "decode_gemv_ring",
        "decode_gemv_ring_new",
        "decode_gemv_local",
        "decode_gemv_local_qkt",
    }
    for op in operators_raw:
        if op["type"] in _GEMV_TYPES:
            if "A" in op.get("inputs", {}):
                a = op["inputs"]["A"]
                a["bank_interleave"] = 2
                if a.get("remapping") is None:
                    a["remapping"] = list(_REMAP_ILV2)
                # Forward-propagate: source output must also be bi=2
                src = a.get("source")
                if isinstance(src, str):
                    for src_op in operators_raw:
                        if src_op["id"] == src:
                            src_out = src_op.get("output", {})
                            src_out["bank_interleave"] = 2
                            if src_out.get("remapping") is None:
                                sdt = src_out.get("dtype", "fp16")
                                src_out["remapping"] = list(_REMAP_ILV2)
                            break
            out = op.get("output", {})
            out["bank_interleave"] = 2
            if out.get("remapping") is None:
                dt = out.get("dtype", "fp16")
                out["remapping"] = list(_REMAP_ILV2)

    # Step 1b: Split decode_gemv_local_qkt into 4 sub-ops by cutting the
    #          B/B' M dimension (decode_attention_length) into 4 equal parts.
    #          Each sub-op runs on ALL 28 slices but computes only 1/4 of the
    #          attention_length.  The 4 partial outputs are then summed by a
    #          prefill_remote_sum_qkt_fp32MN_fp32MN op.
    _QKT_OP_TYPE = "decode_gemv_local_qkt"
    _RSUM_QKT_OLD_TYPE = "prefill_remote_sum_4slice_fp32MN_fp32MN"
    _RSUM_QKT_NEW_TYPE = "prefill_remote_sum_qkt_4slice_fp32MN_fp32MN"
    qkt_idx = None
    rsum_qkt_idx = None
    for i, op in enumerate(operators_raw):
        if op["type"] == _QKT_OP_TYPE and qkt_idx is None:
            qkt_idx = i
        if op["type"] == _RSUM_QKT_OLD_TYPE and rsum_qkt_idx is None:
            rsum_qkt_idx = i
    if qkt_idx is not None and rsum_qkt_idx is not None:
        qkt_op = operators_raw[qkt_idx]
        rsum_op = operators_raw[rsum_qkt_idx]
        # Keep op22 as the original single decode_gemv_local_qkt (NO 4-way split).
        # op26 stays as prefill_remote_sum_qkt_4slice_fp32MN_fp32MN.
        # Its input A golden is a decode_attention_length-sized gemv output
        # plus 3 zero-padded decode_attention_length-sized chunks, total
        # size [decode_attention_length, 4].

        # op26: prefill_remote_sum_qkt_4slice_fp32MN_fp32MN
        # Input A: [1, decode_attention_length, 4]
        # Input B: [1, 1, 1], external placeholder
        # Output: [1, 1, decode_attention_length]
        new_rsum: dict[str, Any] = {
            "id": rsum_op["id"],
            "type": _RSUM_QKT_NEW_TYPE,
            "used_slices": mask_all,
            "inputs": {
                "A": {
                    "bank_interleave": 2,
                    "shape": [
                        1,
                        "decode_attention_length",
                        4,
                    ],
                    "remapping": list(_REMAP_ILV2),
                    "source": qkt_op["id"],
                },
                "B": {
                    "bank_interleave": 2,
                    "shape": [1, 1, 1],
                    "remapping": None,
                    "source": {"type": "external"},
                },
            },
            "output": {
                "shape": rsum_op.get("output", {}).get("shape", [1, 1, "decode_attention_length"]),
                "bank_interleave": 2,
                "remapping": list(_REMAP_ILV2),
            },
        }

        # Replace old rsum op in place; keep qkt op (op22) untouched
        operators_raw[rsum_qkt_idx] = new_rsum

    # Step 2: Bank interleave assignment with consistency propagation
    # Map: op_id -> {"input": {port: ilv}, "output": ilv}
    interleave_map: dict[str, dict[str, Any]] = {}
    for op in operators_raw:
        interleave_map[op["id"]] = {"input": {}, "output": 1}

    # Build producer map: op_id -> which op's output feeds it
    # output_ilv[op_id] = bank_interleave of that op's output
    output_ilv: dict[str, int] = {}

    # First pass: propagate output bank_interleave to downstream inputs
    for op in operators_raw:
        op_id = op["id"]
        # Determine output interleave
        out_bi = op.get("output", {}).get("bank_interleave", 1)
        # If output already has bank_interleave set (e.g. GEMV B/B' have ilv=2),
        # keep it. Otherwise default to 1.
        output_ilv[op_id] = out_bi

    # Propagate: if an input reads from an op whose output is ilv=2, the input must be ilv=2
    for op in operators_raw:
        op_id = op["id"]
        for port, inp in op.get("inputs", {}).items():
            src = inp.get("source")
            src_op_id = None
            if isinstance(src, str):
                src_op_id = src
            if src_op_id and src_op_id in output_ilv and output_ilv[src_op_id] == 2:
                inp["bank_interleave"] = 2
                interleave_map[op_id]["input"][port] = 2

    # Second pass: for each op, pick the largest input/output to set bank_interleave=2
    # Rule: if op has 3 inputs, set 2 to ilv=1 and 1 to ilv=2 (the largest)
    #       if op has 2 inputs, set the larger one to ilv=2 if neither is already ilv=2
    #       output gets ilv=2 only if it's the largest and not constrained by propagation
    for op in operators_raw:
        op_id = op["id"]
        inputs = op.get("inputs", {})
        output = op.get("output", {})

        # Gather all tensors with their sizes
        tensors: list[tuple[str, str, int, str]] = []  # (kind, port, size, dtype)
        for port, inp in inputs.items():
            dtype = inp.get("dtype", "fp32")
            size = _estimate_size(inp.get("shape", []), dtype, config)
            # Skip broadcast inputs: if the source output is a scalar but
            # this input is larger, the data is broadcast, not the same
            # tensor, so bank_interleave shouldn't be set on it.
            src = inp.get("source")
            if isinstance(src, str):
                # Check against source op's output size for broadcast detection
                src_op_entry = None
                for src_op in operators_raw:
                    if src_op["id"] == src:
                        src_op_entry = src_op
                        break
                if src_op_entry is not None:
                    src_out_size = _estimate_size(
                        src_op_entry.get("output", {}).get("shape", []),
                        src_op_entry.get("output", {}).get("dtype", "fp32"),
                        config,
                    )
                    if src_out_size <= 4 and size > 4:
                        continue
            tensors.append(("input", port, size, dtype))
        out_dtype = output.get("dtype", "fp32")
        out_size = _estimate_size(output.get("shape", []), out_dtype, config)
        tensors.append(("output", "", out_size, out_dtype))

        # Find which tensors already have ilv=2 (from propagation)
        already_ilv2: set[str] = set()
        for port, inp in inputs.items():
            if inp.get("bank_interleave", 1) == 2:
                already_ilv2.add(f"input:{port}")

        # If no tensor is already ilv=2, pick the largest non-scalar tensor
        if not already_ilv2 and tensors:
            # Exclude scalar tensors (size <= 4 bytes, e.g. [1,1,1])
            non_scalar = [t for t in tensors if t[2] > 4]
            if non_scalar:
                largest = max(non_scalar, key=lambda t: t[2])
                kind, port, _, dtype = largest
                if kind == "input":
                    inputs[port]["bank_interleave"] = 2
                    interleave_map[op_id]["input"][port] = 2
                    # Forward-propagate: source output must also be bi=2
                    src = inputs[port].get("source")
                    if isinstance(src, str) and src in output_ilv:
                        output_ilv[src] = 2
                else:
                    output["bank_interleave"] = 2
                    output_ilv[op_id] = 2

    # Step 3: Set remapping on ALL bank_interleave=2 tensors.
    #         remapping depends on bank_interleave value, not dtype.
    #         Skip scalar tensors (shape [1,1,1]) — they keep remapping=null.
    for op in operators_raw:
        for port, inp in op.get("inputs", {}).items():
            bi = inp.get("bank_interleave", 1)
            if bi >= 2:
                shp = inp.get("shape", [])
                size = _estimate_size(shp, inp.get("dtype", "fp32"), config)
                if size <= 4:
                    continue  # scalar — keep remapping=null
                inp["remapping"] = _get_remap(bi)
        out = op.get("output", {})
        bi = out.get("bank_interleave", 1)
        if bi >= 2:
            shp = out.get("shape", [])
            size = _estimate_size(shp, out.get("dtype", "fp32"), config)
            if size <= 4:
                continue  # scalar — keep remapping=null
            out["remapping"] = _get_remap(bi)

    # Step 4: Re-propagate output ilv=2 to downstream inputs (after step 2 may have changed outputs)
    for op in operators_raw:
        for port, inp in op.get("inputs", {}).items():
            src = inp.get("source")
            src_op_id = None
            if isinstance(src, str):
                src_op_id = src
            if src_op_id and src_op_id in output_ilv and output_ilv[src_op_id] == 2:
                inp["bank_interleave"] = 2
                bi = inp.get("bank_interleave", 1)
                if "remapping" not in inp or inp.get("remapping") is None:
                    inp["remapping"] = _get_remap(bi)

    # Step 5: Sync output_ilv back to each producer's output dict
    for op in operators_raw:
        op_id = op["id"]
        if op_id in output_ilv:
            out = op.get("output", {})
            if output_ilv[op_id] == 2:
                out["bank_interleave"] = 2
                bi = out.get("bank_interleave", 1)
                if "remapping" not in out or out.get("remapping") is None:
                    out["remapping"] = _get_remap(bi)

    # Step 5b: Force bank_interleave=2 (with remapping) for op15/op20 (K/V
    #          gemv_ring) A and D, and synchronize all related tensors.
    #          Must run AFTER all automatic bi=2 assignment steps (2/3/4/5)
    #          so the override is not overwritten.
    #   op14.D (source of op15/op20.A) → bi=2, remapping
    #   op15.A, op15.D → bi=2, remapping
    #   op16.A (reads op15.D) → bi=2, remapping
    #   op20.A, op20.D → bi=2, remapping
    #   op21.B (reads op20.D) → bi=2, remapping
    _KV_GEMV_IDS = {"op15", "op20"}
    _KV_PRODUCER_IDS = {"op14", "op15", "op20"}
    _KV_CONSUMER_MAP = {"op15": "A", "op20": "A", "op16": "A", "op21": "B"}
    _KV_SRC_MAP = {"op16": "op15", "op21": "op20"}
    for op in operators_raw:
        oid = op["id"]
        # Force producer outputs (op14.D, op15.D, op20.D)
        if oid in _KV_PRODUCER_IDS:
            out = op.get("output", {})
            out["bank_interleave"] = 2
            out["remapping"] = _get_remap(2)
        # Force consumer inputs
        if oid in _KV_CONSUMER_MAP:
            port = _KV_CONSUMER_MAP[oid]
            expected_src = _KV_SRC_MAP.get(oid)
            inp = op.get("inputs", {}).get(port)
            if inp is not None:
                src = inp.get("source")
                if expected_src is None or src == expected_src:
                    inp["bank_interleave"] = 2
                    inp["remapping"] = _get_remap(2)
        # Keep output_ilv in sync so no later step reverts these producers
        if oid in _KV_PRODUCER_IDS and oid in output_ilv:
            output_ilv[oid] = 2

    # Step 5c: Restore the original hardware-validated bank convention —
    #          only GEMV weight ports (B/B') use bank_interleave=2 + remapping;
    #          every other input port and every output stays single-bank with
    #          no remapping.  The experimental bank_interleave=2 propagation
    #          onto activation/output ports broke decode_gemv_local (op29):
    #          golden and hardware outputs no longer matched.
    _GEMV_ILV2_TYPES = ("decode_gemv_ring", "decode_gemv_local", "decode_gemv_local_qkt")
    for op in operators_raw:
        is_gemv = op["type"] in _GEMV_ILV2_TYPES
        for port, inp in op.get("inputs", {}).items():
            if is_gemv and port in ("B", "B'"):
                inp["bank_interleave"] = 2
                inp["remapping"] = list(_REMAP_ILV2)
            else:
                inp.pop("bank_interleave", None)
                inp.pop("remapping", None)
        out = op.get("output", {})
        out.pop("bank_interleave", None)
        out.pop("remapping", None)

    # Step 6: Renumber operators sequentially (op0, op1, op2, ...) and
    #         remap all source references to the new IDs.
    old_to_new: dict[str, str] = {}
    for idx, op in enumerate(operators_raw):
        old_to_new[op["id"]] = f"op{idx}"

    for op in operators_raw:
        op["id"] = old_to_new[op["id"]]
        for inp in op.get("inputs", {}).values():
            src = inp.get("source")
            if isinstance(src, str) and src in old_to_new:
                inp["source"] = old_to_new[src]

    payload: dict[str, Any] = {
        "params": {
            **config,
            "kv_padding_a": int(config.get("kv_padding", 256)),
            "kv_padding_b": int(config.get("kv_padding", 256)) * int(config.get("slice_per_head", 4)),
        },
        "used_slices": used_slices,
        "operators": operators_raw,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, ensure_ascii=False)
    output_path.write_text(
        _compact_inline_arrays(raw) + "\n",
        encoding="utf-8",
    )
    print(f"[decode-program-optimized] wrote {len(operators_raw)} operators to {output_path}")
    return output_path


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
            if inp_spec.get("type"):
                inp["type"] = inp_spec["type"]
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
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Generate an optimized JSON with fused operators and bank_interleave=2.",
    )
    parser.add_argument(
        "--optimized-output",
        type=Path,
        default=BASE_DIR / "model_execplan" / "examples" / "layer0_decode_optimized.json",
        help="Output path for the optimized JSON (used with --optimized).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_program_json(args.config, args.manifest, args.output)
    if args.optimized:
        generate_optimized_program_json(
            args.config, args.manifest, args.optimized_output
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
