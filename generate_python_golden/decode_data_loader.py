"""Load real model weights and pre-generated KV cache for decode golden.

Weights are read from ``model_weights_32/`` (already trimmed to the small-model
dimensions).  The KV cache is generated once by running a mini-prefill pass
and persisted to ``kv_cache/`` so that subsequent decode runs reuse it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "model_weights_32"
KV_CACHE_DIR = BASE_DIR / "kv_cache"


# ---------------------------------------------------------------------------
#  Weight loading
# ---------------------------------------------------------------------------

class LayerWeights(NamedTuple):
    """All weight tensors needed by one decode layer (layer 0 by default)."""
    q_weight: np.ndarray       # (hidden, hidden)            fp16
    q_bias: np.ndarray         # (hidden,)                   fp32
    k_weight: np.ndarray       # (hidden, kv_dim)            fp16
    k_bias: np.ndarray         # (kv_dim,)                   fp32
    v_weight: np.ndarray       # (hidden, kv_dim)            fp16
    v_bias: np.ndarray         # (kv_dim,)                   fp32
    out_weight: np.ndarray     # (hidden, hidden)            fp16
    attn_norm: np.ndarray      # (hidden,)                   fp32
    ffn_gate_weight: np.ndarray   # (hidden, intermediate)   fp16
    ffn_up_weight: np.ndarray     # (hidden, intermediate)   fp16
    ffn_down_weight: np.ndarray   # (intermediate, hidden)   fp16
    ffn_norm: np.ndarray          # (hidden,)                fp32


def _load_bin(path: Path, dtype: np.dtype) -> np.ndarray:
    """Load a raw .bin file and squeeze trivial trailing dimensions."""
    data = np.fromfile(path, dtype=dtype)
    # Files saved as 4-D: e.g. shape 896x896x1x1 → squeeze to 2-D
    # Infer shape from filename: _shapeNxMx…_
    name = path.name
    if "_shape" in name:
        shape_str = name.split("_shape")[1].split("_dtype")[0]
        dims = tuple(int(d) for d in shape_str.split("x") if int(d) != 1)
        data = data.reshape(dims, order="F")
    return data


def load_layer_weights(layer_idx: int = 0) -> LayerWeights:
    """Load all weight tensors for *layer_idx* from ``model_weights_32/``."""
    pfx = f"blk.{layer_idx}"

    def _load(name: str, dtype: np.dtype) -> np.ndarray:
        candidates = sorted(WEIGHTS_DIR.glob(f"{pfx}.{name}*.bin"))
        if not candidates:
            raise FileNotFoundError(f"weight not found: {pfx}.{name}* in {WEIGHTS_DIR}")
        return _load_bin(candidates[0], dtype)

    return LayerWeights(
        q_weight=_load("attn_q.weight", np.float16),
        q_bias=_load("attn_q.bias", np.float32),
        k_weight=_load("attn_k.weight", np.float16),
        k_bias=_load("attn_k.bias", np.float32),
        v_weight=_load("attn_v.weight", np.float16),
        v_bias=_load("attn_v.bias", np.float32),
        out_weight=_load("attn_output.weight", np.float16),
        attn_norm=_load("attn_norm.weight", np.float32),
        ffn_gate_weight=_load("ffn_gate.weight", np.float16),
        ffn_up_weight=_load("ffn_up.weight", np.float16),
        ffn_down_weight=_load("ffn_down.weight", np.float16),
        ffn_norm=_load("ffn_norm.weight", np.float32),
    )


# ---------------------------------------------------------------------------
#  KV cache helpers
# ---------------------------------------------------------------------------

class KVCache(NamedTuple):
    """Pre-computed K / V cache for decode attention (one layer).

    Shapes follow the golden-model convention used inside
    ``build_decode_golden_cases``:

    * k_cache – ``(head_dim, attention_length, num_kv_heads)``  fp16
    * v_cache – ``(head_dim, attention_length, num_kv_heads)``  fp16

    For GQA with ``num_key_value_heads == 1`` these have 1 head.
    """
    k_cache: np.ndarray
    v_cache: np.ndarray

    @property
    def head_dim(self) -> int:
        return self.k_cache.shape[0]

    @property
    def attention_length(self) -> int:
        return self.k_cache.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.k_cache.shape[2]


def _default_kv_cache_path() -> Path:
    return KV_CACHE_DIR / "layer0_kv_cache.npz"


def load_kv_cache(path: Path | None = None) -> KVCache:
    """Load KV cache from disk (or raise FileNotFoundError)."""
    path = path or _default_kv_cache_path()
    if not path.exists():
        raise FileNotFoundError(
            f"KV cache not found at {path}.  "
            f"Run `generate_kv_cache.py` first."
        )
    data = np.load(path)
    return KVCache(
        k_cache=data["k_cache"],
        v_cache=data["v_cache"],
    )


def save_kv_cache(kv: KVCache, path: Path | None = None) -> None:
    """Persist a ``KVCache`` to disk."""
    path = path or _default_kv_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, k_cache=kv.k_cache, v_cache=kv.v_cache)
    print(f"[kv-cache] saved to {path}")


def generate_kv_cache(
    hidden: int = 896,
    head_dim: int = 128,
    num_heads: int = 7,
    num_kv_heads: int = 1,
    attention_length: int = 32,
    random_seed: int = 42,
) -> KVCache:
    """Generate realistic KV cache by running a mini-prefill pass.

    Uses the real model weights from ``model_weights_32/``, runs the Q/K/V
    projections and RoPE on *attention_length* random input tokens, and
    returns the resulting K and V activations (after RoPE for K, before for V).

    This simulates what a real prefill pass would produce as the KV cache
    for layer 0.
    """
    rng = np.random.default_rng(random_seed)

    # Load weights for layer 0
    w = load_layer_weights(0)

    # --- Generate random input tokens (attention_length=32 tokens) ---
    # Each token = hidden-dimensional embedding
    tokens = rng.uniform(-1.0, 1.0, (hidden, attention_length)).astype(np.float32)

    # --- Attention RMS Norm ---
    # Simple RMS norm: x / sqrt(mean(x^2) + eps) * weight
    eps = 1e-6
    rms = np.sqrt(np.mean(tokens ** 2, axis=0, keepdims=True) + eps)
    normed = (tokens / rms) * w.attn_norm.reshape(-1, 1)

    # --- K Projection ---
    # k_weight: (hidden, kv_dim) = (896, 128)
    kv_dim = w.k_weight.shape[1]  # 128 = num_kv_heads * head_dim
    k_proj = np.zeros((kv_dim, attention_length), dtype=np.float32)
    for t in range(attention_length):
        for o in range(kv_dim):
            acc = np.float32(0.0)
            for k in range(hidden):
                acc = np.float32(float(acc) + float(w.k_weight[k, o]) * float(normed[k, t]))
            k_proj[o, t] = acc

    # Add bias
    k_bias = w.k_bias.reshape(-1, 1)  # (kv_dim, 1)
    k_proj += k_bias

    # --- Project back to (head_dim, attention_length, num_kv_heads) ---
    k_cache = k_proj.reshape(head_dim, attention_length, num_kv_heads, order="F").astype(np.float16)

    # --- V Projection ---
    v_proj = np.zeros((kv_dim, attention_length), dtype=np.float32)
    for t in range(attention_length):
        for o in range(kv_dim):
            acc = np.float32(0.0)
            for k in range(hidden):
                acc = np.float32(float(acc) + float(w.v_weight[k, o]) * float(normed[k, t]))
            v_proj[o, t] = acc

    v_bias = w.v_bias.reshape(-1, 1)
    v_proj += v_bias

    v_cache = v_proj.reshape(head_dim, attention_length, num_kv_heads, order="F").astype(np.float16)

    # --- Apply RoPE to K cache ---
    # Standard LLaMA-style RoPE on head_dim=128
    half_hd = head_dim // 2
    theta = 10000.0 ** (-2.0 * np.arange(0, half_hd, dtype=np.float64) / head_dim)
    cos_theta = np.cos(theta).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)

    k_rope = k_cache.astype(np.float32).copy()
    for h in range(num_kv_heads):
        for t in range(attention_length):
            for i in range(half_hd):
                x0 = float(k_cache[i, t, h])
                x1 = float(k_cache[i + half_hd, t, h])
                k_rope[i, t, h]          = np.float32(x0 * float(cos_theta[i]) - x1 * float(sin_theta[i]))
                k_rope[i + half_hd, t, h] = np.float32(x0 * float(sin_theta[i]) + x1 * float(cos_theta[i]))

    k_cache = k_rope.astype(np.float16)

    kv = KVCache(k_cache=k_cache, v_cache=v_cache)

    # Persist automatically on first generation
    save_kv_cache(kv)

    return kv
