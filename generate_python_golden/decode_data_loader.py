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


def _config_dims(config: dict[str, object] | None) -> tuple[int, int, int, int, int]:
    """Return ``(hidden, intermediate, kv_dim, head_dim, kv_heads)``."""

    if config is None:
        return 896, 1792, 128, 128, 1
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    head_dim = int(config["head_dim"])
    kv_heads = int(config["num_key_value_heads"])
    return hidden, intermediate, head_dim * kv_heads, head_dim, kv_heads


def _weights_dir_for_config(config: dict[str, object] | None) -> Path:
    if config is None:
        return WEIGHTS_DIR
    hidden, _, _, _, kv_heads = _config_dims(config)
    if hidden == 896 and kv_heads == 1:
        return WEIGHTS_DIR
    # Full-size weights: prefer the standard generated directory, then the
    # complete original checkpoint directory supplied in the workspace.
    generated = BASE_DIR / "model_weights_full"
    if generated.is_dir():
        return generated
    original = BASE_DIR / "DeepSeek-R1-Distill-Qwen-1.5B-f16"
    if original.is_dir():
        return original
    return generated


def kv_cache_path_for_config(config: dict[str, object] | None) -> Path:
    if config is None:
        return KV_CACHE_DIR / "layer0_kv_cache.npz"
    hidden = int(config["hidden_size"])
    head_dim = int(config["head_dim"])
    kv_heads = int(config["num_key_value_heads"])
    attention_length = int(config["decode_attention_length"])
    # Preserve the legacy small-size cache file so the original small-size
    # golden can be reproduced byte-for-byte where possible.
    if (hidden, head_dim, kv_heads, attention_length) == (896, 128, 1, 32):
        legacy = KV_CACHE_DIR / "layer0_kv_cache.npz"
        if legacy.is_file():
            return legacy
    rope_suffix = "_rope" if str(config.get("rope_mode", "passthrough")) == "table" else ""
    return (
        KV_CACHE_DIR
        / f"layer0_kv_cache_h{hidden}_hd{head_dim}_kv{kv_heads}_a{attention_length}{rope_suffix}.npz"
    )


def load_rope_tables(config: dict[str, object] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the position-dependent NeoX RoPE tables.

    Returns ``cos`` / ``sin`` with shape ``(head_dim//2, 512)`` in F-order,
    matching the prefill golden's ``rope_fp32/*_ne2_512.bin`` layout.
    """

    table_dir = BASE_DIR / "rope_fp32"
    if config is not None:
        configured = config.get("rope_table_dir")
        if configured:
            candidate = Path(str(configured))
            table_dir = candidate if candidate.is_absolute() else BASE_DIR / candidate
    cos_path = table_dir / "rope_neox_cos_float32_ne2_512.bin"
    sin_path = table_dir / "rope_neox_sin_float32_ne2_512.bin"
    cos = np.fromfile(cos_path, dtype=np.float32)
    sin = np.fromfile(sin_path, dtype=np.float32)
    half_dim = 64
    positions = cos.size // half_dim
    return (
        cos.reshape((half_dim, positions), order="F"),
        sin.reshape((half_dim, positions), order="F"),
    )


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


def synthesize_layer_weights(
    config: dict[str, object],
    seed: int | None = None,
) -> LayerWeights:
    """Deterministic synthetic real-size weights (used when 1.5B weights are absent)."""

    hidden, intermediate, kv_dim, _, _ = _config_dims(config)
    rng = np.random.default_rng(
        int(config.get("random_seed", 0)) if seed is None else int(seed)
    )

    def _matrix(rows: int, cols: int) -> np.ndarray:
        return rng.uniform(-0.25, 0.25, (rows, cols)).astype(np.float16)

    def _vector(length: int, low: float, high: float) -> np.ndarray:
        return rng.uniform(low, high, length).astype(np.float32)

    return LayerWeights(
        q_weight=_matrix(hidden, hidden),
        q_bias=_vector(hidden, -0.5, 0.5),
        k_weight=_matrix(hidden, kv_dim),
        k_bias=_vector(kv_dim, -0.5, 0.5),
        v_weight=_matrix(hidden, kv_dim),
        v_bias=_vector(kv_dim, -0.5, 0.5),
        out_weight=_matrix(hidden, hidden),
        attn_norm=_vector(hidden, 0.5, 1.5),
        ffn_gate_weight=_matrix(hidden, intermediate),
        ffn_up_weight=_matrix(hidden, intermediate),
        ffn_down_weight=_matrix(intermediate, hidden),
        ffn_norm=_vector(hidden, 0.5, 1.5),
    )


def load_layer_weights(
    layer_idx: int = 0,
    config: dict[str, object] | None = None,
) -> LayerWeights:
    """Load all weight tensors for *layer_idx*.

    When *config* is supplied the tensors are loaded from the directory that
    matches the real model size (``model_weights_32`` for the small model,
    ``model_weights_full`` for 1.5B) and their shapes are verified.  A missing
    directory or a shape mismatch raises ``FileNotFoundError``/``ValueError``
    so callers can fall back to deterministic synthetic weights.
    """

    weights_dir = _weights_dir_for_config(config)
    hidden, intermediate, kv_dim, head_dim, kv_heads = _config_dims(config)
    pfx = f"blk.{layer_idx}"

    def _load(name: str, dtype: np.dtype) -> np.ndarray:
        candidates = sorted(weights_dir.glob(f"{pfx}.{name}*.bin"))
        if not candidates:
            raise FileNotFoundError(
                f"weight not found: {pfx}.{name}* in {weights_dir}"
            )
        return _load_bin(candidates[0], dtype)

    def _expect(name: str, tensor: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if tensor.shape != shape:
            raise ValueError(
                f"weight {name} has shape {tensor.shape}, expected {shape} "
                f"(config mismatch in {weights_dir})"
            )
        return tensor

    q_weight = _expect("attn_q.weight", _load("attn_q.weight", np.float16), (hidden, hidden))
    q_bias = _expect("attn_q.bias", _load("attn_q.bias", np.float32), (hidden,))
    k_weight = _expect("attn_k.weight", _load("attn_k.weight", np.float16), (hidden, kv_dim))
    k_bias = _expect("attn_k.bias", _load("attn_k.bias", np.float32), (kv_dim,))
    v_weight = _expect("attn_v.weight", _load("attn_v.weight", np.float16), (hidden, kv_dim))
    v_bias = _expect("attn_v.bias", _load("attn_v.bias", np.float32), (kv_dim,))
    out_weight = _expect(
        "attn_output.weight", _load("attn_output.weight", np.float16), (hidden, hidden)
    )
    attn_norm = _expect("attn_norm.weight", _load("attn_norm.weight", np.float32), (hidden,))
    ffn_gate_weight = _expect(
        "ffn_gate.weight", _load("ffn_gate.weight", np.float16), (hidden, intermediate)
    )
    ffn_up_weight = _expect(
        "ffn_up.weight", _load("ffn_up.weight", np.float16), (hidden, intermediate)
    )
    ffn_down_weight = _expect(
        "ffn_down.weight", _load("ffn_down.weight", np.float16), (intermediate, hidden)
    )
    ffn_norm = _expect("ffn_norm.weight", _load("ffn_norm.weight", np.float32), (hidden,))

    return LayerWeights(
        q_weight=q_weight,
        q_bias=q_bias,
        k_weight=k_weight,
        k_bias=k_bias,
        v_weight=v_weight,
        v_bias=v_bias,
        out_weight=out_weight,
        attn_norm=attn_norm,
        ffn_gate_weight=ffn_gate_weight,
        ffn_up_weight=ffn_up_weight,
        ffn_down_weight=ffn_down_weight,
        ffn_norm=ffn_norm,
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


def _default_kv_cache_path(config: dict[str, object] | None = None) -> Path:
    return kv_cache_path_for_config(config)


def load_kv_cache(
    path: Path | None = None,
    config: dict[str, object] | None = None,
) -> KVCache:
    """Load KV cache from disk (or raise FileNotFoundError)."""

    path = path or _default_kv_cache_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"KV cache not found at {path}.  "
            f"Run `generate_kv_cache.py` first."
        )
    data = np.load(path)
    kv = KVCache(k_cache=data["k_cache"], v_cache=data["v_cache"])
    if config is not None:
        expected = (
            int(config["head_dim"]),
            int(config["decode_attention_length"]),
            int(config["num_key_value_heads"]),
        )
        if kv.k_cache.shape != expected or kv.v_cache.shape != expected:
            raise ValueError(
                f"KV cache {path} has shape {kv.k_cache.shape}, expected {expected}"
            )
    return kv


def save_kv_cache(
    kv: KVCache,
    path: Path | None = None,
    config: dict[str, object] | None = None,
) -> None:
    """Persist a ``KVCache`` to disk."""
    path = path or _default_kv_cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, k_cache=kv.k_cache, v_cache=kv.v_cache)
    print(f"[kv-cache] saved to {path}")


def _synth_kv_weights(
    hidden: int,
    kv_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic weights for the KV cache mini-prefill."""

    attn_norm = rng.uniform(0.5, 1.5, hidden).astype(np.float32)
    k_weight = rng.uniform(-0.25, 0.25, (hidden, kv_dim)).astype(np.float16)
    v_weight = rng.uniform(-0.25, 0.25, (hidden, kv_dim)).astype(np.float16)
    k_bias = rng.uniform(-0.5, 0.5, kv_dim).astype(np.float32)
    v_bias = rng.uniform(-0.5, 0.5, kv_dim).astype(np.float32)
    return attn_norm, k_weight, v_weight, k_bias, v_bias


def generate_kv_cache(
    hidden: int = 896,
    head_dim: int = 128,
    num_heads: int = 7,
    num_kv_heads: int = 1,
    attention_length: int = 32,
    random_seed: int = 42,
    weights: object | None = None,
    config: dict[str, object] | None = None,
) -> KVCache:
    """Generate the KV cache by running a mini-prefill pass.

    Uses real weights when supplied (or available on disk); otherwise uses
    deterministic synthetic weights.  The output shapes are
    ``(head_dim, attention_length, num_kv_heads)``.  K is store after RoPE.
    """

    if config is not None:
        hidden = int(config["hidden_size"])
        head_dim = int(config["head_dim"])
        num_kv_heads = int(config["num_key_value_heads"])
        attention_length = int(config["decode_attention_length"])
        random_seed = int(config.get("random_seed", random_seed))

    rng = np.random.default_rng(random_seed)
    kv_dim = num_kv_heads * head_dim

    if weights is None:
        try:
            weights = load_layer_weights(0, config)
        except (FileNotFoundError, ValueError):
            weights = None

    if weights is not None and hasattr(weights, "k_weight"):
        attn_norm = np.asarray(weights.attn_norm, dtype=np.float32)
        k_weight = np.asarray(weights.k_weight, dtype=np.float16)
        v_weight = np.asarray(weights.v_weight, dtype=np.float16)
        k_bias = np.asarray(weights.k_bias, dtype=np.float32)
        v_bias = np.asarray(weights.v_bias, dtype=np.float32)
    else:
        attn_norm, k_weight, v_weight, k_bias, v_bias = _synth_kv_weights(
            hidden, kv_dim, rng
        )

    # --- Generate random input tokens (attention_length tokens) ---
    tokens = rng.uniform(-1.0, 1.0, (hidden, attention_length)).astype(np.float32)

    # --- Attention RMS Norm ---
    eps = 1e-6
    rms = np.sqrt(np.mean(tokens ** 2, axis=0, keepdims=True) + eps)
    normed = (tokens / rms) * attn_norm.reshape(-1, 1)

    # --- K / V projections (numpy matmul for full-size speed) ---
    k_proj = normed.T.astype(np.float32) @ k_weight.astype(np.float32)  # (attn, kv_dim)
    k_proj += k_bias.reshape(1, -1)
    v_proj = normed.T.astype(np.float32) @ v_weight.astype(np.float32)
    v_proj += v_bias.reshape(1, -1)

    # (kv_dim, attn) -> (head_dim, num_kv_heads, attn) -> (head_dim, attn, kv_heads)
    k_cache = (
        k_proj.T.reshape(head_dim, num_kv_heads, attention_length, order="F")
        .transpose(0, 2, 1)
        .astype(np.float16)
    )
    v_cache = (
        v_proj.T.reshape(head_dim, num_kv_heads, attention_length, order="F")
        .transpose(0, 2, 1)
        .astype(np.float16)
    )

    # --- Apply RoPE to K cache ---
    half_hd = head_dim // 2
    table_mode = config is not None and str(config.get("rope_mode", "passthrough")) == "table"
    if table_mode:
        cos_table, sin_table = load_rope_tables(config)
        if attention_length > cos_table.shape[1]:
            raise ValueError(
                f"RoPE table only has {cos_table.shape[1]} positions, "
                f"need {attention_length}"
            )
    else:
        theta = 10000.0 ** (-2.0 * np.arange(0, half_hd, dtype=np.float64) / head_dim)
        cos_theta = np.cos(theta).astype(np.float32)
        sin_theta = np.sin(theta).astype(np.float32)

    k_rope = k_cache.astype(np.float32).copy()
    for h in range(num_kv_heads):
        for t in range(attention_length):
            for i in range(half_hd):
                x0 = float(k_cache[i, t, h])
                x1 = float(k_cache[i + half_hd, t, h])
                if table_mode:
                    c = float(cos_table[i, t])
                    s = float(sin_table[i, t])
                else:
                    c = float(cos_theta[i])
                    s = float(sin_theta[i])
                k_rope[i, t, h] = np.float32(x0 * c - x1 * s)
                k_rope[i + half_hd, t, h] = np.float32(x0 * s + x1 * c)

    kv = KVCache(k_cache=k_rope.astype(np.float16), v_cache=v_cache)
    save_kv_cache(kv, config=config)
    return kv
