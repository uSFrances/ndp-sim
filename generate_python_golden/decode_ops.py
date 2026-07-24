"""Reference models and metadata for the currently supported Decode operators."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DecodeOperatorSpec:
    op_id: str
    name: str
    input_ports: tuple[str, ...]
    slice_policy: str
    hardware_json: str


@dataclass(frozen=True)
class DecodeGoldenCase:
    """A single operator instance within the decode layer trace."""
    instance_id: str          # unique id within the layer, e.g. "attn_rms_norm_summac"
    spec: DecodeOperatorSpec  # operator type this instance uses
    inputs: tuple[np.ndarray, ...]
    output: np.ndarray


SUPPORTED_DECODE_OPERATORS: tuple[DecodeOperatorSpec, ...] = (
    # -- RMS Norm chain (fp32) -----------------------------------------------
    DecodeOperatorSpec(
        "op0", "decode_summac_fp32N_fp32N",
        ("A",), "summac_hidden", "decode_summac_fp32N_fp32N.json",
    ),
    DecodeOperatorSpec(
        "op1", "decode_remote_sum_fp32N_fp32N",
        ("A",), "remote_sum", "decode_remote_sum_fp32N_fp32N.json",
    ),
    DecodeOperatorSpec(
        "op2", "decode_mac_SFU_fp32N_fp32N",
        ("A",), "scalar_replicate", "decode_mac_SFU_fp32N_fp32N.json",
    ),
    # -- RMS Norm scale + cast -----------------------------------------------
    DecodeOperatorSpec(
        "op3", "decode_mul_fp32N_fp32_fp32N",
        ("A", "B"), "hidden_elementwise", "decode_mul_fp32N_fp32N_fp16N.json",
    ),
    DecodeOperatorSpec(
        "op4", "decode_mul_fp32N_fp32N_fp16N",
        ("A", "B"), "hidden_elementwise", "decode_mul_fp32N_fp32N_fp16N.json",
    ),
    # -- GEMV Ring (Q projection) + residual add -----------------------------
    DecodeOperatorSpec(
        "op5", "decode_gemv_ring",
        ("B", "A"), "gemv_ring", "decode_gemv_ring.json",
    ),
    DecodeOperatorSpec(
        "op6", "decode_add_fp16N_fp32N_fp32N",
        ("A", "B"), "hidden_elementwise", "decode_add_fp16N_fp32N_fp32N.json",
    ),
    # -- Element-wise vector ops (RoPE cos/sin/merge) -----------------------
    DecodeOperatorSpec(
        "op7", "decode_mul_fp32N_fp32N_fp32N",
        ("A", "B"), "hidden_elementwise", "decode_mul_fp32N_fp32N_fp16N.json",
    ),
    DecodeOperatorSpec(
        "op8", "decode_mul_fp32N_fp32N_fp32N",
        ("A", "B"), "hidden_elementwise", "decode_mul_fp32N_fp32N_fp16N.json",
    ),
    DecodeOperatorSpec(
        "op9", "decode_add_fp32N_fp32N_fp16N",
        ("A", "B"), "hidden_elementwise", "decode_add_fp16N_fp32N_fp32N.json",
    ),
    # -- Attention: max / sub / sum_rec / softmax-mul ------------------------
    DecodeOperatorSpec(
        "op10", "decode_max_fp32N_fp32N",
        ("A",), "head_reduction", "decode_max_fp32N_fp32N.json",
    ),
    DecodeOperatorSpec(
        "op11", "decode_sub_SFU_fp32N_fp32_fp32N",
        ("A", "B"), "hidden_elementwise", "decode_max_fp32N_fp32N.json",
    ),
    DecodeOperatorSpec(
        "op12", "decode_sum_rec_fp32N_fp32N",
        ("A",), "head_reduction", "decode_sum_rec_fp32N_fp32N.json",
    ),
    DecodeOperatorSpec(
        "op13", "decode_mul_fp32N_fp32_fp16N",
        ("A", "B"), "hidden_elementwise", "decode_mul_fp32N_fp32N_fp16N.json",
    ),
    # -- GEMV Local (SV) ----------------------------------------------------
    DecodeOperatorSpec(
        "op14", "decode_gemv_local",
        ("B", "A"), "gemv_local", "decode_gemv_local.json",
    ),
    # -- Output GEMV + residual ----------------------------------------------
    DecodeOperatorSpec(
        "op15", "decode_add_fp32N_fp16N_fp32N",
        ("A", "B"), "hidden_elementwise", "decode_add_fp16N_fp32N_fp32N.json",
    ),
    # -- FFN: SiLU / gate×up / standalone variants --------------------------
    DecodeOperatorSpec(
        "op16", "decode_silu_fp16N_fp32N",
        ("A",), "hidden_elementwise", "decode_sum_rec_fp32N_fp32N.json",
    ),
    DecodeOperatorSpec(
        "op17", "decode_mul_fp32N_fp16N_fp16N",
        ("A", "B"), "hidden_elementwise", "decode_mul_fp32N_fp32N_fp16N.json",
    ),
    DecodeOperatorSpec(
        "op18", "decode_add_fp16N_fp32N_fp16N",
        ("A", "B"), "hidden_elementwise", "decode_add_fp16N_fp32N_fp32N.json",
    ),
    # -- Additional element-wise ops used in the 43-op layer -----------------
    DecodeOperatorSpec(
        "op_add_fp32_fp32", "decode_add_fp32N_fp32N_fp32N",
        ("A", "B"), "hidden_elementwise", "decode_add_fp16N_fp32N_fp32N.json",
    ),
)

DECODE_OP_REGISTRY: Mapping[str, DecodeOperatorSpec] = {
    spec.name: spec for spec in SUPPORTED_DECODE_OPERATORS
}


def resolve_target_names(target: str) -> list[str]:
    normalized = target.strip()
    if normalized == "all":
        return [spec.name for spec in SUPPORTED_DECODE_OPERATORS]
    if normalized in DECODE_OP_REGISTRY:
        return [normalized]
    prefixed = normalized if normalized.startswith("decode_") else f"decode_{normalized}"
    if prefixed in DECODE_OP_REGISTRY:
        return [prefixed]
    raise ValueError(
        f"unsupported Decode operator {target!r}; expected one of: "
        + ", ".join(DECODE_OP_REGISTRY)
    )


def load_decode_config(path: str | Path) -> dict[str, int | str]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_decode_config(config)
    return config


def validate_decode_config(config: Mapping[str, object]) -> None:
    required = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "sequence_length",
        "slice_per_head",
        "used_slices",
        "kv_cache_initial_length",
        "decode_steps",
        "decode_attention_length",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"decode config is missing required keys: {', '.join(missing)}")

    integer_values = {key: int(config[key]) for key in required}
    if any(value <= 0 for value in integer_values.values()):
        raise ValueError("all decode dimension values must be positive")

    hidden_size = integer_values["hidden_size"]
    intermediate_size = integer_values["intermediate_size"]
    used_slices = integer_values["used_slices"]
    heads = integer_values["num_attention_heads"]
    slice_per_head = integer_values["slice_per_head"]
    attention_length = integer_values["decode_attention_length"]
    head_dim = integer_values["head_dim"]
    sequence_length = integer_values["sequence_length"]
    kv_cache_initial_length = integer_values["kv_cache_initial_length"]

    if heads * slice_per_head != used_slices:
        raise ValueError(
            "num_attention_heads * slice_per_head must equal used_slices "
            f"({heads} * {slice_per_head} != {used_slices})"
        )
    if heads * head_dim != hidden_size:
        raise ValueError(
            "num_attention_heads * head_dim must equal hidden_size "
            f"({heads} * {head_dim} != {hidden_size})"
        )
    if hidden_size % used_slices:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by {used_slices}")
    if intermediate_size % used_slices:
        raise ValueError(
            f"intermediate_size={intermediate_size} must be divisible by {used_slices}"
        )
    if attention_length % slice_per_head:
        raise ValueError(
            f"decode_attention_length={attention_length} must be divisible by "
            f"slice_per_head={slice_per_head}"
        )
    if head_dim % slice_per_head:
        raise ValueError(
            f"head_dim={head_dim} must be divisible by slice_per_head={slice_per_head}"
        )
    if kv_cache_initial_length != sequence_length:
        raise ValueError(
            "kv_cache_initial_length must equal sequence_length for this one-step fixture"
        )
    if attention_length != kv_cache_initial_length:
        raise ValueError(
            "decode_attention_length must equal kv_cache_initial_length; "
            "the current fixture does not append the new token"
        )
    if integer_values["decode_steps"] != 1:
        raise ValueError("the current implementation supports exactly one decode step")


def as_vector(values: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(array.shape[0], 1, 1, order="F")


def fp32_fma(acc: np.float32, a: np.generic, b: np.generic) -> np.float32:
    """Fused-style fp32 accumulation using double precision as a fallback.

    float32 operands multiply exactly in binary64.  ``math.fma`` is used when
    available; otherwise the binary64 multiply/add is rounded once to fp32.
    """

    fma = getattr(math, "fma", None)
    if fma is not None:
        return np.float32(fma(float(a), float(b), float(acc)))
    return np.float32(float(a) * float(b) + float(acc))


def square_sum_fp32(values: np.ndarray) -> np.float32:
    acc = np.float32(0.0)
    for value in np.asarray(values).reshape(-1, order="C"):
        operand = np.float32(value)
        acc = fp32_fma(acc, operand, operand)
    return acc


def summac_partials(values: np.ndarray, slice_count: int) -> np.ndarray:
    vector = np.asarray(values).reshape(-1, order="C")
    if vector.size % slice_count:
        raise ValueError(
            f"summac input length {vector.size} is not divisible by {slice_count}"
        )
    width = vector.size // slice_count
    return np.asarray(
        [square_sum_fp32(vector[index * width : (index + 1) * width]) for index in range(slice_count)],
        dtype=np.float32,
    )


def remote_sum_fp32(values: np.ndarray) -> np.float32:
    acc = np.float32(0.0)
    for value in np.asarray(values, dtype=np.float32).reshape(-1, order="C"):
        acc = np.float32(float(acc) + float(value))
    return acc


def mac_sfu_rec_sqrt(value: np.ndarray, reduction_length: int) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    scale = np.float32(1.0 / reduction_length)
    biased_mean = np.asarray(
        [fp32_fma(np.float32(1.0e-6), item, scale) for item in source.reshape(-1)],
        dtype=np.float32,
    )
    result = np.float32(1.0) / np.sqrt(biased_mean, dtype=np.float32)
    return result.reshape(source.shape, order="C").astype(np.float32)


def head_local_maxima(
    values: np.ndarray,
    heads: int,
    slices_per_head: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != heads:
        raise ValueError(f"expected attention matrix (length, {heads}), got {matrix.shape}")
    if matrix.shape[0] % slices_per_head:
        raise ValueError("attention length must be divisible by slices_per_head")
    width = matrix.shape[0] // slices_per_head
    partials = np.empty((slices_per_head, heads), dtype=np.float32, order="F")
    for head in range(heads):
        for local_slice in range(slices_per_head):
            start = local_slice * width
            partials[local_slice, head] = np.max(matrix[start : start + width, head])
    return partials


def head_local_sum_reciprocals(
    values: np.ndarray,
    heads: int,
    slices_per_head: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != heads:
        raise ValueError(f"expected attention matrix (length, {heads}), got {matrix.shape}")
    if matrix.shape[0] % slices_per_head:
        raise ValueError("attention length must be divisible by slices_per_head")
    width = matrix.shape[0] // slices_per_head
    partials = np.empty((slices_per_head, heads), dtype=np.float32, order="F")
    for head in range(heads):
        for local_slice in range(slices_per_head):
            start = local_slice * width
            local_sum = remote_sum_fp32(matrix[start : start + width, head])
            if local_sum == np.float32(0.0):
                raise ZeroDivisionError("decode_sum_rec received a zero local sum")
            partials[local_slice, head] = np.float32(1.0) / local_sum
    return partials


def gemv_fp32_accumulate(weight: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute ``weight.T @ vector`` with an explicit fp32 accumulation order."""

    matrix = np.asarray(weight)
    input_vector = np.asarray(vector)
    if matrix.ndim != 2 or input_vector.ndim != 1:
        raise ValueError("gemv expects a 2D weight and a 1D vector")
    if matrix.shape[0] != input_vector.size:
        raise ValueError(
            f"gemv K mismatch: weight={matrix.shape}, vector={input_vector.shape}"
        )
    output = np.zeros(matrix.shape[1], dtype=np.float32)
    for out_index in range(matrix.shape[1]):
        acc = np.float32(0.0)
        for k_index in range(matrix.shape[0]):
            acc = fp32_fma(acc, matrix[k_index, out_index], input_vector[k_index])
        output[out_index] = acc
    return output


# ---------------------------------------------------------------------------
#  Simple golden models for element-wise / activation operators
# ---------------------------------------------------------------------------

def _elementwise_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)).astype(np.float32)


def _elementwise_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)).astype(np.float32)


def _vector_sub_scalar(vec: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    s = np.float32(np.asarray(scalar).reshape(-1)[0])
    result = np.empty_like(v)
    for i in range(v.size):
        result[i] = np.float32(float(v[i]) - float(s))
    return result


def _half_swap_vector(v: np.ndarray) -> np.ndarray:
    """Swap first half and second half of a 1-D vector in-place-like copy."""
    vec = np.asarray(v, dtype=np.float32).reshape(-1)
    half = vec.size // 2
    result = np.empty_like(vec)
    result[0:half] = vec[half:]
    result[half:]  = vec[0:half]
    return result
    """RoPE merge exactly matching the Prefill golden ``add_final`` sub-op.

    Prefill golden constructs ``add_in1`` by half-swapping ``mul2_out``::

        add_in1[0      :half] = mul2_out[half:      ]   # -x1*sin → first half
        add_in1[half   :     ] = mul2_out[0     :half]   # +x0*sin → second half

    With sin-table = [+sin | -sin], the element-wise ADD yields the correct
    LLaMA-style RoPE::

        result[i]       = cos_out[i]   + sin_out[i+half]   = x[i]*cos – x[i+half]*sin
        result[i+half]  = cos_out[i+half] + sin_out[i]     = x[i+half]*cos + x[i]*sin
    """
    v = np.asarray(cos_out, dtype=np.float32).reshape(-1)
    s = np.asarray(sin_out, dtype=np.float32).reshape(-1)
    half = v.size // 2
    result = np.empty_like(v)
    # in1: first half ← second half of sin_out, second half ← first half of sin_out
    for i in range(half):
        result[i]       = np.float32(float(v[i])       + float(s[i + half]))
        result[i + half] = np.float32(float(v[i + half]) + float(s[i]))
    return result


def _rope_merge(cos_out: np.ndarray, sin_out: np.ndarray) -> np.ndarray:
    """RoPE cross-pair merge: y0 = cos_out[i] + sin_out[i+half];
    y1 = sin_out[i] + cos_out[i+half] (for each half-pair)."""
    v = np.asarray(cos_out, dtype=np.float32).reshape(-1)
    s = np.asarray(sin_out, dtype=np.float32).reshape(-1)
    half = v.size // 2
    result = np.empty_like(v)
    for i in range(half):
        result[i] = np.float32(float(v[i]) + float(s[i + half]))
        result[i + half] = np.float32(float(s[i]) + float(v[i + half]))
    return result


def _silu_fp32(x: np.ndarray) -> np.ndarray:
    """SiLU: x * sigmoid(x) = x / (1 + exp(-x))."""
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    result = np.empty_like(v)
    for i in range(v.size):
        val = float(v[i])
        # Clip to avoid overflow in exp
        result[i] = np.float32(val / (1.0 + math.exp(-val)))
    return result


def build_decode_golden_cases(
    config: Mapping[str, object],
    weights: object | None = None,   # LayerWeights | None
    kv_cache: object | None = None,  # KVCache | None
) -> list[DecodeGoldenCase]:
    """Simulate one complete decode layer, strictly mirroring the Prefill
    ``layer0_0610_remapped.json`` operator graph (43 op instances).

    When *weights* and *kv_cache* are provided they are used instead of
    random tensors so that the golden matches real model behaviour.
    """
    validate_decode_config(config)
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    num_kv_heads = int(config["num_key_value_heads"])
    attention_length = int(config["decode_attention_length"])
    slice_per_head = int(config["slice_per_head"])
    used_slices = int(config["used_slices"])
    seed = int(config.get("random_seed", 0))
    rng = np.random.default_rng(seed)

    # -- resolve weights -------------------------------------------------
    _W: object = weights  # LayerWeights | None

    cases: list[DecodeGoldenCase] = []
    reg = DECODE_OP_REGISTRY

    def _add(inst_id: str, op_name: str, inputs: tuple[np.ndarray, ...], output: np.ndarray) -> np.ndarray:
        cases.append(DecodeGoldenCase(instance_id=inst_id, spec=reg[op_name], inputs=inputs, output=output))
        return output

    def _hvec(v: np.ndarray, dtype=None) -> np.ndarray:
        return as_vector(v, dtype=dtype)

    def _rope_tables() -> tuple[np.ndarray, np.ndarray]:
        """Build cos/sin tables for local RoPE (no cross-slice data routing).

        cos-table:  same value for pos and pos+half within each head; repeated
                    identically for every head.
        sin-table:  **rearranged** — ``[-sin_raw | +sin_raw]`` (negation in first
                    half) so that when combined with the half-swapped activation
                    at op8, element-wise ADD at op9 yields the correct RoPE.
        """
        half_head = head_dim // 2  # 64
        theta = 10000.0 ** (-2.0 * np.arange(0, half_head, dtype=np.float64) / head_dim)
        cos_theta = np.cos(theta).astype(np.float32)
        sin_theta = np.sin(theta).astype(np.float32)

        # per-head cos: broadcast cos(pos)==cos(pos+half_head)
        cos_1head = np.empty(head_dim, dtype=np.float32)
        cos_1head[0:half_head]       = cos_theta
        cos_1head[half_head:head_dim] = cos_theta

        # per-head sin: REARRANGED — first half: -sin, second half: +sin
        sin_1head = np.empty(head_dim, dtype=np.float32)
        sin_1head[0:half_head]       = -sin_theta   # negated in first half
        sin_1head[half_head:head_dim] =  sin_theta   # positive in second half

        # repeat for all heads → hidden_size
        cos = np.tile(cos_1head, heads)
        sin = np.tile(sin_1head, heads)
        return cos, sin

    def _rope_tables_1head() -> tuple[np.ndarray, np.ndarray]:
        """Single-head RoPE tables — rearranged for local computation."""
        half_head = head_dim // 2
        theta = 10000.0 ** (-2.0 * np.arange(0, half_head, dtype=np.float64) / head_dim)
        cos_theta = np.cos(theta).astype(np.float32)
        sin_theta = np.sin(theta).astype(np.float32)

        cos = np.empty(head_dim, dtype=np.float32)
        cos[0:half_head]       = cos_theta
        cos[half_head:head_dim] = cos_theta

        # REARRANGED: -sin in first half, +sin in second half
        sin = np.empty(head_dim, dtype=np.float32)
        sin[0:half_head]       = -sin_theta
        sin[half_head:head_dim] =  sin_theta
        return cos, sin

    # =========================================================================
    #  Shared input token
    # =========================================================================
    hidden_source = rng.uniform(-1.0, 1.0, hidden).astype(np.float32)

    # op0-op4   Attn RMS Norm
    op0_out = summac_partials(hidden_source, used_slices)
    _add("attn_summac",       "decode_summac_fp32N_fp32N", (_hvec(hidden_source),), _hvec(op0_out))
    op1_out = np.asarray([remote_sum_fp32(op0_out)], dtype=np.float32)
    _add("attn_remote_sum",   "decode_remote_sum_fp32N_fp32N", (_hvec(op0_out),), _hvec(op1_out))
    op2_out = mac_sfu_rec_sqrt(op1_out, hidden)
    _add("attn_mac_SFU",      "decode_mac_SFU_fp32N_fp32N", (_hvec(op1_out),), _hvec(op2_out))
    op3_out = (hidden_source * np.float32(op2_out.item())).astype(np.float32)
    _add("attn_mul_scale",    "decode_mul_fp32N_fp32_fp32N", (_hvec(hidden_source), _hvec(op2_out)), _hvec(op3_out))
    attn_scale = rng.uniform(0.5, 1.5, hidden).astype(np.float32)
    op4_out = (op3_out * attn_scale).astype(np.float16)
    _add("attn_mul_cast",     "decode_mul_fp32N_fp32N_fp16N", (_hvec(attn_scale), _hvec(op3_out)), _hvec(op4_out))

    # op5-op9   Q Projection + RoPE
    if _W is not None and hasattr(_W, "q_weight"):
        q_weight = np.asarray(_W.q_weight, dtype=np.float16)
        res_q = np.asarray(_W.q_bias, dtype=np.float32)
    else:
        q_weight = rng.uniform(-0.25, 0.25, (hidden, hidden)).astype(np.float16)
        res_q = rng.uniform(-0.5, 0.5, hidden).astype(np.float32)
    op5_out = gemv_fp32_accumulate(q_weight, op4_out.reshape(-1)).astype(np.float16)
    _add("q_gemv",        "decode_gemv_ring", (q_weight.reshape(hidden, hidden, 1, order="F"), _hvec(op4_out)), _hvec(op5_out))
    op6_out = _elementwise_add(op5_out, res_q).astype(np.float32)
    _add("q_add_residual","decode_add_fp16N_fp32N_fp32N", (_hvec(op5_out), _hvec(res_q)), _hvec(op6_out))
    cos_q, sin_q = _rope_tables()
    op7_out = _elementwise_mul(op6_out, cos_q)
    _add("q_rope_cos",    "decode_mul_fp32N_fp32N_fp32N", (_hvec(op6_out), _hvec(cos_q)), _hvec(op7_out))
    # op8: sin mul — activation is half-swapped so that element-wise ADD at op9
    # yields the correct RoPE result without cross-slice data routing.
    op6_out_swapped = _half_swap_vector(op6_out)
    op8_out = _elementwise_mul(op6_out_swapped, sin_q)
    _add("q_rope_sin",    "decode_mul_fp32N_fp32N_fp32N", (_hvec(op6_out_swapped), _hvec(sin_q)), _hvec(op8_out))
    # op9: RoPE merge — simple element-wise ADD, no routing needed
    op9_out = (op7_out + op8_out).astype(np.float16)
    _add("q_rope_merge",  "decode_add_fp32N_fp32N_fp16N", (_hvec(op7_out), _hvec(op8_out)), _hvec(op9_out))

    # op10-op14 KV RMS Norm
    # KV uses 4-slice (slice_per_head) unlike attn/ffn which use 28-slice (used_slices).
    # kv_input is 896 dim; hardware pads to 1024 (4×256).  The golden computes
    # summac on the unpadded 896 elements split 4 ways (224 per logical slice)
    # so the partial count matches the hardware's 4 logical slices.
    kv_input = rng.uniform(-1.0, 1.0, hidden).astype(np.float32)
    op10_out = summac_partials(kv_input, slice_per_head)
    _add("kv_summac",       "decode_summac_fp32N_fp32N", (_hvec(kv_input),), _hvec(op10_out))
    op11_out = np.asarray([remote_sum_fp32(op10_out)], dtype=np.float32)
    _add("kv_remote_sum",   "decode_remote_sum_fp32N_fp32N", (_hvec(op10_out),), _hvec(op11_out))
    op12_out = mac_sfu_rec_sqrt(op11_out, hidden)
    _add("kv_mac_SFU",      "decode_mac_SFU_fp32N_fp32N", (_hvec(op11_out),), _hvec(op12_out))
    op13_out = (kv_input * np.float32(op12_out.item())).astype(np.float32)
    _add("kv_mul_scale",    "decode_mul_fp32N_fp32_fp32N", (_hvec(kv_input), _hvec(op12_out)), _hvec(op13_out))
    kv_scale = rng.uniform(0.5, 1.5, hidden).astype(np.float32)
    op14_out = (op13_out * kv_scale).astype(np.float16)
    _add("kv_mul_cast",     "decode_mul_fp32N_fp32N_fp16N", (_hvec(kv_scale), _hvec(op13_out)), _hvec(op14_out))

    # op15-op19 K Projection + RoPE (K output = kv_dim = num_kv_heads*head_dim)
    kv_dim = num_kv_heads * head_dim  # 128 for num_kv_heads=1
    if _W is not None and hasattr(_W, "k_weight"):
        k_weight = np.asarray(_W.k_weight, dtype=np.float16)
        res_k = np.asarray(_W.k_bias, dtype=np.float32)
    else:
        k_weight = rng.uniform(-0.25, 0.25, (hidden, kv_dim)).astype(np.float16)
        res_k = rng.uniform(-0.5, 0.5, kv_dim).astype(np.float32)
    op15_out = gemv_fp32_accumulate(k_weight, op14_out.reshape(-1)).astype(np.float16)
    _add("k_gemv",        "decode_gemv_ring", (k_weight.reshape(hidden, kv_dim, 1, order="F"), _hvec(op14_out)), _hvec(op15_out))
    op16_out = _elementwise_add(op15_out, res_k).astype(np.float32)
    _add("k_add_residual","decode_add_fp16N_fp32N_fp32N", (_hvec(op15_out), _hvec(res_k)), _hvec(op16_out))
    # K RoPE: only kv_dim=128 elements (1 KV head).  Build per-head tables.
    cos_k, sin_k = _rope_tables_1head()
    op17_out = _elementwise_mul(op16_out, cos_k)
    _add("k_rope_cos",    "decode_mul_fp32N_fp32N_fp32N", (_hvec(op16_out), _hvec(cos_k)), _hvec(op17_out))
    # K RoPE sin: half-swapped activation + rearranged sin table → local ADD
    op16_out_swapped = _half_swap_vector(op16_out)
    op18_out = _elementwise_mul(op16_out_swapped, sin_k)
    _add("k_rope_sin",    "decode_mul_fp32N_fp32N_fp32N", (_hvec(op16_out_swapped), _hvec(sin_k)), _hvec(op18_out))
    # op19: RoPE merge — simple element-wise ADD, no routing
    op19_out = (op17_out + op18_out).astype(np.float16)
    _add("k_rope_merge",  "decode_add_fp32N_fp32N_fp16N", (_hvec(op17_out), _hvec(op18_out)), _hvec(op19_out))

    # op20-op21 V Projection
    if _W is not None and hasattr(_W, "v_weight"):
        v_weight = np.asarray(_W.v_weight, dtype=np.float16)
        res_v = np.asarray(_W.v_bias, dtype=np.float32)
    else:
        v_weight = rng.uniform(-0.25, 0.25, (hidden, kv_dim)).astype(np.float16)
        res_v = rng.uniform(-0.5, 0.5, kv_dim).astype(np.float32)
    op20_out = gemv_fp32_accumulate(v_weight, op14_out.reshape(-1)).astype(np.float16)
    _add("v_gemv",        "decode_gemv_ring", (v_weight.reshape(hidden, kv_dim, 1, order="F"), _hvec(op14_out)), _hvec(op20_out))
    op21_out = _elementwise_add(op20_out, res_v).astype(np.float16)
    _add("v_add_residual","decode_add_fp16N_fp32N_fp16N", (_hvec(op20_out), _hvec(res_v)), _hvec(op21_out))

    # op22-op29 Attention (GQA: num_kv_heads=1 shared by 7 Q heads)
    # ── Prefill-consistent pattern: golden produces (N, heads) matrices;
    #     passthrough splits by head (Mode B) → each head replicated to
    #     slice_per_head=4 slices.  Per-slice data = full attention vector
    #     for that head (32 elements after remote_sum). ──
    q_per_head = op9_out.reshape(heads, head_dim).astype(np.float16)

    if kv_cache is not None and hasattr(kv_cache, "k_cache"):
        kc = kv_cache.k_cache   # (head_dim, attention_length, num_kv_heads)
        vc = kv_cache.v_cache
    else:
        kc = rng.uniform(-0.25, 0.25, (head_dim, attention_length, num_kv_heads)).astype(np.float16)
        vc = rng.uniform(-0.25, 0.25, (head_dim, attention_length, num_kv_heads)).astype(np.float16)

    # Broadcast K/V from num_kv_heads to all Q heads for GQA
    k_cache = np.tile(kc, (1, 1, heads // num_kv_heads)).astype(np.float16)   # (128, 32, 7)
    v_cache = np.tile(vc, (1, 1, heads // num_kv_heads)).astype(np.float16)

    local_k = head_dim // slice_per_head  # 32
    qkt_partial = np.empty((attention_length, slice_per_head, heads), dtype=np.float16, order="F")
    for h in range(heads):
        for s in range(slice_per_head):
            start, end = s * local_k, (s + 1) * local_k
            scores_fp32 = gemv_fp32_accumulate(
                k_cache[start:end, :, h].astype(np.float32),
                q_per_head[h, start:end].astype(np.float32),
            )
            scores_fp32 = np.clip(scores_fp32, -65504, 65504)
            qkt_partial[:, s, h] = scores_fp32.astype(np.float16)
    _add("attn_qkt",       "decode_gemv_local",
         (k_cache.reshape(head_dim, attention_length, heads, order="F"),
          q_per_head.T.reshape(head_dim, heads, 1, order="F").astype(np.float16)),
         qkt_partial.reshape(attention_length, slice_per_head, heads, order="F"))

    # Full attention scores per head (sum partials across slices)
    # NOTE: qkt_partial is fp16; sum in fp32 to avoid overflow (4×65504 > fp16 max)
    attn_scores_2d = qkt_partial.astype(np.float32).reshape(attention_length, slice_per_head, heads, order="F").sum(axis=1)  # (32, 7)
    # Apply softmax scale: standard attention = softmax(QK^T / sqrt(head_dim))
    sm_scale = np.float32(1.0 / np.sqrt(float(head_dim)))  # 1/sqrt(128) ≈ 0.088
    attn_scores_2d = attn_scores_2d * sm_scale
    # op23 remote_sum: aggregates 4 slice partials → full (32,) vector per head.
    # The golden produces (32, 7) for Mode B split (7 heads × 4 slices).
    _add("attn_qkt_remote_sum", "decode_remote_sum_fp32N_fp32N",
         (attn_scores_2d.reshape(attention_length, heads, 1, order="F"),),
         attn_scores_2d.reshape(attention_length, heads, 1, order="F"))

    # ── ops 24-28: per-head attention chain (Mode B) ──
    # Each head has a full 32-element attention vector; replicated to
    # slice_per_head=4 slices by the passthrough.
    attn_mask = rng.uniform(-0.5, 0.5, (attention_length, heads)).astype(np.float32)
    op24_out = _elementwise_add(attn_scores_2d, attn_mask)  # (32, 7)
    _add("attn_add_mask",  "decode_add_fp32N_fp32N_fp32N",
         (attn_scores_2d.reshape(attention_length, heads, 1, order="F"),
          attn_mask.reshape(attention_length, heads, 1, order="F")),
         op24_out.reshape(attention_length, heads, 1, order="F"))

    # op25 max: per-head global max → (1, 7)
    op25_out = np.max(op24_out, axis=0, keepdims=True).astype(np.float32)  # (1, 7)
    _add("attn_max",       "decode_max_fp32N_fp32N",
         (op24_out.reshape(attention_length, heads, 1, order="F"),),
         op25_out.reshape(1, heads, 1, order="F"))

    # op26 sub_SFU: scores - max  per head → (32, 7)
    op26_out = op24_out - op25_out  # broadcast (1,7) over (32,7)
    _add("attn_sub_SFU",   "decode_sub_SFU_fp32N_fp32_fp32N",
         (op24_out.reshape(attention_length, heads, 1, order="F"),
          op25_out.reshape(1, heads, 1, order="F")),
         op26_out.reshape(attention_length, heads, 1, order="F"))

    # op27 sum_rec: exp(x-max) → per-head sum → reciprocal → (1, 7)
    exp_attn = np.exp(op26_out).astype(np.float32)  # (32, 7)
    op27_out = (np.float32(1.0) / exp_attn.sum(axis=0, keepdims=True)).astype(np.float32)  # (1, 7)
    _add("attn_sum_rec",   "decode_sum_rec_fp32N_fp32N",
         (op26_out.reshape(attention_length, heads, 1, order="F"),),
         op27_out.reshape(1, heads, 1, order="F"))

    # op28 mul_softmax: exp(x-max) * 1/sum → (32, 7) → fp16
    op28_out = (exp_attn * op27_out).astype(np.float16)  # (32, 7)
    _add("attn_mul_softmax","decode_mul_fp32N_fp32_fp16N",
         (op26_out.reshape(attention_length, heads, 1, order="F"),
          op27_out.reshape(1, heads, 1, order="F")),
         op28_out.reshape(attention_length, heads, 1, order="F"))

    sv_weight = v_cache.astype(np.float16)  # (128, 32, 7) after GQA broadcast
    # Current-token V (kv_dim=128) → broadcast to 7 heads for GQA
    v_cur = np.tile(op21_out.reshape(-1), heads).reshape(heads, kv_dim).T.astype(np.float16)  # (128, 7)
    sv_partial = np.empty((attention_length, slice_per_head, heads), dtype=np.float16, order="F")
    for h in range(heads):
        for s in range(slice_per_head):
            start, end = s * local_k, (s + 1) * local_k
            sv_partial[:, s, h] = gemv_fp32_accumulate(sv_weight[start:end, :, h], v_cur[start:end, h]).astype(np.float16)
    _add("attn_sv",        "decode_gemv_local",
         (sv_weight.reshape(head_dim, attention_length, heads, order="F"),
          v_cur.reshape(head_dim, heads, 1, order="F")),
         sv_partial.reshape(attention_length, slice_per_head, heads, order="F"))

    # op30-op31 Output GEMV + residual
    attn_flat = sv_partial.reshape(-1, order="F")[:hidden].astype(np.float16)
    if _W is not None and hasattr(_W, "out_weight"):
        o_weight = np.asarray(_W.out_weight, dtype=np.float16)
    else:
        o_weight = rng.uniform(-0.25, 0.25, (hidden, hidden)).astype(np.float16)
    op30_out = gemv_fp32_accumulate(o_weight, attn_flat).astype(np.float16)
    _add("out_gemv",       "decode_gemv_ring", (o_weight.reshape(hidden, hidden, 1, order="F"), _hvec(attn_flat)), _hvec(op30_out))
    op31_out = _elementwise_add(op30_out, res_q).astype(np.float32)
    _add("out_add_residual","decode_add_fp32N_fp16N_fp32N", (_hvec(res_q), _hvec(op30_out)), _hvec(op31_out))

    # op32-op36 FFN RMS Norm
    op32_out = summac_partials(op31_out.reshape(-1), used_slices)
    _add("ffn_summac",      "decode_summac_fp32N_fp32N", (_hvec(op31_out.reshape(-1)),), _hvec(op32_out))
    op33_out = np.asarray([remote_sum_fp32(op32_out)], dtype=np.float32)
    _add("ffn_remote_sum",  "decode_remote_sum_fp32N_fp32N", (_hvec(op32_out),), _hvec(op33_out))
    op34_out = mac_sfu_rec_sqrt(op33_out, hidden)
    _add("ffn_mac_SFU",     "decode_mac_SFU_fp32N_fp32N", (_hvec(op33_out),), _hvec(op34_out))
    ffn_src = op31_out.reshape(-1)
    op35_out = (ffn_src * np.float32(op34_out.item())).astype(np.float32)
    _add("ffn_mul_scale",   "decode_mul_fp32N_fp32_fp32N", (_hvec(ffn_src), _hvec(op34_out)), _hvec(op35_out))
    ffn_scale = rng.uniform(0.5, 1.5, hidden).astype(np.float32)
    op36_out = (op35_out * ffn_scale).astype(np.float16)
    _add("ffn_mul_cast",    "decode_mul_fp32N_fp32N_fp16N", (_hvec(ffn_scale), _hvec(op35_out)), _hvec(op36_out))

    # op37-op42 FFN
    if _W is not None and hasattr(_W, "ffn_gate_weight"):
        gate_w = np.asarray(_W.ffn_gate_weight, dtype=np.float16)
        up_w = np.asarray(_W.ffn_up_weight, dtype=np.float16)
        out2_w = np.asarray(_W.ffn_down_weight, dtype=np.float16)
    else:
        gate_w = rng.uniform(-0.25, 0.25, (hidden, intermediate)).astype(np.float16)
        up_w = rng.uniform(-0.25, 0.25, (hidden, intermediate)).astype(np.float16)
        out2_w = rng.uniform(-0.25, 0.25, (intermediate, hidden)).astype(np.float16)
    op37_out = gemv_fp32_accumulate(gate_w, op36_out.reshape(-1)).astype(np.float16)
    _add("ffn_gate_gemv",  "decode_gemv_ring", (gate_w.reshape(hidden, intermediate, 1, order="F"), _hvec(op36_out)), _hvec(op37_out))
    op38_out = gemv_fp32_accumulate(up_w, op36_out.reshape(-1)).astype(np.float16)
    _add("ffn_up_gemv",    "decode_gemv_ring", (up_w.reshape(hidden, intermediate, 1, order="F"), _hvec(op36_out)), _hvec(op38_out))
    op39_out = _silu_fp32(op37_out).astype(np.float32)
    _add("ffn_silu",       "decode_silu_fp16N_fp32N", (_hvec(op37_out),), _hvec(op39_out))
    op40_out = _elementwise_mul(op39_out, op38_out.astype(np.float32)).astype(np.float16)
    _add("ffn_gate_up_mul","decode_mul_fp32N_fp16N_fp16N", (_hvec(op39_out), _hvec(op38_out)), _hvec(op40_out))
    op41_out = gemv_fp32_accumulate(out2_w, op40_out.reshape(-1)).astype(np.float16)
    _add("ffn_out_gemv",   "decode_gemv_ring", (out2_w.reshape(intermediate, hidden, 1, order="F"), _hvec(op40_out)), _hvec(op41_out))
    op42_out = _elementwise_add(op41_out, ffn_src).astype(np.float32)
    _add("ffn_add_residual","decode_add_fp32N_fp16N_fp32N", (_hvec(ffn_src), _hvec(op41_out)), _hvec(op42_out))

    print(f"[decode-layer] {len(cases)} layer instances ({len({c.spec.name for c in cases})} operator types)")
    return cases

