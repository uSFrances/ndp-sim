"""Full-size Decode golden using two 7-head passes.

Hardware has 28 slices and the original small-size kernel handles exactly
7 Q heads (7 * 4 slices = 28).  The full 1.5B model has 12 real Q heads, so
we pad to 14 and run the 7-head per-head computation twice:

    Pass A: Q heads 0..6
    Pass B: Q heads 7..13   (7..11 real, 12..13 padding/zero)

Shared work (Attention RMSNorm, KV RMSNorm/projection/cache, residual, FFN)
is computed once.  Only the per-head part is duplicated:

    Q projection -> Q residual/RoPE -> QK^T -> softmax -> SV -> out partial

The two output-projection partials are combined with the existing
``decode_add_fp32N_fp16N_fp32N`` operator, first adding residual + out_A and
then + out_B, so the residual is only added once.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from decode_ops import (
    DECODE_OP_REGISTRY,
    DecodeGoldenCase,
    _elementwise_add,
    _elementwise_mul,
    _silu_fp32,
    as_vector,
    decode_padded_dims,
    gemv_fp32_accumulate,
    mac_sfu_rec_sqrt,
    pad_matrix_real,
    pad_vector,
    remote_sum_fp32,
    square_sum_fp32,
    validate_decode_config,
)


def _remote_summac_pe_partials(
    a_flat: np.ndarray,
    logical_slices: int,
    per_slice: int,
) -> np.ndarray:
    """Match hardware remote_summac PE grouping.

    The installed per-slice A is laid out as ``[PE=8 fastest, n_per_pe]``
    (F-order of ``[1, 8, n_per_pe]``).  PE ``p`` therefore consumes the
    slice positions ``p, p+8, p+16, ...``.  Each slice computes a local PE
    square-sum; the remote reduction then sums the same PE lane across all
    slices.
    """

    flat = np.asarray(a_flat, dtype=np.float32).reshape(-1)
    total = logical_slices * per_slice
    if flat.size < total:
        padded = np.zeros(total, dtype=np.float32)
        padded[: flat.size] = flat
        flat = padded
    elif flat.size > total:
        flat = flat[:total]

    n_per_pe = per_slice // 8
    if per_slice % 8:
        raise ValueError(f"per_slice={per_slice} must be divisible by 8")
    # C-order index = slice * per_slice + j * 8 + p
    arranged = flat.reshape(logical_slices, n_per_pe, 8)

    out = np.empty(8, dtype=np.float32)
    for pe in range(8):
        local_partials = np.asarray(
            [square_sum_fp32(arranged[sl, :, pe]) for sl in range(logical_slices)],
            dtype=np.float32,
        )
        out[pe] = remote_sum_fp32(local_partials)
    return out


def _real_vector(owner: object | None, attribute: str, length: int, dtype=np.float32):
    if owner is None or not hasattr(owner, attribute):
        return None
    value = np.asarray(getattr(owner, attribute), dtype=dtype).reshape(-1, order="F")
    if value.size < length:
        return None
    return value[:length]


def _real_matrix(owner: object | None, attribute: str, shape: tuple[int, int], dtype=np.float16):
    if owner is None or not hasattr(owner, attribute):
        return None
    value = np.asarray(getattr(owner, attribute), dtype=dtype)
    if value.ndim != 2 or value.shape[0] < shape[0] or value.shape[1] < shape[1]:
        return None
    return value[: shape[0], : shape[1]]


def _q_weight_pass(
    real_weight: np.ndarray,
    head_start: int,
    heads_per_pass: int,
    heads: int,
    head_dim: int,
    hidden: int,
    hidden_p: int,
) -> np.ndarray:
    out = np.zeros((hidden_p, heads_per_pass * head_dim), dtype=np.float16)
    for local_head in range(heads_per_pass):
        head = head_start + local_head
        if head >= heads:
            continue
        src = slice(head * head_dim, (head + 1) * head_dim)
        dst = slice(local_head * head_dim, (local_head + 1) * head_dim)
        out[:hidden, dst] = real_weight[:hidden, src]
    return out


def _q_bias_pass(
    real_bias: np.ndarray,
    head_start: int,
    heads_per_pass: int,
    heads: int,
    head_dim: int,
) -> np.ndarray:
    out = np.zeros(heads_per_pass * head_dim, dtype=np.float32)
    for local_head in range(heads_per_pass):
        head = head_start + local_head
        if head >= heads:
            continue
        src = slice(head * head_dim, (head + 1) * head_dim)
        dst = slice(local_head * head_dim, (local_head + 1) * head_dim)
        out[dst] = real_bias[src]
    return out


def _out_weight_pass(
    real_weight: np.ndarray,
    head_start: int,
    heads_per_pass: int,
    heads: int,
    head_dim: int,
    hidden: int,
    hidden_p: int,
) -> np.ndarray:
    out = np.zeros((heads_per_pass * head_dim, hidden_p), dtype=np.float16)
    for local_head in range(heads_per_pass):
        head = head_start + local_head
        if head >= heads:
            continue
        src = slice(head * head_dim, (head + 1) * head_dim)
        dst = slice(local_head * head_dim, (local_head + 1) * head_dim)
        out[dst, :hidden] = real_weight[src, :hidden]
    return out


def _rope_vectors(
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    position: int,
    heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    if position < 0 or position >= cos_table.shape[1]:
        raise ValueError(f"RoPE position {position} out of range")
    cos_half = np.asarray(cos_table[:half, position], dtype=np.float32)
    sin_half = np.asarray(sin_table[:half, position], dtype=np.float32)
    cos_head = np.concatenate([cos_half, cos_half])
    sin_head = np.concatenate([sin_half, -sin_half])
    return np.tile(cos_head, heads), np.tile(sin_head, heads)


def _half_swap_heads(values: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(heads, head_dim)
    half = head_dim // 2
    out = np.empty_like(array)
    out[:, :half] = array[:, half:]
    out[:, half:] = array[:, :half]
    return out.reshape(-1)


def build_decode_golden_cases_optimized_padded(
    config: Mapping[str, object],
    weights: object | None = None,
    kv_cache: object | None = None,
) -> list[DecodeGoldenCase]:
    validate_decode_config(config)

    dims = decode_padded_dims(config)
    hidden = dims["hidden"]
    hidden_p = dims["hidden_padded"]
    heads = dims["heads"]
    heads_p = dims["heads_padded"]
    head_dim = dims["head_dim"]
    slice_per_head = dims["slice_per_head"]
    attention_passes = dims["attention_passes"]
    heads_per_pass = dims["heads_per_pass"]
    pass_hidden = dims["pass_hidden"]
    used_slices = int(config["used_slices"])
    chunk = dims["chunk"]
    intermediate = int(config["intermediate_size"])
    num_kv_heads = int(config["num_key_value_heads"])
    attention_length = int(config["decode_attention_length"])
    seed = int(config.get("random_seed", 0))
    rng = np.random.default_rng(seed)

    if attention_passes != 2 or heads_per_pass * 2 != heads_p:
        raise ValueError("two-pass builder expects attention_passes=2 and 2*heads_per_pass=heads_padded")
    if pass_hidden != heads_per_pass * head_dim:
        raise ValueError("pass_hidden mismatch")
    if heads_per_pass * slice_per_head != used_slices:
        raise ValueError("pass head/slice mismatch")

    kv_dim = num_kv_heads * head_dim
    kv_slice_count = int(config.get("kv_slice_count", 4))
    kv_padding_a = int(config.get("kv_padding_a", 256))
    rope_mode = str(config.get("rope_mode", "passthrough"))
    rope_position = int(
        config.get("rope_query_position", config.get("kv_cache_initial_length", 0))
    )
    cos_table = sin_table = None
    if rope_mode == "table":
        from decode_data_loader import load_rope_tables

        cos_table, sin_table = load_rope_tables(config)
    weight_owner = weights
    cases: list[DecodeGoldenCase] = []
    reg = DECODE_OP_REGISTRY

    def _add(inst_id, op_name, inputs, output):
        inputs_f = tuple(np.require(np.asarray(t), requirements="F") for t in inputs)
        output_f = np.require(np.asarray(output), requirements="F")
        cases.append(
            DecodeGoldenCase(
                instance_id=inst_id,
                spec=reg[op_name],
                inputs=inputs_f,
                output=output_f,
            )
        )
        return output

    def _hvec(v, dtype=None):
        return as_vector(v, dtype=dtype)

    def _random_vector(length, low, high, dtype=np.float32):
        return rng.uniform(low, high, length).astype(dtype)

    def _padded_random_vector(real_length, low, high, dtype=np.float32):
        return pad_vector(_random_vector(real_length, low, high, dtype), hidden_p, dtype)

    # ------------------------------------------------------------------
    # Real weights
    # ------------------------------------------------------------------
    q_weight_real = _real_matrix(weight_owner, "q_weight", (hidden, hidden))
    if q_weight_real is None:
        q_weight_real = rng.uniform(-0.25, 0.25, (hidden, hidden)).astype(np.float16)
    q_bias_real = _real_vector(weight_owner, "q_bias", hidden)
    if q_bias_real is None:
        q_bias_real = _random_vector(hidden, -0.5, 0.5)

    k_weight = pad_matrix_real(
        _real_matrix(weight_owner, "k_weight", (hidden, kv_dim)),
        (hidden, kv_dim),
        (hidden_p, kv_dim),
        rng=rng,
    )
    k_bias_real = _real_vector(weight_owner, "k_bias", kv_dim)
    if k_bias_real is None:
        k_bias_real = _random_vector(kv_dim, -0.5, 0.5)
    res_k = np.asarray(k_bias_real, dtype=np.float32)

    v_weight = pad_matrix_real(
        _real_matrix(weight_owner, "v_weight", (hidden, kv_dim)),
        (hidden, kv_dim),
        (hidden_p, kv_dim),
        rng=rng,
    )
    v_bias_real = _real_vector(weight_owner, "v_bias", kv_dim)
    if v_bias_real is None:
        v_bias_real = _random_vector(kv_dim, -0.5, 0.5)
    res_v = np.asarray(v_bias_real, dtype=np.float32)

    out_weight_real = _real_matrix(weight_owner, "out_weight", (hidden, hidden))
    if out_weight_real is None:
        out_weight_real = rng.uniform(-0.25, 0.25, (hidden, hidden)).astype(np.float16)

    gate_weight = pad_matrix_real(
        _real_matrix(weight_owner, "ffn_gate_weight", (hidden, intermediate)),
        (hidden, intermediate),
        (hidden_p, intermediate),
        rng=rng,
    )
    up_weight = pad_matrix_real(
        _real_matrix(weight_owner, "ffn_up_weight", (hidden, intermediate)),
        (hidden, intermediate),
        (hidden_p, intermediate),
        rng=rng,
    )
    down_weight = pad_matrix_real(
        _real_matrix(weight_owner, "ffn_down_weight", (intermediate, hidden)),
        (intermediate, hidden),
        (intermediate, hidden_p),
        rng=rng,
    )

    def _norm_scale(attribute: str) -> np.ndarray:
        value = _real_vector(weight_owner, attribute, hidden)
        if value is None:
            value = _random_vector(hidden, 0.5, 1.5)
        return pad_vector(value, hidden_p, np.float32)

    # ------------------------------------------------------------------
    # Shared Attention RMSNorm op0-4
    # ------------------------------------------------------------------
    hidden_source = _padded_random_vector(hidden, -1.0, 1.0)
    op0_out = _remote_summac_pe_partials(
        hidden_source.reshape(8, hidden_p // 8).ravel(order="F"),
        used_slices,
        hidden_p // used_slices,
    )
    _add(
        "attn_remote_summac",
        "prefill_remote_summac_fp32MN_fp32MN",
        (
            np.expand_dims(hidden_source.reshape(8, hidden_p // 8), 2),
            _hvec(np.ones(1, dtype=np.float32)),
        ),
        _hvec(op0_out),
    )
    op1_out = np.asarray([remote_sum_fp32(op0_out)], dtype=np.float32)
    _add("attn_decode_sum", "decode_sum_fp32N_fp32N", (_hvec(op0_out),), _hvec(op1_out))

    op2_out = mac_sfu_rec_sqrt(op1_out, hidden)
    _add("attn_mac_SFU", "decode_mac_SFU_fp32N_fp32N", (_hvec(op1_out),), _hvec(op2_out))

    op3_out = (hidden_source * np.float32(op2_out.item())).astype(np.float32)
    _add(
        "attn_mul_scale",
        "decode_mul_fp32N_fp32_fp32N",
        (_hvec(hidden_source), _hvec(op2_out)),
        _hvec(op3_out),
    )
    attn_scale = _norm_scale("attn_norm")
    op4_out = (op3_out * attn_scale).astype(np.float16)
    _add(
        "attn_mul_cast",
        "decode_mul_fp32N_fp32N_fp16N",
        (_hvec(attn_scale), _hvec(op3_out)),
        _hvec(op4_out),
    )

    # ------------------------------------------------------------------
    # Two per-head Q passes (op5-9, op10-14)
    # ------------------------------------------------------------------
    def _build_q_pass(head_start: int, tag: str, q_input: np.ndarray) -> np.ndarray:
        q_weight = _q_weight_pass(
            q_weight_real, head_start, heads_per_pass, heads, head_dim, hidden, hidden_p
        )
        q_bias = _q_bias_pass(q_bias_real, head_start, heads_per_pass, heads, head_dim)

        q_gemv = gemv_fp32_accumulate(q_weight, q_input.reshape(-1)).astype(np.float16)
        _add(
            f"q_gemv_{tag}",
            "decode_gemv_ring",
            (np.expand_dims(q_weight, 2), _hvec(q_input)),
            _hvec(q_gemv),
        )
        q_add = _elementwise_add(q_gemv, q_bias).astype(np.float32)
        _add(
            f"q_add_residual_{tag}",
            "decode_add_fp16N_fp32N_fp32N",
            (_hvec(q_gemv), _hvec(q_bias)),
            _hvec(q_add),
        )
        if rope_mode == "table":
            q_cos_table, q_sin_table = _rope_vectors(
                cos_table, sin_table, rope_position, heads_per_pass, head_dim
            )
            q_cos = (q_add * q_cos_table).astype(np.float32)
            q_sin = (q_add * q_sin_table).astype(np.float32)
            q_sin_routed = _half_swap_heads(q_sin, heads_per_pass, head_dim)
            q_merge = (q_cos + q_sin_routed).astype(np.float16)
            _add(
                f"q_rope_cos_{tag}",
                "decode_mul_fp32N_fp32N_fp32N",
                (_hvec(q_add), _hvec(q_cos_table)),
                _hvec(q_cos),
            )
            _add(
                f"q_rope_sin_{tag}",
                "decode_mul_fp32N_fp32N_fp32N",
                (_hvec(q_add), _hvec(q_sin_table)),
                _hvec(q_sin),
            )
            _add(
                f"q_rope_merge_{tag}",
                "decode_add_fp32N_fp32N_fp16N",
                (_hvec(q_cos), _hvec(q_sin_routed)),
                _hvec(q_merge),
            )
        else:
            q_cos = np.asarray(q_add, dtype=np.float32).copy()
            _add(
                f"q_rope_cos_{tag}",
                "decode_mul_fp32N_fp32N_fp32N",
                (_hvec(q_add), _hvec(np.ones(pass_hidden, dtype=np.float32))),
                _hvec(q_cos),
            )
            q_sin = np.zeros(pass_hidden, dtype=np.float32)
            _add(
                f"q_rope_sin_{tag}",
                "decode_mul_fp32N_fp32N_fp32N",
                (_hvec(q_add), _hvec(np.zeros(pass_hidden, dtype=np.float32))),
                _hvec(q_sin),
            )
            q_merge = (q_cos + q_sin).astype(np.float16)
            _add(
                f"q_rope_merge_{tag}",
                "decode_add_fp32N_fp32N_fp16N",
                (_hvec(q_cos), _hvec(q_sin)),
                _hvec(q_merge),
            )
        return q_merge

    q_pass_a = _build_q_pass(0, "A", op4_out)
    q_pass_b = _build_q_pass(heads_per_pass, "B", op4_out)

    # ------------------------------------------------------------------
    # Shared KV RMSNorm op15-19
    # ------------------------------------------------------------------
    kv_input = _padded_random_vector(hidden, -1.0, 1.0)
    op15_out = _remote_summac_pe_partials(
        kv_input.reshape(8, hidden_p // 8).ravel(order="F"),
        kv_slice_count,
        kv_padding_a,
    )
    _add(
        "kv_remote_summac",
        "prefill_remote_summac_4slice_fp32MN_fp32MN",
        (
            np.expand_dims(kv_input.reshape(8, hidden_p // 8), 2),
            _hvec(np.ones(1, dtype=np.float32)),
        ),
        _hvec(op15_out),
    )
    op16_out = np.asarray([remote_sum_fp32(op15_out)], dtype=np.float32)
    _add("kv_decode_sum", "decode_sum_fp32N_fp32N", (_hvec(op15_out),), _hvec(op16_out))
    op17_out = mac_sfu_rec_sqrt(op16_out, hidden)
    _add("kv_mac_SFU", "decode_mac_SFU_fp32N_fp32N", (_hvec(op16_out),), _hvec(op17_out))
    op18_out = (kv_input * np.float32(op17_out.item())).astype(np.float32)
    _add(
        "kv_mul_scale",
        "decode_mul_fp32N_fp32_fp32N",
        (_hvec(kv_input), _hvec(op17_out)),
        _hvec(op18_out),
    )
    kv_scale = _norm_scale("attn_norm")
    op19_out = (op18_out * kv_scale).astype(np.float16)
    _add(
        "kv_mul_cast",
        "decode_mul_fp32N_fp32N_fp16N",
        (_hvec(kv_scale), _hvec(op18_out)),
        _hvec(op19_out),
    )

    # ------------------------------------------------------------------
    # Shared K/V projection + cache path op20-26
    # ------------------------------------------------------------------
    op20_out = gemv_fp32_accumulate(k_weight, op19_out.reshape(-1)).astype(np.float16)
    _add(
        "k_gemv",
        "decode_gemv_ring",
        (np.expand_dims(k_weight, 2), _hvec(op19_out)),
        _hvec(op20_out),
    )
    op21_out = _elementwise_add(op20_out, res_k).astype(np.float32)
    _add(
        "k_add_residual",
        "decode_add_fp16N_fp32N_fp32N",
        (_hvec(op20_out), _hvec(res_k)),
        _hvec(op21_out),
    )
    if rope_mode == "table":
        k_cos_table, k_sin_table = _rope_vectors(
            cos_table, sin_table, rope_position, num_kv_heads, head_dim
        )
        op22_out = (op21_out * k_cos_table).astype(np.float32)
        op23_out = (op21_out * k_sin_table).astype(np.float32)
        op23_routed = _half_swap_heads(op23_out, num_kv_heads, head_dim)
        op24_out = (op22_out + op23_routed).astype(np.float16)
        _add(
            "k_rope_cos",
            "decode_mul_fp32N_fp32N_fp32N",
            (_hvec(op21_out), _hvec(k_cos_table)),
            _hvec(op22_out),
        )
        _add(
            "k_rope_sin",
            "decode_mul_fp32N_fp32N_fp32N",
            (_hvec(op21_out), _hvec(k_sin_table)),
            _hvec(op23_out),
        )
        _add(
            "k_rope_merge",
            "decode_add_fp32N_fp32N_fp16N",
            (_hvec(op22_out), _hvec(op23_routed)),
            _hvec(op24_out),
        )
    else:
        op22_out = np.asarray(op21_out, dtype=np.float32).copy()
        _add(
            "k_rope_cos",
            "decode_mul_fp32N_fp32N_fp32N",
            (_hvec(op21_out), _hvec(np.ones(kv_dim, dtype=np.float32))),
            _hvec(op22_out),
        )
        op23_out = np.zeros(kv_dim, dtype=np.float32)
        _add(
            "k_rope_sin",
            "decode_mul_fp32N_fp32N_fp32N",
            (_hvec(op21_out), _hvec(np.zeros(kv_dim, dtype=np.float32))),
            _hvec(op23_out),
        )
        op24_out = (op22_out + op23_out).astype(np.float16)
        _add(
            "k_rope_merge",
            "decode_add_fp32N_fp32N_fp16N",
            (_hvec(op22_out), _hvec(op23_out)),
            _hvec(op24_out),
        )

    op25_out = gemv_fp32_accumulate(v_weight, op19_out.reshape(-1)).astype(np.float16)
    _add(
        "v_gemv",
        "decode_gemv_ring",
        (np.expand_dims(v_weight, 2), _hvec(op19_out)),
        _hvec(op25_out),
    )
    op26_out = _elementwise_add(res_v, op25_out).astype(np.float16)
    _add(
        "v_add_residual",
        "decode_add_fp32N_fp16N_fp16N",
        (_hvec(res_v), _hvec(op25_out)),
        _hvec(op26_out),
    )

    # ------------------------------------------------------------------
    # KV cache / GQA mapping
    # ------------------------------------------------------------------
    valid_cache = (
        kv_cache is not None
        and hasattr(kv_cache, "k_cache")
        and np.asarray(kv_cache.k_cache).shape == (head_dim, attention_length, num_kv_heads)
    )
    if valid_cache:
        kc = np.asarray(kv_cache.k_cache, dtype=np.float16)
        vc = np.asarray(kv_cache.v_cache, dtype=np.float16)
    else:
        kc = _random_vector(head_dim * attention_length * num_kv_heads, -0.25, 0.25).reshape(
            head_dim, attention_length, num_kv_heads, order="F"
        ).astype(np.float16)
        vc = _random_vector(head_dim * attention_length * num_kv_heads, -0.25, 0.25).reshape(
            head_dim, attention_length, num_kv_heads, order="F"
        ).astype(np.float16)

    if rope_mode != "table":
        from decode_ops import _rope_inverse_rot

        kc = _rope_inverse_rot(kc)
    repeats = max(1, heads // num_kv_heads)

    def q_to_kv(head: int) -> int:
        if head >= heads:
            return head % num_kv_heads
        return min(head // repeats, num_kv_heads - 1)

    # ------------------------------------------------------------------
    # Two per-head attention passes (op27-35, op36-44)
    # ------------------------------------------------------------------
    local_k = head_dim // slice_per_head
    sm_scale = np.float32(1.0 / np.sqrt(float(head_dim)))

    def _build_attention_pass(head_start: int, tag: str, q_out: np.ndarray) -> np.ndarray:
        q_per_head = q_out.reshape(heads_per_pass, head_dim).astype(np.float16)

        k_cache_pass = np.empty(
            (head_dim, attention_length, heads_per_pass), dtype=np.float16, order="F"
        )
        v_cache_pass = np.empty(
            (head_dim, attention_length, heads_per_pass), dtype=np.float16, order="F"
        )
        for local_head in range(heads_per_pass):
            head = head_start + local_head
            kv_head = q_to_kv(head)
            k_cache_pass[:, :, local_head] = kc[:, :, kv_head]
            v_cache_pass[:, :, local_head] = vc[:, :, kv_head]

        qkt_partial = np.empty(
            (attention_length, slice_per_head, heads_per_pass),
            dtype=np.float32,
            order="F",
        )
        for local_head in range(heads_per_pass):
            for local_slice in range(slice_per_head):
                start = local_slice * local_k
                end = start + local_k
                qkt_partial[:, local_slice, local_head] = gemv_fp32_accumulate(
                    k_cache_pass[start:end, :, local_head].astype(np.float32),
                    q_per_head[local_head, start:end].astype(np.float32),
                )

        _add(
            f"attn_qkt_{tag}",
            "decode_gemv_local_qkt",
            (
                k_cache_pass,
                np.asarray(np.expand_dims(q_per_head.T, 2), dtype=np.float16),
            ),
            qkt_partial,
        )

        attn_scores_2d = qkt_partial.sum(axis=1)
        _add(
            f"attn_qkt_remote_sum_{tag}",
            "prefill_remote_sum_qkt_4slice_fp32MN_fp32MN",
            (qkt_partial, _hvec(np.ones(1, dtype=np.float32))),
            np.expand_dims(attn_scores_2d, 2),
        )

        attn_mask = rng.uniform(-0.5, 0.5, (attention_length, heads_per_pass)).astype(np.float32)
        for local_head in range(heads_per_pass):
            if head_start + local_head >= heads:
                attn_mask[:, local_head] = 0.0

        op_masked = attn_scores_2d * sm_scale + attn_mask
        _add(
            f"attn_add_mask_{tag}",
            "decode_mac_fp32N_fp32N_fp32N",
            (np.expand_dims(attn_scores_2d, 2), np.expand_dims(attn_mask, 2)),
            np.expand_dims(op_masked, 2),
        )

        op_max = np.max(op_masked, axis=0, keepdims=True).astype(np.float32)
        _add(
            f"attn_max_{tag}",
            "decode_max_fp32N_fp32N",
            (np.expand_dims(op_masked, 2),),
            np.expand_dims(op_max, 2),
        )

        op_exp = np.exp(op_masked - op_max).astype(np.float32)
        _add(
            f"attn_sub_SFU_{tag}",
            "decode_sub_SFU_fp32N_fp32_fp32N",
            (np.expand_dims(op_masked, 2), np.expand_dims(op_max, 2)),
            np.expand_dims(op_exp, 2),
        )

        op_rec = (np.float32(1.0) / op_exp.sum(axis=0, keepdims=True)).astype(np.float32)
        _add(
            f"attn_sum_rec_{tag}",
            "decode_sum_rec_fp32N_fp32N",
            (np.expand_dims(op_exp, 2),),
            np.expand_dims(op_rec, 2),
        )

        op_soft = (op_exp * op_rec).astype(np.float16)
        _add(
            f"attn_mul_softmax_{tag}",
            "decode_mul_fp32N_fp32_fp16N",
            (np.expand_dims(op_exp, 2), np.expand_dims(op_rec, 2)),
            np.expand_dims(op_soft, 2),
        )

        sv_partial = np.empty(
            (attention_length, slice_per_head, heads_per_pass),
            dtype=np.float16,
            order="F",
        )
        for local_head in range(heads_per_pass):
            for local_slice in range(slice_per_head):
                start = local_slice * local_k
                end = start + local_k
                sv_partial[:, local_slice, local_head] = gemv_fp32_accumulate(
                    np.asarray(v_cache_pass[start:end, :, local_head].T, dtype=np.float32),
                    op_soft[:, local_head].astype(np.float32),
                ).astype(np.float16)

        _add(
            f"attn_sv_{tag}",
            "decode_gemv_local",
            (
                v_cache_pass,
                np.asarray(np.expand_dims(op_soft, 2), dtype=np.float16),
            ),
            sv_partial,
        )

        out_weight = _out_weight_pass(
            out_weight_real, head_start, heads_per_pass, heads, head_dim, hidden, hidden_p
        )
        attn_flat = sv_partial.reshape(-1, order="F")[:pass_hidden].astype(np.float16)
        out_partial = gemv_fp32_accumulate(out_weight, attn_flat).astype(np.float16)
        _add(
            f"out_gemv_{tag}",
            "decode_gemv_ring",
            (np.expand_dims(out_weight, 2), _hvec(attn_flat)),
            _hvec(out_partial),
        )
        return out_partial

    out_a = _build_attention_pass(0, "A", q_pass_a)
    out_b = _build_attention_pass(heads_per_pass, "B", q_pass_b)

    # ------------------------------------------------------------------
    # Combine output-projection partials (residual only once) op45-46
    # ------------------------------------------------------------------
    combined_a = _elementwise_add(hidden_source, out_a).astype(np.float32)
    _add(
        "out_add_residual_A",
        "decode_add_fp32N_fp16N_fp32N",
        (_hvec(hidden_source), _hvec(out_a)),
        _hvec(combined_a),
    )
    combined = _elementwise_add(combined_a, out_b).astype(np.float32)
    _add(
        "out_add_residual_B",
        "decode_add_fp32N_fp16N_fp32N",
        (_hvec(combined_a), _hvec(out_b)),
        _hvec(combined),
    )

    # ------------------------------------------------------------------
    # Shared FFN op47-57
    # ------------------------------------------------------------------
    ffn_src = combined.reshape(-1)
    # op47 reads op46's D directly (source operator), so the hardware sees the
    # natural hidden vector, not the interleaved external-A layout used by op0.
    op47_out = _remote_summac_pe_partials(
        ffn_src,
        used_slices,
        hidden_p // used_slices,
    )
    _add(
        "ffn_remote_summac",
        "prefill_remote_summac_fp32MN_fp32MN",
        (
            _hvec(ffn_src),
            _hvec(np.ones(1, dtype=np.float32)),
        ),
        _hvec(op47_out),
    )
    op48_out = np.asarray([remote_sum_fp32(op47_out)], dtype=np.float32)
    _add("ffn_decode_sum", "decode_sum_fp32N_fp32N", (_hvec(op47_out),), _hvec(op48_out))
    op49_out = mac_sfu_rec_sqrt(op48_out, hidden)
    _add("ffn_mac_SFU", "decode_mac_SFU_fp32N_fp32N", (_hvec(op48_out),), _hvec(op49_out))
    op50_out = (ffn_src * np.float32(op49_out.item())).astype(np.float32)
    _add(
        "ffn_mul_scale",
        "decode_mul_fp32N_fp32_fp32N",
        (_hvec(ffn_src), _hvec(op49_out)),
        _hvec(op50_out),
    )
    ffn_scale = _norm_scale("ffn_norm")
    op51_out = (op50_out * ffn_scale).astype(np.float16)
    _add(
        "ffn_mul_cast",
        "decode_mul_fp32N_fp32N_fp16N",
        (_hvec(ffn_scale), _hvec(op50_out)),
        _hvec(op51_out),
    )

    op52_out = gemv_fp32_accumulate(gate_weight, op51_out.reshape(-1)).astype(np.float16)
    _add(
        "ffn_gate_gemv",
        "decode_gemv_ring",
        (np.expand_dims(gate_weight, 2), _hvec(op51_out)),
        _hvec(op52_out),
    )
    op53_out = gemv_fp32_accumulate(up_weight, op51_out.reshape(-1)).astype(np.float16)
    _add(
        "ffn_up_gemv",
        "decode_gemv_ring",
        (np.expand_dims(up_weight, 2), _hvec(op51_out)),
        _hvec(op53_out),
    )
    op54_out = _silu_fp32(op52_out).astype(np.float32)
    _add("ffn_silu", "decode_silu_fp16N_fp32N", (_hvec(op52_out),), _hvec(op54_out))
    op55_out = _elementwise_mul(op54_out, op53_out.astype(np.float32)).astype(np.float16)
    _add(
        "ffn_gate_up_mul",
        "decode_mul_fp32N_fp16N_fp16N",
        (_hvec(op54_out), _hvec(op53_out)),
        _hvec(op55_out),
    )
    op56_out = gemv_fp32_accumulate(down_weight, op55_out.reshape(-1)).astype(np.float16)
    _add(
        "ffn_out_gemv",
        "decode_gemv_ring",
        (np.expand_dims(down_weight, 2), _hvec(op55_out)),
        _hvec(op56_out),
    )
    op57_out = _elementwise_add(op56_out, ffn_src).astype(np.float32)
    _add(
        "ffn_add_residual",
        "decode_add_fp32N_fp16N_fp32N",
        (_hvec(ffn_src), _hvec(op56_out)),
        _hvec(op57_out),
    )

    print(
        f"[decode-layer-full] {len(cases)} layer instances "
        f"({len({c.spec.name for c in cases})} operator types, "
        f"{attention_passes} passes x {heads_per_pass} heads, "
        f"hidden {hidden}->{hidden_p})"
    )
    return cases
