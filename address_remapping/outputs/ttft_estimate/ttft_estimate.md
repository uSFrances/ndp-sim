# TTFT Estimate for `examples/configs/config.json`

This note gives a standalone TTFT estimate for [config.json](/H:/dev/projects/ndp-sim/address_remapping/examples/configs/config.json) without modifying any existing project code.

## Assumptions

- Logical model config:
  - `hidden_size = 1536`
  - `intermediate_size = 8960`
  - `num_attention_heads = 12`
  - `num_key_value_heads = 2`
  - `head_dim = 128`
  - `sequence_length = 8`
  - `slice_per_head = 4`
  - `used_slices = 28`
  - `num_hidden_layers = 28`
- Hardware execution equivalence:
  - `28` slices form `7` clusters
  - `4` slices compute one `q_head`
  - because `12` heads cannot fill `7` clusters evenly, use head padding to `14` heads
  - execution-side hidden width becomes `14 * 128 = 1792`
  - FFN keeps `intermediate_size = 8960`
- Performance priors:
  - the 7 dominant `ring_gemm` ops use `92%` effective GEMM utilization
  - non-`ring_gemm` ops are estimated with effective bandwidth `16 B / slice-cycle`
  - slice clock is `1 GHz`, so `1 slice-cycle = 1 ns`
- Scope:
  - this is a core-layer TTFT estimate
  - it does not include host/runtime overhead, embedding, or lm head unless they are already part of the layer graph

## Execution Interpretation

The layer template in [layer0_padding_0529.json](/H:/dev/projects/ndp-sim/address_remapping/examples/graphs/layer0/layer0_padding_0529.json) is structurally useful, but its built-in params are smaller (`896/1792/7/1/32`). For this estimate, only the operator structure is reused; the actual arithmetic uses the target config plus the execution-side padding assumptions above.

The 7 dominant `ring_gemm` ops are:

- `op5`: Q projection
- `op15`: K projection
- `op20`: V projection
- `op30`: attention output projection
- `op37`: FFN gate projection
- `op38`: FFN up projection
- `op41`: FFN down projection

Important latency interpretation:

- Q path uses `14` execution heads on `7` clusters, so Q-side work takes `2` waves.
- Attention output projection also follows the padded hidden width, so it is also counted as `2` head waves.
- K/V are replicated across clusters, but that replication is parallel across the `7` clusters. It increases total work, but not layer latency by `7x`. For TTFT latency, K and V are each counted once at the per-wave cluster latency.

## Ring-GEMM Estimate

Use:

```text
cycles = (2 * M * K * N_local) / (0.92 * 256)
```

where `256 ops / slice-cycle` is the project peak GEMM throughput.

### 1. Q projection (`op5`)

- per-slice local output width stays `head_dim / slice_per_head = 128 / 4 = 32`
- one wave:
  - `M = 8`
  - `K = 1792`
  - `N_local = 32`
- cycles per wave:

```text
2 * 8 * 1792 * 32 / (0.92 * 256) = 3895.65
```

- two waves for `14` padded heads:

```text
T_q = 7791.30 cycles
```

### 2. K projection (`op15`)

- each cluster computes complete K for `2` KV heads
- per-slice local width:

```text
N_local = num_key_value_heads * head_dim / slice_per_head
        = 2 * 128 / 4
        = 64
```

- cycles:

```text
T_k = 2 * 8 * 1792 * 64 / (0.92 * 256) = 7791.30 cycles
```

### 3. V projection (`op20`)

- same shape as K:

```text
T_v = 7791.30 cycles
```

### 4. Attention output projection (`op30`)

- same per-wave local width as Q output projection
- one wave:

```text
2 * 8 * 1792 * 32 / (0.92 * 256) = 3895.65
```

- two waves:

```text
T_attn_out = 7791.30 cycles
```

### 5. FFN gate projection (`op37`)

- keep `intermediate_size = 8960`
- local output width:

```text
N_local = 8960 / 28 = 320
```

- cycles:

```text
T_ffn_gate = 2 * 8 * 1792 * 320 / (0.92 * 256) = 38956.52 cycles
```

### 6. FFN up projection (`op38`)

- same as gate:

```text
T_ffn_up = 38956.52 cycles
```

### 7. FFN down projection (`op41`)

- local output width returns to padded hidden width:

```text
N_local = 1792 / 28 = 64
```

- cycles:

```text
T_ffn_down = 2 * 8 * 8960 * 64 / (0.92 * 256) = 38956.52 cycles
```

### Ring-GEMM subtotal

```text
T_ring_gemm
= T_q + T_k + T_v + T_attn_out + T_ffn_gate + T_ffn_up + T_ffn_down
= 148035.98 cycles
```

Rounded:

```text
T_ring_gemm ≈ 148.0k cycles per layer
```

## Non-Ring-GEMM Estimate

For the remaining ops, use:

```text
cycles = total_bytes / 16
```

This bucket includes:

- RMSNorm chain
- bias/add/mul around Q/K/V paths
- local attention GEMMs and softmax chain
- FFN SiLU and elementwise multiply
- residual adds

Given `sequence_length = 8`, these non-`ring_gemm` tensors are small compared with the 7 dominant GEMMs. A practical coarse estimate is:

```text
T_non_ring ≈ 8k to 12k cycles per layer
```

Use `10k cycles` as the nominal point estimate.

## Per-Layer TTFT Estimate

### Nominal

```text
T_layer_nominal = 148035.98 + 10000
                = 158035.98 cycles
                ≈ 158.0k cycles
```

### Range

- optimistic:

```text
148035.98 + 8000 = 156035.98 cycles
```

- conservative:

```text
148035.98 + 12000 = 160035.98 cycles
```

So:

```text
T_layer ≈ 156k to 160k cycles
```

## Whole-Network TTFT Estimate

For `28` layers:

### Nominal

```text
T_total_nominal = 28 * 158035.98
                = 4425007.44 cycles
```

At `1 GHz`:

```text
TTFT_nominal = 4425007.44 ns
             = 4.425 ms
```

### Range

- optimistic:

```text
28 * 156035.98 = 4369007.44 cycles = 4.369 ms
```

- conservative:

```text
28 * 160035.98 = 4481007.44 cycles = 4.481 ms
```

## Final Number

Recommended reported value:

```text
TTFT ≈ 4.43 ms
```

Recommended report range:

```text
TTFT ≈ 4.37 ms to 4.48 ms
```

## Notes for Discussion

- This estimate is intentionally dominated by the 7 `ring_gemm` ops, matching the stated profiling intuition.
- The biggest sensitivity is not the non-GEMM bandwidth term; it is whether the padded `14-head` execution really requires `2` Q-side waves and whether K/V replication is fully parallel across clusters.
- If later measurements show that K/V replication is partially serialized, the true TTFT will be higher than this note.
- If later measurements show that some FFN GEMMs do not sustain the same `92%` as `ffn_gate`, the first place to revise is the three FFN GEMM terms.
