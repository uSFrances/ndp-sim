# Layer0 Roofline vs Measured Summary Tables

## Model Parameters

- Model config: `examples\configs\config.json`
- Model name: `deepseek1.5b`
- TTFT layers: `28`
- TTFT frequency: `800 MHz`
- Layer graph: `layer0`; template TTFT multiplies layer0 cycles, model-scaled TTFT recomputes op sizes from the model config.
- `hidden_size` = `1536`
- `execution_hidden_size` = `1792`
- `intermediate_size` = `8960`
- `num_attention_heads` = `12`
- `padded_attention_heads` = `14`
- `num_key_value_heads` = `2`
- `head_dim` = `128`
- `num_hidden_layers` = `28`
- `requested_sequence_length` = `8`
- `sequence_length` = `32`
- `slice_per_head` = `4`
- `used_slices` = `28`
- `kv_padding` = `512`
- `kv_padding_a` = `256`
- `kv_padding_b` = `1024`
- `clusters` = `7`
- `attention_waves` = `2`
- `kv_heads_per_cluster` = `1`
- Layer graph params: `hidden_size=896`, `intermediate_size=1792`, `sequence_length=32`, `num_hidden_layers=1`

## Summary Metrics

| Metric | Scope | AXI pull roofline | Projected measured |
| --- | --- | --- | --- |
| GEMM compute utilization | GEMM-only cycles | 100.00% | 89.80% |
| GEMM compute utilization | Full-layer cycles | 91.76% | 28.54% |
| non-GEMM bandwidth utilization | non-GEMM-only cycles | 100.00% | 3.76% |
| non-GEMM bandwidth utilization | Full-layer cycles | 8.24% | 2.56% |
| Whole-layer bandwidth utilization | Full-layer cycles | 8.24% | 2.56% |
| GEMM time share | Cycles | 91.76% | 31.78% |
| non-GEMM time share | Cycles | 8.24% | 68.22% |

## Model-Scaled Summary

| Scenario | Per-layer cycles | Total cycles | TTFT |
| --- | --- | --- | --- |
| Projected measured | 1,811,956 | 50,734,762 | 63.418 ms |
| Projected measured with centralized global remote-sum | 3,860,035 | 108,080,987 | 135.101 ms |
| Projected measured with Ring2Ring remote-sum | 1,811,956 | 50,734,762 | 63.418 ms |
| AXI pull roofline | 563,584 | 15,780,352 | 19.725 ms |
| Centralized global roofline | 570,068 | 15,961,904 | 19.952 ms |
| Ring2Ring n2n roofline | 563,584 | 15,780,352 | 19.725 ms |

## Model-Scaled Operator Projection

This table recomputes each operator's work or bytes from the target model config, then projects cycles using the calibration layer's measured GEMM throughput or measured non-GEMM effective bandwidth.

- Target model: `deepseek1.5b`
- Target layers: `28`
- Target sequence length: `32`
- Target execution hidden size: `1792`


### Model-Scaled GEMM Operators

| Op ID | Operator | Type | Model work ops | Projected measured cycles | Layer share | Projected compute util | AXI roofline cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| op37 | ffn_gate | prefill_gemm_ring_4slice | 36,700,160 | 158,780 | 8.76% | 90.29% | 143,360 |
| op38 | ffn_up | prefill_gemm_ring_4slice | 36,700,160 | 157,920 | 8.72% | 90.78% | 143,360 |
| op41 | ffn_down | prefill_gemm_ring_4slice | 36,700,160 | 157,560 | 8.70% | 90.99% | 143,360 |
| op5 | q_gen | prefill_gemm_ring_4slice | 7,340,032 | 32,044 | 1.77% | 89.48% | 28,672 |
| op30 | atten_out | prefill_gemm_ring_4slice | 7,340,032 | 32,016 | 1.77% | 89.56% | 28,672 |
| op20 | v_gen | prefill_gemm_ring_4slice | 3,670,016 | 17,915 | 0.99% | 80.02% | 14,336 |
| op15 | k_gen | prefill_gemm_ring_4slice | 3,670,016 | 15,790 | 0.87% | 90.79% | 14,336 |
| op29 | local_gemm_sv | prefill_gemm_local | 131,072 | 1,990 | 0.11% | 25.73% | 512 |
| op22 | local_gemm_qkt | prefill_gemm_local_qkt | 131,072 | 1,858 | 0.10% | 27.56% | 512 |

### Model-Scaled non-GEMM Operators

| Op ID | Operator | Type | Input shape | Output shape | Remote-sum geometry | Model bytes | Projected measured cycles | Layer share | Projected BW util | AXI roofline cycles | Centralized roofline cycles | Ring2Ring roofline cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op23 | qkt_remote_sum | prefill_remote_sum_4slice_fp32MN_fp32MN | A=[1 x 4 x 1024] | out=[1 x 1 x 1024] | fan-in=4 partial slices -> 1 result (1024 elements) | 40,960 | 414,022 | 22.85% | 0.31% | 1,280 | 6,144 | 1,280 |
| op12 | k_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | A=[1792 x 1 x 32] | out=[1 x 1 x 32] | - | 229,504 | 172,128 | 9.50% | 4.17% | 7,172 | 7,172 | 7,172 |
| op2 | q_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | A=[1792 x 1 x 32] | out=[1 x 1 x 32] | - | 229,504 | 164,956 | 9.10% | 4.35% | 7,172 | 7,172 | 7,172 |
| op34 | ffn_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | A=[1792 x 1 x 32] | out=[1 x 1 x 32] | - | 229,504 | 152,405 | 8.41% | 4.71% | 7,172 | 7,172 | 7,172 |
| op8 | q_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 49,152 | 69,232 | 3.82% | 2.22% | 1,536 | 1,536 | 1,536 |
| op18 | k_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 24,576 | 34,752 | 1.92% | 2.21% | 768 | 768 | 768 |
| op33 | ffn_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | A=[1 x 28 x 32] | out=[1 x 1 x 32] | fan-in=28 partial slices -> 1 result (32 elements) | 3,712 | 34,432 | 1.90% | 0.34% | 116 | 880 | 116 |
| op1 | q_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | A=[1 x 28 x 32] | out=[1 x 1 x 32] | fan-in=28 partial slices -> 1 result (32 elements) | 3,712 | 34,279 | 1.89% | 0.34% | 116 | 880 | 116 |
| op14 | k_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | B=[1 x 32 x 256]; A=[1 x 1 x 256] | out=[1 x 32 x 256] | - | 50,176 | 26,218 | 1.45% | 5.98% | 1,568 | 1,568 | 1,568 |
| op6 | q_bias_add | prefill_add_fp16MN_fp32N_fp32MN | B=[1 x 32 x 64]; A=[1 x 1 x 64] | out=[1 x 32 x 64] | - | 33,024 | 20,393 | 1.13% | 5.06% | 1,032 | 1,032 | 1,032 |
| op40 | ffn_gate_mul | prefill_mul_fp32MN_fp16MN_fp16MN | A=[1 x 32 x 320]; B=[1 x 32 x 320] | out=[1 x 32 x 320] | - | 81,920 | 17,795 | 0.98% | 14.39% | 2,560 | 2,560 | 2,560 |
| op16 | k_bias_add | prefill_add_fp16MN_fp32N_fp32MN | B=[1 x 32 x 64]; A=[1 x 1 x 64] | out=[1 x 32 x 64] | - | 16,512 | 9,507 | 0.52% | 5.43% | 516 | 516 | 516 |
| op9 | q_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 40,960 | 9,216 | 0.51% | 13.89% | 1,280 | 1,280 | 1,280 |
| op7 | q_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 49,152 | 9,132 | 0.50% | 16.82% | 1,536 | 1,536 | 1,536 |
| op36 | ffn_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | B=[1 x 32 x 64]; A=[1 x 1 x 64] | out=[1 x 32 x 64] | - | 12,544 | 8,056 | 0.44% | 4.87% | 392 | 392 | 392 |
| op39 | ffn_silu | prefill_silu_fp16MN_fp32MN | A=[1 x 32 x 320] | out=[1 x 32 x 320] | - | 61,440 | 5,365 | 0.30% | 35.79% | 1,920 | 1,920 | 1,920 |
| op4 | q_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | B=[1 x 32 x 64]; A=[1 x 1 x 64] | out=[1 x 32 x 64] | - | 12,544 | 4,896 | 0.27% | 8.01% | 392 | 392 | 392 |
| op11 | k_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | A=[1 x 4 x 32] | out=[1 x 1 x 32] | fan-in=4 partial slices -> 1 result (32 elements) | 640 | 4,836 | 0.27% | 0.41% | 20 | 112 | 20 |
| op17 | k_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 24,576 | 4,774 | 0.26% | 16.09% | 768 | 768 | 768 |
| op24 | qkt_score_add | prefill_add_fp32MN_fp32MN_fp32MN | A=[1 x 32 x 32]; B=[1 x 32 x 32] | out=[1 x 32 x 32] | - | 24,576 | 4,688 | 0.26% | 16.38% | 768 | 768 | 768 |
| op19 | k_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 20,480 | 4,448 | 0.25% | 14.39% | 640 | 640 | 640 |
| op13 | k_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | A=[1 x 32 x 256]; B=[1 x 1 x 32] | out=[1 x 32 x 256] | - | 65,664 | 4,396 | 0.24% | 46.68% | 2,052 | 2,052 | 2,052 |
| op10 | k_norm_summac | prefill_summac_fp32MN_fp32MN | A=[1 x 32 x 256] | out=[1 x 1 x 32] | - | 32,896 | 4,365 | 0.24% | 23.55% | 1,028 | 1,028 | 1,028 |
| op31 | atten_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 20,480 | 3,394 | 0.19% | 18.86% | 640 | 640 | 640 |
| op28 | softmax_scale | prefill_mul_fp32MN_fp32M_fp16MN | A=[1 x 32 x 32]; B=[1 x 1 x 32] | out=[1 x 32 x 32] | - | 12,544 | 2,898 | 0.16% | 13.53% | 392 | 392 | 392 |
| op21 | v_bias_add | prefill_add_V_fp16MN_fp32N_fp16MN | B=[1 x 32 x 64]; A=[1 x 1 x 64] | out=[1 x 32 x 64] | - | 12,416 | 2,548 | 0.14% | 15.22% | 388 | 388 | 388 |
| op35 | ffn_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | A=[1 x 32 x 64]; B=[1 x 1 x 32] | out=[1 x 32 x 64] | - | 16,512 | 2,451 | 0.14% | 21.05% | 516 | 516 | 516 |
| op26 | softmax_sub | prefill_sub_SFU_fp32MN_fp32M_fp32MN | A=[1 x 32 x 32]; B=[1 x 1 x 32] | out=[1 x 32 x 32] | - | 16,640 | 2,162 | 0.12% | 24.05% | 520 | 520 | 520 |
| op42 | ffn_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | A=[1 x 32 x 64]; B=[1 x 32 x 64] | out=[1 x 32 x 64] | - | 20,480 | 1,732 | 0.10% | 36.95% | 640 | 640 | 640 |
| op25 | softmax_max | prefill_max_fp32MN_fp32MN | A=[1 x 32 x 32] | out=[1 x 1 x 32] | - | 8,448 | 1,468 | 0.08% | 17.98% | 264 | 264 | 264 |
| op27 | softmax_sum_rec | prefill_sum_rec_fp32MN_fp32MN | A=[1 x 32 x 32] | out=[1 x 1 x 32] | - | 8,448 | 1,392 | 0.08% | 18.97% | 264 | 264 | 264 |
| op32 | ffn_norm_summac | prefill_summac_fp32MN_fp32MN | A=[1 x 32 x 64] | out=[1 x 1 x 32] | - | 8,320 | 1,300 | 0.07% | 20.00% | 260 | 260 | 260 |
| op3 | q_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | A=[1 x 32 x 64]; B=[1 x 1 x 32] | out=[1 x 32 x 64] | - | 16,512 | 1,288 | 0.07% | 40.06% | 516 | 516 | 516 |
| op0 | q_norm_summac | prefill_summac_fp32MN_fp32MN | A=[1 x 32 x 64] | out=[1 x 1 x 32] | - | 8,320 | 1,158 | 0.06% | 22.45% | 260 | 260 | 260 |
