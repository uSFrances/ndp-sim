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

| Metric | Scope | Roofline | Measured |
| --- | --- | --- | --- |
| GEMM compute utilization | GEMM-only cycles | 100.00% | 87.73% |
| GEMM bandwidth utilization | GEMM-only cycles | 28.10% | 24.65% |
| GEMM compute utilization | Full-layer cycles | 43.76% | 17.21% |
| GEMM bandwidth utilization | Full-layer cycles | 12.30% | 4.84% |
| non-GEMM bandwidth utilization | non-GEMM-only cycles | 37.90% | 10.43% |
| non-GEMM bandwidth utilization | Full-layer cycles | 21.32% | 8.38% |
| Whole-layer bandwidth utilization | Full-layer cycles | 33.61% | 13.22% |
| Template TTFT | 28 layers @ 800 MHz | 5.938 ms | 15.096 ms |
| Model-scaled TTFT | 28 layers @ 800 MHz | 19.725 ms | 46.963 ms |
| Model-scaled per-layer cycles | Target model operator sizes | 563,584 | 1,341,799 |
| GEMM time share | Cycles | 43.76% | 19.62% |
| non-GEMM time share | Cycles | 56.24% | 80.38% |

## GEMM Operators

| Op ID | Kernel | Type | Work ops | Roofline cycles | Measured cycles | Roofline layer time share | Layer time share | Roofline compute util | Measured compute util | Roofline bandwidth util | Measured bandwidth util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op37 | ffn_gate | ring_gemm_fp16_fp16_fp16 | 3,670,016 | 14,336 | 15,878 | 8.45% | 3.68% | 100.00% | 90.29% | 26.34% | 23.78% |
| op38 | ffn_up | ring_gemm_fp16_fp16_fp16 | 3,670,016 | 14,336 | 15,792 | 8.45% | 3.66% | 100.00% | 90.78% | 26.34% | 23.91% |
| op41 | ffn_down | ring_gemm_fp16_fp16_fp16 | 3,670,016 | 14,336 | 15,756 | 8.45% | 3.65% | 100.00% | 90.99% | 26.34% | 23.97% |
| op20 | v_gen | ring_gemm_fp16_fp16_fp16 | 2,097,152 | 8,192 | 10,237 | 4.83% | 2.37% | 100.00% | 80.02% | 32.03% | 25.63% |
| op15 | k_gen | ring_gemm_fp16_fp16_fp16 | 2,097,152 | 8,192 | 9,023 | 4.83% | 2.09% | 100.00% | 90.79% | 32.03% | 29.08% |
| op5 | q_gen | ring_gemm_fp16_fp16_fp16 | 1,835,008 | 7,168 | 8,011 | 4.23% | 1.86% | 100.00% | 89.48% | 26.79% | 23.97% |
| op30 | atten_out | ring_gemm_fp16_fp16_fp16 | 1,835,008 | 7,168 | 8,004 | 4.23% | 1.86% | 100.00% | 89.56% | 26.79% | 23.99% |
| op29 | local_gemm_sv | gemm_local_fp16_fp16_fp16 | 65,536 | 256 | 995 | 0.15% | 0.23% | 100.00% | 25.73% | 75.00% | 19.30% |
| op22 | local_gemm_qkt | gemm_local_qkt_fp16_fp16_fp32 | 65,536 | 256 | 929 | 0.15% | 0.22% | 100.00% | 27.56% | 100.00% | 27.56% |

## non-GEMM Operators

| Op ID | Operator | Type | Input shape | Output shape | Remote-sum geometry | Total bytes | Roofline cycles | Measured cycles | Roofline layer time share | Layer time share | Roofline bandwidth util | Measured bandwidth util | Roofline global BW util | Measured global BW util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op23 | qkt_remote_sum | prefill_remote_sum_fp32MN_fp32MN_2d | inA=[4 x 1024] | out=[1 x 1024] | fan_in=4 | 573,440 | 57,344 | 165,609 | 33.80% | 38.40% | 100.00% | 34.63% | 100.00% | 34.63% |
| op33 | ffn_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | inA=[28 x 32] | out=[1 x 32] | fan_in=28 | 103,936 | 12,544 | 33,245 | 7.39% | 7.71% | 100.00% | 37.73% | 100.00% | 37.73% |
| op1 | q_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | inA=[28 x 32] | out=[1 x 32] | fan_in=28 | 103,936 | 12,544 | 33,097 | 7.39% | 7.67% | 100.00% | 37.90% | 100.00% | 37.90% |
| op14 | k_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | inA=[1 x 256]; inB=[32 x 256] | out=[32 x 256] | - | 50,176 | 1,568 | 26,218 | 0.92% | 6.08% | 100.00% | 5.98% | N/A | N/A |
| op18 | k_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 12,288 | 384 | 17,376 | 0.23% | 4.03% | 100.00% | 2.21% | N/A | N/A |
| op8 | q_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 12,288 | 384 | 17,308 | 0.23% | 4.01% | 100.00% | 2.22% | N/A | N/A |
| op13 | k_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | inA=[32 x 256]; inB=[1 x 32] | out=[32 x 256] | - | 65,664 | 2,052 | 4,396 | 1.21% | 1.02% | 100.00% | 46.68% | N/A | N/A |
| op10 | k_norm_summac | prefill_summac | inA=[32 x 256] | out=[1 x 32] | - | 32,896 | 1,028 | 4,365 | 0.61% | 1.01% | 100.00% | 23.55% | N/A | N/A |
| op36 | ffn_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | inA=[1 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 6,272 | 196 | 4,028 | 0.12% | 0.93% | 100.00% | 4.87% | N/A | N/A |
| op6 | q_bias_add | prefill_add_fp16MN_fp32N_fp32MN | inA=[1 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 6,272 | 196 | 3,873 | 0.12% | 0.90% | 100.00% | 5.06% | N/A | N/A |
| op11 | k_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | inA=[4 x 32] | out=[1 x 32] | fan_in=4 | 17,920 | 1,792 | 3,869 | 1.06% | 0.90% | 100.00% | 46.32% | 100.00% | 46.32% |
| op16 | k_bias_add | prefill_add_fp16MN_fp32N_fp32MN | inA=[1 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 6,272 | 196 | 3,611 | 0.12% | 0.84% | 100.00% | 5.43% | N/A | N/A |
| op40 | ffn_gate_mul | prefill_mul_fp32MN_fp16MN_fp16MN | inA=[32 x 64]; inB=[32 x 64] | out=[32 x 64] | - | 16,384 | 512 | 3,559 | 0.30% | 0.83% | 100.00% | 14.39% | N/A | N/A |
| op4 | q_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | inA=[1 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 6,272 | 196 | 2,448 | 0.12% | 0.57% | 100.00% | 8.01% | N/A | N/A |
| op17 | k_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 12,288 | 384 | 2,387 | 0.23% | 0.55% | 100.00% | 16.09% | N/A | N/A |
| op24 | qkt_score_add | prefill_add_fp32MN_fp32MN_fp32MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 12,288 | 384 | 2,344 | 0.23% | 0.54% | 100.00% | 16.38% | N/A | N/A |
| op9 | q_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 10,240 | 320 | 2,304 | 0.19% | 0.53% | 100.00% | 13.89% | N/A | N/A |
| op7 | q_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 12,288 | 384 | 2,283 | 0.23% | 0.53% | 100.00% | 16.82% | N/A | N/A |
| op19 | k_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 10,240 | 320 | 2,224 | 0.19% | 0.52% | 100.00% | 14.39% | N/A | N/A |
| op31 | atten_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 10,240 | 320 | 1,697 | 0.19% | 0.39% | 100.00% | 18.86% | N/A | N/A |
| op28 | softmax_scale | prefill_mul_fp32MN_fp32M_fp16MN | inA=[32 x 32]; inB=[1 x 32] | out=[32 x 32] | - | 6,272 | 196 | 1,449 | 0.12% | 0.34% | 100.00% | 13.53% | N/A | N/A |
| op35 | ffn_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | inA=[32 x 32]; inB=[1 x 32] | out=[32 x 32] | - | 8,320 | 260 | 1,235 | 0.15% | 0.29% | 100.00% | 21.05% | N/A | N/A |
| op26 | softmax_sub | prefill_sub_SFU_fp32MN_fp32MN_fp32MN | inA=[32 x 32]; inB=[1 x 32] | out=[32 x 32] | - | 8,320 | 260 | 1,081 | 0.15% | 0.25% | 100.00% | 24.05% | N/A | N/A |
| op39 | ffn_silu | prefill_silu_fp16MN_fp32MN | inA=[32 x 64] | out=[32 x 64] | - | 12,288 | 384 | 1,073 | 0.23% | 0.25% | 100.00% | 35.79% | N/A | N/A |
| op21 | v_bias_add | prefill_add_V_fp16MN_fp32N_fp16MN | inA=[1 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 4,224 | 132 | 867 | 0.08% | 0.20% | 100.00% | 15.22% | N/A | N/A |
| op42 | ffn_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | inA=[32 x 32]; inB=[32 x 32] | out=[32 x 32] | - | 10,240 | 320 | 866 | 0.19% | 0.20% | 100.00% | 36.95% | N/A | N/A |
| op25 | softmax_max | prefill_max | inA=[32 x 32] | out=[1 x 32] | - | 4,224 | 132 | 734 | 0.08% | 0.17% | 100.00% | 17.98% | N/A | N/A |
| op27 | softmax_sum_rec | prefill_sum_rec_fp32MN_fp32MN | inA=[32 x 32] | out=[1 x 32] | - | 4,224 | 132 | 696 | 0.08% | 0.16% | 100.00% | 18.97% | N/A | N/A |
| op32 | ffn_norm_summac | prefill_summac | inA=[32 x 32] | out=[1 x 32] | - | 4,224 | 132 | 660 | 0.08% | 0.15% | 100.00% | 20.00% | N/A | N/A |
| op3 | q_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | inA=[32 x 32]; inB=[1 x 32] | out=[32 x 32] | - | 8,320 | 260 | 649 | 0.15% | 0.15% | 100.00% | 40.06% | N/A | N/A |
| op0 | q_norm_summac | prefill_summac | inA=[32 x 32] | out=[1 x 32] | - | 4,224 | 132 | 588 | 0.08% | 0.14% | 100.00% | 22.45% | N/A | N/A |
| op12 | k_norm_mac_sfu | prefill_mac_SFU | inA=[1 x 32] | out=[1 x 32] | - | 256 | 8 | 192 | 0.00% | 0.04% | 100.00% | 4.17% | N/A | N/A |
| op2 | q_norm_mac_sfu | prefill_mac_SFU | inA=[1 x 32] | out=[1 x 32] | - | 256 | 8 | 184 | 0.00% | 0.04% | 100.00% | 4.35% | N/A | N/A |
| op34 | ffn_norm_mac_sfu | prefill_mac_SFU | inA=[1 x 32] | out=[1 x 32] | - | 256 | 8 | 170 | 0.00% | 0.04% | 100.00% | 4.71% | N/A | N/A |

## Model-Scaled Operator Projection

This table recomputes each operator's work or bytes from the target model config, then projects cycles using the calibration layer's measured GEMM throughput or measured non-GEMM effective bandwidth.

- Target model: `deepseek1.5b`
- Target layers: `28`
- Target sequence length: `32`
- Target execution hidden size: `1792`


### Model-Scaled GEMM Operators

| Op ID | Operator | Type | Model work ops | Projected measured cycles | Layer share | Projected compute util | AXI roofline cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| op37 | ffn_gate | prefill_gemm_ring_4slice | 36,700,160 | 158,780 | 11.83% | 90.29% | 143,360 |
| op38 | ffn_up | prefill_gemm_ring_4slice | 36,700,160 | 157,920 | 11.77% | 90.78% | 143,360 |
| op41 | ffn_down | prefill_gemm_ring_4slice | 36,700,160 | 157,560 | 11.74% | 90.99% | 143,360 |
| op5 | q_gen | prefill_gemm_ring_4slice | 7,340,032 | 32,044 | 2.39% | 89.48% | 28,672 |
| op30 | atten_out | prefill_gemm_ring_4slice | 7,340,032 | 32,016 | 2.39% | 89.56% | 28,672 |
| op20 | v_gen | prefill_gemm_ring_4slice | 3,670,016 | 17,915 | 1.34% | 80.02% | 14,336 |
| op15 | k_gen | prefill_gemm_ring_4slice | 3,670,016 | 15,790 | 1.18% | 90.79% | 14,336 |
| op29 | local_gemm_sv | prefill_gemm_local | 131,072 | 1,990 | 0.15% | 25.73% | 512 |
| op22 | local_gemm_qkt | prefill_gemm_local_qkt | 131,072 | 1,858 | 0.14% | 27.56% | 512 |

### Model-Scaled non-GEMM Operators

| Op ID | Operator | Type | Input shape | Output shape | Remote-sum geometry | Model bytes | Projected measured cycles | Layer share | Projected BW util | AXI roofline cycles | Centralized roofline cycles | Ring2Ring roofline cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op12 | k_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | A=[1 x 32] | out=[1 x 32] | - | 229,504 | 172,128 | 12.83% | 4.17% | 7,172 | 7,172 | 7,172 |
| op2 | q_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | A=[1 x 32] | out=[1 x 32] | - | 229,504 | 164,956 | 12.29% | 4.35% | 7,172 | 7,172 | 7,172 |
| op34 | ffn_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | A=[1 x 32] | out=[1 x 32] | - | 229,504 | 152,405 | 11.36% | 4.71% | 7,172 | 7,172 | 7,172 |
| op8 | q_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 49,152 | 69,232 | 5.16% | 2.22% | 1,536 | 1,536 | 1,536 |
| op18 | k_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 24,576 | 34,752 | 2.59% | 2.21% | 768 | 768 | 768 |
| op14 | k_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | B=[32 x 256]; A=[1 x 256] | out=[32 x 256] | - | 50,176 | 26,218 | 1.95% | 5.98% | 1,568 | 1,568 | 1,568 |
| op6 | q_bias_add | prefill_add_fp16MN_fp32N_fp32MN | B=[32 x 64]; A=[1 x 64] | out=[32 x 64] | - | 33,024 | 20,393 | 1.52% | 5.06% | 1,032 | 1,032 | 1,032 |
| op40 | ffn_gate_mul | prefill_mul_fp32MN_fp16MN_fp16MN | A=[32 x 320]; B=[32 x 320] | out=[32 x 320] | - | 81,920 | 17,795 | 1.33% | 14.39% | 2,560 | 2,560 | 2,560 |
| op23 | qkt_remote_sum | prefill_remote_sum_4slice_fp32MN_fp32MN | A=[4 x 1024] | out=[1 x 1024] | fan_in=4 | 40,960 | 14,787 | 1.10% | 8.66% | 1,280 | 6,144 | 1,280 |
| op16 | k_bias_add | prefill_add_fp16MN_fp32N_fp32MN | B=[32 x 64]; A=[1 x 64] | out=[32 x 64] | - | 16,512 | 9,507 | 0.71% | 5.43% | 516 | 516 | 516 |
| op9 | q_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 40,960 | 9,216 | 0.69% | 13.89% | 1,280 | 1,280 | 1,280 |
| op7 | q_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 49,152 | 9,132 | 0.68% | 16.82% | 1,536 | 1,536 | 1,536 |
| op36 | ffn_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | B=[32 x 64]; A=[1 x 64] | out=[32 x 64] | - | 12,544 | 8,056 | 0.60% | 4.87% | 392 | 392 | 392 |
| op39 | ffn_silu | prefill_silu_fp16MN_fp32MN | A=[32 x 320] | out=[32 x 320] | - | 61,440 | 5,365 | 0.40% | 35.79% | 1,920 | 1,920 | 1,920 |
| op4 | q_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | B=[32 x 64]; A=[1 x 64] | out=[32 x 64] | - | 12,544 | 4,896 | 0.36% | 8.01% | 392 | 392 | 392 |
| op17 | k_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 24,576 | 4,774 | 0.36% | 16.09% | 768 | 768 | 768 |
| op24 | qkt_score_add | prefill_add_fp32MN_fp32MN_fp32MN | A=[32 x 32]; B=[32 x 32] | out=[32 x 32] | - | 24,576 | 4,688 | 0.35% | 16.38% | 768 | 768 | 768 |
| op19 | k_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 20,480 | 4,448 | 0.33% | 14.39% | 640 | 640 | 640 |
| op13 | k_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | A=[32 x 256]; B=[1 x 32] | out=[32 x 256] | - | 65,664 | 4,396 | 0.33% | 46.68% | 2,052 | 2,052 | 2,052 |
| op10 | k_norm_summac | prefill_summac_fp32MN_fp32MN | A=[32 x 256] | out=[1 x 32] | - | 32,896 | 4,365 | 0.33% | 23.55% | 1,028 | 1,028 | 1,028 |
| op31 | atten_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 20,480 | 3,394 | 0.25% | 18.86% | 640 | 640 | 640 |
| op28 | softmax_scale | prefill_mul_fp32MN_fp32M_fp16MN | A=[32 x 32]; B=[1 x 32] | out=[32 x 32] | - | 12,544 | 2,898 | 0.22% | 13.53% | 392 | 392 | 392 |
| op21 | v_bias_add | prefill_add_V_fp16MN_fp32N_fp16MN | B=[32 x 64]; A=[1 x 64] | out=[32 x 64] | - | 12,416 | 2,548 | 0.19% | 15.22% | 388 | 388 | 388 |
| op35 | ffn_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | A=[32 x 64]; B=[1 x 32] | out=[32 x 64] | - | 16,512 | 2,451 | 0.18% | 21.05% | 516 | 516 | 516 |
| op26 | softmax_sub | prefill_sub_SFU_fp32MN_fp32M_fp32MN | A=[32 x 32]; B=[1 x 32] | out=[32 x 32] | - | 16,640 | 2,162 | 0.16% | 24.05% | 520 | 520 | 520 |
| op42 | ffn_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | A=[32 x 64]; B=[32 x 64] | out=[32 x 64] | - | 20,480 | 1,732 | 0.13% | 36.95% | 640 | 640 | 640 |
| op25 | softmax_max | prefill_max_fp32MN_fp32MN | A=[32 x 32] | out=[1 x 32] | - | 8,448 | 1,468 | 0.11% | 17.98% | 264 | 264 | 264 |
| op27 | softmax_sum_rec | prefill_sum_rec_fp32MN_fp32MN | A=[32 x 32] | out=[1 x 32] | - | 8,448 | 1,392 | 0.10% | 18.97% | 264 | 264 | 264 |
| op32 | ffn_norm_summac | prefill_summac_fp32MN_fp32MN | A=[32 x 64] | out=[1 x 32] | - | 8,320 | 1,300 | 0.10% | 20.00% | 260 | 260 | 260 |
| op3 | q_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | A=[32 x 64]; B=[1 x 32] | out=[32 x 64] | - | 16,512 | 1,288 | 0.10% | 40.06% | 516 | 516 | 516 |
| op33 | ffn_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | A=[28 x 32] | out=[1 x 32] | fan_in=28 | 3,712 | 1,230 | 0.09% | 9.43% | 116 | 880 | 116 |
| op1 | q_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | A=[28 x 32] | out=[1 x 32] | fan_in=28 | 3,712 | 1,224 | 0.09% | 9.48% | 116 | 880 | 116 |
| op0 | q_norm_summac | prefill_summac_fp32MN_fp32MN | A=[32 x 64] | out=[1 x 32] | - | 8,320 | 1,158 | 0.09% | 22.45% | 260 | 260 | 260 |
| op11 | k_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | A=[4 x 32] | out=[1 x 32] | fan_in=4 | 640 | 173 | 0.01% | 11.58% | 20 | 112 | 20 |

## Remote-Sum Transport Comparison

Ring2Ring remote-sum model: each slice first reads its local partial result, then receives the other slice partial results through the slice-to-slice n2n datapath. The ring datapath bandwidth is modeled as 256 bit/cycle = 32 B/cycle per slice.

AXI-pull model: all active slices read every `fan_in` partial in their group through AXI, including their local partial. The global read traffic is `active_slices * fan_in * output_elements * dtype_bytes` and is divided by the total 8 B/cycle global AXI bandwidth.

For each remote-sum op: `local_read_bytes = output_elements * dtype_bytes`, `ring_transfer_bytes = (fan_in - 1) * output_elements * dtype_bytes`, `local_write_bytes = output_elements * output_dtype_bytes`, and `ring2ring_roofline_cycles = max(local_read_bytes / local_bw + ring_transfer_bytes / 32, local_write_bytes / local_bw, reduction_ops / general_peak)`.

Centralized-global remote-sum projection: each group has one central slice read all `fan_in` partials, perform the reduction, then return the result to the other `(fan_in - 1)` slices. Read and return traffic are both accumulated across `active_slices / fan_in` groups and use the total 8 B/cycle global AXI bandwidth. This projection keeps all non-remote-sum measured cycles unchanged and scales each remote-sum measured cycle count by `centralized_global_roofline_cycles / axi_pull_roofline_cycles`.

### Layer Roofline Summary

| Metric | Scope | Measured | Projected measured with centralized global remote-sum | Projected measured with Ring2Ring remote-sum | AXI pull roofline | Centralized global roofline | Ring2Ring n2n roofline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM compute utilization | GEMM-only cycles | 87.73% | 87.73% | 87.73% | 100.00% | 100.00% | 100.00% |
| GEMM bandwidth utilization | GEMM-only cycles | 24.65% | 24.65% | 24.65% | 28.10% | 28.10% | 28.10% |
| GEMM compute utilization | Full-layer cycles | 17.21% | 27.07% | 37.57% | 43.76% | 65.66% | 86.15% |
| GEMM bandwidth utilization | Full-layer cycles | 4.84% | 7.61% | 10.56% | 12.30% | 18.45% | 24.21% |
| non-GEMM bandwidth utilization | non-GEMM-only cycles | 10.43% | 19.07% | 32.01% | 37.90% | 93.16% | 101.17% |
| non-GEMM bandwidth utilization | Full-layer cycles | 8.38% | 13.18% | 18.30% | 21.32% | 31.99% | 14.02% |
| Whole-layer bandwidth utilization | Full-layer cycles | 13.22% | 20.79% | 28.86% | 33.61% | 50.44% | 38.23% |
| GEMM time share | Cycles | 19.62% | 30.85% | 42.83% | 43.76% | 65.66% | 86.15% |
| non-GEMM time share | Cycles | 80.38% | 69.15% | 57.17% | 56.24% | 34.34% | 13.85% |
| Total cycles | Cycles | 431,310 | 274,291 | 197,596 | 169,652 | 113,060 | 86,180 |
| Template TTFT | 28 layers @ 800 MHz | 15.096 ms | 9.600 ms | 6.916 ms | 5.938 ms | 3.957 ms | 3.016 ms |
| Model-scaled TTFT | 28 layers @ 800 MHz | 46.963 ms | 49.523 ms | 46.963 ms | 19.725 ms | 19.952 ms | 19.725 ms |
| Remote-sum cycles | Cycles | 235,820 | 78,801 | 2,106 | 84,224 | 27,632 | 752 |
| Speedup vs measured | Measured total / scenario total | 1.00x | 1.57x | 2.18x | N/A | N/A | N/A |

### Remote-Sum Operators

| Operator | Type | Input shape | Output shape | Reduction geometry | Fan-in | AXI roofline cycles | Centralized global roofline cycles | Ring2Ring roofline cycles | Speedup | Measured cycles | Projected measured cycles (centralized global) | Projected measured cycles (Ring2Ring) | AXI roofline layer share | Ring2Ring roofline layer share | Ring transfer bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qkt_remote_sum | prefill_remote_sum_fp32MN_fp32MN_2d | inA=[4 x 1024] | out=[1 x 1024] | fan_in=4 | 4 | 57,344 | 25,088 | 512 | 112.00x | 165,609 | 72,454 | 1,479 | 33.80% | 0.59% | 12,288 |
| q_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | inA=[28 x 32] | out=[1 x 32] | fan_in=28 | 28 | 12,544 | 880 | 112 | 112.00x | 33,097 | 2,322 | 296 | 7.39% | 0.13% | 3,456 |
| ffn_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | inA=[28 x 32] | out=[1 x 32] | fan_in=28 | 28 | 12,544 | 880 | 112 | 112.00x | 33,245 | 2,332 | 297 | 7.39% | 0.13% | 3,456 |
| k_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | inA=[4 x 32] | out=[1 x 32] | fan_in=4 | 4 | 1,792 | 784 | 16 | 112.00x | 3,869 | 1,693 | 35 | 1.06% | 0.02% | 384 |
