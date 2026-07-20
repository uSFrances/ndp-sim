# Layer0 Roofline vs Measured Summary Tables

## Model Parameters

- Model config: `examples\configs\smallsize_config.json`
- Model name: `smallsize`
- TTFT layers: `1`
- TTFT frequency: `800 MHz`
- Layer graph: `layer0`; template TTFT multiplies layer0 cycles, model-scaled TTFT recomputes op sizes from the model config.
- `hidden_size` = `896`
- `execution_hidden_size` = `896`
- `intermediate_size` = `1792`
- `num_attention_heads` = `7`
- `padded_attention_heads` = `7`
- `num_key_value_heads` = `1`
- `head_dim` = `128`
- `num_hidden_layers` = `1`
- `requested_sequence_length` = `32`
- `sequence_length` = `32`
- `slice_per_head` = `4`
- `used_slices` = `28`
- `kv_padding` = `512`
- `kv_padding_a` = `256`
- `kv_padding_b` = `1024`
- `clusters` = `7`
- `attention_waves` = `1`
- `kv_heads_per_cluster` = `1`
- Layer graph params: `hidden_size=896`, `intermediate_size=1792`, `sequence_length=32`, `num_hidden_layers=1`

## Summary Metrics

| Metric | Scope | AXI pull roofline | Projected measured |
| --- | --- | --- | --- |
| GEMM compute utilization | GEMM-only cycles | 100.00% | 87.81% |
| GEMM compute utilization | Full-layer cycles | 75.83% | 10.02% |
| non-GEMM bandwidth utilization | non-GEMM-only cycles | 100.00% | 3.60% |
| non-GEMM bandwidth utilization | Full-layer cycles | 24.17% | 3.19% |
| Whole-layer bandwidth utilization | Full-layer cycles | 24.17% | 3.19% |
| GEMM time share | Cycles | 75.83% | 11.41% |
| non-GEMM time share | Cycles | 24.17% | 88.59% |

## Model-Scaled Summary

| Scenario | Per-layer cycles | Total cycles | TTFT |
| --- | --- | --- | --- |
| Projected measured | 720,751 | 720,751 | 0.901 ms |
| Projected measured with centralized global remote-sum | 2,147,797 | 2,147,797 | 2.685 ms |
| Projected measured with Ring2Ring remote-sum | 720,751 | 720,751 | 0.901 ms |
| AXI pull roofline | 95,198 | 95,198 | 0.119 ms |
| Centralized global roofline | 99,762 | 99,762 | 0.125 ms |
| Ring2Ring n2n roofline | 95,198 | 95,198 | 0.119 ms |

## Model-Scaled Operator Projection

This table recomputes each operator's work or bytes from the target model config, then projects cycles using the calibration layer's measured GEMM throughput or measured non-GEMM effective bandwidth.

- Target model: `smallsize`
- Target layers: `1`
- Target sequence length: `32`
- Target execution hidden size: `896`


### Model-Scaled GEMM Operators

| Op ID | Operator | Type | Model work ops | Projected measured cycles | Layer share | Projected compute util | AXI roofline cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| op37 | ffn_gate | prefill_gemm_ring_4slice | 3,670,016 | 15,878 | 2.20% | 90.29% | 14,336 |
| op38 | ffn_up | prefill_gemm_ring_4slice | 3,670,016 | 15,792 | 2.19% | 90.78% | 14,336 |
| op41 | ffn_down | prefill_gemm_ring_4slice | 3,670,016 | 15,756 | 2.19% | 90.99% | 14,336 |
| op20 | v_gen | prefill_gemm_ring_4slice | 1,835,008 | 8,957 | 1.24% | 80.02% | 7,168 |
| op5 | q_gen | prefill_gemm_ring_4slice | 1,835,008 | 8,011 | 1.11% | 89.48% | 7,168 |
| op30 | atten_out | prefill_gemm_ring_4slice | 1,835,008 | 8,004 | 1.11% | 89.56% | 7,168 |
| op15 | k_gen | prefill_gemm_ring_4slice | 1,835,008 | 7,895 | 1.10% | 90.79% | 7,168 |
| op29 | local_gemm_sv | prefill_gemm_local | 65,536 | 995 | 0.14% | 25.73% | 256 |
| op22 | local_gemm_qkt | prefill_gemm_local_qkt | 65,536 | 929 | 0.13% | 27.56% | 256 |

### Model-Scaled non-GEMM Operators

| Op ID | Operator | Type | Model bytes | Projected measured cycles | Layer share | Projected BW util | AXI roofline cycles | Centralized roofline cycles | Ring2Ring roofline cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op23 | qkt_remote_sum | prefill_remote_sum_4slice_fp32MN_fp32MN | 20,480 | 207,011 | 28.72% | 0.31% | 640 | 3,584 | 640 |
| op12 | k_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | 114,816 | 86,112 | 11.95% | 4.17% | 3,588 | 3,588 | 3,588 |
| op2 | q_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | 114,816 | 82,524 | 11.45% | 4.35% | 3,588 | 3,588 | 3,588 |
| op34 | ffn_norm_mac_sfu | prefill_mac_SFU_fp32MN_fp32MN | 114,816 | 76,245 | 10.58% | 4.71% | 3,588 | 3,588 | 3,588 |
| op33 | ffn_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | 3,712 | 34,432 | 4.78% | 0.34% | 116 | 880 | 116 |
| op1 | q_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | 3,712 | 34,279 | 4.76% | 0.34% | 116 | 880 | 116 |
| op14 | k_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 50,176 | 26,218 | 3.64% | 5.98% | 1,568 | 1,568 | 1,568 |
| op18 | k_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 17,376 | 2.41% | 2.21% | 384 | 384 | 384 |
| op8 | q_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 17,308 | 2.40% | 2.22% | 384 | 384 | 384 |
| op6 | q_bias_add | prefill_add_fp16MN_fp32N_fp32MN | 8,256 | 5,098 | 0.71% | 5.06% | 258 | 258 | 258 |
| op11 | k_norm_remote_sum | prefill_remote_sum_fp32MN_fp32MN | 640 | 4,836 | 0.67% | 0.41% | 20 | 112 | 20 |
| op16 | k_bias_add | prefill_add_fp16MN_fp32N_fp32MN | 8,256 | 4,753 | 0.66% | 5.43% | 258 | 258 | 258 |
| op13 | k_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 65,664 | 4,396 | 0.61% | 46.68% | 2,052 | 2,052 | 2,052 |
| op10 | k_norm_summac | prefill_summac_fp32MN_fp32MN | 32,896 | 4,365 | 0.61% | 23.55% | 1,028 | 1,028 | 1,028 |
| op36 | ffn_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 6,272 | 4,028 | 0.56% | 4.87% | 196 | 196 | 196 |
| op40 | ffn_gate_mul | prefill_mul_fp32MN_fp16MN_fp16MN | 16,384 | 3,559 | 0.49% | 14.39% | 512 | 512 | 512 |
| op4 | q_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 6,272 | 2,448 | 0.34% | 8.01% | 196 | 196 | 196 |
| op17 | k_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 2,387 | 0.33% | 16.09% | 384 | 384 | 384 |
| op24 | qkt_score_add | prefill_add_fp32MN_fp32MN_fp32MN | 12,288 | 2,344 | 0.33% | 16.38% | 384 | 384 | 384 |
| op9 | q_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | 10,240 | 2,304 | 0.32% | 13.89% | 320 | 320 | 320 |
| op7 | q_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 2,283 | 0.32% | 16.82% | 384 | 384 | 384 |
| op19 | k_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | 10,240 | 2,224 | 0.31% | 14.39% | 320 | 320 | 320 |
| op31 | atten_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | 10,240 | 1,697 | 0.24% | 18.86% | 320 | 320 | 320 |
| op28 | softmax_scale | prefill_mul_fp32MN_fp32M_fp16MN | 6,272 | 1,449 | 0.20% | 13.53% | 196 | 196 | 196 |
| op21 | v_bias_add | prefill_add_V_fp16MN_fp32N_fp16MN | 6,208 | 1,274 | 0.18% | 15.22% | 194 | 194 | 194 |
| op35 | ffn_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 8,320 | 1,235 | 0.17% | 21.05% | 260 | 260 | 260 |
| op26 | softmax_sub | prefill_sub_SFU_fp32MN_fp32M_fp32MN | 8,320 | 1,081 | 0.15% | 24.05% | 260 | 260 | 260 |
| op39 | ffn_silu | prefill_silu_fp16MN_fp32MN | 12,288 | 1,073 | 0.15% | 35.79% | 384 | 384 | 384 |
| op42 | ffn_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | 10,240 | 866 | 0.12% | 36.95% | 320 | 320 | 320 |
| op25 | softmax_max | prefill_max_fp32MN_fp32MN | 4,224 | 734 | 0.10% | 17.98% | 132 | 132 | 132 |
| op27 | softmax_sum_rec | prefill_sum_rec_fp32MN_fp32MN | 4,224 | 696 | 0.10% | 18.97% | 132 | 132 | 132 |
| op32 | ffn_norm_summac | prefill_summac_fp32MN_fp32MN | 4,224 | 660 | 0.09% | 20.00% | 132 | 132 | 132 |
| op3 | q_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 8,320 | 649 | 0.09% | 40.06% | 260 | 260 | 260 |
| op0 | q_norm_summac | prefill_summac_fp32MN_fp32MN | 4,224 | 588 | 0.08% | 22.45% | 132 | 132 | 132 |
