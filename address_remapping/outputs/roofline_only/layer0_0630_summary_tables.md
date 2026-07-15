# Layer0 Roofline vs Measured Summary Tables

## Summary Metrics

| Metric | Scope | Value |
| --- | --- | --- |
| GEMM compute utilization | GEMM-only cycles | 64.93% |
| GEMM bandwidth utilization | GEMM-only cycles | 18.25% |
| GEMM compute utilization | Full-layer cycles | 10.31% |
| GEMM bandwidth utilization | Full-layer cycles | 2.90% |
| Whole-layer bandwidth utilization | Full-layer cycles | 4.49% |
| GEMM time share | Measured cycles | 15.88% |
| non-GEMM time share | Measured cycles | 84.12% |

## GEMM Operators

| Kernel | Type | Work ops | Measured cycles | Compute util | Bandwidth util |
| --- | --- | --- | --- | --- | --- |
| q_gen | ring_gemm_fp16_fp16_fp16 | 1,835,008 | 8,015 | 89.43% | 23.96% |
| k_gen | ring_gemm_fp16_fp16_fp16 | 2,097,152 | 9,022 | 90.80% | 29.08% |
| v_gen | ring_gemm_fp16_fp16_fp16 | 2,097,152 | 10,305 | 79.50% | 25.46% |
| local_gemm_qkt | gemm_local_qkt_fp16_fp16_fp32 | 65,536 | 922 | 27.77% | 27.77% |
| local_gemm_sv | gemm_local_fp16_fp16_fp16 | 65,536 | 18,053 | 1.42% | 1.06% |
| atten_out | ring_gemm_fp16_fp16_fp16 | 1,835,008 | 20,776 | 34.50% | 9.24% |
| ffn_gate | ring_gemm_fp16_fp16_fp16 | 3,670,016 | 15,902 | 90.15% | 23.75% |
| ffn_up | ring_gemm_fp16_fp16_fp16 | 3,670,016 | 15,633 | 91.70% | 24.15% |
| ffn_down | ring_gemm_fp16_fp16_fp16 | 3,670,016 | 15,718 | 91.21% | 24.02% |

## non-GEMM Operators

| Operator | Type | Total bytes | Measured cycles | Bandwidth util |
| --- | --- | --- | --- | --- |
| q_norm_summac | prefill_summac | 4,224 | 588 | 22.45% |
| q_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 256 | 33,097 | 0.02% |
| q_norm_mac_sfu | prefill_mac_SFU | 256 | 184 | 4.35% |
| q_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 8,320 | 649 | 40.06% |
| q_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 6,272 | 2,496 | 7.85% |
| q_bias_add | prefill_add_fp16MN_fp32N_fp32MN | 6,272 | 3,874 | 5.06% |
| q_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 2,283 | 16.82% |
| q_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 17,308 | 2.22% |
| q_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | 10,240 | 8,606 | 3.72% |
| k_norm_summac | prefill_summac | 32,896 | 4,365 | 23.55% |
| k_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 256 | 3,869 | 0.21% |
| k_norm_mac_sfu | prefill_mac_SFU | 256 | 178 | 4.49% |
| k_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 65,664 | 4,396 | 46.68% |
| k_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 50,176 | 26,260 | 5.97% |
| k_bias_add | prefill_add_fp16MN_fp32N_fp32MN | 6,272 | 3,592 | 5.46% |
| k_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 2,359 | 16.28% |
| k_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 17,308 | 2.22% |
| k_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | 10,240 | 8,606 | 3.72% |
| v_bias_add | prefill_add_V_fp16MN_fp32N_fp16MN | 4,224 | 862 | 15.31% |
| qkt_remote_sum | prefill_remote_sum_fp32MN_fp32MN_2d | 8,192 | 165,609 | 0.15% |
| qkt_score_add | prefill_add_fp32MN_fp32MN_fp32MN | 12,288 | 48,801 | 0.79% |
| softmax_max | prefill_max | 4,224 | 31,510 | 0.42% |
| softmax_sub | prefill_sub_SFU_fp32MN_fp32MN_fp32MN | 8,320 | 51,991 | 0.50% |
| softmax_sum_rec | prefill_sum_rec_fp32MN_fp32MN | 4,224 | 31,520 | 0.42% |
| softmax_scale | prefill_mul_fp32MN_fp32M_fp16MN | 6,272 | 41,061 | 0.48% |
| atten_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | 10,240 | 50,897 | 0.63% |
| ffn_norm_summac | prefill_summac | 4,224 | 708 | 18.64% |
| ffn_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 256 | 33,231 | 0.02% |
| ffn_norm_mac_sfu | prefill_mac_SFU | 256 | 200 | 4.00% |
| ffn_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 8,320 | 1,235 | 21.05% |
| ffn_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 6,272 | 3,956 | 4.95% |
| ffn_silu | prefill_silu_fp16MN_fp32MN | 12,288 | 1,073 | 35.79% |
| ffn_gate_mul | prefill_mul_fp32MN_fp16MN_fp16MN | 16,384 | 2,307 | 22.19% |
| ffn_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | 10,240 | 866 | 36.95% |
