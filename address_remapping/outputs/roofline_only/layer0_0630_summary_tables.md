# Layer0 Roofline vs Measured Summary Tables

## Model Parameters

- Model config: `examples\configs\config.json`
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
| non-GEMM bandwidth utilization | non-GEMM-only cycles | 12.66% | 3.48% |
| non-GEMM bandwidth utilization | Full-layer cycles | 7.12% | 2.80% |
| Whole-layer bandwidth utilization | Full-layer cycles | 19.42% | 7.64% |
| Template TTFT | 28 layers @ 800 MHz | 5.938 ms | 15.096 ms |
| Model-scaled TTFT | 28 layers @ 800 MHz | 19.725 ms | 63.418 ms |
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

| Op ID | Operator | Type | Total bytes | Roofline cycles | Measured cycles | Roofline layer time share | Layer time share | Roofline bandwidth util | Measured bandwidth util | Roofline global BW util | Measured global BW util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op23 | qkt_remote_sum | prefill_remote_sum_fp32MN_fp32MN_2d | 20,480 | 57,344 | 165,609 | 33.80% | 38.40% | 100.00% | 34.63% | 100.00% | 34.63% |
| op33 | ffn_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 3,712 | 12,544 | 33,245 | 7.39% | 7.71% | 100.00% | 37.73% | 100.00% | 37.73% |
| op1 | q_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 3,712 | 12,544 | 33,097 | 7.39% | 7.67% | 100.00% | 37.90% | 100.00% | 37.90% |
| op14 | k_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 50,176 | 1,568 | 26,218 | 0.92% | 6.08% | 100.00% | 5.98% | N/A | N/A |
| op18 | k_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 384 | 17,376 | 0.23% | 4.03% | 100.00% | 2.21% | N/A | N/A |
| op8 | q_rope_mul_b | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 384 | 17,308 | 0.23% | 4.01% | 100.00% | 2.22% | N/A | N/A |
| op13 | k_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 65,664 | 2,052 | 4,396 | 1.21% | 1.02% | 100.00% | 46.68% | N/A | N/A |
| op10 | k_norm_summac | prefill_summac | 32,896 | 1,028 | 4,365 | 0.61% | 1.01% | 100.00% | 23.55% | N/A | N/A |
| op36 | ffn_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 6,272 | 196 | 4,028 | 0.12% | 0.93% | 100.00% | 4.87% | N/A | N/A |
| op6 | q_bias_add | prefill_add_fp16MN_fp32N_fp32MN | 6,272 | 196 | 3,873 | 0.12% | 0.90% | 100.00% | 5.06% | N/A | N/A |
| op11 | k_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 640 | 1,792 | 3,869 | 1.06% | 0.90% | 100.00% | 46.32% | 100.00% | 46.32% |
| op16 | k_bias_add | prefill_add_fp16MN_fp32N_fp32MN | 6,272 | 196 | 3,611 | 0.12% | 0.84% | 100.00% | 5.43% | N/A | N/A |
| op40 | ffn_gate_mul | prefill_mul_fp32MN_fp16MN_fp16MN | 16,384 | 512 | 3,559 | 0.30% | 0.83% | 100.00% | 14.39% | N/A | N/A |
| op4 | q_norm_apply | prefill_mul_fp32MN_fp32N_fp16MN | 6,272 | 196 | 2,448 | 0.12% | 0.57% | 100.00% | 8.01% | N/A | N/A |
| op17 | k_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 384 | 2,387 | 0.23% | 0.55% | 100.00% | 16.09% | N/A | N/A |
| op24 | qkt_score_add | prefill_add_fp32MN_fp32MN_fp32MN | 12,288 | 384 | 2,344 | 0.23% | 0.54% | 100.00% | 16.38% | N/A | N/A |
| op9 | q_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | 10,240 | 320 | 2,304 | 0.19% | 0.53% | 100.00% | 13.89% | N/A | N/A |
| op7 | q_rope_mul_a | prefill_mul_fp32MN_fp32MN_fp32MN | 12,288 | 384 | 2,283 | 0.23% | 0.53% | 100.00% | 16.82% | N/A | N/A |
| op19 | k_rope_out | prefill_add_fp32MN_fp32MN_fp16MN | 10,240 | 320 | 2,224 | 0.19% | 0.52% | 100.00% | 14.39% | N/A | N/A |
| op31 | atten_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | 10,240 | 320 | 1,697 | 0.19% | 0.39% | 100.00% | 18.86% | N/A | N/A |
| op28 | softmax_scale | prefill_mul_fp32MN_fp32M_fp16MN | 6,272 | 196 | 1,449 | 0.12% | 0.34% | 100.00% | 13.53% | N/A | N/A |
| op35 | ffn_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 8,320 | 260 | 1,235 | 0.15% | 0.29% | 100.00% | 21.05% | N/A | N/A |
| op26 | softmax_sub | prefill_sub_SFU_fp32MN_fp32MN_fp32MN | 8,320 | 260 | 1,081 | 0.15% | 0.25% | 100.00% | 24.05% | N/A | N/A |
| op39 | ffn_silu | prefill_silu_fp16MN_fp32MN | 12,288 | 384 | 1,073 | 0.23% | 0.25% | 100.00% | 35.79% | N/A | N/A |
| op21 | v_bias_add | prefill_add_V_fp16MN_fp32N_fp16MN | 4,224 | 132 | 867 | 0.08% | 0.20% | 100.00% | 15.22% | N/A | N/A |
| op42 | ffn_residual_add | prefill_add_fp32MN_fp16MN_fp32MN | 10,240 | 320 | 866 | 0.19% | 0.20% | 100.00% | 36.95% | N/A | N/A |
| op25 | softmax_max | prefill_max | 4,224 | 132 | 734 | 0.08% | 0.17% | 100.00% | 17.98% | N/A | N/A |
| op27 | softmax_sum_rec | prefill_sum_rec_fp32MN_fp32MN | 4,224 | 132 | 696 | 0.08% | 0.16% | 100.00% | 18.97% | N/A | N/A |
| op32 | ffn_norm_summac | prefill_summac | 4,224 | 132 | 660 | 0.08% | 0.15% | 100.00% | 20.00% | N/A | N/A |
| op3 | q_norm_scale | prefill_mul_fp32MN_fp32M_fp32MN | 8,320 | 260 | 649 | 0.15% | 0.15% | 100.00% | 40.06% | N/A | N/A |
| op0 | q_norm_summac | prefill_summac | 4,224 | 132 | 588 | 0.08% | 0.14% | 100.00% | 22.45% | N/A | N/A |
| op12 | k_norm_mac_sfu | prefill_mac_SFU | 256 | 8 | 192 | 0.00% | 0.04% | 100.00% | 4.17% | N/A | N/A |
| op2 | q_norm_mac_sfu | prefill_mac_SFU | 256 | 8 | 184 | 0.00% | 0.04% | 100.00% | 4.35% | N/A | N/A |
| op34 | ffn_norm_mac_sfu | prefill_mac_SFU | 256 | 8 | 170 | 0.00% | 0.04% | 100.00% | 4.71% | N/A | N/A |

## Remote-Sum Transport Comparison

Ring2Ring remote-sum model: each slice first reads its local partial result, then receives the other slice partial results through the slice-to-slice n2n datapath. The ring datapath bandwidth is modeled as 256 bit/cycle = 32 B/cycle per slice.

For each remote-sum op: `local_read_bytes = output_elements * dtype_bytes`, `ring_transfer_bytes = (fan_in - 1) * output_elements * dtype_bytes`, `local_write_bytes = output_elements * output_dtype_bytes`, and `ring2ring_roofline_cycles = max(local_read_bytes / local_bw, ring_transfer_bytes / 32, local_write_bytes / local_bw, reduction_ops / general_peak)`.

Centralized-global remote-sum projection: one slice reads all partial results through global AXI, performs the reduction, then sends the reduced result back to the other participating slices through global AXI. The global AXI bandwidth is modeled as 128 bit at half slice frequency, i.e. 8 B/cycle. This projection keeps all non-remote-sum measured cycles unchanged and scales each remote-sum measured cycle count by `centralized_global_roofline_cycles / axi_pull_roofline_cycles`.

### Layer Roofline Summary

| Metric | Scope | Measured | Projected measured with centralized global remote-sum | Projected measured with Ring2Ring remote-sum | AXI pull roofline | Centralized global roofline | Ring2Ring n2n roofline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM compute utilization | GEMM-only cycles | 87.73% | 87.73% | 87.73% | 100.00% | 100.00% | 100.00% |
| GEMM bandwidth utilization | GEMM-only cycles | 24.65% | 24.65% | 24.65% | 28.10% | 28.10% | 28.10% |
| GEMM compute utilization | Full-layer cycles | 17.21% | 35.23% | 37.65% | 43.76% | 81.69% | 86.29% |
| GEMM bandwidth utilization | Full-layer cycles | 4.84% | 9.90% | 10.58% | 12.30% | 22.96% | 24.25% |
| non-GEMM bandwidth utilization | non-GEMM-only cycles | 3.48% | 9.58% | 10.73% | 12.66% | 72.58% | 102.37% |
| non-GEMM bandwidth utilization | Full-layer cycles | 2.80% | 5.73% | 6.13% | 7.12% | 13.29% | 14.04% |
| Whole-layer bandwidth utilization | Full-layer cycles | 7.64% | 15.63% | 16.71% | 19.42% | 36.25% | 38.29% |
| GEMM time share | Cycles | 19.62% | 40.16% | 42.91% | 43.76% | 81.69% | 86.29% |
| non-GEMM time share | Cycles | 80.38% | 59.84% | 57.09% | 56.24% | 18.31% | 13.71% |
| Total cycles | Cycles | 431,310 | 210,736 | 197,196 | 169,652 | 90,884 | 86,040 |
| Template TTFT | 28 layers @ 800 MHz | 15.096 ms | 7.376 ms | 6.902 ms | 5.938 ms | 3.181 ms | 3.011 ms |
| Model-scaled TTFT | 28 layers @ 800 MHz | 63.418 ms | 47.438 ms | 46.472 ms | 19.725 ms | 19.952 ms | 19.725 ms |
| Remote-sum cycles | Cycles | 235,820 | 15,246 | 1,706 | 84,224 | 5,456 | 612 |
| Speedup vs measured | Measured total / scenario total | 1.00x | 2.05x | 2.19x | N/A | N/A | N/A |

### Remote-Sum Operators

| Operator | Type | Fan-in | AXI roofline cycles | Ring2Ring roofline cycles | Speedup | Measured cycles | Projected measured cycles | AXI roofline layer share | Ring2Ring roofline layer share | Ring transfer bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qkt_remote_sum | prefill_remote_sum_fp32MN_fp32MN_2d | 4 | 57,344 | 384 | 149.33x | 165,609 | 1,109 | 33.80% | 0.45% | 12,288 |
| q_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 28 | 12,544 | 108 | 116.15x | 33,097 | 285 | 7.39% | 0.13% | 3,456 |
| ffn_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 28 | 12,544 | 108 | 116.15x | 33,245 | 286 | 7.39% | 0.13% | 3,456 |
| k_norm_remote_sum | prefill_remote_sum_Mfp32_Mfp32 | 4 | 1,792 | 12 | 149.33x | 3,869 | 26 | 1.06% | 0.01% | 384 |
