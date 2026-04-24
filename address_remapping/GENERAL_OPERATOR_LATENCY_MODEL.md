# 普通算子 Latency 模型总览

本文描述当前工程中“普通算子（非 ring_gemm）”的 latency 计算方式、模型构建流程、组成模块，以及已考虑/未考虑的情况。

## 1. 适用范围

本模型适用于 `src/address_remapping/performance.py` 中 `_analyze_op` 的普通算子分支，即：

- 不走 `ring_gemm_fp16_fp16_fp16` 专用微流水线路径
- 典型如：`prefill_summac`、`prefill_remote_sum`、`prefill_mac_SFU`、`prefill_mul` 等

## 2. 总体思路

模型不是单一公式，而是由以下子模型组合：

1. 地址请求生成模型
2. stream 级统计模型
3. bank 闭环调度模型（读写仲裁 + 回压）
4. 算子计算量模型
5. 顶层 bound 汇总模型

最终用 max-bound 得到算子 latency。

## 3. 构建流程（从图到 latency）

### 3.1 图解析与请求物化

对每个 op 的每个输入端口：

- 判定 role/AG（`_classify_input_stream`）
- 根据 edge remap 或 source tensor 生成 `PhysicalRequest`

对输出端口：

- 生成 writeback requests（固定 `ag4`）

请求字段包含：

- `ag_id`, `role`
- `bank_id`, `row_id`, `col_id`
- `physical_addr`

### 3.2 stream 级统计（每个 AG 流）

`_analyze_request_stream` 会计算：

- `request_count`
- `issue_cycles = ceil(request_count / ag_issue_rate)`
- `bank_cycles`
- `row_switch_penalty_cycles`
- `adjusted_stream_cycles`

这些值主要用于诊断与辅助报告；普通算子最终 memory bound 以闭环 timeline 为准。

### 3.3 闭环 bank timeline（核心）

`_simulate_per_bank_timeline` 做离散事件调度，关键机制包括：

- 读写请求按 AG issue 节奏释放
- 两级仲裁：
  - 一级优先同 phase + 同 open row
  - 二级处理其余请求
- 写队列深度限制（`controller_write_queue_depth`）
- slice 侧写缓冲阈值触发阻塞（`slice_write_buffer_depth`）
- 强制写排空（slice blocked 时仅允许写请求被选中）

输出：

- `memory_timeline_cycles`
- `phase_switch_penalty_cycles`
- `row_switch_penalty_cycles`
- `rw_switch_count`, `row_switch_count`
- `q_w_full_cycles`, `slice_blocked_cycles`
- `arbiter1_wins`, `arbiter2_wins`

### 3.4 计算量与 compute bound

`_estimate_compute` 按 op 类型估计 `op_work`：

- `prefill_summac/prefill_sum_rec/prefill_max`：
  $$
  op\_work=\max(output\_elements,\ input\_elements-output\_elements)
  $$
- `remote_sum`：当前用 `output_elements` 近似
- 其余普通算子：当前用 `output_elements` 近似

再算：

$$
compute\_bound=\left\lceil\frac{op\_work}{general\_peak\_ops\_per\_cycle}\right\rceil
$$

### 3.5 顶层汇总

普通算子最终：

$$
latency_{op}=\max\left(compute\_bound,\ memory\_timeline\_bound,\ ag\_issue\_bound\right)
$$

其中：

- `compute_bound`: 来自 `_estimate_compute`
- `memory_timeline_bound`: `bank_timeline.memory_timeline_cycles`
- `ag_issue_bound`: 所有 stream 的 `issue_cycles` 最大值

## 4. 模型包含了哪些部分

1. 地址映射与 remap 对 bank/row 分布的影响
2. AG 发射速率约束
3. bank 内 row-hit / row-switch / read-write phase-switch 时序
4. controller 写队列容量与 slice 写缓冲回压
5. 仲裁策略对请求顺序与切换损耗的影响
6. compute 与 memory 的 bound 竞争关系

## 5. 已考虑的典型情况

1. 单 bank 高集中访问（容易形成长串 row-hit）
2. 多 row 访问导致 row-switch penalty 累积
3. 读写交替导致 phase-switch penalty
4. 写压力升高导致强制写排空
5. remap 与 baseline 在请求分布上的差异对 latency 的影响

## 6. 当前简化与潜在误差来源

1. 非 ring 普通算子的 compute work 目前较粗粒度：
  - 许多类型使用 `output_elements` 近似
2. 对 SFU/特殊流水线固定开销未显式建模
3. 未显式引入更细粒度的片上执行流水细节（例如 issue-to-execute 固定延迟、功能单元冲突等）
4. `remote_sum` fan-in 细节尚未完全体现在 compute work 中

因此，在某些小算子或特殊算子上会出现“memory timeline 主导且整体低估”的情况。

## 7. 如何阅读输出中的 latency 分解

看每个 op 的这几块：

1. `analytical_model.compute_bound_cycles`
2. `analytical_model.memory_access_bound_cycles`
3. `analytical_model.ag_issue_bound_cycles`
4. `bank_timeline`（看 row/phase 切换、回压指标）
5. `hardware_measured_cycles`（与 analytical 对比）

先判断谁是主导 bound，再回看 `bank_timeline` 和 stream 统计解释原因。

## 8. 建议的后续增强方向

1. 针对 SFU 类算子引入类型化 compute 模型（每元素多操作或固定 pipeline 开销）
2. 为 remote_sum 增加显式 fan-in 元数据并修正 work 估计
3. 增加按算子类型的校准参数（可选）
4. 保留闭环调度主干不变，优先增强 compute 端可解释性
