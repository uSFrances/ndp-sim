# ring_gemm latency 组成与计算说明（ring_gemm_bias）

## 1. 结论先看

本案例（baseline 模式）中，ring_gemm 的最终 latency 为：

- Final latency = 726 cycles

总判定公式是：

- latency = max(compute_pipeline_completion, memory_access_bound, ag_issue_bound, ring_transfer_bound)

代入本例数值：

- compute_pipeline_completion = 384
- memory_access_bound = 726
- ag_issue_bound = 256
- ring_transfer_bound = 256

所以：

- latency = max(384, 726, 256, 256) = 726

结论：本例是 memory bound（由内存时间线主导）。

---

## 2. 参数总览（按层次）

### 2.1 顶层四个 bound

- compute_bound_cycles = 256
- memory_access_bound_cycles = 726
- ag_issue_bound_cycles = 256
- ring_transfer_bound_cycles = 256

其中最终 latency 取上述关键路径相关项的最大值。

### 2.2 本问题的关键几何参数

- ring_participants = 4
- M = 64, N = 128, K_total = 16（来自图）
- output_tile_m = 32, output_tile_n = 32
- output_tile_count = 2（M 方向 2，N 方向 1）
- total_k_tiles_per_output_tile = 8
- total_coarse_compute_events = output_tile_count × total_k_tiles_per_output_tile = 2 × 8 = 16
- pe_micro_ops_per_output_tile = 16
- per_micro_op_compute_cycles = 1

---

## 3. Compute 侧 latency 怎么来的

## 3.1 理论 compute bound

- work_ops = 65536
- peak_compute_ops_per_cycle = 256
- compute_bound_cycles = work_ops / peak = 65536 / 256 = 256

这个 256 是理想纯计算下界。

## 3.2 实际 compute pipeline completion

模型里实际参与最终 max 比较的是 compute pipeline completion：

- compute_pipeline_completion = 384

其等价分解可写成：

- compute_pipeline_completion = idle_before_first + active_compute + idle_between

本例数值：

- idle_before_first = 58
- active_compute = 256（16 个 coarse event，每个 event 做 16 个 micro-op，每个 micro-op 1 cycle）
- idle_between = 70

校验：

- 58 + 256 + 70 = 384

### 3.3 idle 的组成

- idle_before_first = 58
  - event0 启动条件：start0 = max(pe_available0, ring_ready0, b_ready0, psum_dep0)
  - 本例：max(0, 58, 14, 0) = 58
  - 所以首拍主要被 ring_ready0 限制。

- idle_between = 70
  - 定义：sum(max(0, start[i] - end[i-1]))，i=1..15
  - 来自微块之间等待，主要由 b_ready 与 ring_ready 的到达时序造成。

### 3.4 ping-pong 统计口径

报告中还给了一个等价统计：

- ping_pong_startup_cycles = 76
- ping_pong_steady_cycles = 308
- ping_pong_pipeline_cycles = 384

即：

- 76 + 308 = 384

其中 76 与 local A 首次装载时间一致（local_a_read_cycles）。

---

## 4. Ring 传输相关参数怎么构成

### 4.1 局部与环上传输量

- local_a_bytes = 512 B
- ring_a_total_bytes = 1536 B

因为参与者为 4，A 需要沿 ring 传给其余 3 个 hop，故总环上传输量约为本地 A 的 3 倍。

### 4.2 带宽与每 hop 时间

- ring_bandwidth_bytes_per_cycle = 32 B/cycle
- microtile_bytes = a_buffer_bytes = 128 B
- per_tile_ring_a_transfer_cycles = ceil(128 / 32) = 4 cycles

### 4.3 ring_transfer_bound 的形成

- ring_transfer_bound_cycles = 256

这对应时间线中 ring link completion 的最晚到达时间（所有 coarse event 所需 ring A 到位的最晚时刻）。

另外有一个总传输量统计：

- ring_a_transfer_cycles = 48

它是总量口径（按 A tile 数与 hop 数累计），不是最终关键路径口径；最终参与 latency max 的是 ring_transfer_bound（本例 256）。

---

## 5. AG 发射上界怎么来

- ag_issue_bound_cycles = max(stream.issue_cycles)

本例最大值是：

- ag_issue_bound_cycles = 256

它表示地址发生/请求发射速率带来的上界，不是最终瓶颈（因为 256 < 726）。

---

## 6. Memory access bound（726）怎么来的

该项来自 ring_bank_timeline 的 memory_timeline_cycles：

- memory_access_bound_cycles = 726

本例时间线特征：

- read_priority_policy = reads_before_writes_until_full
- forced_drain_count = 1
- per_bank_completion_cycles: bank0 = 726
- phase_switch_penalty_cycles = 28
- row_switch_penalty_cycles = 168
- per_bank_breakdown(bank0):
  - row_switch_count = 6
  - phase_switch_count = 1
  - read_request_count = 96
  - write_request_count = 129

### 6.1 可分解理解

把 bank 服务时间拆成三部分看更直观：

- 基础请求服务时间（不含额外切换惩罚）
  - first request latency + 后续请求按 return interval 推进
- 行切换与读写相位切换惩罚
  - row_switch_count × row_switch_penalty + phase_switch_count × row_switch_penalty
- 由 release 时机、读优先策略、buffer 占用与 forced drain 触发带来的调度空隙

在本例中，这些因素叠加后，memory 时间线末端落在 726，成为最终瓶颈。

---

## 7. 从参数到最终 latency 的一页式链路

1. 几何与硬件决定每个 coarse event 需要的数据与计算粒度。
2. 由 work_ops 得到 compute_bound = 256。
3. 由 ring/B/psum/PE 依赖排程得到 compute_pipeline_completion = 384。
4. 由 ring 到位最晚时刻得到 ring_transfer_bound = 256。
5. 由 AG 发射速率得到 ag_issue_bound = 256。
6. 由 bank 事件循环（读优先、phase/row 切换、forced drain）得到 memory_access_bound = 726。
7. 最终 latency 取最大值：726。

---

## 8. 本例所有关键数值清单

- Final latency: 726
- compute_bound: 256
- compute_pipeline_completion: 384
- memory_access_bound: 726
- ag_issue_bound: 256
- ring_transfer_bound: 256
- local_a_read_cycles: 76
- ping_pong_startup: 76
- ping_pong_steady: 308
- ping_pong_pipeline: 384
- idle_before_first: 58
- idle_between: 70
- ring_bandwidth: 32 B/cycle
- per_tile_ring_a_transfer: 4
- ring_participants: 4
- forced_drain_count: 1
- row_switch_penalty_total: 168
- phase_switch_penalty_total: 28

以上即 ring_gemm latency 的完整组成与本例对应数值。