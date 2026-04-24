# RMSNorm 中 Summac 与 mac_SFU 的 Latency 计算拆解

本文基于当前工程实现与输出结果，说明 `rmsnorm_withbaseaddr.json` 中：

- `op0: prefill_summac`
- `op2: prefill_mac_SFU`

的 analytical latency 如何计算、由哪些项构成，以及每一项如何得到。

## 1. 统一主公式（普通算子路径）

当前普通算子（非 ring_gemm）统一使用：

$$
latency_{op}=\max\left(compute\_bound,\ memory\_timeline\_bound,\ ag\_issue\_bound\right)
$$

实现位置：

- `src/address_remapping/performance.py` 中 `_analyze_op`
- 普通算子分支（`else`）

其中三个 bound 分别是：

- `compute_bound`: 来自 `_estimate_compute`
- `memory_timeline_bound`: 来自 `_simulate_per_bank_timeline` 的 `memory_timeline_cycles`
- `ag_issue_bound`: 各 stream 的 `issue_cycles` 取最大

## 2. op0 (prefill_summac) 逐项计算

### 2.1 输入与输出规模

在 `examples/graphs/rmsnorm_withbaseaddr.json`：

- 输入 A: `[1, 32, 32]`，元素数 `1024`
- 输出: `[1, 1, 32]`，元素数 `32`

### 2.2 Compute bound

`prefill_summac` 属于 reduce 类分支，工作量估计为：

$$
op\_work=\max\left(output\_elements,\ input\_elements-output\_elements\right)
$$

代入：

$$
op\_work=\max(32,\ 1024-32)=992
$$

general 峰值算力：

$$
peak=16\ \text{ops/cycle}
$$

故：

$$
compute\_bound=\left\lceil\frac{992}{16}\right\rceil=62
$$

### 2.3 AG issue bound

请求数由 shape 与 block 大小决定：

$$
request\_count=\left\lceil\frac{total\_bytes}{16\ \text{B/block}}\right\rceil
$$

- 读流 A: `4096 B -> 256` requests
- 写回流: `128 B -> 8` requests

`ag_issue_rate=1`，所以：

- 读流 `issue_cycles=256`
- 写流 `issue_cycles=8`

取最大：

$$
ag\_issue\_bound=256
$$

### 2.4 Memory timeline bound

闭环 bank timeline 输出（baseline, op0）为：

- `memory_timeline_cycles=652`
- `read_request_count=256`
- `write_request_count=8`
- `row_switch_count=3`
- `phase_switch_count=1`

时序常量（slice-cycle 域）：

- `request_latency_cycles=14`
- `row_switch_penalty_cycles=28`
- `bank_return_interval_cycles=2`

按单 bank 请求序列可还原为：

1. 基础串行代价（264 个请求，首个 14，其余按 row-hit 2）

$$
14 + (264-1)\times 2 = 540
$$

2. 切换附加代价

$$
3\times 28 + 1\times 28 = 112
$$

3. 合计

$$
memory\_timeline\_bound = 540 + 112 = 652
$$

### 2.5 最终 latency 与误差

$$
latency=\max(62,652,256)=652
$$

输出中：

- analytical: `652`
- measured: `841`

误差：

- 绝对差: `189` cycles
- measured/analytical: `1.2899x`
- 低估比例（相对 measured）: `22.47%`

## 3. op2 (prefill_mac_SFU) 逐项计算

### 3.1 输入与输出规模

在 `examples/graphs/rmsnorm_withbaseaddr.json`：

- 输入 A: `[1, 1, 32]`，元素数 `32`
- 输出: `[1, 1, 32]`，元素数 `32`

### 3.2 Compute bound

`prefill_mac_SFU` 当前走默认分支：

$$
op\_work = output\_elements = 32
$$

$$
compute\_bound=\left\lceil\frac{32}{16}\right\rceil=2
$$

### 3.3 AG issue bound

- 读流 A: `128 B -> 8` requests
- 写回流: `128 B -> 8` requests
- `ag_issue_rate=1`

因此：

$$
ag\_issue\_bound = \max(8,8)=8
$$

### 3.4 Memory timeline bound

闭环 bank timeline 输出（baseline, op2）为：

- `memory_timeline_cycles=72`
- `read_request_count=8`
- `write_request_count=8`
- `row_switch_count=0`
- `phase_switch_count=1`

按同一组时序常量还原：

1. 前 8 个同 phase 请求：

$$
14 + 7\times 2 = 28
$$

2. 一次读写 phase 切换代价：

$$
28 + 2 = 30
$$

3. 后 7 个同 phase 请求：

$$
7\times 2 = 14
$$

4. 合计：

$$
memory\_timeline\_bound = 28 + 30 + 14 = 72
$$

### 3.5 最终 latency 与误差

$$
latency=\max(2,72,8)=72
$$

输出中：

- analytical: `72`
- measured: `160`

误差：

- 绝对差: `88` cycles
- measured/analytical: `2.2222x`
- 低估比例（相对 measured）: `55.00%`

## 4. 这两个算子的主要差异点

- `summac`：读请求量大（256），memory timeline 由大量 read + 少量 write + 行/相位切换主导。
- `mac_SFU`：请求量小（8+8），在当前模型里 compute 仅计作 32 ops，导致总 latency 几乎完全由 memory timeline 主导，容易低估 SFU 相关真实执行开销。

## 5. 实务建议（面向后续校准）

针对 `prefill_mac_SFU`，可优先考虑：

- 增加 `effective_ops_per_element`（提升 compute bound）
- 或增加 `fixed_pipeline_overhead_cycles`（补偿小算子固定开销）

然后用该图中的 `op2` 作为首个标定点，再检查是否会破坏其它图中的一致性。
