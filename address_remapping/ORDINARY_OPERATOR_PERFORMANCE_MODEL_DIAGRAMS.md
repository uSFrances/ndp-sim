# 普通算子性能分析建模图解

## 🎯 目标

这份说明只覆盖 `address_remapping` 当前 **普通算子** 的性能分析主路径，不展开 `ring_gemm` 专用微流水路径。

它回答 5 个问题：

1. 请求是怎么从图和张量变成 `PhysicalRequest` 的？
2. 请求是怎么分配到多个 AG、多个 bank 的？
3. 每个 bank/controller 维护了哪些状态？
4. `RR` 切行、`RW` 切换、写回阻塞是怎么在模型里出现的？
5. 最后 `latency` 是怎么从 `compute / memory timeline / AG issue` 三类约束汇总出来的？

对应代码主干在：

- [performance.py](/H:/dev/projects/ndp-sim/address_remapping/src/address_remapping/performance.py)
- [BANK_CONTROLLER_COST_MODEL_RULES.md](/H:/dev/projects/ndp-sim/address_remapping/BANK_CONTROLLER_COST_MODEL_RULES.md)
- [GENERAL_OPERATOR_LATENCY_MODEL.md](/H:/dev/projects/ndp-sim/address_remapping/GENERAL_OPERATOR_LATENCY_MODEL.md)

---

## 🧭 总览图

下面这张图对应普通算子主路径的 5 层建模：

1. 图与张量解析
2. 地址请求物化
3. stream 级统计
4. 闭环 bank timeline
5. bound 汇总与 roofline 对照

```mermaid
flowchart LR
    accTitle: 普通算子性能分析总览
    accDescr: 从 graph 输入到请求物化、stream 统计、bank timeline、compute bound 和最终 latency 汇总的普通算子性能分析主链路。

    graph["📄 Graph / Tensors<br/>shape + dtype + base_addr<br/>mode = baseline / remap / interleave"]
    op["🧩 _analyze_op<br/>按输入端口遍历 op"]
    req["📦 PhysicalRequest 物化<br/>role + ag_id + bank,row,col"]
    stream["📊 Stream 级统计<br/>request_count / issue_cycles<br/>row_switch_penalty / adjusted_stream_cycles"]
    timeline["🏦 闭环 Bank Timeline<br/>多 bank ready queue + 两级仲裁 + 回压"]
    compute["🧮 Compute Bound<br/>_estimate_compute"]
    roof["📈 True Roofline<br/>ops / bytes"]
    merge["📌 普通算子 Latency 汇总<br/>max(compute, memory_timeline, ag_issue)"]
    report["📝 Performance JSON / MD<br/>bank_timeline + analytical_model + roofline"]

    graph --> op --> req --> stream
    req --> timeline
    op --> compute
    req --> roof
    stream --> merge
    timeline --> merge
    compute --> merge
    roof -. 理论上界对照 .-> merge
    merge --> report

    classDef source fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef process fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef core fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef output fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class graph source
    class op,req,stream,compute,roof process
    class timeline,merge core
    class report output
```

---

## 🧱 请求物化：从 Tensor 到 `PhysicalRequest`

普通算子的请求生成入口在 `_analyze_op`：

- 输入端口：
  - `_classify_input_stream`
  - `_requests_from_edge_result` 或 `_requests_from_source_tensor`
- 输出端口：
  - `_requests_from_output_tensor`
- 然后统一进入 `PhysicalRequest`

### 1. 请求物化示意图

```mermaid
flowchart TD
    accTitle: 请求物化示意图
    accDescr: 普通算子输入输出张量如何结合 role、AG、address transform 和 base_addr，生成 PhysicalRequest。

    inA["输入 Tensor A<br/>shape / dtype / base_addr"]
    inB["输入 Tensor B / bias / aux<br/>shape / dtype / base_addr"]
    out["输出 Tensor<br/>shape / dtype / base_addr"]

    classify["role / AG 判定<br/>_classify_input_stream"]
    transform["地址变换对象 P<br/>baseline / remap / remap_interleave"]
    materialize["请求物化<br/>logical_addr -> P -> physical_addr"]

    reqA["PhysicalRequest(read)<br/>request_id<br/>ag_id=ag0/ag1/...<br/>role=A/B/aux<br/>bank_id,row_id,col_id"]
    reqW["PhysicalRequest(write)<br/>request_id<br/>ag_id=ag4<br/>role=writeback<br/>bank_id,row_id,col_id"]

    inA --> classify
    inB --> classify
    classify --> transform
    out --> transform
    transform --> materialize
    materialize --> reqA
    materialize --> reqW

    classDef tensor fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef logic fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef req fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class inA,inB,out tensor
    class classify,transform,materialize logic
    class reqA,reqW req
```

### 2. `PhysicalRequest` 里最关键的字段

```mermaid
classDiagram
    accTitle: PhysicalRequest 关键字段
    accDescr: 普通算子性能分析里每个请求最关键的字段，以及它们分别服务于地址映射、bank 调度和 trace 输出。

    class PhysicalRequest {
      +request_id
      +tensor_name
      +edge_name
      +ag_id
      +role
      +logical_addr
      +base_addr
      +address_transform
      +physical_addr
      +slice_id
      +bank_id
      +row_id
      +col_id
    }
```

### 3. mode 影响的是哪一层

- `baseline`
  - 输入/输出按 identity transform 物化
- `remap`
  - 使用 solver 推导出的地址变换
- `remap_interleave`
  - 在 remap 基础上进一步把 bank 维度做 interleave

也就是说：

- **mode 改的是请求分布**
- 然后请求分布再去影响：
  - 哪些 bank 被激活
  - row hit / row miss 怎么出现
  - `RR` / `RW` 切换如何累积

---

## 🏦 多 Bank 请求分发：每个 Bank 维护什么

`_simulate_per_bank_timeline` 是普通算子建模的核心。  
它不是“把所有请求塞进一个静态大公式”，而是一个 **per-bank 状态 + 全局闭环推进** 的离散事件模型。

### 1. 4-bank 抽象布局图

```mermaid
flowchart LR
    accTitle: 4-bank 控制器抽象布局图
    accDescr: 普通算子请求按 bank 分发到 4 个 bank，每个 bank 维护本地状态，控制器同时维护全局写队列占用和 slice 阻塞状态。

    subgraph ags["输入请求流"]
      ag0["AG0<br/>A stream"]
      ag1["AG1<br/>B / aux stream"]
      ag4["AG4<br/>writeback stream"]
    end

    subgraph ctrl["闭环控制器层"]
      release["release_due_requests()<br/>按 release_cycle 注入"]
      pending["slice_pending_writes<br/>等待进入 controller queue"]
      global["全局状态<br/>now / write_queue_occupancy<br/>slice_blocked / forced_drain_count"]
    end

    subgraph banks["每个 bank 一份本地状态"]
      b0["Bank0<br/>phase<br/>open_row<br/>cycles<br/>row_switch_count<br/>phase_switch_count"]
      b1["Bank1<br/>phase<br/>open_row<br/>cycles<br/>row_switch_count<br/>phase_switch_count"]
      b2["Bank2<br/>phase<br/>open_row<br/>cycles<br/>row_switch_count<br/>phase_switch_count"]
      b3["Bank3<br/>phase<br/>open_row<br/>cycles<br/>row_switch_count<br/>phase_switch_count"]
    end

    ag0 --> release
    ag1 --> release
    ag4 --> pending
    pending --> release
    release --> b0
    release --> b1
    release --> b2
    release --> b3
    global --- pending
    global --- b0
    global --- b1
    global --- b2
    global --- b3

    classDef stream fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef control fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef bank fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class ag0,ag1,ag4 stream
    class release,pending,global control
    class b0,b1,b2,b3 bank
```

### 2. 模型里“全局状态”和“bank 本地状态”的分工

```mermaid
flowchart TB
    accTitle: 全局状态与 Bank 本地状态分工
    accDescr: 闭环 bank timeline 模型中，全局控制器状态负责写回拥塞和时间推进，本地 bank 状态负责 row 和 phase 相关成本。

    subgraph global["全局控制器状态 _ClosedLoopControllerState"]
      g1["now"]
      g2["write_queue_occupancy"]
      g3["q_w_full_cycles"]
      g4["slice_blocked_cycles"]
      g5["forced_drain_count"]
      g6["rw_switch_count / row_switch_count"]
      g7["arbiter1_wins / arbiter2_wins"]
    end

    subgraph local["单 bank 状态 _BankTimelineState"]
      l1["cycles"]
      l2["phase"]
      l3["open_row"]
      l4["row_switch_count"]
      l5["phase_switch_count"]
      l6["read_request_count"]
      l7["write_request_count"]
    end

    classDef globalStyle fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef localStyle fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class g1,g2,g3,g4,g5,g6,g7 globalStyle
    class l1,l2,l3,l4,l5,l6,l7 localStyle
```

---

## 🔀 仲裁、切行、回压：闭环 Bank Timeline 到底怎么跑

这一层对应：

- `_bank_request_delta`
- `_apply_bank_request`
- `_simulate_per_bank_timeline`

### 1. 单 bank 的 row / phase 成本模型

这部分是切行和切换成本最核心的抽象：

- 初始第一次请求：
  - `request_latency_cycles`
- 同 `phase` 且同 `row`：
  - `bank_return_interval_cycles`
- 同 `phase` 但换 `row`：
  - `row_switch_penalty_cycles + bank_return_interval_cycles`
- `read <-> write` phase 切换：
  - 当前模型也记成 `row_switch_penalty_cycles + bank_return_interval_cycles`

```mermaid
stateDiagram-v2
    accTitle: 单 Bank Phase 和 Row 状态图
    accDescr: 单个 bank 维护当前 phase 和 open row，并根据命中、换行和读写切换计算不同的服务代价。

    [*] --> Empty
    Empty --> ReadRowX: 首次读请求\ncost = request_latency
    Empty --> WriteRowY: 首次写请求\ncost = request_latency

    ReadRowX --> ReadRowX: 同 row 继续读\ncost = bank_return_interval
    ReadRowX --> ReadRowY: 读内切行\ncost = row_switch + return_interval
    ReadRowX --> WriteRowY: 读写切换\ncost = row_switch + return_interval

    WriteRowY --> WriteRowY: 同 row 继续写\ncost = bank_return_interval
    WriteRowY --> WriteRowZ: 写内切行\ncost = row_switch + return_interval
    WriteRowY --> ReadRowX: 写读切换\ncost = row_switch + return_interval
```

### 2. 两级仲裁规则：为什么会偏好 continuation traffic

```mermaid
flowchart TD
    accTitle: 两级仲裁规则
    accDescr: 控制器优先选择同 phase 且同 row 的候选请求，只有一级仲裁没有命中时才考虑其余请求。

    ready["当前 bank ready queue"]
    split["按 bank_state.phase / open_row 分类"]
    a1["Arbiter-1<br/>同 phase + 同 row"]
    a2["Arbiter-2<br/>其余候选"]
    pick1["若 Arbiter-1 非空<br/>选择 request_id 最小者"]
    pick2["否则从 Arbiter-2 选<br/>request_id 最小者"]
    serve["服务该请求<br/>更新 phase / open_row / cycles"]

    ready --> split
    split --> a1 --> pick1 --> serve
    split --> a2 --> pick2 --> serve

    classDef queue fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef arb fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef action fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    class ready,split queue
    class a1,a2 arb
    class pick1,pick2,serve action
```

这就是为什么：

- 同一 row、同一 phase 的长串请求更容易持续被服务
- `RR` 连续命中很多时，`arbiter1_wins` 会很高
- 一旦 phase 或 row 频繁切换，`arbiter2_wins` 和切换惩罚就会升高

### 3. `RR` 切行和 `RW` 切换在模型里怎么出现

```mermaid
sequenceDiagram
    accTitle: RR 切行与 RW 切换的建模语义
    accDescr: 普通算子模型中，RR 切行和 RW 切换都表现为 bank_state 的 row 或 phase 变化，并由 _apply_bank_request 负责累计计数。

    participant Q as Ready Queue
    participant B as Bank State
    participant M as Metrics

    Q->>B: 取出下一条 read(row=4)
    B->>B: 当前也是 read, open_row=4
    B-->>M: 不增加 row_switch_count
    B-->>M: 不增加 phase_switch_count

    Q->>B: 下一条 read(row=5)
    B->>B: 同 phase, 但 row 变化
    B-->>M: row_switch_count += 1

    Q->>B: 下一条 write(row=9)
    B->>B: phase 从 read -> write
    B-->>M: phase_switch_count += 1
    B-->>M: 全局 rw_switch_count += 1
```

### 4. 写回回压为什么会反过来堵住读

这是普通算子模型和“纯静态 flat queue 模型”最大的区别之一。

```mermaid
sequenceDiagram
    accTitle: 写回回压传播路径
    accDescr: 当写队列满时，控制器停止接受新的写请求，slice 端写缓冲继续堆积，最终 slice_blocked 触发，读请求也会间接受阻。

    participant Slice as Slice 侧请求生成
    participant Pending as slice_pending_writes
    participant Ctrl as Controller write queue
    participant Bank as Bank service

    Slice->>Pending: 生成 write requests
    Pending->>Ctrl: 尝试注入 controller queue
    Ctrl-->>Pending: queue full / ready_w = 0
    Pending-->>Slice: 本地 write buffer 持续累积
    Slice-->>Slice: 达到 slice_write_buffer_depth
    Slice->>Slice: slice_blocked = 1
    Slice-->>Slice: 暂停进一步读流释放
    Bank->>Ctrl: 服务 write, occupancy 下降
    Ctrl-->>Pending: 恢复可注入
    Pending-->>Slice: 压力下降
    Slice->>Slice: 清除 slice_blocked
```

### 5. 回压状态机：什么时候进入 forced drain

```mermaid
stateDiagram-v2
    accTitle: 写回回压与 Forced Drain 状态机
    accDescr: 普通算子 bank timeline 中，写缓冲积压达到阈值会进入 slice_blocked，并在强制写排空期间优先服务写请求。

    [*] --> Normal
    Normal --> PendingGrow: 写请求持续到达
    PendingGrow --> SliceBlocked: len(slice_pending_writes) >= slice_write_buffer_depth
    SliceBlocked --> ForcedDrain: force_write_drain = true
    ForcedDrain --> Recovering: write_queue_occupancy 下降\nslice_pending_writes 下降
    Recovering --> Normal: slice_blocked 清零
```

---

## 📐 Latency 是怎么汇总出来的

普通算子不走 ring-gemm 专用重叠公式，它的 latency 汇总非常直接：

```text
latency = max(compute_bound, memory_timeline_bound, ag_issue_bound)
```

### 1. 三类 bound 对比图

```mermaid
flowchart LR
    accTitle: 普通算子 latency 汇总
    accDescr: 普通算子最终延迟由 compute bound、memory timeline bound 和 AG issue bound 三者取最大值得到。

    compute["compute_bound<br/>_estimate_compute"]
    memory["memory_timeline_bound<br/>_simulate_per_bank_timeline"]
    issue["ag_issue_bound<br/>max(issue_cycles across streams)"]
    maxop["latency = max(...)"]
    report["analytical_model.estimated_latency_cycles"]

    compute --> maxop
    memory --> maxop
    issue --> maxop
    maxop --> report

    classDef bound fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef result fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class compute,memory,issue,maxop bound
    class report result
```

### 2. `compute_bound` 怎么估计

普通算子的 compute work 由 `_estimate_compute` 给出：

- `prefill_summac / prefill_sum_rec / prefill_max`
  - 近似为 reduction work
- `remote_sum`
  - 当前用 `output_elements` 近似
- 其他普通算子
  - 当前基本按 `output_elements` 近似

然后：

```text
compute_bound = ceil(op_work / general_peak_ops_per_cycle)
```

### 3. `true_roofline` 和行为级 timeline 的区别

```mermaid
flowchart TB
    accTitle: True Roofline 与行为级 Timeline 的区别
    accDescr: Roofline 给出理论 compute/bandwidth 上界，而 bank timeline 给出考虑 bank 调度、切换和回压后的行为级 memory bound。

    roof["True Roofline<br/>只看总 ops / 总 bytes / 峰值带宽 / 峰值算力"]
    tl["Bank Timeline Bound<br/>看请求分布、bank 状态、row/phase 切换、写回回压"]
    compare["两者一起看：<br/>roofline 是理论上界<br/>timeline 是当前行为级实现的 memory 约束"]

    roof --> compare
    tl --> compare

    classDef roofStyle fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef tlStyle fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef cmp fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    class roof roofStyle
    class tl tlStyle
    class compare cmp
```

换句话说：

- `true_roofline`
  - 回答“如果只看总量，理论最好能多快”
- `bank_timeline`
  - 回答“考虑 bank 行为、切换、仲裁、写回压力后，这个普通算子在当前模型里会被 memory 约束到什么程度”

---

## 🧪 案例落地：以 `rmsnorm_mul_withbaseaddr` 为例

为了让上面的抽象图有落点，可以把 `rmsnorm_mul_withbaseaddr` 理解成下面这个请求图景：

```mermaid
flowchart LR
    accTitle: rmsnorm_mul_withbaseaddr 请求图景
    accDescr: 以广播式 element-wise mul 为例，请求从 A、B、writeback 三路进入普通算子性能模型，并在 bank timeline 中竞争服务。

    A["A stream<br/>大批量 read requests"]
    B["B stream<br/>较小向量 / 广播输入"]
    W["writeback stream<br/>输出写回 requests"]
    Timeline["普通算子闭环 bank timeline"]
    Metrics["输出指标<br/>rw_switch_count<br/>slice_blocked_cycles<br/>arbiter1_wins ..."]

    A --> Timeline
    B --> Timeline
    W --> Timeline
    Timeline --> Metrics

    classDef rd fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef wr fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef core fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    class A,B rd
    class W wr
    class Timeline,Metrics core
```

你在读这个 case 的 JSON 输出时，建议按这个顺序看：

1. `streams`
   - 每个 AG 流自己有多少请求、issue 多久
2. `bank_timeline`
   - 是否有大量 `rw_switch_count`
   - 是否有 `q_w_full_cycles / slice_blocked_cycles`
   - `arbiter1_wins` 是否远高于 `arbiter2_wins`
3. `analytical_model`
   - 到底是 `compute_bound`、`memory_timeline_bound` 还是 `ag_issue_bound` 在主导
4. `hardware_measured_cycles`
   - 看模型和硬件是总 latency 偏了，还是 trace 机制解释偏了

---

## 🗺️ 输出指标阅读图

下面这张图可以当成“拿到 performance.json 以后先看什么”的速查表。

```mermaid
flowchart TB
    accTitle: 普通算子输出指标阅读图
    accDescr: performance.json 中 bank_timeline 和 analytical_model 相关字段的物理意义与典型解释关系图。

    mem["memory_timeline_cycles<br/>闭环 bank 调度总耗时"]
    rw["rw_switch_count<br/>读写相位切换次数"]
    row["row_switch_count<br/>同 phase 内切行次数"]
    qfull["q_w_full_cycles<br/>controller write queue 满了多久"]
    blocked["slice_blocked_cycles<br/>slice 因写压而阻塞多久"]
    arb["arbiter1_wins / arbiter2_wins<br/>continuation traffic 偏好强不强"]
    hwcmp["hardware_measured_cycles 对比<br/>看总 latency 偏差还是 trace 机制偏差"]

    mem --> explain1["高：memory 主导明显"]
    rw --> explain2["高：读写交替频繁，phase switch 多"]
    row --> explain3["高：row locality 差，切行多"]
    qfull --> explain4["高：写队列常满，回压明显"]
    blocked --> explain5["高：写压力已经反向堵住读流"]
    arb --> explain6["Arbiter1 高：同 row 同 phase continuation 强"]
    hwcmp --> explain7["和模型一起读：定位是总量问题还是机制问题"]

    classDef metric fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef explain fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    class mem,rw,row,qfull,blocked,arb,hwcmp metric
    class explain1,explain2,explain3,explain4,explain5,explain6,explain7 explain
```

### 最后可以压缩成一句话

普通算子的性能分析不是一个单一公式，而是：

> **先把张量访问物化成带 `bank,row,col` 的请求，再用带两级仲裁和写回回压的闭环 bank timeline 计算 memory 行为，最后和 compute bound / AG issue bound 取最大值。**

---

## 📎 代码映射速查

| 主题 | 代码入口 |
| --- | --- |
| mode 级汇总 | `_analyze_mode` |
| 普通算子主入口 | `_analyze_op` |
| stream 统计 | `_build_stream_reports` |
| 单 bank 请求代价 | `_bank_request_delta` |
| bank 状态更新 | `_apply_bank_request` |
| 闭环 bank timeline | `_simulate_per_bank_timeline` |
| compute bound | `_estimate_compute` |
| true roofline | `_true_roofline_from_totals` |

这些图如果要继续往下细化，下一步最自然的扩展就是：

- 再单独画一份 `summac` / `mul` 的 trace 节奏对照图
- 或补一个“baseline vs remap_interleave 请求分布差异图”
