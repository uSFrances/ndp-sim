# ring_gemm M32N1792K896 28-slices 硬件 Trace 空泡分析

## 结论摘要

这次更新后的硬件 trace 中，slice completion 显示的总时延是 **15532 cycles**，而当前理论 compute roofline 仍然是 **14336 cycles**，两者相差：

```text
15532 - 14336 = 1196 cycles
```

按更新后的真实 `AG1/AG2 + ping/pong + row/col` 语义重建后，当前这 `1196 cycles` 可以拆成三部分：

1. **B-side event-ready lag：858 cycles（5.5% of total latency）**
2. **completion tail：326 cycles（2.1% of total latency）**
3. **剩余小额残差：12 cycles**

也就是：

```text
1196 = 858 + 326 + 12
```

其中最重要的结论是：

- 这组 case 的主要空泡依然来自 **B ping/pong event ready 节拍落后于理想供数节拍**
- 这次真正暴露给 PE 的 bubble **几乎全部落在 `ping`**
- 尾部还剩一段 `A-side + writeback` completion tail

---

## Trace 与配置对应关系

本次分析使用的 trace 目录：

- `golden/ring_gemm/M32N1792K896_28slices`

关键日志：

- `local_hub_req_bank0.log` / `local_hub_req_bank1.log`
- `local_hub_req_bank2.log` / `local_hub_req_bank3.log`
- `bank0_frame.log` / `bank1_frame.log`
- `bank2_frame.log` / `bank3_frame.log`

当前有效 slice 的起点在：

- `slice start (cycle=0)` 对应时间戳：`9736000 ns`

四个 bank 在有效 slice 内的统计如下：

| bank | read count | write count | last read cycle | last write cycle | read row transitions | slice completed |
|---|---:|---:|---:|---:|---:|---:|
| bank0 | 131 | 129 | 15312 | 15532 | 1 | 15532 |
| bank1 | 130 | 128 | 15314 | 15530 | 0 | 15532 |
| bank2 | 3639 | 0 | 15204 | - | 55 | 15532 |
| bank3 | 3639 | 0 | 15206 | - | 55 | 15532 |

这里最重要的四个时间点是：

1. `B-side` (`bank2/bank3`) 的最后一次读：**15206**
2. `A-side` (`bank0/bank1`) 的最后一次读：**15314**
3. `bank0/bank1` 的最后一次 writeback：**15532**
4. `slice completed`：**15532**

---

## 真实硬件语义：这次怎么定义 `B-event ready`

这次分析严格按真实硬件语义重建：

- `AG1` 负责填 `B ping buffer`
- `AG2` 负责填 `B pong buffer`
- 每个 AG 有两个通道，并且自己做 bank-interleave
- PE 阵列按：
  - `ping -> pong -> ping -> pong -> ...`
  - 的顺序交替消费

从 `local_hub_req_bank2/3.log` 对齐后的通道映射是：

- `AG1 / ping`
  - `bank2 -> ReqCh 2`
  - `bank3 -> ReqCh 3`
- `AG2 / pong`
  - `bank2 -> ReqCh 4`
  - `bank3 -> ReqCh 5`

通道请求数完全对称：

- `bank2`
  - `ReqCh 2`: `1792`
  - `ReqCh 4`: `1792`
- `bank3`
  - `ReqCh 3`: `1792`
  - `ReqCh 5`: `1792`

因此：

- `ping` event 数 = `448`
- `pong` event 数 = `448`
- 全局交替 B-event 数 = `896`

### 真实地址模式

按对齐后的真实 `row/col`，前几个 event 的边界是：

#### `AG1 / ping`

- `event0`: `bank2/3 row0 col 0,1,2,3`
- `event1`: `bank2/3 row0 col 8,9,10,11`
- `event2`: `bank2/3 row0 col 16,17,18,19`
- ...
- `event7`: `bank2/3 row0 col 56,57,58,59`
- `event8`: 切到 `row1 col 0,1,2,3`

#### `AG2 / pong`

- `event0`: `bank2/3 row0 col 8,9,10,11`
- `event1`: `bank2/3 row0 col 16,17,18,19`
- `event2`: `bank2/3 row0 col 24,25,26,27`
- ...
- `event7`: 切到 `row1 col 0,1,2,3`
- `event8`: `row1 col 8,9,10,11`

因此：

- 两条流都在共享 `bank2/bank3` 的服务槽位
- 但当前这版 trace 里，**真正拖慢 PE 的是 `ping` 这一侧**

### `B-event ready` 的定义

```text
ping_ready(k) = max(AG1@bank2 当前 4 条完成的最后时刻,
                    AG1@bank3 当前 4 条完成的最后时刻)

pong_ready(k) = max(AG2@bank2 当前 4 条完成的最后时刻,
                    AG2@bank3 当前 4 条完成的最后时刻)
```

然后 PE 看到的全局 B-event 序列是：

```text
ping0, pong0, ping1, pong1, ping2, pong2, ...
```

---

## 1. B-side event-ready lag：858 cycles

理想情况下：

- 全局交替 event 的理想节拍：`16 cycles / event`
- `896` 个全局 B-event 的最后一个理想 ready 时间是：

```text
ideal_last_ready = 28 + 16 * (896 - 1) = 14348
```

而 trace 重建出的最后一个全局 B-event 实际 ready 时间是：

```text
actual_last_ready = 15206
```

因此：

```text
B-side event-ready lag = 15206 - 14348 = 858 cycles
```

这说明：

- 即使 `AG1/AG2` 已经分别给 `ping/pong` 供数
- 即使 `bank2/bank3` 已经错开
- 即使每个 AG 自己做了 bank-interleave
- 下一次要算的 buffer 仍然没有做到严格按理想节拍 ready

### same-buffer ready gap 分布

如果只看同一个 buffer 流自己的 ready-to-ready 间隔：

#### `ping` gap 分布

- `34 cycles`：`279` 次
- `70 cycles`：`55` 次
- `24 cycles`：`55` 次
- `8 cycles`：`55` 次

#### `pong` gap 分布

- `34 cycles`：`334` 次
- `46 cycles`：`55` 次
- `22 cycles`：`55` 次

这说明：

1. **steady-state 已经不是理想 `32 cycles`**
   - 两条流大多数时候都是 `34`
   - 也就是同一个 buffer 的 ready-to-ready 节拍本身就慢了 `2 cycles`

2. **row boundary 会带来更大的慢拍**
   - `ping` 在 row-boundary 处常见 `70`
   - `pong` 在 row-boundary 处常见 `46`

3. **这次 row-boundary 更重的是 `ping`**
   - `70 - 32 = 38`
   - `46 - 32 = 14`

所以当前这版 trace 和昨天那版最大的区别是：

**这次是 `ping` 的 row-boundary 慢拍更重，最终也主要由 `ping` 侧把 PE 卡住。**

---

## 2. PE 真正看到的 bubble：为什么几乎都落在 `ping`

上面的 `858 cycles` 是最后 ready 节拍相对理想节拍的累计落后。  
如果你想回答：

- PE 阵列在真实执行时，**哪几步真的空等了**
- 每一步真正暴露给 PE 的 bubble 有多大

那要看的是：

```text
pe_next_available_cycle = previous_actual_compute_start + 16
actual_compute_start_cycle = max(pe_next_available_cycle, actual_ready_cycle)
exposed_bubble_cycles = max(actual_ready_cycle - pe_next_available_cycle, 0)
```

### bubble 分布

当前这组 `896` 个全局 event 的 PE-visible bubble 分布是：

| exposed bubble | count | 解释 |
|---:|---:|---|
| 0 | 841 | PE 轮到这一步时，buffer 已经 ready |
| 16 | 53 | PE 空出来后，还要再等整整一拍 |
| 20 | 1 | startup 的第一次等待 |
| 12 | 1 | 一次较小 stall |

并且：

- 总 PE-visible bubble 和：`880 cycles`
- 出现正 bubble 的 event 数：`55`
- 正 bubble 分布按 buffer 看：
  - `ping = 55`
  - `pong = 0`

这说明：

**这次真正暴露给 PE 的 stall，几乎全部落在 `ping` 上。**

### 最主要的模式：53 次完整 `16-cycle` stall

绝大多数正 bubble 是：

```text
actual_ready_cycle - pe_next_available_cycle = 16
```

也就是：

- PE 已经空出来了
- 但当前轮到的 `ping` 还差整整一个 coarse 的时间槽才 ready

第一批典型事件是：

- `global event 48 = ping24`
- `global event 64 = ping32`
- `global event 80 = ping40`
- `global event 96 = ping48`
- ...

可以看到，这些 stall 是高度周期性的：

- 从 `ping24` 开始
- 之后几乎每隔 `8` 个 `ping event`
- 就出现一次 `16-cycle` bubble

这和 `ping` 的 row-boundary 周期完全一致。

### startup bubble

还有一次启动气泡：

- `global event 0 = ping0`
- `pe_next_available = 28`
- `actual_ready = 48`

所以：

```text
48 - 28 = 20 cycles
```

另外还有一次较小 stall：

- `global event 32 = ping16`
- `pe_next_available = 560`
- `actual_ready = 572`

所以：

```text
572 - 560 = 12 cycles
```

---

## 3. `ping/pong` 的 8 请求 fill span：这次还是 `pong` 更拖

如果你关心的是：

- 一个 `ping` event 的 8 个请求，从第 0 个请求完成到第 7 个请求完成，跨了多久
- 对应的 `pong` 是不是更拖

那可以看：

```text
event_ready_span_cycles = last_request_cycle - first_request_cycle
```

当前这组 trace 的平均 span 是：

- `ping` 平均 span：`10.067 cycles`
- `pong` 平均 span：`24.955 cycles`

所以从 “8 个请求什么时候全部到齐” 的角度看：

**`pong` 的 fill 过程仍然系统性比 `ping` 更分散、更拖。**

这说明两件事可以同时成立：

1. `pong` 的 event 自身 fill span 更长
2. 但真正暴露给 PE 的 bubble 却主要是 `ping`

原因是：

- `pong` 虽然自己更拖
- 但很多时候它仍然会在 PE 轮到它之前 ready
- 而 `ping` 在某些 row-boundary 事件上，恰好会在 `pong -> ping` 切换点上迟到
- 所以 PE-visible bubble 主要落在 `ping`

---

## 4. completion tail：326 cycles

这次的 tail 可以从四个关键时间点直接读出来：

- 最后一个 `B-event ready`：`15206`
- `bank0/bank1` 最后一次 A-side 读：`15314`
- `bank0/bank1` 最后一次 writeback：`15532`
- `slice completed`：`15532`

因此：

```text
completion_tail = 15532 - 15206 = 326 cycles
```

这 `326 cycles` 还可以进一步拆开：

1. **B-side 已结束，但 A-side 还没结束**

```text
15314 - 15206 = 108 cycles
```

2. **A-side 结束后，bank0/bank1 还要继续 drain writeback**

```text
15532 - 15314 = 218 cycles
```

所以：

```text
326 = 108 + 218
```

这说明：

- 这次 tail 不是单一的 “最后 writeback 尾巴”
- 而是：
  - `B-side` 结束
  - `A-side` 还剩一点
  - 然后再排空 writeback

---

## 5. 剩余 12 cycles：小额 residual

把前面两大块都减掉之后，还剩：

```text
15532 - 14336 - 858 - 326 = 12 cycles
```

这 `12 cycles` 很小，可以理解成：

- 事件边界对齐误差
- startup/drain 的小额边界开销

---

## 6. 最终闭合账

这组更新后的 `28-slice` trace 总账可以闭合成：

```text
measured_total = ideal_compute + B_event_ready_lag + completion_tail + residual
```

代入数值：

```text
15532 = 14336 + 858 + 326 + 12
```

因此，**理论时间和实际时间的差距主要落在两处**：

1. **B-side event-ready lag：858 cycles**
   - 而且这次 PE-visible bubble 几乎都体现在 `ping`

2. **completion tail：326 cycles**
   - 包括 `B-side` 结束后 `A-side` 的剩余读，以及随后 `bank0/bank1` 的 writeback drain

---

## 建议给 PPT 的一句话

**相对 14336-cycle 的理论 compute roofline，更新后的 28-slice 硬件 trace 的 15532-cycle 实测主要多出两部分开销：B-side `ping/pong` event-ready 累计落后约 858 cycles，以及 `B-side` 结束后 `A-side + writeback` completion tail 约 326 cycles；剩余仅约 12 cycles 为小额边界开销。**
