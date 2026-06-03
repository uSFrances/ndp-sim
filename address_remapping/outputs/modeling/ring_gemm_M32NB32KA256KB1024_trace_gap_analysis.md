# ring_gemm M32NB32KA256KB1024 硬件 Trace 空泡分析

## 结论摘要

这组硬件 trace 对应的实测总时延是 **9005 cycles**，而当前理论 compute roofline 是 **8192 cycles**，两者相差：

```text
9005 - 8192 = 813 cycles
```

按更新后的真实 `ping/pong + row/col` 语义重建，当前这 `813 cycles` 可以拆成三部分：

1. **B-side event-ready lag：475 cycles**
2. **尾部 writeback drain：308 cycles**
3. **剩余小额残差：30 cycles**

也就是：

```text
813 = 475 + 308 + 30
```

其中最重要的结论是：

- 这组 case 的主要空泡 **不是** A-side；
- 也 **不只是** “writeback 最后拖尾”；
- 最大头的损失来自 **B ping/pong event ready 节拍逐渐落后于理想供数节拍**。
- `pe_visible_bubble_table.csv` 说明：**真正暴露给 PE 的 stall 几乎全部发生在周期性的 `ping` event 上**
- `ping_pong_event_span_comparison.csv` 说明：**`pong` 的 8 请求 fill span 几乎总是更长，但大多数 `pong` 仍能在 PE 轮到它之前准备好**

---

## Trace 与配置对应关系

本次分析使用的 trace 目录：

- `golden/ring_gemm/M32NB32KA256KB1024`

关键日志：

- `local_hub_req_bank0.log` / `local_hub_req_bank1.log`
- `local_hub_req_bank2.log` / `local_hub_req_bank3.log`
- `bank0_frame.log` / `bank1_frame.log`
- `bank2_frame.log` / `bank3_frame.log`

当前有效 slice 的起点在：

- `slice start (cycle=0)` 对应时间戳：`12244000 ns`

从当前 trace 看，硬件实际的数据分布关系是：

- `A` 读主要落在 `bank0/bank1`
- `B` 读主要落在 `bank2/bank3`
- **writeback 实际是分散到 `bank0` 和 `bank1` 的**

也就是说，最终 output drain 并不是只打在 `bank0`。

从 `bank*_frame.log` 中看到的 slice 内请求统计如下：

| bank | read count | write count | last read cycle | last write cycle | row change count |
|---|---:|---:|---:|---:|---:|
| bank0 | 521 | 64 | 8785 | 9005 | 9 |
| bank1 | 520 | 63 | 8787 | 9003 | 8 |
| bank2 | 2112 | 0 | 8697 | - | 64 |
| bank3 | 2112 | 0 | 8697 | - | 64 |

这里最重要的两个事实是：

1. **B-side (`bank2/bank3`) 的最后一次读在 cycle 8697 结束**
2. **slice 最终在 cycle 9005 才完成**

因此，最后一定存在一个：

```text
9005 - 8697 = 308 cycles
```

的尾部 drain。

---

## 真实硬件语义：这次怎么定义 `B-event ready`

这次分析不再使用之前的“每个 bank 每 4 条请求一组”的粗分法，而是严格按你给出的真实硬件语义重建：

- `AG1` 负责填 `B ping buffer`
- `AG2` 负责填 `B pong buffer`
- 每个 AG 有两个通道，并且自己做 bank-interleave
- PE 阵列按：
  - `ping -> pong -> ping -> pong -> ...`
  - 的顺序交替消费

对当前这组 trace，对齐后的通道对应关系是：

- `AG1 / ping`
  - `bank2 -> ReqCh 2`
  - `bank3 -> ReqCh 3`
- `AG2 / pong`
  - `bank2 -> ReqCh 4`
  - `bank3 -> ReqCh 5`

### 真实地址模式

更新后的 `local_hub` 地址译码表明，B-side 的真实 event 边界是：

#### `AG1 / ping`

- `event0`: `bank2/3 col 0,1,2,3`
- `event1`: `bank2/3 col 8,9,10,11`
- `event2`: `bank2/3 col 16,17,18,19`
- ...
- `event7`: `bank2/3 col 56,57,58,59`
- `event8`: 切到下一行 `row1` 的 `col 0,1,2,3`

#### `AG2 / pong`

- `event0`: `bank2/3 col 4,5,6,7`
- `event1`: `bank2/3 col 12,13,14,15`
- `event2`: `bank2/3 col 20,21,22,23`
- ...
- `event7`: `bank2/3 col 60,61,62,63`
- `event8`: 切到下一行 `row1` 的 `col 4,5,6,7`

这和“一个 row 有 64 个 column、每个请求是 128 bit、两个 AG 在 row 内交错铺开”是一致的。

### `B-event ready` 的定义

因此，当前真正合理的定义是：

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

这一点很重要，因为：

- AG 想发出的顺序
- 和 bank 最终完成返回的顺序

并不完全一样；真正决定 PE 是否空等的，是 **buffer 被填满 ready 的真实 cycle**。

---

## 1. 尾部 writeback drain：308 cycles

这部分依然是最容易直接从 trace 上读出来的。

### 现象

- `bank2/bank3` 的最后一次读：`cycle = 8697`
- `bank0/bank1` 仍然继续写到：
  - `bank0 last write = 9005`
  - `bank1 last write = 9003`
- slice completion 标记：
  - `cycle = 9005`

因此：

```text
tail_drain = 9005 - 8697 = 308 cycles
```

### 含义

这说明：

- B-side 已经把最后一批原始读请求读完了
- 但 output writeback 还没有完全清空
- 最终 slice completion 被 `bank0/bank1` 的尾部写回拖到了 `9005`

### 结论

**尾部 writeback drain 占了额外 813 cycles 中的 308 cycles。**

换成比例：

```text
308 / 9005 ≈ 3.42%
308 / 813 ≈ 37.9%
```

所以 writeback 是显著因素，但它不是最大头。

---

## 2. B-side event-ready lag：475 cycles

这部分是本次最关键的空泡来源。

### 为什么这次不再直接用旧的 493

旧版 `493-cycle` 结论是基于更粗的 bank-side grouping。  
这次按更新后的真实 `ping/pong + row/col` 语义重建后，新的结果是：

```text
actual last B-event ready = 8679
ideal  last B-event ready = 8204
delta = 475 cycles
```

所以现在应以：

- **`475 cycles` = AG-aware / ping-pong-aware / row-col-aware 的 B-side event-ready lag**

作为主结论。

### 理想节拍是什么

这里要区分两层：

#### 全局交替 B-event

PE 按：

```text
ping, pong, ping, pong, ...
```

消费，因此**全局交替 event** 的理想节拍是：

```text
16 cycles / event
```

#### 同一个 buffer 流

但如果只看 `ping` 自己，或者只看 `pong` 自己：

- 同一个 `ping` 再次被消费，需要隔两次全局 coarse
- 所以同一个 buffer 的理想 ready-to-ready 间隔是：

```text
32 cycles / same-buffer event
```

### 真实 ready 序列

按更新后的真实 `row/col` 重建，前 12 个 `ping/pong ready` 分别是：

#### `ping`

| event | cols | actual ready |
|---|---|---:|
| 0 | `0,1,2,3` | 47 |
| 1 | `8,9,10,11` | 63 |
| 2 | `16,17,18,19` | 79 |
| 3 | `24,25,26,27` | 93 |
| 4 | `32,33,34,35` | 129 |
| 5 | `40,41,42,43` | 163 |
| 6 | `48,49,50,51` | 197 |
| 7 | `56,57,58,59` | 231 |
| 8 | `row1: 0,1,2,3` | 301 |
| 9 | `row1: 8,9,10,11` | 309 |
| 10 | `row1: 16,17,18,19` | 333 |
| 11 | `row1: 24,25,26,27` | 367 |

#### `pong`

| event | cols | actual ready |
|---|---|---:|
| 0 | `4,5,6,7` | 49 |
| 1 | `12,13,14,15` | 65 |
| 2 | `20,21,22,23` | 81 |
| 3 | `28,29,30,31` | 111 |
| 4 | `36,37,38,39` | 145 |
| 5 | `44,45,46,47` | 179 |
| 6 | `52,53,54,55` | 213 |
| 7 | `60,61,62,63` | 247 |
| 8 | `row1: 4,5,6,7` | 293 |
| 9 | `row1: 12,13,14,15` | 315 |
| 10 | `row1: 20,21,22,23` | 349 |
| 11 | `row1: 28,29,30,31` | 383 |

### 事件级 lag 的物理含义

这个 `475 cycles` 表示：

- 即使 `AG1/AG2` 已经分别给 `ping/pong` 供数
- 即使 `bank2/bank3` 已经错开
- 即使每个 AG 自己做了 bank interleave
- **下一次要算的 buffer 仍然没有做到严格按理想节拍 ready**

所以更准确的 headline 应写成：

- **B-side event-ready lag = 475 cycles**

而不是单纯一句：

- row-switch penalty
- 或 AG conflict

---

## 3. 这 475 cycles 是怎么逐步积出来的

### steady-state 已经不是理想 `32`

同一个 buffer 的相邻 ready gap 统计如下：

#### `ping` gap 分布

- `34 cycles`：`158` 次
- `70 cycles`：`31` 次
- `8 cycles`：`31` 次
- `24 cycles`：`31` 次

#### `pong` gap 分布

- `34 cycles`：`190` 次
- `46 cycles`：`31` 次
- `22 cycles`：`31` 次

这说明：

1. **平时就不是理想的 `32 cycles`**
   - 大多数时候是 `34`
   - 也就是 same-buffer steady-state 本身就慢了 `2 cycles`

2. **一到 row boundary 会出现更大的慢拍**
   - `ping` 会跳到 `70`
   - `pong` 会跳到 `46`

因此 lag 会不断累积。

### 为什么 `ping` 第一次切行是 `70`

看第一次切行前后：

#### `ping event 7`

- `bank2`: `(221, row0 col56), (225, col57), (229, col58), (231, col59)`
- `bank3`: `(221, row0 col56), (225, col57), (229, col58), (231, col59)`
- ready = `231`

#### `ping event 8`

- `bank2`: `(287, row1 col0), (291, col1), (295, col2), (299, col3)`
- `bank3`: `(289, row1 col0), (293, col1), (297, col2), (301, col3)`
- ready = `301`

所以：

```text
301 - 231 = 70 cycles
```

而理想 same-buffer gap 是 `32`，因此这次切行对应的**可见慢拍**是：

```text
70 - 32 = 38 cycles
```

### 为什么 `pong` 第一次切行是 `46`

同样看第一次切行前后：

#### `pong event 7`

- `bank2`: `(223, row0 col60), (227, col61), (243, col62), (245, col63)`
- `bank3`: `(223, row0 col60), (227, col61), (245, col62), (247, col63)`
- ready = `247`

#### `pong event 8`

- `bank2`: `(257, row1 col4), (269, col5), (289, col6), (293, col7)`
- `bank3`: `(257, row1 col4), (269, col5), (287, col6), (291, col7)`
- ready = `293`

所以：

```text
293 - 247 = 46 cycles
```

对应 same-buffer 的可见慢拍是：

```text
46 - 32 = 14 cycles
```

### 这说明了什么

现在可以非常明确地说：

- `ping` 和 `pong` 的切行慢拍并不一样
- `ping` 第一次切行更重，表现为 `+38`
- `pong` 第一次切行较轻，表现为 `+14`
- 这不是简单固定的 DRAM row-switch 参数
- 而是：
  - row-switch 恢复
- AG1/AG2 共享 `bank2/bank3` 槽位
- bank2/bank3 两边完成时间取 `max`
- 共同叠加后的事件级可见结果

---

## 4. PE 真正看到的 bubble：`pe_visible_bubble_table.csv`

上面的 `475 cycles` 是 **最终 ready 节拍相对理想节拍的累计落后**。  
但如果你想回答：

- PE 阵列在真实执行时，**哪几步真的空等了**
- 每一步真正暴露给 PE 的 bubble 有多大

那就应该看：

- `ring_gemm_M32NB32KA256KB1024_pe_visible_bubble_table.csv`

这张表的定义是：

```text
pe_next_available_cycle = previous_actual_compute_start + 16
actual_compute_start_cycle = max(pe_next_available_cycle, actual_ready_cycle)
exposed_bubble_cycles = max(actual_ready_cycle - pe_next_available_cycle, 0)
```

含义是：

- `pe_next_available_cycle`
  - PE 算完上一个 global event 后，最早空出来的时间
- `actual_ready_cycle`
  - 当前这个 `ping/pong` 真实 ready 的时间
- `exposed_bubble_cycles`
  - 当前这一步真正让 PE 额外空等了多少 cycles

### 统计分布

当前更新后的 trace 下，`512` 个全局 event 的 PE-visible bubble 分布是：

- `481` 次：`0`
- `29` 次：`16`
- `1` 次：`19`
- `1` 次：`14`

这意味着：

1. **大多数 event 并不会直接让 PE 空等**
   - `481` 次 bubble 为 `0`
   - 说明 buffer 在 PE 真正轮到它之前已经 ready

2. **最主要的可见 bubble 是 29 次完整的 `16-cycle` stall**
   - 这 `16` 不是 DRAM 参数
   - 而是：

```text
actual_ready_cycle - pe_next_available_cycle = 16
```

   - 也就是：
     - PE 已经空出来了
     - 但当前轮到的 `ping` 还差整整一个 coarse 的时间槽才 ready

3. **还有 1 次 startup bubble = 19**
   - 对应 `global event 0 = ping0`
   - `actual_ready(47) - pe_next_available(28) = 19`

4. **还有 1 次较小 bubble = 14**
   - 也是一个 `ping` 侧 row-boundary stall

### 29 次 `16-cycle` bubble 分别是谁

这 `29` 次全部发生在 `buffer = ping` 上，对应：

```text
global event:
48, 64, 80, 96, 112, 128, 144, 160, 176, 192,
208, 224, 240, 256, 272, 288, 304, 320, 336, 352,
368, 384, 400, 416, 432, 448, 464, 480, 496
```

对应的 `ping event` 编号是：

```text
ping24, ping32, ping40, ping48, ping56, ping64, ping72, ping80, ping88, ping96,
ping104, ping112, ping120, ping128, ping136, ping144, ping152, ping160, ping168, ping176,
ping184, ping192, ping200, ping208, ping216, ping224, ping232, ping240, ping248
```

这说明一个很强的规律：

- **周期性的 PE-visible bubble 基本都落在 `pong -> ping` 的切换点**
- `pong` 多数时候都能在 PE 轮到它之前 ready
- 真正卡住 PE 的是某些 row-boundary 之后的 `ping`

### 例子：为什么 `global48 = ping24` 会让 PE 等 16 cycles

在表里：

- `global47 = pong23`
  - `actual_compute_start = 813`
  - `actual_compute_end = 829`
- 所以：

```text
global48 的 pe_next_available_cycle = 829
```

而：

- `global48 = ping24`
  - `actual_ready_cycle = 845`

所以：

```text
exposed_bubble = 845 - 829 = 16 cycles
```

这就是这 `16-cycle` bubble 的来源：

- `pong23` 已经算完
- PE 在 `829` 就空出来了
- 但 `ping24` 要到 `845` 才真正填满
- 所以 PE 白等了一个 `16-cycle` 的 coarse 槽位

### 为什么 `pong24` 虽然也跨 row，却没有额外 bubble

`global49 = pong24` 这一行里：

- `actual_ready_cycle = 837`
- `global48 = ping24` 的计算区间是：
  - `845 -> 861`

因此等到 PE 真轮到 `pong24` 时：

- `pong24` 已经在 `837` 就 ready 了
- 提前等在 buffer 里

所以：

- `pong24` 自己虽然也跨 row
- 但它并没有额外让 PE 空等

这个现象非常重要：

**某个 event 自己 ready 得慢，不一定直接变成 PE bubble；只有当它慢到超过 `pe_next_available_cycle`，才会变成真实可见 stall。**

---

## 5. `ping_i` 和 `pong_i` 的 8 请求 fill span 对比：`ping_pong_event_span_comparison.csv`

如果你想看的是：

- 一个 `ping event` 从第 0 个请求完成到第 7 个请求完成一共花了多久
- 对应的 `pong event` 是不是更拖

那应该看：

- `ring_gemm_M32NB32KA256KB1024_b_event_request_span_table.csv`
- 以及配套的：
- `ring_gemm_M32NB32KA256KB1024_ping_pong_event_span_comparison.csv`

这里定义：

```text
event_ready_span_cycles = last_request_cycle - first_request_cycle
```

也就是：

- 一个 `ping/pong event`
- 它的 8 个请求从第一个完成到最后一个完成
- 横跨了多少 cycles

### 总体统计

把 `ping_i` 和对应的 `pong_i` 放在同一行比较后，结果是：

- 总共 `256` 组 `ping_i / pong_i`
- `pong` 更慢：`253` 组
- 一样快：`3` 组
- `ping` 更慢：`0` 组

这说明：

**几乎所有对应的 `ping_i / pong_i` 里，`pong_i` 的 8 请求 fill span 都比 `ping_i` 更长。**

### 典型前几组

前 20 组里最典型的模式是：

- `event0`: `ping 28`, `pong 28`, 差 `0`
- `event1`: `ping 28`, `pong 28`, 差 `0`
- `event2`: `ping 28`, `pong 28`, 差 `0`
- `event3`: `ping 20`, `pong 36`, 差 `16`
- `event4`: `ping 10`, `pong 24`, 差 `14`
- `event5`: `ping 10`, `pong 24`, 差 `14`
- `event6`: `ping 10`, `pong 24`, 差 `14`
- `event7`: `ping 10`, `pong 24`, 差 `14`
- `event8`: `ping 14`, `pong 36`, 差 `22`

这说明：

- 当 `ping` 填得比较快时
- 对应的 `pong` 往往确实更慢

### 为什么分到两个 bank 也不一定会很快 ready

“分到两个 bank”只说明：

- 有并行带宽潜力

但 event ready 看的是：

- **这 8 个请求里最慢的那个什么时候到**

所以只要：

- 两个 bank 的完成节拍不同步
- `AG1/AG2` 在 `bank2/bank3` 上共享服务槽位
- row boundary 让后半段请求变慢

那么：

- 即使 8 个请求分散到了两个 bank
- 最后一个请求仍然可能被拖后
- 整个 event 的 fill span 仍然会很长

当前这组 trace 的结果正是这样：

- `ping` 平均 span 明显更短
- `pong` 平均 span 明显更长
- 说明 `pong` 这条流的 8 请求完成过程系统性更分散、更拖

### 这和 PE-visible bubble 的关系

这里还要再区分一次：

- `pong` 的 fill span 几乎总是比 `ping` 更长
- 但 **PE-visible bubble 却主要出现在 `ping` 上**

原因是：

- `pong` 虽然慢，但很多时候它会在 PE 轮到它之前就先准备好
- `ping` 的某些 row-boundary 事件虽然未必比 `pong` span 更长很多
- 但它恰好落在 `pong -> ping` 的切换点上
- 所以更容易把 delay 直接暴露给 PE

因此可以把这两张表合起来理解：

- `ping_pong_event_span_comparison.csv`
  - 解释：为什么从 event fill 过程上看，`pong` 更拖
- `pe_visible_bubble_table.csv`
  - 解释：为什么真正暴露给 PE 的 stall 却主要落在 `ping`

---

## 6. 为什么明明有 AG1/AG2 + ping/pong，`B-event ready` 仍然会延迟

这里要把“有 overlap 能力”和“能否准时 ready”分开。

### ping/pong 解决了什么

`B ping/pong` 的作用是：

- 当前 coarse 在算的时候
- 下一批 B 可以提前装到另一个 buffer
- 从而允许 overlap

它解决的是：

- **能不能边算边准备下一批 B**

### ping/pong 没有保证什么

它并不保证：

- `AG1/AG2` 的 issue 节拍一定完全理想
- `bank2/bank3` 的完成节拍一定严格稳定
- 两个 AG 在共享 `bank2/bank3` 时一定不会互相影响
- `ping/pong` 一定能严格每 `16 cycles` 一个全局 event ready

从当前 trace 的真实完成序列可以直接看出：

- `bank2` 上 `ReqCh2` 和 `ReqCh4` 交替占用完成槽位
- `bank3` 上 `ReqCh3` 和 `ReqCh5` 也交替占用完成槽位

因此，对单个 AG 来说：

- 它拿不到某个 bank 的全部吞吐
- 它只能拿到该 bank 的一部分服务时隙

这就是为什么：

- 即使有 `AG1/AG2 + ping/pong`
- `B-event ready` 仍然会慢

更准确的说法是：

**AG1/ping 和 AG2/pong 在 `bank2/bank3` 上共享服务节拍；steady-state 就已经把 same-buffer event 节拍从理想 `32` 拉成常见的 `34`，而 row boundary 处又进一步拉成长慢拍，最终表现成 `475-cycle` 的 B-side event-ready lag。**

---

## 7. 剩余 30 cycles：小额 residual

上面两部分已经解释了：

```text
475 + 308 = 783 cycles
```

而总额外开销是：

```text
813 cycles
```

因此还剩：

```text
813 - 783 = 30 cycles
```

这 `30 cycles` 目前更适合保守地看作：

- 小额边界性开销
- 包括：
  - 最后一个 AG-aware `B-event ready` (`8679`)
  - 到原始 `bank2/bank3` 最后一条读完成 (`8697`) 之间的 `18 cycles`
  - 再加上约 `12 cycles` 的 ring/boundary-like 小额残差

也就是说：

```text
30 ≈ 18 + 12
```

这部分不是主因，但它把总账补齐了。

---

## 8. 最终闭合账

把这次硬件实测按更新后的真实 `ping/pong + row/col` 语义收口，就是：

```text
9005
= 8192                       # ideal compute roofline
+ 475                        # B-side event-ready lag (updated AG-aware result)
+ 308                        # final writeback drain tail
+ 30                         # small residual / boundary-like overhead
```

也就是：

```text
9005 = 8192 + 475 + 308 + 30
```

---

## 建议给 PPT 的一句话

如果你想把这次 10% 左右的空泡写成一句 PPT 结论，建议这样写：

**相对 8192-cycle 的理论 compute roofline，9005-cycle 的硬件实测主要多了两部分开销：一部分是按真实 `ping/pong + row/col` 语义重建后的 B-side event-ready 节拍累计落后约 475 cycles，另一部分是尾部 writeback drain 约 308 cycles；剩余约 30 cycles 为小额边界性开销。**

如果想更短一点：

**The 9005-cycle hardware runtime is mainly explained by 475 cycles of AG-aware B-side event-ready lag and a 308-cycle final writeback tail, with only ~30 cycles of residual boundary overhead.**

---

## 备注

1. 这份分析现在采用的是**更新后的真实硬件语义**：
   - `AG1 -> ping`
   - `AG2 -> pong`
   - 每个 AG 两个通道
   - event 边界按真实 `row/col` 重建
2. 这份分析的主口径是：
   - `B-event ready` 的最终可见滞后
   - 而不是 bank-local 切行恢复的简单求和
3. `row switch` 的影响是明确存在的，但它不是在 `row change` 标记点瞬间以固定大 gap 出现，而是体现在：
   - row boundary 前后
   - 同一个 `ping/pong event` 的 ready gap 被明显拉长
4. 因此当前最保守也最准确的说法是：
   - `475 cycles` 是 **更新后的 B-side event-ready lag**
   - 其中 row-switch recovery 是重要组成部分
   - 同时还叠加了 AG1/AG2 共享 `bank2/bank3` 服务节拍带来的 steady-state 慢拍
