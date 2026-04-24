# Bank Controller Arbitration and Backpressure Rules (for Cost Modeling)

## 1. Scope
This document summarizes the scheduling and backpressure behavior described for normal operators (e.g., `mul` in `rms_norm`) and translates it into modeling-ready rules.

## 2. Controller Architecture (Single Bank View)

### 2.1 Two-level arbitration
Per bank controller has two arbitration layers every cycle:

1. Arbiter-1: only considers requests that do **not** require row switch.
   - Condition: same row as previous served request.
   - And same request type as previous served request (read/read or write/write).
2. Arbiter-2: only considers requests that **require** row switch.

Priority rule:
- If Arbiter-1 outputs a valid winner, Arbiter-2 result is discarded in this cycle.
- Arbiter-2 is used only when Arbiter-1 has no valid candidate.

This creates a strict preference for row-buffer-friendly continuation traffic.

### 2.2 Write queue and ready behavior
- Write request queue depth is fixed: `K_w = 16`.
- If write queue is full, controller deasserts `ready` for write channel.
- Slice side then cannot send new write requests/data to this controller.

### 2.3 Backpressure propagation path
When controller write `ready=0`:
1. Slice cannot inject more writes to controller.
2. Slice local FIFOs / pipelines may still absorb traffic temporarily.
3. Once slice local FIFOs/pipelines are also full:
   - loop control is blocked;
   - further read generation is blocked as a secondary effect.
4. To break deadlock-like pressure buildup, controller must serve write requests (drain write queue), then restore progress.

This means write congestion can indirectly throttle read arrival rate.

## 3. Implications for Performance Modeling

### 3.1 Why a simple static bank model is inaccurate
A model that assumes read/write requests always compete in one flat queue misses:
- strict two-level arbitration priority;
- continuation bias (same-row + same-type) over switching requests;
- finite write queue overflow and upstream backpressure;
- closed-loop coupling where write stalls suppress future reads.

### 3.2 Required state variables (minimum)
For each bank/controller, track at least:
- previous served tuple: `(last_row, last_type)`;
- queue lengths by class:
  - `Q_hit_same` (eligible for Arbiter-1),
  - `Q_switch` (eligible for Arbiter-2),
  - `Q_w` (write queue, capacity 16);
- controller write ready flag `ready_w = (Q_w < 16)`;
- slice-side occupancy state (`fifo/pipeline used`), and a `slice_blocked` bit.

## 4. Recommended Cost Model: Hybrid Epoch Model

Use a cycle-accurate discrete-event simulator if possible. If analytical speed is required, use an **epoch-based hybrid model** with short windows (e.g., 32-256 cycles).

### 4.1 Per-epoch update flow
For each epoch:

1. Arrival estimation:
   - Estimate raw read/write generation from operator dataflow.
   - Apply upstream stall factor if `slice_blocked=1`.

2. Classify incoming requests:
   - Place into `Q_hit_same` if same row + same type as `(last_row,last_type)`.
   - Else place into `Q_switch`.
   - Writes additionally consume `Q_w` capacity.

3. Service decision (priority discipline):
   - Serve from `Q_hit_same` first when non-empty.
   - Else serve from `Q_switch`.
   - Update `(last_row,last_type)` after each service.

4. Timing/cost accounting:
   - row-hit service cost for continuation class.
   - row-miss / row-switch cost for switch class.
   - optional read/write turnaround penalties.

5. Backpressure propagation:
   - If `Q_w == 16`, set `ready_w=0`.
   - Track slice fifo/pipeline fill level; if saturated, set `slice_blocked=1`.
   - Under `slice_blocked=1`, reduce or stop new read generation.

6. Recovery:
   - When write drain reduces `Q_w < 16`, re-enable write injection.
   - As slice occupancy drops below threshold, clear `slice_blocked`.

### 4.2 Practical approximation equations
A lightweight approximation can use:

- Effective read arrival:
  `lambda_r_eff = lambda_r_raw * (1 - P_slice_blocked)`
- Effective service under strict priority:
  `mu_switch_eff ~= mu_switch * max(0, 1 - rho_hit_same)`
- Write-full probability (rough first-order):
  model `Q_w` as finite queue (`M/G/1/K`-like) to estimate `P(Q_w=16)`.

Then feed `P(Q_w=16)` into slice blocking estimator.

## 5. Validation Strategy

1. Build microbench traces with controllable row locality and read/write ratios.
2. Compare model predictions against simulator/hardware traces for:
   - throughput,
   - average read latency,
   - write queue occupancy distribution,
   - stall cycles at slice loop control.
3. Calibrate only a small set of constants:
   - row-hit latency,
   - row-switch latency,
   - turnaround penalty,
   - slice local buffering depth/effective threshold.

## 6. Suggested Research References

### 6.1 Memory controller scheduling (directly relevant)
- Rixner et al., “Memory Access Scheduling,” ISCA 2000.
- Mutlu and Moscibroda, “Parallelism-Aware Batch Scheduling (PAR-BS),” ISCA 2008.
- Kim et al., “ATLAS: A Scalable and High-Performance Scheduling Algorithm for Multiple Memory Controllers,” HPCA 2010.
- Subramanian et al., “BLISS: Balancing Performance, Fairness and Complexity in Memory Access Scheduling,” HPCA 2014.
- Kim et al., “A Case for Exploiting Subarray-Level Parallelism (SALP) in DRAM,” ISCA 2012.

### 6.2 Accelerator/dataflow cost-model methodology (modeling style reference)
- Chen et al., “Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs,” ISSCC 2016 / JSSC 2017.
- Sze et al., “Efficient Processing of Deep Neural Networks: A Tutorial and Survey,” Proc. IEEE 2017.
- Kwon et al., “MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings,” MICRO 2019.
- Parashar et al., “Timeloop: A Systematic Approach to DNN Accelerator Evaluation,” ISPASS 2019.

## 7. Recommended Next Implementation Step

Implement a small “bank service kernel” in your analytical model with explicit state transitions for:
- two-level arbitration priority,
- write-queue-full backpressure,
- slice-blocked feedback to read generation.

This kernel can then be reused for all ordinary operators (`mul`, `add`, etc.), with only traffic-generation parameters changed per operator.
