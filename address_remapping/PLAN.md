# Address Remapping Design Manual and Project Plan

This document is the long-lived design and maintenance manual for the project.

For day-to-day setup and commands, use:

- [README.md](/home/liudy/workspace/dev/projects/address_remapping/README.md)

This file focuses on:

- hardware assumptions
- mode semantics
- roofline and analytical model definitions
- trace and validation architecture
- current implementation state
- maintenance and extension rules

## 1. Project Summary

`address_remapping` is a Python project for:

- inferring 128-bit-granularity address remapping between producer and consumer tensor layouts
- analyzing operator-to-operator data movement cost under different hardware/layout policies
- exporting request traces for Ramulator validation
- comparing an analytical performance model against DRAM-side reference behavior

The project now supports three analysis modes:

1. `baseline`
2. `remap`
3. `remap_interleave`

Their meanings are fixed and should be interpreted as follows:

- `baseline`
  - no hardware layout-align/remap optimization is available
  - if two adjacent operators require different memory layouts, the system must insert a **software relayout stage**
  - that software relayout reads the tensor from memory, uses the `4x4` general array to reorder data, then writes the next operator’s required layout back to memory
- `remap`
  - hardware supports 128-bit-granularity data layout alignment and address remapping
  - no software relayout is inserted for those edges
- `remap_interleave`
  - same as `remap`, plus bank-aware interleaving to spread requests more evenly across banks

The project also computes a **true roofline** that is intentionally independent of mode.  
That roofline is based only on workload `ops/bytes` and hardware peak compute/bandwidth, not on request ordering or address mapping details.

## 2. Canonical Workspace

The canonical workspace is the WSL-local repo:

- [address_remapping](/home/liudy/workspace/dev/projects/address_remapping)

The Windows-mounted copy under `/mnt/h/...` is not the primary development workspace anymore.

## 3. Documentation Responsibilities

- [README.md](/home/liudy/workspace/dev/projects/address_remapping/README.md)
  - quick start
  - repository navigation
  - setup commands
  - common CLI usage
  - output locations
- [PLAN.md](/home/liudy/workspace/dev/projects/address_remapping/PLAN.md)
  - design truth
  - hardware model
  - semantic meaning of each analysis mode
  - validation and Ramulator integration model
  - current accepted implementation state

## 4. Repository Layout

### Inputs

- Graph examples:
  - [examples/graphs](/home/liudy/workspace/dev/projects/address_remapping/examples/graphs)
- Performance/hardware configs:
  - [examples/configs](/home/liudy/workspace/dev/projects/address_remapping/examples/configs)

### Outputs

- Solver outputs:
  - [outputs/solver](/home/liudy/workspace/dev/projects/address_remapping/outputs/solver)
- Performance + validation outputs:
  - [outputs/performance](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance)
- Test scratch outputs:
  - [outputs/tests](/home/liudy/workspace/dev/projects/address_remapping/outputs/tests)

### Main source files

- CLI:
  - [cli.py](/home/liudy/workspace/dev/projects/address_remapping/src/address_remapping/cli.py)
- Performance analysis:
  - [performance.py](/home/liudy/workspace/dev/projects/address_remapping/src/address_remapping/performance.py)
- Validation and Ramulator integration:
  - [validation.py](/home/liudy/workspace/dev/projects/address_remapping/src/address_remapping/validation.py)
- Solver:
  - [solver.py](/home/liudy/workspace/dev/projects/address_remapping/src/address_remapping/solver.py)
- Tests:
  - [test_solver.py](/home/liudy/workspace/dev/projects/address_remapping/tests/test_solver.py)

## 5. Hardware Model

### 5.1 Addressing / memory organization

The current model assumes:

- 1 slice = 4 banks
- slice frequency = `1GHz`
- memory/bank frequency = `500MHz`
- each bank bandwidth = `128 bit / bank-cycle`
- aggregate slice bandwidth = `4 * 128 bit / bank-cycle = 64 B / bank-cycle`
- converted to slice-cycle domain:
  - `64 B / bank-cycle @ 500MHz`
  - `32 B / slice-cycle @ 1GHz`

### 5.2 Timing domains

The model uses two clock domains:

- `bank-cycle`
  - used for DRAM timing parameters
  - `tRCD`, `tRP`, `tCL`, `tBL`
- `slice-cycle`
  - used for all externally reported latency/roofline/performance outputs
  - all memory timing is converted into this domain before reporting

The conversion is:

- `slice_cycles = bank_cycles * (slice_frequency_hz / memory_frequency_hz)`
- with current defaults:
  - `slice_cycles = bank_cycles * 2`

### 5.3 Compute arrays

#### GEMM specialized array

The GEMM compute model is a tensor-core style `m8n8k2` unit:

- input A tile: `8x2`
- input B tile: `2x8`
- output tile: `8x8`
- one cycle completes the full `m8n8k2` micro-tile
- total MAC lanes per cycle:
  - `8 * 8 * 2 = 128 MAC`
- project convention:
  - `1 MAC = 2 ops`
- therefore GEMM peak compute:
  - `256 ops / slice-cycle`

#### General array

The general-purpose array is `4x4` and is used for:

- non-GEMM compute
- software relayout stages in `baseline`

### 5.4 Current config fields

The active performance config includes:

- `hardware.address_space`
- `hardware.clocks`
- `hardware.dram`
- `hardware.compute`
- `hardware.ag_issue_rate`
- `performance.overlap`

The design intent is now:

- `HardwareSpec`
  - describes what the machine is
  - topology, frequencies, queues, timing, array shape, and derived peaks
- `PerformanceConfig`
  - describes how the analytical model estimates overlap
  - not the hardware resource itself

The loader now accepts only this structured format.
Legacy flat config fields are intentionally rejected.

See:

- [performance_config.json](/home/liudy/workspace/dev/projects/address_remapping/examples/configs/performance_config.json)

## 6. Performance Model

### 6.1 Two distinct layers

The project intentionally separates:

1. **True Roofline**
2. **Analytical Mode-Aware Model**

They are not the same.

### 6.1.1 Physical-first address generation

The accepted request-generation semantics are now:

- AG emits final physical memory requests
- every modeled request carries:
  - `logical_addr`
  - `base_addr`
  - address transform object `P`
  - `physical_addr`
- `slice_id / bank_id / row_id / col_id` are decoded from `physical_addr`

The address chain is:

- `logical_addr -> P -> base_addr -> physical_addr`

`P` is materialized as an explicit address transform object in the implementation and is exposed in:

- mode-level reports
- op-level reports
- request traces

All tensors used by the performance model, including intermediate tensors declared for model outputs, must provide explicit `base_addr`.
This project does not auto-allocate tensor addresses.

### 6.1.2 `ring_gemm` communication semantics

`ring_gemm_fp16_fp16_fp16` is not modeled as a plain two-input local-memory GEMM.

- `tensorA`
  - first tile is read from the local slice memory
  - remaining tiles are received from other slices over a single-direction ring
  - execution is modeled as a ping-pong pipeline across tiles:
    - local A into the first buffer
    - next remote A into the other buffer
    - compute and ring transfer overlap at tile granularity
- `tensorB`
  - remains a local memory input
- ring bandwidth
  - fixed at `256 bit / slice-cycle = 32 B / slice-cycle`
- `ring_scope`
  - `cluster` means `4` participating slices
  - `global` means `28` participating slices
- the current analytical model treats ring transfer and GEMM compute as fully overlapped
  - more specifically: overlap is expressed through a tile-level ping-pong pipeline, not a single bulk max model

### 6.2 True Roofline

`true_roofline` is:

- independent of `baseline/remap/remap_interleave`
- computed from the original operator graph only
- based on:
  - total workload ops
  - total workload bytes
  - peak compute capability
  - peak memory bandwidth

It does **not** include baseline-only software relayout overhead.

For GEMM:

- `work_ops = 2 * M * N * K`
- `peak_compute_ops_per_cycle = 256`
- `compute_bound_cycles = ceil((2MNK) / 256)`

For memory:

- the current top-level peak memory bandwidth is `32 B / slice-cycle`

The roofline summary reports:

- `work_ops`
- `total_bytes`
- `arithmetic_intensity_ops_per_byte`
- `compute_bound_cycles`
- `bandwidth_bound_cycles`
- `roofline_cycles`

### 6.3 Analytical mode-aware model

This is the request-order-aware analytical model used to estimate actual performance under each mode.

It includes:

- AG issue behavior
- bank distribution
- row hit/miss/empty behavior
- same-bank conflicts
- row-switch hiding under interleave
- overlap assumptions
- software relayout cost in `baseline`

The key reported fields are:

- `estimated_total_cycles`
- `compute_bound_cycles`
- `memory_access_bound_cycles`
- `ag_issue_bound_cycles`
- `lower_bound_cycles`
- `latency_to_lower_bound_ratio`

## 7. Mode Semantics

### `baseline`

The baseline is now the **software relayout baseline**.

Rules:

- if adjacent operators are already layout-compatible:
  - no software relayout is inserted
- if adjacent operators require incompatible layouts:
  - insert a `software_relayout` stage between them

This relayout stage:

- reads producer-layout data from memory
- performs reordering on the `4x4` general array
- writes consumer-layout data back to memory

This cost is part of:

- `modes.baseline.op_breakdown`
- `modes.baseline.total_latency_cycles`
- `mode_summaries.baseline`

### `remap`

The system assumes hardware 128-bit layout alignment + address remapping exists.

- no software relayout is inserted
- requests are materialized directly in the aligned target layout

### `remap_interleave`

Same as `remap`, but request placement is additionally spread across banks to hide row-switch penalties and increase effective parallelism.

## 8. Request / Trace Model

The project can materialize requests as `PhysicalRequest` records with:

- `request_id`
- `tensor_name`
- `edge_name`
- `ag_id`
- `role`
- `slice_id`
- `bank_id`
- `row_id`
- `col_id`

When `--emit-trace` is enabled, each mode can generate:

- request-level JSON trace
- Ramulator `LD/ST 0xADDR` trace
- Ramulator YAML config

In `baseline`, if software relayout is inserted, those extra read/write requests are also included in the emitted trace.

## 9. Validation Model

Validation has two layers:

### 9.1 Internal validation

Internal sanity checks include:

- `round_robin_hides_more_row_switch`
- `interleave_preserves_request_count`
- `same_bank_conflict_worse_than_cross_bank`

### 9.2 External reference validation

The project can run Ramulator 2.0 as an external DRAM-side reference.

Current WSL-native executable path:

- [ramulator2](/home/liudy/workspace/dev/projects/address_remapping/third_party/ramulator2/build-linux/ramulator2)

Validation outputs include:

- `validation_summary`
- `validation_overview`
- `reference_mode_summaries`
- `reference_results`
- `comparison_to_reference`
- `calibration_notes`

### Validation-readable summary

`validation_overview` is meant to be the first thing to read.  
It contains:

- internal validation pass/fail
- number of passed cases
- reference validation status
- best modeled mode
- best reference mode
- whether modeled/reference ordering matches
- confidence level
- cycle and timing domains
- whether baseline includes software relayout

## 10. JSON Output Structure

The main performance JSON now has a readable top-down structure:

1. `overview`
2. `mode_summaries`
3. `hardware`
4. `performance_config`
5. `graph_summary`
6. `true_roofline`
7. `modes`
8. `summary_markdown`
9. `validation` (if enabled)

### Readable summary fields

#### `overview`

Contains:

- graph name/size
- best mode by estimated latency
- mode ordering
- cycle domain
- memory timing domain
- baseline latency
- top-level true roofline cycles
- whether roofline is compute- or bandwidth-bound

#### `mode_summaries`

For each mode:

- estimated latency
- speedup vs baseline
- latency vs true roofline
- latency vs analytical lower bound
- compute/memory/AG bounds
- software relayout counts and bytes
- top latency contributors

#### `modes`

This is the full detailed breakdown:

- `total_latency_cycles`
- `analytical_model`
- `op_breakdown`
- `edge_breakdown`
- optional `request_trace`

`op_breakdown` may include:

- `kind = "op"`
- `kind = "relayout"`

## 11. Operational Workflow

The recommended operational path is:

1. Use the WSL-local repository as the only active workspace.
2. Set up the environment and Ramulator in WSL.
3. Run graph solve and performance analysis from WSL.
4. Export traces and validate against Ramulator from the same workspace.
5. Read results first through the JSON summary layers, then inspect detailed breakdowns only when needed.

The exact day-to-day commands are intentionally kept in:

- [README.md](/home/liudy/workspace/dev/projects/address_remapping/README.md)

## 12. Key Output Files

For `ring_gemm_bias`, the main outputs are under:

- [ring_gemm_bias](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias)

Important files:

- [ring_gemm_bias_performance.json](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance.json)
- [ring_gemm_bias_performance.md](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance.md)
- [ring_gemm_bias_performance_validation.json](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance_validation.json)
- [ring_gemm_bias_performance_validation.md](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance_validation.md)

## 13. Current Status

The current implementation state is:

- WSL-native Ramulator build is working
- `analyze-performance --validate` is working end-to-end
- true roofline and analytical model are separated
- validation has a readable summary layer
- output directories are reorganized
- baseline is modeled as a software-relayout baseline
- GEMM roofline now uses the `m8n8k2` tensor-core interpretation

## 14. Test / Acceptance Status

Current regression suite:

- `40` tests passing

Coverage includes:

- graph solving
- output path behavior
- true roofline invariance across modes
- validation summary generation
- WSL Ramulator executable detection
- software relayout insertion for mismatch cases
- GEMM tensor-core peak modeling

## 15. Design Principles

- Keep **true roofline** independent from mode.
- Keep **mode-aware analytical model** explicit and separate.
- Treat `baseline` as a realistic no-hardware-optimization path, not just a different bit ordering.
- Report all final latencies in **slice-cycle**.
- Keep DRAM timing parameters in **bank-cycle**.
- Preserve trace export and Ramulator validation compatibility for all modes.
- Keep [README.md](/home/liudy/workspace/dev/projects/address_remapping/README.md) short and operational.
- Keep [PLAN.md](/home/liudy/workspace/dev/projects/address_remapping/PLAN.md) as the design source of truth.
