# AGENTS.md

This file is the project entry guide for coding agents and human maintainers working in this repository.

## Do Agents Need `AGENTS.md`?

No. Codex does not require an `AGENTS.md` file to work on a repository.

But if an `AGENTS.md` file exists at the project root, it is a good place to record:

- project-specific conventions
- key architectural assumptions
- preferred workflows
- things that are easy to get wrong

For this repository, that is useful, because `address_remapping` has several non-obvious remap semantics and backfill rules.

## Repository Purpose

`address_remapping` is used to:

- solve tensor address remapping between operator layouts
- add DRAM-aware physical placement such as bank interleave
- backfill external graph JSONs with hardware-facing remap configuration
- generate performance-model request streams
- export Ramulator traces and compare analytical vs reference behavior

## Most Important Files

- [README.md](README.md)
  - usage-oriented project overview
- [PLAN.md](PLAN.md)
  - longer design and modeling document
- [src/address_remapping/solver.py](src/address_remapping/solver.py)
  - remap solver
- [src/address_remapping/rmsnorm_bridge.py](src/address_remapping/rmsnorm_bridge.py)
  - external graph normalization and remap backfill
- [src/address_remapping/cli.py](src/address_remapping/cli.py)
  - CLI entry points
- [src/address_remapping/performance.py](src/address_remapping/performance.py)
  - performance modeling
- [src/address_remapping/addressing.py](src/address_remapping/addressing.py)
  - address transform application
- [src/address_remapping/json_format.py](src/address_remapping/json_format.py)
  - pretty JSON renderer with compact one-line remap/permutation lists
- [examples/configs/performance_config.json](examples/configs/performance_config.json)
  - hardware/performance/solver config
- [examples/graphs](examples/graphs)
  - example graph inputs
- [tests/test_solver.py](tests/test_solver.py)
  - solver, bridge, CLI, and many regression tests

## Core Remap Semantics

### Hardware-Facing Remap Direction

Exported and backfilled remap lists use hardware semantics:

- `remapping[new_bit] = old_bit`

Example:

- `remapping = [0, 5, 6, 1, 2, 3, 4, ...]`

means:

- `new_addr[0] = old_addr[0]`
- `new_addr[1] = old_addr[5]`
- `new_addr[2] = old_addr[6]`
- `new_addr[3] = old_addr[1]`

Internal runtime helpers may still use inverse mappings; do not assume internal `permutation` storage and exported JSON have the same direction unless you verify the field.

### Two-Stage Remap Model

The solver distinguishes:

- `P_layout`
  - layout alignment between producer and consumer
- `P_physical`
  - physical field placement for bank interleave
- `P_out`
  - producer-side writeback remap
  - normally `P_physical ∘ P_layout`
- `P_in`
  - consumer-side read remap
  - normally `P_physical`

### External Input Rule

External tensors are assumed to already match the consumer layout.

So for external inputs:

- no layout solve is done
- only `P_physical` is solved
- backfilled input remap is physical-only

### Terminal Output Rule

Leaf outputs with no downstream consumer must still honor `solver.bank_interleave[op_type].out`.

For these outputs:

- no layout alignment is needed
- output remap is physical-only

### Auxiliary `B'` Rule

For:

- `prefill_gemm_ring_4slice`
- `prefill_gemm_local`
- `prefill_gemm_local_qkt`

the auxiliary input `B'` must mirror `B`:

- same `remapping`
- same `bank_interleave`

### Remote-Sum Default Input Remap

These external op types have a built-in default remap on input `A` before bank interleave is applied:

- `prefill_remote_sum_fp32MN_fp32MN`
- `prefill_remote_sum_4slice_fp32MN_fp32MN`

Default remap:

- `[0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 1, 2, 3, 4, 5]`

This is used to move original address bits `1..5` into the slice field for remote access.

When bank interleave is enabled, final input remap is:

- `P_final = P_default_remote ∘ P_physical`

This logic is table-driven in `rmsnorm_bridge.py`.

## Solver Config Semantics

`examples/configs/performance_config.json` contains:

- `hardware`
- `performance`
- `solver`

`solver.bank_interleave` is keyed by canonical operator type and port:

```json
"solver": {
  "bank_interleave": {
    "ring_gemm_fp16_fp16_fp16": {
      "inA": 2,
      "inB": 2,
      "out": 2
    }
  }
}
```

Important:

- only `1 / 2 / 4` are currently supported
- unspecified ports default to `1`
- backfill should use canonical op type, not external alias, when looking up values

## Backfilled JSON Conventions

Backfilled graph/operator JSON now includes:

- final `remapping`
- `bank_interleave`

For readability:

- `bank_interleave` is inserted before `shape`
- `remapping` is rendered on one line

The backfilled graph JSON intentionally does not keep all intermediate matrices and permutations.

Those stay in solver result outputs.

## Where Intermediate Solver Details Live

If you need to inspect `P_layout`, `P_physical`, and `P_total`, look at solver result JSON, not the remapped graph JSON.

Typical outputs:

- `outputs/solver/..._remapped.json`
  - final configuration for graph/operator payloads
- `outputs/solver/..._solver_results.json`
  - detailed solver artifacts for inspection

## Common CLI Workflows

### Solve Graph

```bash
python -m address_remapping.cli solve-graph examples/graphs/ring_gemm/ring_gemm_bias.json --config examples/configs/performance_config.json
```

### Fill Remapping

```bash
python -m address_remapping.cli fill-remapping examples/graphs/layer0/layer0_padding_0529.json --config examples/configs/performance_config.json --output outputs/solver/layer0_bank2_interleave/layer0_padding_bankinterleave2_remapped.json --dump-solver-results outputs/solver/layer0_bank2_interleave/layer0_padding_bankinterleave2_solver_results.json
```

### Analyze Performance

```bash
python -m address_remapping.cli analyze-performance examples/graphs/ring_gemm/ring_gemm_bias.json --config examples/configs/performance_config.json
```

## Testing Guidance

Before closing a solver/bridge change, prefer targeted regressions in:

- [tests/test_solver.py](tests/test_solver.py)

Especially re-check:

- remap direction
- external input physical-only remap
- `B'` mirrors `B`
- remote-sum default input remap
- leaf output remap from `out`
- `bank_interleave` backfill fields

## Editing Guidance

- Keep hardware-facing exported remaps as `new_bit -> old_bit`
- Do not silently flip direction in one layer only
- If a change affects backfilled JSON, regenerate representative outputs under `outputs/solver`
- If a change affects formatting, use `json_format.py` rather than one-off regexes in multiple files

