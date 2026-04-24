# address_remapping

`address_remapping` is a Python library and CLI for:

- inferring 128-bit-granularity address remapping between producer and consumer tensor layouts
- analyzing operator-to-operator data movement cost under different hardware/layout policies
- exporting request traces for Ramulator validation
- comparing an analytical performance model against DRAM-side reference behavior

## Document split

This repository now keeps two long-lived documents with different purposes:

- [README.md](/home/liudy/workspace/dev/projects/address_remapping/README.md)
  - day-to-day usage
  - repository layout
  - setup commands
  - common CLI workflows
  - where outputs go
- [PLAN.md](/home/liudy/workspace/dev/projects/address_remapping/PLAN.md)
  - design manual
  - hardware assumptions
  - mode semantics
  - roofline and analytical model definitions
  - validation architecture
  - current implementation status and maintenance principles

If you are trying to run the project, start here.
If you are trying to understand why the model behaves this way, read `PLAN.md`.

## Canonical workspace

The canonical workspace is the WSL-local repository:

- [address_remapping](/home/liudy/workspace/dev/projects/address_remapping)

The Windows-mounted copy under `/mnt/h/...` is no longer the primary development workspace.

## Repository layout

### Inputs

- Graph examples:
  - [examples/graphs](/home/liudy/workspace/dev/projects/address_remapping/examples/graphs)
- Performance and hardware configs:
  - [examples/configs](/home/liudy/workspace/dev/projects/address_remapping/examples/configs)

The config split is now:

- `hardware`
  - machine resources and timing
  - address space
  - clocks
  - DRAM organization
  - compute arrays
  - AG issue rate
- `performance`
  - analytical-model assumptions only
  - currently overlap ratios such as read overlap and writeback overlap

Only the structured config format is supported now.
The older flat mixed format is no longer accepted.

### Outputs

- Solver outputs:
  - [outputs/solver](/home/liudy/workspace/dev/projects/address_remapping/outputs/solver)
- Performance and validation outputs:
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

## Input model

The recommended input format is:

- `shape_bindings`: symbolic problem sizes such as `M`, `K`
- `tensors`: graph tensors with explicit metadata, including required `base_addr`
- `model`: a compact list of statements like `out = op_name(arg0, arg1)`

Operator port layout and tensor memory dtype come from the internal op registry.

Each tensor participating in performance analysis must provide a stable `base_addr`.
Intermediate tensors produced inside `model` should also be predeclared in `tensors` with at least:

- `base_addr`

The request-generation chain is now:

- `logical_addr -> P -> base_addr -> physical_addr`

Where:

- `logical_addr`
  - AG-local logical request index
- `P`
  - explicit address transform object derived from solver remap results
- `physical_addr`
  - the final AG-issued memory request address used for trace export and Ramulator validation

For `ring_gemm_fp16_fp16_fp16`, performance analysis now models a dedicated ring2ring path:

- `A`
  - first comes from local memory on the current slice
  - then remaining A tiles arrive from other slices over a single-direction ring
  - local A and ring A are modeled as a tile-level ping-pong pipeline
- `B`
  - remains a local memory input
- ring bandwidth
  - `256 bit / slice-cycle = 32 B / slice-cycle`
- `ring_scope`
  - `cluster = 4 slices`
  - `global = 28 slices`

### Registry model

- `layout.dtype` means the memory dtype of the tensor stored in memory
- connected ops communicate through one fixed memory tensor dtype
- address remapping only cares about memory dtype and memory layout
- no separate `compute_dtype` is modeled in v1

### Example model file

```json
{
  "shape_bindings": {"M": 128, "K": 64},
  "tensors": {
    "x_fp32": {"dtype": "fp32", "shape": {"M": "M", "K": "K"}},
    "bias_fp32": {"dtype": "fp32", "shape": {"M": "M", "K": "K"}},
    "skip_fp32": {"dtype": "fp32", "shape": {"M": "M", "K": "K"}}
  },
  "model": [
    "act_fp32 = add_MN_N(x_fp32, bias_fp32)",
    "post_gemm_fp32 = gemm_local(act_fp32, skip_fp32)",
    "out_fp32 = add_MN_MN(post_gemm_fp32, skip_fp32)"
  ]
}
```

## Registered ops in v1

The built-in registry currently includes:

- `rmsnorm`
- `mul_MN_N`
- `gemm_local`
- `gemm_ring`
- `add_MN_N`
- `rope`
- `softmax`
- `add_MN_MN`
- `silu`
- `mul_MN_MN`

## Recommended workflow

Linux/WSL is the recommended path for Ramulator-backed validation.

### WSL setup

```bash
cd /home/liudy/workspace/dev/projects/address_remapping
bash scripts/setup_ramulator_wsl.sh
source .venv/bin/activate
```

The existing Windows helper at [setup_ramulator_windows.ps1](/home/liudy/workspace/dev/projects/address_remapping/scripts/setup_ramulator_windows.ps1) remains available for native Windows experiments, but WSL/Linux is the primary supported workflow for Ramulator execution.

## Debug workflow

The repository now includes WSL-oriented VS Code debug configuration under:

- [launch.json](/home/liudy/workspace/dev/projects/address_remapping/.vscode/launch.json)
- [settings.json](/home/liudy/workspace/dev/projects/address_remapping/.vscode/settings.json)
- [tasks.json](/home/liudy/workspace/dev/projects/address_remapping/.vscode/tasks.json)

Recommended way to debug:

1. Open the WSL-local repo in VS Code Remote WSL.
2. Make sure the selected interpreter is:
   - `/home/liudy/workspace/dev/projects/address_remapping/.venv/bin/python`
3. Set breakpoints in:
   - [cli.py](/home/liudy/workspace/dev/projects/address_remapping/src/address_remapping/cli.py)
   - [performance.py](/home/liudy/workspace/dev/projects/address_remapping/src/address_remapping/performance.py)
   - [validation.py](/home/liudy/workspace/dev/projects/address_remapping/src/address_remapping/validation.py)
4. Open the Run and Debug panel and choose one of:
   - `CLI: solve-graph (ring_gemm_bias)`
   - `CLI: analyze-performance`
   - `CLI: analyze-performance + validate`
   - `CLI: run-validation`
   - `Tests: current file`
   - `Tests: all`

The debug configs set `justMyCode = false`, so you can step through the project code without the debugger aggressively skipping frames.

## Common commands

### One-command runs via Makefile

You can keep repeatable CLI commands in [Makefile](Makefile) and run them with `make`.

```bash
make help
make perf-rmsnorm-mul
make roofline-rmsnorm-mul-all
make perf-and-roofline-rmsnorm-mul
```

These targets automatically:

- load conda from `/cluster/home/liudy/anaconda3/etc/profile.d/conda.sh`
- activate `conda base`
- set `PYTHONPATH=src`
- run the corresponding `address_remapping.cli` subcommand

### Run tests

```bash
python -m unittest discover -s tests -v
```

### Solve graph remap relations

```bash
python -m address_remapping.cli solve-graph \
  examples/graphs/ring_gemm_bias.json
```

If no explicit `--output` is provided, solver results go under:

- [outputs/solver](/home/liudy/workspace/dev/projects/address_remapping/outputs/solver)

### Run performance analysis

```bash
python -m address_remapping.cli analyze-performance \
  examples/graphs/ring_gemm_bias.json \
  --config examples/configs/performance_config.json
```

### Run performance analysis with trace export and validation

```bash
python -m address_remapping.cli analyze-performance \
  examples/graphs/ring_gemm_bias.json \
  --config examples/configs/performance_config.json \
  --emit-trace \
  --validate \
  --ramulator-root third_party/ramulator2
```

### Run standalone validation

```bash
python -m address_remapping.cli run-validation \
  examples/graphs/ring_gemm_bias.json \
  --output outputs/performance/ring_gemm_bias/ring_gemm_bias_performance.json \
  --config examples/configs/performance_config.json \
  --ramulator-root third_party/ramulator2
```

## Output layout

For a case such as `ring_gemm_bias`, the main performance outputs live under:

- [outputs/performance/ring_gemm_bias](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias)

Important files include:

- [ring_gemm_bias_performance.json](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance.json)
- [ring_gemm_bias_performance.md](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance.md)
- [ring_gemm_bias_performance_validation.json](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance_validation.json)
- [ring_gemm_bias_performance_validation.md](/home/liudy/workspace/dev/projects/address_remapping/outputs/performance/ring_gemm_bias/ring_gemm_bias_performance_validation.md)

When `--emit-trace` is enabled, the same case directory may also contain:

- `*_trace_<mode>.json`
- `*_ramulator_<mode>.trace`
- `*_ramulator_<mode>.yaml`
- `*.yaml.stdout.txt`
- `*.yaml.stderr.txt`

## How to read the performance JSON

Read the output in this order:

1. `overview`
2. `mode_summaries`
3. `true_roofline`
4. `validation.validation_overview`
5. `validation.reference_mode_summaries`
6. `modes` for full detail

The key idea is:

- `true_roofline`
  - mode-independent theoretical upper bound
- `mode_summaries`
  - concise per-mode analytical estimate
- `validation`
  - comparison against Ramulator-side reference behavior

The exact design semantics for `baseline`, `remap`, `remap_interleave`, software relayout, timing domains, and GEMM modeling are documented in:

- [PLAN.md](/home/liudy/workspace/dev/projects/address_remapping/PLAN.md)
