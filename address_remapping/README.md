# address_remapping

`address_remapping` is a Python toolkit for tensor address remapping and performance analysis.

It is used to:

- solve address-bit remapping between producer and consumer tensor layouts
- backfill external graph JSONs with hardware-facing remap configuration
- model DRAM/bank-interleave-aware data movement
- generate request streams and Ramulator traces
- compare layer0 measured cycles against roofline estimates
- project model-level TTFT from a target model config

## Quick Start

From the repository root:

```powershell
make help
make roofline-layer0
```

If native Windows PowerShell does not have `make`, use the wrapper command shown in
[Default Layer0 Report](#default-layer0-report), or run the same Makefile targets from
Git Bash, MSYS2, or WSL.

The standard layer0 measured-vs-roofline report is written to:

```text
outputs/roofline_only/deepseek1.5b/layer0_0630_roofline_vs_measured.json
outputs/roofline_only/deepseek1.5b/layer0_0630_roofline_vs_measured.csv
outputs/roofline_only/deepseek1.5b/layer0_0630_summary_tables.md
```

To estimate TTFT for another model config while reusing the same layer0 template and measured efficiency:

```powershell
make roofline-layer0-model MODEL_CONFIG=examples/configs/llama3_8b_config.json MODEL_NAME=llama3_8b
```

To generate the smallsize model's own report:

```powershell
make roofline-layer0-smallsize
```

To override sequence length without editing the model JSON:

```powershell
make roofline-layer0-model MODEL_CONFIG=examples/configs/llama3_8b_config.json MODEL_NAME=llama3_8b SEQUENCE_LENGTH=2048
```

## Three Different Inputs

The performance report uses three different inputs. They are easy to confuse.

### 1. Layer Graph Template

Example:

```text
examples/graphs/layer0/layer0_0630.json
```

This describes the operator topology and symbolic shapes for one layer:

- op ids such as `op5`, `op22`, `op41`
- op types such as `prefill_gemm_ring_4slice`
- input/output shape expressions
- dependency topology

It is not the target model config. For model-level TTFT, this file is used as the per-layer operator template.

For the current default report, `layer0_0630.json` corresponds to the smallsize model/template used for hardware
measurement and calibration. It provides measured operator cycles and layer topology, not the DeepSeek 1.5B model
dimensions.

### 2. Hardware/Performance Config

Example:

```text
examples/configs/performance_config.json
```

This contains hardware and analytical-model assumptions:

- compute peak
- local memory bandwidth
- DRAM/bank settings
- solver bank-interleave settings
- performance overlap assumptions

### 3. Target Model Config

Example:

```text
examples/configs/config.json
examples/configs/llama3_8b_config.json
```

`examples/configs/config.json` is the default DeepSeek 1.5B target-model config and includes
`"model_name": "deepseek1.5b"`. `examples/configs/smallsize_config.json` records the smallsize model dimensions
used by the current layer0 measured template.

This contains model dimensions:

- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `num_hidden_layers`
- `sequence_length`
- `slice_per_head`
- `used_slices`
- `kv_padding`

The model-level TTFT path recomputes per-op work/bytes from this file, then projects cycles using current layer0 measured utilization/bandwidth and roofline assumptions.

## Recommended Makefile Commands

### Default Layer0 Report

```powershell
make roofline-layer0
```

Equivalent long command:

```powershell
python scripts\compare_layer0_roofline_vs_measured.py examples\graphs\layer0\layer0_0630.json `
  --config examples\configs\performance_config.json `
  --measured golden\layer0\op_cycle_summary.json `
  --model-config examples\configs\config.json `
  --model-name deepseek1.5b `
  --frequency-mhz 800 `
  --sequence-multiple 32
```

### Swap Model Config

```powershell
make roofline-layer0-model MODEL_CONFIG=examples/configs/llama3_8b_config.json MODEL_NAME=llama3_8b
```

### Override Sequence Length

```powershell
make roofline-layer0-model MODEL_CONFIG=examples/configs/llama3_8b_config.json MODEL_NAME=llama3_8b SEQUENCE_LENGTH=2048
```

### Change Frequency

```powershell
make roofline-layer0-model MODEL_CONFIG=examples/configs/config.json FREQUENCY_MHZ=1000
```

### Generic Measured-vs-Roofline Report

```powershell
make roofline-vs-measured GRAPH=examples/graphs/layer0/layer0_0630.json
```

Useful variables:

```text
GRAPH              layer graph/template
LAYER0_GRAPH       default layer0 graph for roofline-layer0
CONFIG             hardware/performance config
MODEL_CONFIG       target model config for model-scaled TTFT
MODEL_NAME         target model name used in output filenames
MEASURED           measured per-op cycle summary
FREQUENCY_MHZ      TTFT frequency
SEQUENCE_LENGTH    optional sequence_length override
SEQUENCE_MULTIPLE  sequence rounding multiple
ROOFLINE_OUT_PREFIX output path prefix
```

By default the Makefile writes each model into its own output directory:

```text
outputs/roofline_only/deepseek1.5b/layer0_0630_summary_tables.md
outputs/roofline_only/smallsize/layer0_0630_summary_tables.md
```

If you use a different model config, pass a matching `MODEL_NAME` so outputs are easy to distinguish:

```powershell
make roofline-layer0-model MODEL_CONFIG=examples/configs/llama3_8b_config.json MODEL_NAME=llama3_8b
```

## Python Entrypoints

The stable user-facing wrapper is:

```powershell
python scripts\compare_layer0_roofline_vs_measured.py ...
```

The implementation lives in the modular package:

```powershell
python -m performance.report ...
```

When running `python -m performance.report` directly, make sure `src` is on `PYTHONPATH` unless the package is installed:

```powershell
$env:PYTHONPATH='src'
python -m performance.report examples\graphs\layer0\layer0_0630.json --config examples\configs\performance_config.json --measured golden\layer0\op_cycle_summary.json --model-config examples\configs\config.json
```

The wrapper script automatically inserts `src` into `sys.path`, so it is usually the safer command to use manually.

## Model-Scaled TTFT Semantics

The report contains two related but different notions:

- Template layer0 measured/roofline:
  - uses the exact graph/template shapes and measured op cycles
  - useful for understanding the current layer0 run
- Model-scaled TTFT:
  - reads `--model-config`
  - recomputes each op's work/bytes for that model
  - applies current layer0 measured compute utilization or measured effective bandwidth
  - multiplies per-layer projected cycles by `num_hidden_layers`
  - converts cycles to milliseconds using `--frequency-mhz`

Formula:

```text
TTFT_ms = per_layer_cycles * num_hidden_layers / frequency_hz * 1000
```

The JSON output includes:

```text
model_scaled_ttft_summary
```

The Markdown output includes:

- `Model Parameters`
- `Summary Metrics`
- `Model-Scaled Summary`
- `Model-Scaled Operator Projection`
- `Model-Scaled GEMM Operators`
- `Model-Scaled non-GEMM Operators`
- model-scaled rows for:
  - per-layer cycles
  - total cycles
  - TTFT
  - measured projection
  - projected measured with centralized global remote-sum
  - projected measured with Ring2Ring remote-sum
  - AXI pull roofline
  - centralized global roofline
  - Ring2Ring n2n roofline

The Markdown report is model-centric: it does not include smallsize calibration operator tables. The calibration data
is still preserved in JSON fields such as `operators` for traceability, while `model_scaled_ttft_summary.operators`
contains the target-model per-op projection.

## Source Layout

Important files:

```text
src/address_remapping/solver.py          remap solver
src/address_remapping/rmsnorm_bridge.py  external graph normalization and remap backfill
src/address_remapping/cli.py             address-remap CLI
src/address_remapping/performance.py     address-level performance model
src/address_remapping/addressing.py      address transform helpers
src/address_remapping/json_format.py     compact JSON formatter

src/performance/model_config.py          model config and execution shape derivation
src/performance/roofline.py              measured-vs-roofline helper exports
src/performance/ttft.py                  model-scaled TTFT projection
src/performance/report.py                Markdown/CSV/JSON report CLI

scripts/compare_layer0_roofline_vs_measured.py  compatibility wrapper
tests/test_solver.py                            regression tests
```

## Core Remap Semantics

Exported/backfilled remap lists use hardware-facing direction:

```text
remapping[new_bit] = old_bit
```

The solver distinguishes:

- `P_layout`: layout alignment between producer and consumer
- `P_physical`: physical field placement for bank interleave
- `P_out`: producer-side writeback remap, normally `P_physical compose P_layout`
- `P_in`: consumer-side read remap, normally `P_physical`

External tensors are assumed to already match the consumer layout, so external inputs only receive physical remapping.

Leaf outputs with no downstream consumer still honor `solver.bank_interleave[op_type].out`.

For `prefill_gemm_ring_4slice`, `prefill_gemm_local`, and `prefill_gemm_local_qkt`, auxiliary input `B'` mirrors `B`.

Remote-sum external inputs have a table-driven default remap in `rmsnorm_bridge.py`; when bank interleave is enabled, the final input remap composes the default remote remap with physical placement.

## Common Solver Commands

Solve/backfill a graph:

```powershell
make fill-remapping GRAPH=examples/graphs/layer0/layer0_0630.json
```

Force bank interleave:

```powershell
make fill-remapping-bank2 GRAPH=examples/graphs/layer0/layer0_0630.json
```

Direct CLI equivalent:

```powershell
python -m address_remapping.cli fill-remapping examples\graphs\layer0\layer0_0630.json `
  --config examples\configs\performance_config.json `
  --output outputs\solver\layer0_0630\layer0_0630_remapped.json `
  --dump-solver-results outputs\solver\layer0_0630\layer0_0630_solver_results.json
```

## Testing

Run the core regression tests:

```powershell
python -m unittest discover -s tests -v
```

For performance-report changes, also run:

```powershell
python -m py_compile scripts\compare_layer0_roofline_vs_measured.py src\performance\*.py
make roofline-layer0
```

Then inspect:

```text
outputs/roofline_only/deepseek1.5b/layer0_0630_summary_tables.md
outputs/roofline_only/deepseek1.5b/layer0_0630_roofline_vs_measured.json
outputs/roofline_only/smallsize/layer0_0630_summary_tables.md
```

## Maintenance Notes

- Keep exported remaps in `new_bit -> old_bit` direction.
- If backfilled JSON changes, regenerate representative outputs under `outputs/solver`.
- If report semantics change, regenerate representative outputs under `outputs/roofline_only`.
- Prefer adding repeatable workflows to `Makefile` instead of relying on long commands in notes.
- Keep long design explanations in `PLAN.md`; keep README focused on how to run and interpret the current repo.
