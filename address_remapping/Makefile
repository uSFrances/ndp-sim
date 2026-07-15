PYTHON ?= python
export PYTHONPATH := src
GRAPH ?= examples/graphs/layer0/layer0.json
CONFIG ?= examples/configs/performance_config.json
GRAPH_STEM := $(basename $(notdir $(GRAPH)))
OUT_DIR ?= outputs/performance/$(GRAPH_STEM)
SOLVER_DIR ?= outputs/solver/$(GRAPH_STEM)
REMAP_OUTPUT ?= $(SOLVER_DIR)/$(GRAPH_STEM)_remapped.json
SOLVER_OUTPUT ?= $(SOLVER_DIR)/$(GRAPH_STEM)_solver_results.json
MEASURED ?= golden/layer0/op_cycle_summary.json
ROOFLINE_OUT_PREFIX ?= outputs/roofline_only/$(GRAPH_STEM)
BANK_INTERLEAVE ?=
BANK_INTERLEAVE_ARG := $(if $(BANK_INTERLEAVE),--bank-interleave $(BANK_INTERLEAVE),)

.PHONY: help solve-graph fill-remapping fill-remapping-bank1 fill-remapping-bank2 solve-layer0 fill-layer0 perf roofline roofline-all roofline-vs-measured run-all

help:
	@echo "Available targets:"
	@echo "  make solve-graph GRAPH=examples/graphs/xxx.json"
	@echo "  make fill-remapping GRAPH=examples/graphs/xxx.json BANK_INTERLEAVE=2"
	@echo "  make fill-remapping-bank1 GRAPH=examples/graphs/xxx.json"
	@echo "  make fill-remapping-bank2 GRAPH=examples/graphs/xxx.json"
	@echo "  make solve-layer0"
	@echo "  make fill-layer0"
	@echo "  make perf GRAPH=examples/graphs/xxx.json"
	@echo "  make roofline GRAPH=examples/graphs/xxx.json MODE=baseline"
	@echo "  make roofline-all GRAPH=examples/graphs/xxx.json"
	@echo "  make roofline-vs-measured GRAPH=examples/graphs/layer0/layer0_0630.json"
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  GRAPH=$(GRAPH)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  MEASURED=$(MEASURED)"
	@echo "  ROOFLINE_OUT_PREFIX=$(ROOFLINE_OUT_PREFIX)"
	@echo "  REMAP_OUTPUT=$(REMAP_OUTPUT)"
	@echo "  SOLVER_OUTPUT=$(SOLVER_OUTPUT)"
	@echo "  OUT_DIR=$(OUT_DIR)"
	@echo "  BANK_INTERLEAVE=$(BANK_INTERLEAVE)"

solve-graph:
	$(PYTHON) -m address_remapping.cli solve-graph $(GRAPH) --output $(SOLVER_OUTPUT) $(BANK_INTERLEAVE_ARG)

fill-remapping:
	$(PYTHON) -m address_remapping.cli fill-remapping $(GRAPH) --config $(CONFIG) --output $(REMAP_OUTPUT) --dump-solver-results $(SOLVER_OUTPUT) $(BANK_INTERLEAVE_ARG)

fill-remapping-bank1:
	$(MAKE) fill-remapping BANK_INTERLEAVE=1 REMAP_OUTPUT=$(SOLVER_DIR)/$(GRAPH_STEM)_bank1_remapped.json SOLVER_OUTPUT=$(SOLVER_DIR)/$(GRAPH_STEM)_bank1_solver_results.json

fill-remapping-bank2:
	$(MAKE) fill-remapping BANK_INTERLEAVE=2 REMAP_OUTPUT=$(SOLVER_DIR)/$(GRAPH_STEM)_bank2_remapped.json SOLVER_OUTPUT=$(SOLVER_DIR)/$(GRAPH_STEM)_bank2_solver_results.json

solve-layer0:
	$(MAKE) fill-remapping GRAPH=examples/graphs/layer0/layer0.json

fill-layer0:
	$(MAKE) fill-remapping GRAPH=examples/graphs/layer0/layer0.json

perf:
	$(PYTHON) -m address_remapping.cli analyze-performance $(GRAPH) --config $(CONFIG) --output $(OUT_DIR)/$(GRAPH_STEM)_performance.json

MODE ?= baseline

roofline:
	$(PYTHON) -m address_remapping.cli plot-roofline $(GRAPH) --config $(CONFIG) --mode $(MODE) --output $(OUT_DIR)/$(GRAPH_STEM)_roofline_$(MODE).svg

roofline-all:
	$(PYTHON) -m address_remapping.cli plot-roofline $(GRAPH) --config $(CONFIG) --mode baseline --output $(OUT_DIR)/$(GRAPH_STEM)_roofline_baseline.svg
	$(PYTHON) -m address_remapping.cli plot-roofline $(GRAPH) --config $(CONFIG) --mode remap --output $(OUT_DIR)/$(GRAPH_STEM)_roofline_remap.svg
	$(PYTHON) -m address_remapping.cli plot-roofline $(GRAPH) --config $(CONFIG) --mode remap_interleave --output $(OUT_DIR)/$(GRAPH_STEM)_roofline_remap_interleave.svg

roofline-vs-measured:
	$(PYTHON) scripts/compare_layer0_roofline_vs_measured.py $(GRAPH) --config $(CONFIG) --measured $(MEASURED) --output-prefix $(ROOFLINE_OUT_PREFIX)

run-all: perf roofline-all
