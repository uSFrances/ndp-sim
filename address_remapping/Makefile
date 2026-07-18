PYTHON ?= python
export PYTHONPATH := src
GRAPH ?= examples/graphs/layer0/layer0.json
CONFIG ?= examples/configs/performance_config.json
LAYER0_GRAPH ?= examples/graphs/layer0/layer0_0630.json
MODEL_CONFIG ?= examples/configs/config.json
MODEL_NAME ?= deepseek1.5b
FREQUENCY_MHZ ?= 800
SEQUENCE_MULTIPLE ?= 32
SEQUENCE_LENGTH ?=
GRAPH_STEM := $(basename $(notdir $(GRAPH)))
LAYER0_GRAPH_STEM := $(basename $(notdir $(LAYER0_GRAPH)))
OUT_DIR ?= outputs/performance/$(GRAPH_STEM)
SOLVER_DIR ?= outputs/solver/$(GRAPH_STEM)
REMAP_OUTPUT ?= $(SOLVER_DIR)/$(GRAPH_STEM)_remapped.json
SOLVER_OUTPUT ?= $(SOLVER_DIR)/$(GRAPH_STEM)_solver_results.json
MEASURED ?= golden/layer0/op_cycle_summary.json
ROOFLINE_OUT_PREFIX ?= outputs/roofline_only/$(MODEL_NAME)/$(GRAPH_STEM)
SMALLSIZE_MODEL_CONFIG ?= examples/configs/smallsize_config.json
SMALLSIZE_MODEL_NAME ?= smallsize
BANK_INTERLEAVE ?=
BANK_INTERLEAVE_ARG := $(if $(BANK_INTERLEAVE),--bank-interleave $(BANK_INTERLEAVE),)
SEQUENCE_LENGTH_ARG := $(if $(SEQUENCE_LENGTH),--sequence-length $(SEQUENCE_LENGTH),)

.PHONY: help solve-graph fill-remapping fill-remapping-bank1 fill-remapping-bank2 solve-layer0 fill-layer0 perf roofline roofline-all roofline-vs-measured roofline-layer0 roofline-layer0-model roofline-layer0-smallsize run-all

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
	@echo "  make roofline-layer0"
	@echo "  make roofline-layer0-model MODEL_CONFIG=examples/configs/config.json MODEL_NAME=deepseek1.5b SEQUENCE_LENGTH=2048"
	@echo "  make roofline-layer0-smallsize"
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  GRAPH=$(GRAPH)"
	@echo "  LAYER0_GRAPH=$(LAYER0_GRAPH)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  MODEL_CONFIG=$(MODEL_CONFIG)"
	@echo "  MODEL_NAME=$(MODEL_NAME)"
	@echo "  SMALLSIZE_MODEL_CONFIG=$(SMALLSIZE_MODEL_CONFIG)"
	@echo "  SMALLSIZE_MODEL_NAME=$(SMALLSIZE_MODEL_NAME)"
	@echo "  FREQUENCY_MHZ=$(FREQUENCY_MHZ)"
	@echo "  SEQUENCE_MULTIPLE=$(SEQUENCE_MULTIPLE)"
	@echo "  SEQUENCE_LENGTH=$(SEQUENCE_LENGTH)"
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
	$(PYTHON) scripts/compare_layer0_roofline_vs_measured.py $(GRAPH) --config $(CONFIG) --measured $(MEASURED) --output-prefix $(ROOFLINE_OUT_PREFIX) --model-config $(MODEL_CONFIG) --model-name $(MODEL_NAME) --frequency-mhz $(FREQUENCY_MHZ) --sequence-multiple $(SEQUENCE_MULTIPLE) $(SEQUENCE_LENGTH_ARG)

roofline-layer0:
	$(MAKE) roofline-vs-measured GRAPH=$(LAYER0_GRAPH) ROOFLINE_OUT_PREFIX=outputs/roofline_only/$(MODEL_NAME)/$(LAYER0_GRAPH_STEM)

roofline-layer0-model:
	$(MAKE) roofline-layer0 MODEL_CONFIG=$(MODEL_CONFIG) MODEL_NAME=$(MODEL_NAME) FREQUENCY_MHZ=$(FREQUENCY_MHZ) SEQUENCE_MULTIPLE=$(SEQUENCE_MULTIPLE) SEQUENCE_LENGTH=$(SEQUENCE_LENGTH)

roofline-layer0-smallsize:
	$(MAKE) roofline-layer0 MODEL_CONFIG=$(SMALLSIZE_MODEL_CONFIG) MODEL_NAME=$(SMALLSIZE_MODEL_NAME) FREQUENCY_MHZ=$(FREQUENCY_MHZ) SEQUENCE_MULTIPLE=$(SEQUENCE_MULTIPLE)

run-all: perf roofline-all
