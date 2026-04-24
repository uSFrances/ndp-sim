SHELL := /bin/bash

CONDA_SH := /cluster/home/liudy/anaconda3/etc/profile.d/conda.sh
PYTHONPATH_EXPORT := PYTHONPATH=src

# Usage:
#   make perf GRAPH=examples/graphs/xxx.json
#   make roofline-all GRAPH=examples/graphs/xxx.json
#   make run-all GRAPH=examples/graphs/xxx.json
GRAPH ?= examples/graphs/rmsnorm_withbaseaddr.json
CONFIG ?= examples/configs/performance_config.json
GRAPH_STEM := $(basename $(notdir $(GRAPH)))
OUT_DIR ?= outputs/performance/$(GRAPH_STEM)

.PHONY: help perf roofline roofline-all run-all perf-rmsnorm-mul roofline-rmsnorm-mul roofline-rmsnorm-mul-all perf-and-roofline-rmsnorm-mul

help:
	@echo "Available targets:"
	@echo "  make perf GRAPH=examples/graphs/xxx.json"
	@echo "  make roofline GRAPH=examples/graphs/xxx.json MODE=baseline"
	@echo "  make roofline-all GRAPH=examples/graphs/xxx.json"
	@echo "  make run-all GRAPH=examples/graphs/xxx.json"
	@echo "Variables:"
	@echo "  GRAPH=$(GRAPH)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  OUT_DIR=$(OUT_DIR)"

perf:
	@source $(CONDA_SH) && conda activate base && \
	$(PYTHONPATH_EXPORT) python -m address_remapping.cli analyze-performance \
	$(GRAPH) \
	--config $(CONFIG) \
	--output $(OUT_DIR)/$(GRAPH_STEM)_performance.json

MODE ?= baseline

roofline:
	@source $(CONDA_SH) && conda activate base && \
	$(PYTHONPATH_EXPORT) python -m address_remapping.cli plot-roofline \
	$(GRAPH) \
	--config $(CONFIG) \
	--mode $(MODE) \
	--output $(OUT_DIR)/$(GRAPH_STEM)_roofline_$(MODE).svg

roofline-all:
	@source $(CONDA_SH) && conda activate base && \
	$(PYTHONPATH_EXPORT) python -m address_remapping.cli plot-roofline \
	$(GRAPH) \
	--config $(CONFIG) \
	--mode baseline \
	--output $(OUT_DIR)/$(GRAPH_STEM)_roofline_baseline.svg && \
	$(PYTHONPATH_EXPORT) python -m address_remapping.cli plot-roofline \
	$(GRAPH) \
	--config $(CONFIG) \
	--mode remap \
	--output $(OUT_DIR)/$(GRAPH_STEM)_roofline_remap.svg && \
	$(PYTHONPATH_EXPORT) python -m address_remapping.cli plot-roofline \
	$(GRAPH) \
	--config $(CONFIG) \
	--mode remap_interleave \
	--output $(OUT_DIR)/$(GRAPH_STEM)_roofline_remap_interleave.svg

run-all: perf roofline-all

# Backward-compatible aliases for older command names.
perf-rmsnorm-mul:
	@$(MAKE) perf GRAPH=examples/graphs/rmsnorm_mul_withbaseaddr.json

roofline-rmsnorm-mul:
	@$(MAKE) roofline GRAPH=examples/graphs/rmsnorm_mul_withbaseaddr.json MODE=baseline

roofline-rmsnorm-mul-all:
	@$(MAKE) roofline-all GRAPH=examples/graphs/rmsnorm_mul_withbaseaddr.json

perf-and-roofline-rmsnorm-mul:
	@$(MAKE) run-all GRAPH=examples/graphs/rmsnorm_mul_withbaseaddr.json