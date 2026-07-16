PYTHON ?= python

.DEFAULT_GOAL := prefill

.PHONY: prefill all inputs weights golden single_op \
	decode decode_golden decode_single_op decode_graphs decode_layer0 decode_execplan \
	decode_package decode_validate clean help

# Preserve the original bare `make` behavior: run the Prefill pipeline.
prefill: golden single_op

# Explicitly request both phases with `make all`.
all: prefill decode

inputs:
	@echo "=> Generating dummy inputs based on config.json..."
	$(PYTHON) generate_seq_input.py

weights:
	@echo "=> Generating sliced weights based on config.json..."
	$(PYTHON) weight_gen.py

golden: inputs weights
	@echo "=> Generating Prefill Python Golden data..."
	$(PYTHON) deepseek1.5b_3_time_golden_smallsize.py

single_op: golden
	@echo "=> Generating Prefill per-slice operator data..."
	$(PYTHON) run_single_op.py

decode_golden:
	@echo "=> Generating deterministic Decode Python Golden data..."
	$(PYTHON) deepseek1.5b_decode_golden.py

decode_single_op: decode_golden
	@echo "=> Generating Decode per-slice operator data..."
	$(PYTHON) run_single_op_decode.py

decode_graphs:
	@echo "=> Generating Decode execution-plan operator graphs..."
	$(PYTHON) generate_decode_execplan_inputs.py

decode_layer0: decode_graphs
	@echo "=> Merging Decode operator graphs into layer0_decode.json..."
	$(PYTHON) gen_layer0_oplist.py --decode

decode_execplan: decode_layer0
	@echo "=> Generating Decode bitstreams and execution-plan instructions..."
	$(PYTHON) model_execplan/main.py model_execplan/examples/layer0_decode.json

decode_package: decode_single_op decode_execplan
	@echo "=> Assembling the self-contained Decode SCA loading package..."
	$(PYTHON) assemble_decode_package.py

decode_validate: decode_package
	@echo "=> Validating Decode Golden, slice data, bitstreams, and instructions..."
	$(PYTHON) validate_decode_outputs.py

# Complete Phase A, Phase B, and the software-only Phase C integration.
decode: decode_validate

clean:
	@echo "=> Cleaning Prefill generated folders..."
	rm -rf inputs model_weights_small python_golden python_golden_debug

help:
	@echo "Available targets:"
	@echo "  make / make prefill       - run the original Prefill pipeline"
	@echo "  make decode_golden        - generate python_golden_decode"
	@echo "  make decode_single_op     - generate single_op_data/install_decode"
	@echo "  make decode_graphs        - generate the 10 Decode operator graph JSONs"
	@echo "  make decode_layer0        - generate model_execplan/examples/layer0_decode.json"
	@echo "  make decode_execplan      - generate Decode bitstreams and instructions"
	@echo "  make decode_package       - assemble every sca_cfg path into one package"
	@echo "  make decode_validate      - validate all generated software artifacts"
	@echo "  make decode               - run the complete Decode software pipeline"
	@echo "  make all                  - run Prefill and Decode"
