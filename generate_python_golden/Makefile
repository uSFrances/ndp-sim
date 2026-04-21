PYTHON = python3

.PHONY: all inputs weights golden single_op clean help

# 默认目标，执行完整的生成流程
all: golden single_op

# 1. 产生 inputs 文件
inputs:
	@echo "=> Generating dummy inputs based on config.json..."
	$(PYTHON) create_dummy_inputs.py

# 2. 产生 weight 文件
weights:
	@echo "=> Generating sliced weights based on config.json..."
	$(PYTHON) weight_gen.py

# 3. 产生 python_golden 数据（依赖输入和权重）
golden: inputs weights
	@echo "=> Running main model script to generate python_golden data..."
	$(PYTHON) deepseek1.5b_3_time_golden_smallsize.py

# 4. 运行单算子的 slice 分割
single_op: golden
	@echo "=> Running single operator slice extraction based on config.json..."
	$(PYTHON) run_single_op.py

# 清理生成的文件和文件夹
clean:
	@echo "=> Cleaning up generated folders..."
	rm -rf inputs
	rm -rf model_weights_small
	rm -rf python_golden
	rm -rf python_golden_debug

# 打印帮助信息
help:
	@echo "可用命令:"
	@echo "  make         - 执行完整的生成流程 (inputs -> weights -> golden -> single_op)"
	@echo "  make inputs  - 仅产生 inputs 文件"
	@echo "  make weights - 仅产生 weight 文件"
	@echo "  make golden  - 产生 python_golden 数据 (自动依赖 inputs 和 weights)"
	@echo "  make single_op - 仅运行 target_op 的算子分割代码 (依赖 golden)"
	@echo "  make clean   - 清理所有生成的文件和文件夹"
