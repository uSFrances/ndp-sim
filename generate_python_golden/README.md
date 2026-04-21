# Generate Python Golden

这是一个用于一键生成模型输入、按配置裁剪权重及生成 `python_golden` 数据的自动化工具。

## 依赖
- Python 3
- numpy
- tqdm

可以通过以下命令安装依赖：
```bash
pip install numpy tqdm
```

## 配置
核心参数在 `config.json` 中定义。调整这些参数会影响输入维度以及模型推理时的形状。
额外支持 `"target_op"` 参数用于指定需要提取局部数据的模块（比如 `"rms_norm"`、`"gemm"` 或是 `"all"`），相应的代码须放在 `single_op_data/` 文件夹中。

## 核心功能与运行机制

本项目是一个两阶段的自动化工具链，主要用于为硬件仿真生成和处理标准答案数据（Golden Data）。

### 第一阶段：Python Golden 数据生成 (`make golden`)
本阶段目标是**模拟一个 Transformer 模型层的计算过程，并将每一个算子的输入和输出张量以 `.bin` 格式保存下来**。
- **需要输入**：`config.json` 配置及 `DeepSeek-R1-Distill-Qwen-1.5B-f16/` 原始权重文件夹。
- **运行流程**：
  1. `create_dummy_inputs.py`: 创建虚拟输入数据（存入 `inputs/`）。
  2. `weight_gen.py`: 提取并裁剪适用于当前配置的小尺寸模型权重（存入 `model_weights_small/`）。
  3. `deepseek1.5b_3_time_golden_smallsize.py`: 模拟执行单层 Transformer，保存所有计算节点的 Tensor 数据至 `python_golden/`，相关复杂算子内部步骤存于 `python_golden/sub_ops/`。

### 第二阶段：单算子数据切片与重排 (`make single_op`)
本阶段目标是**读取第一阶段生成的黄金数据，并按照特定硬件的内存布局要求进行分片 (Slice) 和数据流重排 (Relayout)**。
- **需要输入**：第一阶段产出的 `python_golden/` 和 `python_golden/sub_ops/` 文件夹数据。
- **运行流程**：
  1. `run_single_op.py`: 读取 `config.json` 中的 `"target_op"` 字段。
  2. `single_op_data/relayout_*.py`: 脚本会根据指定的算子读取对应输入，执行如 M8N2M4N 类似的数据层级置换与切分。
  3. **最终输出**：生成的文件按硬件目录结构（如 `op0/slice00/`）输出至 `single_op_data/install/` 目录下，格式包含 `.bin` 及 `128-bit .txt`，可直接供硬件模拟器加载使用。

## 使用方法

本仓库提供了一个 `Makefile` 来简化执行流程。打开终端，进入本目录，输入以下命令即可：

- **一键执行完整流程 (推荐)**
  ```bash
  make
  ```
  该命令会依次执行：生成虚拟输入 -> 生成/裁剪权重 -> 产生 python_golden 数据 -> 进行算子切片与重排。

- **部分执行流程**
  - `make inputs`: 仅生成虚拟输入。
  - `make weights`: 仅执行权重裁剪。
  - `make golden`: 执行到第一阶段，仅生成黄金数据不进行切片重排。
  - `make single_op`: 在已有黄金数据的基础上，依据 `config.json` 更新执行切片代码。

- **清理生成的数据**
  ```bash
  make clean
  ```
