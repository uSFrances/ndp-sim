# generate_python_golden 使用说明

这个目录的作用是：根据 `config.json` 生成 smallsize 模型输入和权重，运行 Python 模型得到完整 golden 数据，再把完整 golden 数据按硬件 slice/relayout 规则转换成 `model_execplan` 使用的数据目录，最后拼成 layer0/layer0_physic。

整体流程分为四步：

```text
一、设置 config，生成 smallsize 输入和权重
二、运行 Python 模型，生成完整 python_golden
三、对完整数据按算子分 slice 和 relayout
四、把 relayout 好的数据按 layer0_op_listing.json 拼成 layer0/layer0_physic
```

最后一节单独说明 `Makefile`。

## 一、设置 Config 并生成 Smallsize 模型参数

### 1. 配置文件

主配置文件是：

```text
ndp-sim/generate_python_golden/config.json
```

常用备份配置：

| 文件 | 用途 |
| --- | --- |
| `config_small.json` | smallsize 配置备份。 |
| `config_full.json` | full/1.5B 配置备份。 |

`config.json` 中主要字段含义：

| 字段 | 含义 |
| --- | --- |
| `hidden_size` | 隐藏层维度。 |
| `intermediate_size` | FFN 中间层维度。 |
| `num_attention_heads` | Q attention head 数。 |
| `num_key_value_heads` | KV head 数。 |
| `head_dim` | 每个 head 的维度。 |
| `num_hidden_layers` | 需要运行的 transformer layer 数。 |
| `sequence_length` | token 数。 |
| `slice_per_head` | 每个 head 切成几个 slice。 |
| `used_slices` | 总 slice 数，当前通常为 28。 |
| `kv_padding` / `kv_padding_a` / `kv_padding_b` | KV 相关 padding 大小。 |
| `target_op` | `run_single_op.py` 要处理的目标算子。 |

`target_op` 常用值：

| `target_op` | 行为 |
| --- | --- |
| `all` | 运行 `gemm`、`rmsnorm`、`rope`、`softmax`。 |
| `rmsnorm` / `rms_norm` | 只运行 RMSNorm relayout。 |
| `softmax` | 只运行 Softmax relayout。 |
| `rope` | 只运行 RoPE relayout。 |
| `gemm` | 只运行 GEMM relayout。 |

### 2. 生成输入 Input

当前 Makefile 的 `inputs` 目标调用：

```bash
python generate_seq_input.py
```

脚本路径：

```text
ndp-sim/generate_python_golden/generate_seq_input.py
```

输入：

```text
ndp-sim/generate_python_golden/inputs_good/*.bin
ndp-sim/generate_python_golden/config.json
```

功能：

- 从 `inputs_good/` 读取已有 8-token 输入。
- 根据 `config.json` 中的 `sequence_length` 扩展 token 维度。
- 通过重复原始 token 数据生成目标长度输入。

输出：

```text
ndp-sim/generate_python_golden/python_golden_custom_seq/
```

补充脚本：

```text
create_dummy_inputs.py
```

这个脚本会根据 `config.json` 生成随机输入到：

```text
ndp-sim/generate_python_golden/inputs/
```

包含：

| 文件 | 内容 |
| --- | --- |
| `inp_embd_shape...dtype_f32.bin` | embedding 输入。 |
| `leaf_12_shape...dtype_f32.bin` | softmax mask 输入。 |
| `leaf_395_shape...dtype_i32.bin` | int32 标量输入。 |

注意：当前 Makefile 用的是 `generate_seq_input.py`，不是 `create_dummy_inputs.py`。

### 3. 生成权重 Weight

Makefile 的 `weights` 目标调用：

```bash
python weight_gen.py
```

脚本路径：

```text
ndp-sim/generate_python_golden/weight_gen.py
```

输入：

```text
ndp-sim/generate_python_golden/DeepSeek-R1-Distill-Qwen-1.5B-f16/
ndp-sim/generate_python_golden/config.json
```

原始权重文件名格式：

```text
<tensor_name>__dtype=<f16|f32|i32>__shape=<shape>.bin
```

功能：

- 读取原始 1.5B/f16 权重。
- 根据 `config.json` 裁切到 smallsize 需要的大小。
- 保存为模型脚本可直接加载的 `.bin`。

输出：

```text
ndp-sim/generate_python_golden/model_weights_full/
```

主要裁切规则：

| 权重 | 裁切规则 |
| --- | --- |
| `attn_norm.weight` / `ffn_norm.weight` / `output_norm.weight` | 裁到 `hidden_size`。 |
| `attn_q.weight` / `attn_output.weight` | 裁到 `hidden_size x hidden_size`。 |
| `attn_k.weight` / `attn_v.weight` | 裁到 `hidden_size x (head_dim*num_key_value_heads)`。 |
| `ffn_gate.weight` / `ffn_up.weight` | 裁到 `hidden_size x intermediate_size`。 |
| `ffn_down.weight` | 裁到 `intermediate_size x hidden_size`。 |
| bias | 按对应输出维度裁切。 |

## 二、生成完整 Python Golden 数据

生成完整模型 golden 的主脚本是：

```text
ndp-sim/generate_python_golden/deepseek1.5b_3_time_golden_smallsize.py
```

Makefile 的 `golden` 目标会调用：

```bash
python deepseek1.5b_3_time_golden_smallsize.py
```

输入：

```text
ndp-sim/generate_python_golden/config.json
ndp-sim/generate_python_golden/model_weights_full/
ndp-sim/generate_python_golden/inputs_full/
```

说明：

- `model_weights_full/` 来自 `weight_gen.py`。
- `inputs_full/` 是当前脚本实际加载的输入目录。
- 如果你改成加载 `inputs_32/` 或其他目录，需要看脚本底部 `input_folder` 的设置。

输出：

```text
ndp-sim/generate_python_golden/python_golden/
ndp-sim/generate_python_golden/python_golden/sub_ops/
```

输出文件命名格式：

```text
<node_name>_shape<dim0>x<dim1>x<dim2>x<dim3>_dtype_<f32|f16|i32>.bin
```

两类输出：

| 输出目录 | 内容 |
| --- | --- |
| `python_golden/` | 主算子输入输出，例如 matmul、add、mul、silu 等节点。 |
| `python_golden/sub_ops/` | 复杂算子拆开的子算子输入输出，例如 RMSNorm、RoPE、Softmax 的内部步骤。 |

这个脚本主要做：

- 加载输入和权重到 `TensorStore`。
- 运行 transformer layer。
- 对每个重要节点保存 input/output。
- 对 RMSNorm、RoPE、Softmax 等复杂算子保存 subop 数据。
- 保存 dtype 信息到文件名中，后续 relayout 会根据 `dtype_f32` / `dtype_f16` 处理。

相关历史脚本：

| 脚本 | 用途 |
| --- | --- |
| `deepseek1.5b_3_time_golden.py` | 较早的 full-size/固定路径版本。 |
| `deepseek1.5b_3_time_golden_smallsize_0527.py` | 旧 smallsize 备份。 |
| `deepseek1.5b_3_time_golden_smallsize copy.py` | 临时备份，不作为主流程入口。 |

## 三、分 Slice 和 Relayout

完整 golden 生成后，需要把 `python_golden/` 和 `python_golden/sub_ops/` 中的数据转换为硬件可读的单算子目录。

入口脚本：

```text
ndp-sim/generate_python_golden/run_single_op.py
```

Makefile 的 `single_op` 目标调用：

```bash
python run_single_op.py
```

输入：

```text
ndp-sim/generate_python_golden/config.json
ndp-sim/generate_python_golden/python_golden/
ndp-sim/generate_python_golden/python_golden/sub_ops/
```

输出：

```text
ndp-sim/model_execplan/data/<op_category>/<op_prefix>/install/
ndp-sim/model_execplan/data/<op_category>/<op_prefix>/install_beforerelayout/
```

通用输出结构：

```text
install/opX/sliceYY/matrix_A_linearized_128bit.bin
install/opX/sliceYY/matrix_B_linearized_128bit.bin
install/opX/sliceYY/matrix_C_linearized_128bit.bin
install/opX/sliceYY/matrix_D_linearized_128bit.bin
install/opX/sliceYY/matrix_*.txt
install/opX/sliceYY/matrix_*_decimal_1d.txt
install/opX/sliceYY/matrix_*_hex.txt
```

`matrix` 约定：

| Matrix | 含义 |
| --- | --- |
| `matrix_A` | 算子输入 A。 |
| `matrix_B` | 算子输入 B。 |
| `matrix_C` | 第三个输入，例如 softmax mask。 |
| `matrix_D` | 算子输出。 |

注意：有些脚本会根据硬件端口要求交换 in0/in1，所以原始 golden 的 `in0` 不一定总是 `matrix_A`。

### 1. Regular 算子

脚本：

```text
single_op_data/relayout_regular.py
```

处理对象：

- 普通 add。
- 普通 mul。
- unary/silu。
- residual add。
- attention/ffn norm 后的 elementwise 算子。

输入：

```text
python_golden/*.bin
```

输出：

```text
model_execplan/data/regular/<prefix>/install/
model_execplan/data/regular/<prefix>/install_beforerelayout/
```

relayout 规则：

- 默认对二维矩阵按 M8N 方向重排。
- 部分 in0/in1 根据硬件 A/B 端口规则交换。
- `blk.0_Vcur-0-add_op-add` 的 `matrix_B`、`matrix_D` 使用 N8M/转置相关处理。
- `blk.0_ffn_silu-0_op-unary` 的 before-relayout debug 数据按 F-order 保存，便于和物理 M/N 方向对齐。

### 2. RMSNorm 算子

脚本：

```text
single_op_data/relayout_rmsnorm.py
```

处理对象：

- RMSNorm 内部 `sum_mac`。
- `remote_sum`。
- `mac_SFU`。
- 最后的 norm/mul 输出。

输入：

```text
python_golden/sub_ops/*.bin
```

输出：

```text
model_execplan/data/rmsnorm/<prefix>/install/
model_execplan/data/rmsnorm/<prefix>/install_beforerelayout/
model_execplan/data/rmsnorm/<prefix>_kv/install/
model_execplan/data/rmsnorm/<prefix>_kv/install_beforerelayout/
```

relayout 规则：

- 按 `used_slices` 或 `slice_per_head` 分 slice。
- 常规 RMSNorm 使用 M8N 相关重排。
- KV 版本会额外生成 `<prefix>_kv` 目录。
- KV case 会先对输入做平方和，再广播/分配到对应 slice。

### 3. RoPE 算子

脚本：

```text
single_op_data/relayout_rope.py
```

处理对象：

- Q/K 的 RoPE 子算子。
- RoPE 内部 add/mul 等步骤。

输入：

```text
python_golden/sub_ops/*.bin
```

输出：

```text
model_execplan/data/rope/<prefix>/install/
model_execplan/data/rope/<prefix>/install_beforerelayout/
```

relayout 规则：

- 按 head 和 `slice_per_head` 切分，总共通常 28 个 slice。
- 有独立 head 轴的数据按 head 切。
- 单头控制参数会广播到对应 head/slice。
- `op1/matrix_B` 有专用切片逻辑。
- 每个 slice 内使用 M8N 相关重排。

### 4. Softmax 算子

脚本：

```text
single_op_data/relayout_softmax.py
```

处理对象：

- Softmax 的 `add`。
- `max`。
- `sub_SFU`，也就是 `exp(x-xmax)`。
- `sum_rec`。
- 最后的 `mul`。

输入：

```text
python_golden/sub_ops/*.bin
```

输出：

```text
model_execplan/data/softmax/<prefix>/install/
model_execplan/data/softmax/<prefix>/install_beforerelayout/
```

relayout 规则：

- `op0/matrix_A` 在 relayout 前用 fp32 乘以 `sqrt(128)`。
- mask 等第三输入会映射为 `matrix_C`。
- max/sum_rec 这类向量数据会按 head 广播给对应 slice。
- 矩阵类数据按 M8N 相关规则重排。

### 5. GEMM Local 算子

脚本：

```text
single_op_data/relayout_gemm_local.py
```

处理对象：

- attention scores 的 local QK^T。
- softmax 后概率乘 V，也就是 SV。

输入：

```text
python_golden/*.bin
```

输出：

```text
model_execplan/data/gemm_local/<prefix>/install/
model_execplan/data/gemm_local/<prefix>/install_beforerelayout/
```

relayout 规则：

- 根据文件名前缀区分 `attn_scores` 和 `attn/SV`。
- 按 head 和 `slice_per_head` 切分。
- `matrix_A`、`matrix_B`、`matrix_D` 使用不同的 M8N/N8M/GEMM-local 重排。

### 6. Remote Sum 算子

脚本：

```text
single_op_data/relayout_remote_sum.py
```

处理对象：

- local GEMM 后的 remote_sum。

输入：

```text
python_golden/*.bin
```

输出：

```text
model_execplan/data/gemm_local/<prefix>/install/
model_execplan/data/gemm_local/<prefix>/install_beforerelayout/
```

relayout 规则：

- 输入 `matrix_A` 按 `slice_per_head` 在 N 维切分。
- 输出 `matrix_D` 取对应 slice。
- 当前 M8N 含义是：先取 M 方向 8 个，再切换 N 列。

### 7. KV 相关 `mul_MN_N`

脚本：

```text
single_op_data/relayout_mul_MN_N_kv.py
```

处理对象：

- KV 路径中的 `mul_MN_N`。

输入：

```text
python_golden/*.bin
```

输出：

```text
model_execplan/data/mul_MN_N_kv/<prefix>/install/
model_execplan/data/mul_MN_N_kv/<prefix>/install_beforerelayout/
```

relayout 规则：

- 根据 `config.json` 构造目标文件名集合。
- 部分算子会交换 in0/in1 到 `matrix_B`/`matrix_A`。
- 按 4 slice/head 和 head 数进行广播或切分。

### 8. GEMM Ring / 通用 GEMM

脚本：

```text
single_op_data/relayout_gemm.py
single_op_data/relayout_gemm_ring.py
```

运行示例：

```bash
python single_op_data/relayout_gemm.py --target-op all
```

可选参数：

```bash
python single_op_data/relayout_gemm.py \
  --target-op all \
  --layer-id 0 \
  --input-dir <golden_dir> \
  --output-dir <output_dir>
```

默认输入：

```text
single_op_data/golde_data/
```

默认输出：

```text
single_op_data/outputs/<target-op>/
```

输出子目录：

| 目录 | 内容 |
| --- | --- |
| `install_beforerelayout/` | 切片后、relayout 前的数据。 |
| `install_after_ring/` | ring 重排后的中间数据。 |
| `install_logic/` | 逻辑 slice 顺序的数据。 |
| `install/` | 物理 slice 映射后的数据。 |

主要 relayout 规则：

| 端口 | 规则 |
| --- | --- |
| weight/input0 | N8K2N4K。 |
| activation/input1 | L8K2L4K。 |
| output | L8N8L4N4N2L1。 |

`relayout_gemm_old.py` 是旧版 GEMM relayout，保留用于复现实验，不建议新流程优先使用。

## 四、拼接 Relayout 数据并生成 Layer0

完整 layer0 数据拼接脚本：

```text
single_op_data/relayout_layer0.py
```

运行：

```bash
python single_op_data/relayout_layer0.py
```

输入：

```text
model_execplan/layer0.json
model_execplan/layer0_op_listing.json
model_execplan/data/rmsnorm/<prefix>/install/
model_execplan/data/regular/<prefix>/install/
model_execplan/data/rope/<prefix>/install/
model_execplan/data/softmax/<prefix>/install/
model_execplan/data/gemm_local/<prefix>/install/
model_execplan/data/gemm_ring/<prefix>/install/
model_execplan/data/mul_MN_N_kv/<prefix>/install/
```

`layer0_op_listing.json` 定义最终 layer0 中每个 op 来自哪个单算子模板，例如：

```json
{
  "op0": "rmsnorm::op0",
  "op24": "softmax::op0",
  "op42": "add_fp32MN_fp16MN_fp32MN_out::op0"
}
```

第一阶段输出：

```text
model_execplan/data/layer0/install/opX/sliceYY/
```

第二阶段输出：

```text
model_execplan/data/layer0_physic/install/opX/sliceYY/
```

拼接规则：

- 先按 `layer0_op_listing.json` 从各单算子目录复制对应 `install/opX`。
- 普通数据先生成 `data/layer0/install`。
- 拼接时会读取 `layer0.json` 中每个 op 的完整 `type`，用于判断特殊规则；`layer0_op_listing.json` 继续用于定位单算子数据来源。
- 当 `layer0.json` 中的 op 类型是 `prefill_add_fp16MN_fp32N_fp32MN`、`add_fp16MN_fp32N_fp32MN`、`prefill_mul_fp32MN_fp32N_fp16MN` 或 `mul_fp32MN_fp32N_fp16MN` 时，脚本不会复制 `install/opX`，而是改用同一单算子目录下的 `install_beforerelayout/opX`，用于保留未重排版本。
- 命中上述规则时，脚本会打印 layer0 op、算子类型和实际来源文件夹，例如 `op6 -> regular/<prefix>/install_beforerelayout/op0`。
- 再复制成 `data/layer0_physic`。
- `layer0_physic` 会把 slice 从逻辑顺序复原到物理顺序。

物理 slice 顺序：

```python
order = [0,2,3,1,5,4,6,7,8,10,11,9,15,14,12,13,16,17,19,18,20,21,23,22,26,24,25,27]
```

含义：

```text
源 slice i 的真实物理 slice id 是 order[i]
最终目录按 slice00, slice01, ..., slice27 排列
```

最后再补充/覆盖 `gemm_ring` 数据，因为 `gemm_ring` 不参与普通数据的 slice 复原。

## Makefile 模块

Makefile 路径：

```text
ndp-sim/generate_python_golden/Makefile
```

### Makefile 目标

| 命令 | 调用脚本 | 作用 |
| --- | --- | --- |
| `make` / `make all` | `golden single_op` | 执行 golden 生成和单算子 relayout。 |
| `make inputs` | `python generate_seq_input.py` | 生成/扩展输入。 |
| `make weights` | `python weight_gen.py` | 从原始权重裁切 smallsize 权重。 |
| `make golden` | `inputs weights deepseek1.5b_3_time_golden_smallsize.py` | 先生成输入和权重，再生成完整 `python_golden`。 |
| `make single_op` | `python run_single_op.py` | 根据 `target_op` 生成单算子 relayout 数据和执行计划。 |
| `make clean` | `rm -rf ...` | 删除部分生成目录。 |
| `make help` | echo | 打印帮助信息。 |

### Makefile 实际执行顺序

执行：

```bash
make
```

等价于：

```bash
make golden
make single_op
```

其中 `make golden` 又依赖：

```bash
make inputs
make weights
python deepseek1.5b_3_time_golden_smallsize.py
```

所以完整展开为：

```text
1. python generate_seq_input.py
2. python weight_gen.py
3. python deepseek1.5b_3_time_golden_smallsize.py
4. python run_single_op.py
```

### 常用命令

进入目录：

```bash
cd ndp-sim/generate_python_golden
```

完整流程：

```bash
make
```

只生成输入：

```bash
make inputs
```

只生成权重：

```bash
make weights
```

只生成完整 Python golden：

```bash
make golden
```

只做单算子 relayout：

```bash
make single_op
```

只拼 layer0：

```bash
python single_op_data/relayout_layer0.py
```

清理：

```bash
make clean
```

### 注意事项

- `make single_op` 依赖 `make golden`，所以直接运行它会先重新生成 golden。
- 如果你只想重跑某个 relayout 脚本，可以直接运行对应 `single_op_data/relayout_*.py`。
- `make clean` 当前删除的是 `inputs/`、`model_weights_small/`、`python_golden/`、`python_golden_debug/`，不会清理所有 `model_execplan/data` 输出。
- 改 `config.json` 后，建议重新运行输入、权重、golden 和 relayout，避免 shape/dtype 不一致。
