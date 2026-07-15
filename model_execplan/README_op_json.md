# 生成 layer0_physic.json 的流程

本文说明如何从 `model_execplan/op_json/` 里的单算子模板生成完整的
`model_execplan/layer0_physic.json`。

整体流程分两步：

1. 用 `gen_layer0_oplist.py` 把多个 `op_json/*.json` 模板按 layer0 顺序拼成逻辑图 `layer0.json`。
2. 用 `address_remapping.cli fill-remapping` 给 `layer0.json` 填入物理 remapping 信息，生成 `layer0_physic.json`。

## 1. 输入文件

### 模型参数

`gen_layer0_oplist.py` 会读取：

```text
ndp-sim/generate_python_golden/config.json
```

其中的 `hidden_size`、`intermediate_size`、`num_attention_heads`、`used_slices` 等参数会写入生成文件的顶层 `params` 字段。

### 单算子模板

所有单算子模板位于：

```text
ndp-sim/model_execplan/op_json/
```

这些模板描述每类算子的输入、输出、source 关系、dtype、shape、config 等信息。常用模板包括：

```text
rmsnorm.json
rmsnorm_kv.json
rope.json
rope_k.json
softmax.json
gemm_ring.json
gemm_ring_k.json
gemm_ring_v.json
gemm_ring_ffn_gate.json
gemm_ring_ffn_up.json
gemm_ring_ffn_out.json
gemm_local_qkt.json
gemm_local_sv.json
prefill_mul_fp32MN_fp32N_fp16MN.json
prefill_mul_fp32MN_fp32N_fp16MN_kv.json
prefill_add_fp16MN_fp32N_fp32MN.json
prefill_add_fp16MN_fp32N_fp32MN_k.json
prefill_add_V_fp16MN_fp32N_fp16MN.json
prefill_remote_sum_4slice_fp32MN_fp32MN.json
prefill_silu_fp16MN_fp32MN.json
prefill_mul_fp32MN_fp16MN_fp16MN.json
prefill_add_fp32MN_fp16MN_fp32MN_residual.json
prefill_add_fp32MN_fp16MN_fp32MN_out.json
```

## 2. 生成逻辑版 layer0.json

在仓库根目录执行：

```bash
cd ndp-sim
python model_execplan/gen_layer0_oplist.py
```

默认输出：

```text
ndp-sim/model_execplan/layer0.json
ndp-sim/model_execplan/layer0_op_listing.json
```

其中：

- `layer0.json`：完整 layer0 的逻辑算子图，还没有填物理 remapping。
- `layer0_op_listing.json`：记录每个全局 `opX` 来自哪个模板文件中的哪个局部 op，便于之后对照数据和调试。

如果要自定义输出路径：

```bash
python model_execplan/gen_layer0_oplist.py \
  --output model_execplan/layer0.json
```

如果要临时指定 op 拼接顺序：

```bash
python model_execplan/gen_layer0_oplist.py \
  --ops rmsnorm mul_fp32MN_fp32N_fp16MN gemm_ring softmax \
  --output model_execplan/layer0.json
```

更常用的方式是直接修改 `model_execplan/gen_layer0_oplist.py` 里的 `DEFAULT_OPLIST`。

## 3. 默认 layer0 op 顺序

当前 `DEFAULT_OPLIST` 的顺序大致对应：

```text
Q path:
  rmsnorm
  mul_fp32MN_fp32N_fp16MN
  gemm_ring
  add_fp16MN_fp32N_fp32MN
  rope

K path:
  rmsnorm_kv
  mul_fp32MN_fp32N_fp16MN_kv
  gemm_ring_k
  add_fp16MN_fp32N_fp32MN_k
  rope_k

V path:
  gemm_ring_v
  add_V_fp16MN_fp32N_fp16MN

attention:
  gemm_local_qkt
  remote_sum_fp32MN_fp32MN
  softmax
  gemm_local_sv
  gemm_ring
  add_fp32MN_fp16MN_fp32MN_residual

FFN:
  rmsnorm
  mul_fp32MN_fp32N_fp16MN
  gemm_ring_ffn_gate
  gemm_ring_ffn_up
  silu
  mul_fp32MN_fp16MN_fp16MN
  gemm_ring_ffn_out
  add_fp32MN_fp16MN_fp32MN_out
```

生成时脚本会把每个模板内部的 `op0`、`op1` 重新编号成全局连续的 `op0`、`op1`、...。
模板里的 `source: op-1`、`source: op-2` 会被解释为引用前一个或前两个全局算子。

## 4. 生成 layer0_physic.json

`layer0.json` 只是逻辑连接关系，真正用于 execution plan 的完整文件需要填入 `remapping`、`bank_interleave` 等物理布局字段。

在仓库根目录执行：

```bash
cd ndp-sim
PYTHONPATH=address_remapping/src python -m address_remapping.cli fill-remapping \
  model_execplan/layer0.json \
  --output model_execplan/layer0_physic.json
```

如果需要同时保存 solver 结果，便于检查每条 producer-consumer 边的 remapping：

```bash
PYTHONPATH=address_remapping/src python -m address_remapping.cli fill-remapping \
  model_execplan/layer0.json \
  --output model_execplan/layer0_physic.json \
  --dump-solver-results
```

默认 solver 结果会写到：

```text
ndp-sim/address_remapping/outputs/solver/layer0/layer0_solver_results.json
```

生成后的目标文件是：

```text
ndp-sim/model_execplan/layer0_physic.json
```

这个文件包含：

- 顶层 `params`
- 顶层 `used_slices`
- 完整 `operators`
- 每个 tensor 的 `shape`
- 每个 tensor 的 `dtype`
- 每个 tensor 的 `source`
- 已填好的 `remapping`
- 已填好的 `bank_interleave`

## 5. 验证 layer0_physic.json

生成后可以用 execution plan 生成器验证：

```bash
cd ndp-sim
python model_execplan/main.py model_execplan/layer0_physic.json
```

主要输出目录：

```text
ndp-sim/model_execplan/output/layer0_physic/
```

常见输出包括：

```text
jsons/opX_*.json
config/opX/parsed_bitstream.txt
config/opX/mapping_review.json
install/execplan_opX.txt
install/opX/sliceXX/matrix_A_linearized_128bit.txt
install/opX/sliceXX/matrix_B_linearized_128bit.txt
install/opX/sliceXX/matrix_C_linearized_128bit.txt
install/opX/sliceXX/matrix_D_linearized_128bit.txt
sca_cfg.json
sca_cfg_D.json
instructions_explained.txt
```

## 6. 和数据 relayout 的关系

`layer0_physic.json` 只描述完整算子图和物理 remapping，不负责生成 tensor 数据。

数据拼接和 relayout 主要由：

```text
ndp-sim/generate_python_golden/single_op_data/relayout_layer0.py
```

完成。这个脚本会参考：

```text
ndp-sim/model_execplan/layer0_op_listing.json
```

把各个单算子目录下已经 relayout 好的数据拼到：

```text
ndp-sim/model_execplan/data/layer0_physic/install/
```

所以推荐顺序是：

1. 先用 `gen_layer0_oplist.py` 生成 `layer0.json` 和 `layer0_op_listing.json`。
2. 再用 `address_remapping.cli fill-remapping` 生成 `layer0_physic.json`。
3. 再跑各单算子的 golden/relayout 脚本。
4. 最后用 `relayout_layer0.py` 拼完整 layer0 数据。

## 7. 最常用命令汇总

从 `LLM_python_golden` 目录开始：

```bash
cd ndp-sim

# 1. 由 op_json 模板拼完整逻辑图
python model_execplan/gen_layer0_oplist.py

# 2. 填入物理 remapping，生成 layer0_physic.json
PYTHONPATH=address_remapping/src python -m address_remapping.cli fill-remapping \
  model_execplan/layer0.json \
  --output model_execplan/layer0_physic.json

# 3. 用 layer0_physic.json 生成 execution plan
python model_execplan/main.py model_execplan/layer0_physic.json
```
