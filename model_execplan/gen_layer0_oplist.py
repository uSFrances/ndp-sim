import argparse
import copy
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OP_JSON_DIR = BASE_DIR / "op_json"

# 直接在这里改你要拼的 op 顺序
DEFAULT_OPLIST = [
    "rmsnorm",
    "mul_fp32MN_fp32N_fp16MN",
    "gemm_ring",
    "add_fp16MN_fp32N_fp32MN",
    "rope",

    "rmsnorm_kv",
    "mul_fp32MN_fp32N_fp16MN_kv",
    "gemm_ring_k",
    "add_fp16MN_fp32N_fp32MN_k",
    "rope_k",

    "gemm_ring_v",
    "add_V_fp16MN_fp32N_fp16MN",
    # "prefill_qkt_kt_view_fp16_fp16"

    "gemm_local_qkt",
    "remote_sum_fp16MN_fp32MN",
    "softmax",
    "gemm_local_sv",
    "gemm_ring",

    "add_fp32MN_fp16MN_fp32MN_residual",
    "rmsnorm",
    "mul_fp32MN_fp32N_fp16MN",
    "gemm_ring_ffn_gate",
    "gemm_ring_ffn_up",
    "silu",
    "mul_fp32MN_fp16MN_fp16MN",
    "gemm_ring_ffn_out",
    "add_fp32MN_fp16MN_fp32MN_out"
]

# 可识别 op 名称到模板文件的映射
ALIAS_TO_FILE = {
    "softmax": "softmax.json",
    "rope": "rope.json",
    "rope_k": "rope_k.json",
    "rmsnorm": "rmsnorm.json",
    "rmsnorm_kv": "rmsnorm_kv.json",
    "gemm_ring": "gemm_ring.json",
    "gemm_ring_k": "gemm_ring_k.json",
    "gemm_ring_v": "gemm_ring_v.json",
    "gemm_ring_ffn_gate": "gemm_ring_ffn_gate.json",
    "gemm_ring_ffn_up": "gemm_ring_ffn_up.json",
    "gemm_ring_ffn_out": "gemm_ring_ffn_out.json",
    "gemm_local_qkt": "gemm_local_qkt.json",
    "gemm_local_sv": "gemm_local_sv.json",
    "layer0": "layer0.json",

    "mul_fp32MN_fp32N_fp16MN": "prefill_mul_fp32MN_fp32N_fp16MN.json",
    "mul_fp32MN_fp32N_fp16MN_kv": "prefill_mul_fp32MN_fp32N_fp16MN_kv.json",
    "add_fp16MN_fp32N_fp32MN": "prefill_add_fp16MN_fp32N_fp32MN.json",
    "add_fp16MN_fp32N_fp32MN_k": "prefill_add_fp16MN_fp32N_fp32MN_k.json",
    "add_V_fp16MN_fp32N_fp16MN": "prefill_add_V_fp16MN_fp32N_fp16MN.json",
    "mul_fp32MN_fp16MN_fp16MN": "prefill_mul_fp32MN_fp16MN_fp16MN.json",
    "add_fp32MN_fo32MN_fp16MN": "prefill_add_fp32MN_fp32MN_fp16MN.json",
    "add_fp32MN_fp16MN_fp32MN_residual": "prefill_add_fp32MN_fp16MN_fp32MN_residual.json",
    "add_fp32MN_fp16MN_fp32MN_out": "prefill_add_fp32MN_fp16MN_fp32MN_out.json",

    "remote_sum_fp16MN_fp32MN": "prefill_remote_sum_4slice_fp16MN_fp32MN.json",

    "silu": "prefill_silu_fp16MN_fp32MN.json"
}


def load_template(op_name: str) -> dict:
    file_name = ALIAS_TO_FILE.get(op_name, op_name)
    path = Path(file_name)
    if not path.is_absolute():
        path = OP_JSON_DIR / path.name

    if not path.exists():
        raise FileNotFoundError(f"找不到模板文件: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rewrite_source_refs(obj, local_id_map, global_ops):
    """
    obj: 当前正在处理的算子字典或其子结构
    local_id_map: 当前文件内部的 ID 映射 (如 "op0" -> "op10")
    global_ops: 已经合并进总列表的所有算子列表 (merged_ops)
    """
    if isinstance(obj, dict):
        if "source" in obj:
            src_val = obj["source"]
            raw_id = None
            
            # 提取原始 ID (字符串或字典格式)
            if isinstance(src_val, str):
                raw_id = src_val
            elif isinstance(src_val, dict) and "type" in src_val:
                raw_id = src_val["type"]

            if raw_id:
                # --- 逻辑 A: 处理负数偏移引用 (op-1, op-2, ...) ---
                if raw_id.startswith("op-"):
                    try:
                        # 提取数字，例如 "op-2" -> 2
                        offset = int(raw_id.split("-")[1])
                        
                        # 检查全局列表中是否有足够的算子可以引用
                        if len(global_ops) >= offset:
                            # 倒数第 offset 个算子的新全局 ID
                            target_op = global_ops[-offset]
                            obj["source"] = target_op["id"]
                        else:
                            # 如果引用超出了范围（比如第一个算子写 op-1），设为 external
                            obj["source"] = {"type": "external"}
                    except (ValueError, IndexError):
                        pass # 格式不对则跳过

                # --- 逻辑 B: 处理正常的内部引用 (op0, op1, ...) ---
                elif raw_id in local_id_map:
                    obj["source"] = local_id_map[raw_id]

        # 递归处理
        for v in obj.values():
            rewrite_source_refs(v, local_id_map, global_ops)
            
    elif isinstance(obj, list):
        for item in obj:
            rewrite_source_refs(item, local_id_map, global_ops)

def merge_templates(oplist: list[str]) -> dict:
    merged_ops = []
    op_idx = 0
    used_slices = 0
    op_mapping = {}

    # 1. 读取 params (config.json)
    params = {}
    config_path = BASE_DIR.parent / "generate_python_golden" / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            params = json.load(f)

    for i, op_name in enumerate(oplist):
        tpl = load_template(op_name)
        used_slices = max(used_slices, int(tpl.get("used_slices", 0)))

        local_id_map = {}
        operators = tpl.get("operators", [])
        current_file_new_ops = []

        # 第一遍：先生成当前文件所有算子的全局新 ID
        for op in operators:
            new_op = copy.deepcopy(op)
            old_id = new_op["id"]
            new_id = f"op{op_idx}"
            op_idx += 1
            
            local_id_map[old_id] = new_id
            new_op["id"] = new_id
            current_file_new_ops.append(new_op)
            
            # 新增：记录当前全局 op 属于哪个原始模板的哪个旧 id
            op_mapping[new_id] = f"{op_name}::{old_id}"

        # 第二遍：重写引用 (此时 merged_ops 包含了之前所有文件的算子)
        for op in current_file_new_ops:
            rewrite_source_refs(op, local_id_map, merged_ops)
        
        # 将当前文件处理完的算子加入全局列表
        merged_ops.extend(current_file_new_ops)

    return {
        "params": params,
        "used_slices": used_slices,
        "operators": merged_ops,
    }, op_mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops",
        nargs="*",
        default=DEFAULT_OPLIST,
        help="要合并的 op 列表，可直接写 softmax/rope/rmsnorm 或 json 文件名",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "layer0.json"),
        help="输出 layer0.json 路径",
    )
    args = parser.parse_args()

    result, op_mapping = merge_templates(args.ops)
    out_path = Path(args.output)
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已生成: {out_path}")

    # 新增：保存 op 到模板文件来源的映射列表
    mapping_out_path = out_path.with_name(out_path.stem + "_mapping.json")
    with mapping_out_path.open("w", encoding="utf-8") as f:
        json.dump(op_mapping, f, ensure_ascii=False, indent=2)
    print(f"已生成映射: {mapping_out_path}")

if __name__ == "__main__":
    main()
