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

    "gemm_ring",
    "add_fp16MN_fp32N_fp32MN",
    "rope",

    "gemm_ring",
    "add_fp16MN_fp32N_fp32MN",
    # "prefill_add_V_MN_N_fp16_fp32_fp16"
    # "prefill_qkt_kt_view_fp16_fp16"

    "gemm_local",
    "softmax",
    "gemm_local",
    "gemm_ring",

    "add_fp32MN_fp16MN_fp32MN",
    "rmsnorm",
    "mul_fp32MN_fp32N_fp16MN",
    "gemm_ring_fnn",
    "silu",
    "gemm_ring_fnn",
    "mul_fp32MN_fp16MN_fp16MN",
    "gemm_ring_fnn",
    "add_fp32MN_fp16MN_fp32MN"
]

# 可识别 op 名称到模板文件的映射
ALIAS_TO_FILE = {
    "softmax": "softmax.json",
    "rope": "rope.json",
    "rmsnorm": "rmsnorm.json",
    "gemm_ring": "gemm_ring.json",
    "gemm_ring_fnn": "gemm_ring_fnn.json",
    "gemm_local": "gemm_local.json",
    "layer0": "layer0.json",

    "mul_fp32MN_fp32N_fp16MN": "prefill_mul_fp32MN_fp32N_fp16MN.json",
    "add_fp16MN_fp32N_fp32MN": "prefill_add_fp16MN_fp32N_fp32MN.json",
    "mul_fp32MN_fp16MN_fp16MN": "prefill_mul_fp32MN_fp16MN_fp16MN.json",
    "add_fp32MN_fo32MN_fp16MN": "prefill_add_fp32MN_fp32MN_fp16MN.json",
    "add_fp32MN_fp16MN_fp32MN": "prefill_add_fp32MN_fp16MN_fp32MN.json",

    "silu": "prefill_silu_fp32MN_fp16MN.json"
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


def rewrite_source_refs(obj, local_id_map, bridge_last_id, state):
    """
    递归处理 source 引用转换
    state: 包含 bridge_used 标记的字典，用于跨递归调用共享状态
    """
    if isinstance(obj, dict):
        if "source" in obj and isinstance(obj["source"], str):
            old_src = obj["source"]
            
            # 只有当 source 明确写为 "op0" 时，考虑跨文件衔接
            if old_src == "op0":
                # 如果有衔接点，且在这个算子组里还没被用过
                if bridge_last_id is not None and not state["bridge_used"]:
                    obj["source"] = bridge_last_id
                    state["bridge_used"] = True  # 标记已消耗，后续 op0 将指向本地
                
                # 如果是第一个文件，且还没处理过起始点，设为 external
                elif bridge_last_id is None and not state["bridge_used"]:
                    obj["source"] = {"type": "external"}
                    state["bridge_used"] = True
                
                # 如果衔接点已经用过了，或者根本没法衔接，则回退到本地映射
                elif old_src in local_id_map:
                    obj["source"] = local_id_map[old_src]
            
            # 普通的内部引用 (如 op1 -> opX)
            elif old_src in local_id_map:
                obj["source"] = local_id_map[old_src]
        
        for v in obj.values():
            rewrite_source_refs(v, local_id_map, bridge_last_id, state)
    elif isinstance(obj, list):
        for item in obj:
            rewrite_source_refs(item, local_id_map, bridge_last_id, state)

def merge_templates(oplist: list[str]) -> dict:
    merged_ops = []
    op_idx = 0
    used_slices = 0
    last_op_id_from_prev_file = None 

    # 1. 读取 params
    # params = {}
    # config_path = BASE_DIR.parent / "generate_python_golden" / "config.json"
    # if config_path.exists():
    #     with config_path.open("r", encoding="utf-8") as f:
    #         params = json.load(f)

    for i, op_name in enumerate(oplist):
        tpl = load_template(op_name)
        used_slices = max(used_slices, int(tpl.get("used_slices", 0)))

        local_id_map = {}
        operators = tpl.get("operators", [])
        current_file_new_ops = []
        
        # 记录当前的衔接点
        bridge_id = last_op_id_from_prev_file

        # 第一遍：先生成所有新 ID 映射
        for op in operators:
            new_op = copy.deepcopy(op)
            old_id = new_op["id"]
            new_id = f"op{op_idx}"
            op_idx += 1
            
            local_id_map[old_id] = new_id
            new_op["id"] = new_id
            current_file_new_ops.append(new_op)

        # 第二遍：重写引用
        # 每个文件重置一次 state，确保 bridge_id 在当前文件只被消耗一次
        state = {"bridge_used": False}
        for op in current_file_new_ops:
            rewrite_source_refs(op, local_id_map, bridge_id, state)
        
        merged_ops.extend(current_file_new_ops)
        if current_file_new_ops:
            last_op_id_from_prev_file = current_file_new_ops[-1]["id"]

    return {
        # "params": params,
        "used_slices": used_slices,
        "operators": merged_ops,
    }

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

    result = merge_templates(args.ops)
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()
