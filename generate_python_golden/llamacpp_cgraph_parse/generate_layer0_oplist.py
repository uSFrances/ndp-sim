import os
import copy
import json
import argparse


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
    "add_fp32MN_fp16MN_fp32MN_residual": "prefill_add_fp32MN_fp16MN_fp32MN_residual.json",
    "add_fp32MN_fp16MN_fp32MN_out": "prefill_add_fp32MN_fp16MN_fp32MN_out.json",

    "remote_sum_fp32MN_fp32MN": "prefill_remote_sum_4slice_fp32MN_fp32MN.json",

    "silu": "prefill_silu_fp16MN_fp32MN.json"
}


def get_fp_type(str: str) -> str:
    result = "f32"
    match str:
        case "f16":
            result = "fp16"
        case "f32":
            result = "fp32"
        case _:
            raise ValueError(f"Unsupported float type: {str}.")
    return result


def get_oplist(cgraph_path: str) -> list[str]:
    # Load cgraph.json
    cgraph = {}
    if not os.path.exists(cgraph_path):
        raise FileNotFoundError(f"找不到cgraph文件: {cgraph_path}")
    with open(cgraph_path, 'r') as f:
        cgraph = json.load(f)

    n_nodes = cgraph.get("n_nodes", 0)
    nodes = cgraph.get("nodes", [])
    leafs = cgraph.get("leafs", [])

    def get_node(id: str) -> dict:
        type, idx = id.split(".")
        if type == "node":
            return nodes[int(idx)]
        elif type == "leaf":
            return leafs[int(idx)]
        else:
            raise ValueError(f"Invalid node id: {id}.")

    oplist = []
    # Parse cgraph
    for node in nodes:
        id = node.get("id")
        name = node.get("name")
        dtype_str = node.get("dtype")
        shape = node.get("ne", [])
        op = node.get("op")
        srcs = node.get("srcs", [])

        # TODO: only generate one layer now!
        if name == "norm-1":
            break

        match op:
            case "RMS_NORM":
                # TODO: rmsnorm_kv
                oplist.append("rmsnorm")
            case "MUL":
                # TODO: Unsupported type combination?
                # TODO: mul_kv select
                # parse fp type
                #src0 = get_node(srcs[0])
                #src1 = get_node(srcs[1])
                #src0_type = get_fp_type(src0.get("dtype", ""))
                #src1_type = get_fp_type(src1.get("dtype", ""))
                #dst_type = get_fp_type(dtype_str)
                #op = f"mul_{src0_type}MN_{src1_type}N_{dst_type}MN"
                op = "mul_fp32MN_fp32N_fp16MN"
                oplist.append(op)
            case "MUL_MAT":
                # TODO: gemm type select
                oplist.append("gemm_ring")
            case "ADD":
                # TODO: Unsupported type combination?
                # TODO: add type select
                # parse fp type
                #src0 = get_node(srcs[0])
                #src1 = get_node(srcs[1])
                #src0_type = get_fp_type(src0.get("dtype", ""))
                #src1_type = get_fp_type(src1.get("dtype", ""))
                #dst_type = get_fp_type(dtype_str)
                #op = f"add_{src0_type}MN_{src1_type}N_{dst_type}MN"
                op = "add_fp16MN_fp32N_fp32MN"
                oplist.append(op)
            case "ROPE":
                # TODO: rope_k
                oplist.append("rope")
            case "SOFT_MAX":
                oplist.append("softmax")
            case "GLU":
                # TODO: not match: silu vs swiglu!!
                oplist.append("silu")
            case _:
                continue
    return oplist


def load_template(op_name: str, template_dir: str) -> dict:
    file_name = ALIAS_TO_FILE.get(op_name, op_name)
    file_path = os.path.join(template_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到模板文件: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
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

def merge_templates(oplist: list[str], template_dir: str, config_path: str) -> tuple[dict, dict]:
    # 1. 读取 params (config.json)
    params = {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # 2. 解析并合并 oplist
    op_idx = 0
    used_slices = 0
    merged_ops = []
    op_mapping = {}
    for i, op_name in enumerate(oplist):
        tpl = load_template(op_name, template_dir)
        used_slices = max(used_slices, int(tpl.get("used_slices", 0)))

        local_id_map = {}
        operators = tpl.get("operators", [])
        current_file_new_ops = []

        # 第一遍：先生成当前文件所有算子的全局新 ID
        for op in operators:
            # generate new id for op
            old_id = op["id"]
            new_id = f"op{op_idx}"
            local_id_map[old_id] = new_id
            op_idx += 1
            
            new_op = copy.deepcopy(op)
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


if __name__ == "__main__":
    # Default arguments
    output_default = os.path.join(os.path.dirname(__file__), "../../model_execplan/examples/layer0_decode.json")
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the cgraph json file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output oplist file path.")
    parser.add_argument("--template", type=str, required=True, help="Directory of template json files.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json.")
    args = parser.parse_args()

    # get oplist
    oplist = get_oplist(args.input)
    # merge templates
    result, op_mapping = merge_templates(oplist, args.template, args.config)

    # Output result
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已生成: {output_path}")

    # 新增：保存 op 到模板文件来源的映射列表
    output_basename : str = os.path.basename(output_path)
    mapping_output_basename = output_basename.replace(".json", "_op_listing.json")
    mapping_output_path = os.path.join(output_dir, mapping_output_basename)
    with open(mapping_output_path, "w", encoding="utf-8") as f:
        json.dump(op_mapping, f, ensure_ascii=False, indent=2)
    print(f"已生成映射: {mapping_output_path}")