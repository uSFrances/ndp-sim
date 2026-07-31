import json
import os
import shutil
import glob
import re
from pathlib import Path

BEFORE_RELAYOUT_TEMPLATE_NAMES = {
    "prefill_add_fp16mn_fp32n_fp32mn",
    "add_fp16mn_fp32n_fp32mn",
    "prefill_mul_fp32mn_fp32n_fp16mn",
    "mul_fp32mn_fp32n_fp16mn",
}

def _resolve_search_name(lower_name, count):
    search_name = lower_name
    if lower_name == "rmsnorm" and count == 1:
        search_name = "rmsnorm_second"
    elif lower_name == "mul_fp32mn_fp32n_fp16mn" and count == 1:
        search_name = "mul_fp32mn_fp32n_fp16mn_second"
    elif lower_name == "gemm_ring":
        if count == 1:
            search_name = "gemm_ring_second"
    return search_name

def get_category_and_prefix(template_name, base_data_dir, count):
    """
    根据模板名推导 category 和 prefix。
    优先使用精确/关键字映射表，再回退到原有启发式。
    使用了基于 op 编号回滚检测到的 count 来改变搜索名，防止内部多 op 被截断。
    """
    lower_name = template_name.lower()
    search_name = _resolve_search_name(lower_name, count)

    # 精确/关键字映射（优先）
    special_map = {
        # "gemm_ring": (None, None),  # 跳过
        "mul_fp32mn_fp32n_fp16mn_kv": ("mul_MN_N_kv", "blk.0_attn_norm-0_op-mul"),
        "mul_fp32mn_fp32n_fp16mn": ("regular", "blk.0_attn_norm-0_op-mul"),
        "add_fp16mn_fp32n_fp32mn_k": ("regular", "blk.0_Kcur-0-add_op-add"),
        "add_fp16mn_fp32n_fp32mn": ("regular", "blk.0_Qcur-0-add_op-add"),
        "add_v_fp16mn_fp32n_fp16mn": ("regular", "blk.0_Vcur-0-add_op-add"),
        "add_fp32mn_fp16mn_fp32mn_residual": ("regular", "blk.0_ffn_inp-0_op-add"),
        "add_fp32mn_fp16mn_fp32mn_out": ("regular", "blk.0_l"),  # 以你提供的前缀为准
        "mul_fp32mn_fp32n_fp16mn_k": ("mul_MN_N_kv", "blk.0_attn_norm-0_op-mul"),  # 备用映射
        "mul_fp32mn_fp32n_fp16mn_second": ("regular", "blk.0_ffn_norm-0_op-mul"),
        "mul_fp32mn_fp16mn_fp16mn": ("regular", "blk.0_ffn_gate_par-0_op-mul"),
        "rope_k": ("rope", "blk.0_Kcur-0_op-rope"),
        "rope": ("rope", "blk.0_Qcur-0_op-rope"),
        "rmsnorm_kv": ("rmsnorm", "blk.0_norm-0_op-rms_norm_kv"),
        "rmsnorm_second": ("rmsnorm", "blk.0_norm_ffn-0_op-rms_norm"),
        "remote_sum_fp32mn_fp32mn": ("gemm_local", "blk.0_node_0_attn_scores_op-remote_sum"),
        "gemm_local_sv": ("gemm_local", "blk.0_node_0_attn"),
        "silu": ("regular", "blk.0_ffn_silu-0_op-unary"),

        # "gemm_ring": ("gemm_ring", "q_gen"),
        # "gemm_ring_k": ("gemm_ring", "k_gen"),
        # "gemm_ring_v": ("gemm_ring", "v_gen"),
        # "gemm_ring_second": ("gemm_ring", "atten_final"),
        # "gemm_ring_ffn_gate": ("gemm_ring", "ffn_gate"),
        # "gemm_ring_ffn_up": ("gemm_ring", "ffn_up"),
        # "gemm_ring_ffn_out": ("gemm_ring", "ffn_out"),

        # "gemm_ring": ("gemm_local", "blk.0_node_0_attn"),
        # "gemm_ring_k": ("gemm_local", "blk.0_node_0_attn"),
        # "gemm_ring_v": ("gemm_local", "blk.0_node_0_attn"),
        # "gemm_ring_second": ("gemm_local", "blk.0_node_0_attn"),
        # "gemm_ring_ffn_gate": ("gemm_local", "blk.0_node_0_attn"),
        # "gemm_ring_ffn_up": ("gemm_local", "blk.0_node_0_attn"),
        # "gemm_ring_ffn_out": ("gemm_local", "blk.0_node_0_attn"),
    }

    if search_name in special_map:
        return special_map[search_name]

    # 关键字匹配：模板名中包含任意 special_map key 则采用对应映射
    for key, (cat, pref) in special_map.items():
        if key in search_name:
            return cat, pref

    # 原有启发式
    if "rmsnorm" in search_name:
        category = "rmsnorm"
    elif "softmax" in search_name:
        category = "softmax"
    elif "rope" in search_name:
        category = "rope"
    elif "gemm_local" in search_name or "attn" in search_name:
        category = "gemm_local"
    else:
        category = "regular"

    category_dir = os.path.join(base_data_dir, category)
    if not os.path.exists(category_dir):
        return category, None

    prefixes = sorted([d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))])

    # gemm_local：优先找 mul_mat 相关前缀
    if category == "gemm_local":
        for p in prefixes:
            if "mul_mat" in p.lower():
                return category, p

    # 其他类别的简单启发式
    for p in prefixes:
        if "ffn" in lower_name and "ffn" in p: return category, p
        if "kv" in lower_name and ("key" in p or "kcur" in p.lower()): return category, p

    return category, prefixes[0] if prefixes else None

def should_use_before_relayout(template_info):
    """根据算子类型判断是否使用未 relayout 的数据。"""
    normalized_name = template_info.lower()
    return normalized_name in BEFORE_RELAYOUT_TEMPLATE_NAMES

def load_layer0_operator_types(layer0_json_path):
    """读取 layer0.json 中每个 op 的完整算子类型。"""
    if not layer0_json_path.exists():
        print(f"⚠️ layer0 json not found: {layer0_json_path}")
        return {}

    with layer0_json_path.open("r", encoding="utf-8") as f:
        layer0_config = json.load(f)

    op_types = {}
    for operator in layer0_config.get("operators", []):
        op_id = operator.get("id")
        op_type = operator.get("type")
        if op_id and op_type:
            op_types[op_id] = op_type
    return op_types

def build_layer0():
    current_dir = Path(__file__).resolve().parent
    base_data_dir = current_dir.parent.parent / "model_execplan" / "data"
    mapping_file = current_dir.parent.parent / "model_execplan" / "layer0_op_listing.json"
    layer0_json_file = current_dir.parent.parent / "model_execplan" / "layer0.json"
    layer0_install_dir = base_data_dir / "layer0" / "install"

    if not mapping_file.exists():
        print(f"❌ Mapping file not found: {mapping_file}")
        return

    with mapping_file.open("r", encoding="utf-8") as f:
        op_mapping = json.load(f)
    op_types = load_layer0_operator_types(layer0_json_file)

    if layer0_install_dir.exists():
        shutil.rmtree(layer0_install_dir)
    layer0_install_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting to assemble layer0 in: {layer0_install_dir}")

    block_counts = {}
    last_ops = {}

    for new_op, template_info in op_mapping.items():
        parts = template_info.split("::")
        if len(parts) != 2:
            print(f"  ⚠️ Invalid map info: {template_info} for {new_op}")
            continue

        template_name, old_op = parts
        lower_name = template_name.lower()

        # 根据 old_op 的编号判断是否进入了新的一个同名 block（比如重新从 op0 开始）
        match = re.search(r'\d+', old_op)
        op_num = int(match.group()) if match else 0

        last_num = last_ops.get(lower_name, -1)
        if op_num <= last_num:
            # 编号跌回或者相等，说明开启了一个新同名模板块
            block_counts[lower_name] = block_counts.get(lower_name, 0) + 1
        last_ops[lower_name] = op_num

        count = block_counts.get(lower_name, 0)
        category, prefix = get_category_and_prefix(template_name, base_data_dir, count)

        # 若该前缀属于 op-mul（mul 算子），强制使用 op0（避免 mapping 写成 op1 导致查找错误）
        if prefix and "_op-mul" in prefix:
            old_op = "op0"

        if not prefix:
            print(f"  ⚠️ Could not find prefix for {template_name} in category {category}. Skipping {new_op}.")
            continue

        op_type = op_types.get(new_op, template_name)
        source_stage = "install_beforerelayout" if should_use_before_relayout(op_type) else "install"
        src_op_dir = base_data_dir / category / prefix / source_stage / old_op
        dst_op_dir = layer0_install_dir / new_op

        if src_op_dir.exists():
            shutil.copytree(src_op_dir, dst_op_dir)
            if source_stage == "install_beforerelayout":
                print(f"🔁 Using install_beforerelayout [{new_op}]: {op_type} -> {category}/{prefix}/{source_stage}/{old_op}")
            print(f"✅ Merged [{new_op}]: {category}/{prefix}/{source_stage} ({old_op}) -> {dst_op_dir.relative_to(base_data_dir)}")
        else:
            print(f"  ❌ Source NOT FOUND: {src_op_dir}")

    print("\n🎉 layer0 assembly complete!")

    # 生成物理布局版本
    create_layer0_physic(layer0_install_dir, op_mapping)

def _inject_gemm_ring_ops_to_physic(phys_install, base_data_dir, op_mapping):
    gemm_ring_map = {
        "gemm_ring": "q_gen",
        "gemm_ring_k": "k_gen",
        "gemm_ring_v": "v_gen",
        "gemm_ring_second": "atten_final",
        "gemm_ring_ffn_gate": "ffn_gate",
        "gemm_ring_ffn_up": "ffn_up",
        "gemm_ring_ffn_out": "ffn_out",
    }

    block_counts = {}
    last_ops = {}

    for new_op, template_info in op_mapping.items():
        parts = template_info.split("::")
        if len(parts) != 2:
            continue

        template_name, old_op = parts
        lower_name = template_name.lower()

        match = re.search(r"\d+", old_op)
        op_num = int(match.group()) if match else 0

        last_num = last_ops.get(lower_name, -1)
        if op_num <= last_num:
            block_counts[lower_name] = block_counts.get(lower_name, 0) + 1
        last_ops[lower_name] = op_num

        count = block_counts.get(lower_name, 0)
        search_name = _resolve_search_name(lower_name, count)

        if search_name not in gemm_ring_map:
            continue

        prefix = gemm_ring_map[search_name]
        src_op_dir = base_data_dir / "gemm_ring" / prefix / "install" / old_op
        if not src_op_dir.exists():
            fallback = base_data_dir / "gemm_ring" / prefix / "install" / "op0"
            if fallback.exists():
                src_op_dir = fallback
            else:
                print(f"  ❌ gemm_ring source NOT FOUND: {src_op_dir}")
                continue

        dst_op_dir = phys_install / new_op
        if dst_op_dir.exists():
            shutil.rmtree(dst_op_dir)
        shutil.copytree(src_op_dir, dst_op_dir)
        print(f"✅ Injected gemm_ring [{new_op}]: gemm_ring/{prefix} ({src_op_dir.name})")

def create_layer0_physic(layer0_install_dir, op_mapping):
    """
    复制 layer0 到 layer0_physic，复原普通数据的 slice 编号后再补充 gemm_ring。
    输入位置 i 的真实 slice ID 为 order[i]，输出按 slice00..slice27 排列。
    """
    base_data_dir = layer0_install_dir.parent.parent  # .../data
    src_layer0 = base_data_dir / "layer0"
    dst_layer0 = base_data_dir / "layer0_physic"

    if not src_layer0.exists():
        print(f"⚠️ source layer0 not found: {src_layer0}")
        return

    if dst_layer0.exists():
        shutil.rmtree(dst_layer0)
    shutil.copytree(src_layer0, dst_layer0)
    phys_install = dst_layer0 / "install"

    # 输入位置 i 对应的真实 slice ID（长度 28）。
    order = [0,2,3,1,5,4,6,7,8,10,11,9,15,14,12,13,16,17,19,18,20,21,23,22,26,24,25,27]

    for op_dir in sorted(phys_install.iterdir()):
        if not op_dir.is_dir():
            continue

        # 收集原始 slice 目录
        orig_slices = {}
        for p in op_dir.glob("slice*"):
            if p.is_dir():
                try:
                    idx = int(p.name.replace("slice", ""))
                    orig_slices[idx] = p
                except ValueError:
                    continue

        # 创建临时目录放重排后的 slice
        temp_dir = op_dir.parent / f"{op_dir.name}_reorder_tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        # 源 slice i 标记为真实 ID order[i]，再按真实 ID 复原为连续目录。
        for src_idx, tgt_idx in enumerate(order):
            src_path = orig_slices.get(src_idx)
            dst_slice = temp_dir / f"slice{tgt_idx:02d}"
            if src_path and src_path.exists():
                shutil.copytree(src_path, dst_slice)
            else:
                # 若源 slice 不存在，创建空目录占位
                dst_slice.mkdir()
                print(f"  ⚠️ Missing slice{src_idx:02d} under {op_dir.name}, created empty slice{tgt_idx:02d}")

        # 删除原有 slice 目录（仅删除匹配的 sliceNNN，而保留其他文件）
        for p in list(op_dir.glob("slice*")):
            if p.is_dir():
                shutil.rmtree(p)

        # 移动新 slice 到原 op 目录
        for new_p in sorted(temp_dir.iterdir()):
            shutil.move(str(new_p), str(op_dir))

        # 清理临时目录
        if temp_dir.exists():
            temp_dir.rmdir()

    # 普通数据重排完成后，再补充/覆盖 gemm_ring；gemm_ring 不参与上述重排。
    _inject_gemm_ring_ops_to_physic(phys_install, base_data_dir, op_mapping)

    print(f"✅ layer0_physic generated at: {dst_layer0}")

if __name__ == "__main__":
    build_layer0()
