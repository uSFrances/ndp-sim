import json
import os
import shutil
import glob
from pathlib import Path

def get_category_and_prefix(template_name, base_data_dir):
    """
    根据模板名推导 category 和 prefix。
    gemm_local 会优先在 gemm_local 目录下找包含 mul_mat 的前缀。
    """
    lower_name = template_name.lower()
    if "rmsnorm" in lower_name:
        category = "rmsnorm"
    elif "softmax" in lower_name:
        category = "softmax"
    elif "rope" in lower_name:
        category = "rope"
    elif "gemm_local" in lower_name or "attn" in lower_name:
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

def build_layer0():
    current_dir = Path(__file__).resolve().parent
    base_data_dir = current_dir.parent.parent / "model_execplan" / "data"
    mapping_file = current_dir.parent.parent / "model_execplan" / "layer0_mapping.json"
    layer0_install_dir = base_data_dir / "layer0" / "install"

    if not mapping_file.exists():
        print(f"❌ Mapping file not found: {mapping_file}")
        return

    with mapping_file.open("r", encoding="utf-8") as f:
        op_mapping = json.load(f)

    if layer0_install_dir.exists():
        shutil.rmtree(layer0_install_dir)
    layer0_install_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting to assemble layer0 in: {layer0_install_dir}")

    for new_op, template_info in op_mapping.items():
        parts = template_info.split("::")
        if len(parts) != 2:
            print(f"  ⚠️ Invalid map info: {template_info} for {new_op}")
            continue

        template_name, old_op = parts
        category, prefix = get_category_and_prefix(template_name, base_data_dir)

        if not prefix:
            print(f"  ⚠️ Could not find prefix for {template_name} in category {category}. Skipping {new_op}.")
            continue

        src_op_dir = base_data_dir / category / prefix / "install" / old_op
        dst_op_dir = layer0_install_dir / new_op

        if src_op_dir.exists():
            shutil.copytree(src_op_dir, dst_op_dir)
            print(f"✅ Merged [{new_op}]: {category}/{prefix} ({old_op}) -> {dst_op_dir.relative_to(base_data_dir)}")
        else:
            print(f"  ❌ Source NOT FOUND: {src_op_dir}")

    print("\n🎉 layer0 assembly complete!")

if __name__ == "__main__":
    build_layer0()
