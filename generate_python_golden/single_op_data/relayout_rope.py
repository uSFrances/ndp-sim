import numpy as np
import os
import glob
import re
import struct
import json

MODEL_PARAMS = {
    "hidden_size": 896,
    "intermediate_size": 1792,
    "num_attention_heads": 7,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "sequence_length": 32,
    "slice_per_head": 4,
    "used_slices": 28,
    "kv_padding": 256,
}

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

def load_model_params(config_path=CONFIG_PATH):
    params = dict(MODEL_PARAMS)
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Config not found, using defaults: {config_path}")
        return params

    for key in params:
        if key in config:
            params[key] = config[key]
    return params

MODEL_PARAMS = load_model_params()

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def float16_to_bin(f):
    return bin(struct.unpack('<H', struct.pack('<e', np.float16(f)))[0])[2:].zfill(16)

def dtype_from_filename(filepath):
    match = re.search(r"_dtype_(f16|f32|float16|float32)", os.path.basename(filepath).lower())
    if not match:
        raise ValueError(f"Cannot determine dtype from filename: {filepath}")
    return np.float16 if match.group(1) in ("f16", "float16") else np.float32

def convert_to_decimal_txt(bin_path, rows=None, cols=None, file_dtype=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）；对于 relayout 后的数据一维展开"""
    if file_dtype is None:
        file_dtype = dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=file_dtype)
    
    # 执行过 relayout 后的数据，直接按一维展开输出
    if "beforerelayout" not in bin_path:
        txt_path = bin_path.replace('.bin', '_decimal_1d.txt')
        with open(txt_path, 'w') as f:
            f.write("\n".join(f"{float(v):.10g}" for v in data) + "\n")
        return

    if rows is None or cols is None:
        rows, cols = data.size, 1
    if rows * cols != data.size:
        print(f"  ⚠️ Decimal reshape mismatch: {bin_path}, fallback to Nx1")
        rows, cols = data.size, 1

    # 统一按固有的 F-style 解释原始物理二维形状
    matrix = data.reshape((rows, cols), order='F')
    txt_path = bin_path.replace('.bin', f'_decimal_{rows}x{cols}.txt')
    with open(txt_path, 'w') as f:
        for r in range(rows):
            f.write(",".join(f"{float(v):.10g}" for v in matrix[r]))
            f.write("\n")

def convert_to_128bit_txt(bin_path, rows=None, cols=None, file_dtype=None):
    """按真实 dtype 输出每行 128-bit：8个float16 或4个float32。"""
    if file_dtype is None:
        file_dtype = dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=file_dtype)
    values_per_line = 8 if file_dtype == np.float16 else 4
    remainder = len(data) % values_per_line
    if remainder:
        data = np.concatenate(
            (data, np.zeros(values_per_line - remainder, dtype=file_dtype))
        )

    txt_path = bin_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as f:
        for i in range(0, len(data), values_per_line):
            converter = float16_to_bin if file_dtype == np.float16 else float_to_bin
            bins = [converter(value) for value in data[i:i + values_per_line]]
            f.write("".join(reversed(bins)) + "\n")

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols, file_dtype=file_dtype)

def relayout_slice_M8_N(slice_data):
    """
    对一个子切片进行硬件标准的 M8_N 重排。
    【核心修正】：这里的输入维度直接对应物理算子轴，即 (物理N维, 物理M维)。
    因为底层数据是 F-style（列优先）存储，所以第二维（物理M维）才是内存中变化最快的连续轴。
    
    重排规则：外层按步长 8 遍历变化最快的物理 M 轴，中层遍历物理 N 轴，内层连续读取 8 个连续的 M 轴元素。
    """
    phys_N, phys_M = slice_data.shape
    relayout_data = []
    
    # 外层按步长 8 遍历变化最快的物理 M 轴
    for m_outer in range(0, phys_M, 8):
        limit = min(m_outer + 8, phys_M)
        # 中层遍历物理 N 轴（切分轴）
        for n_idx in range(phys_N):
            # 提取连续的 M 轴切片块（对 F-style 而言，此块在底层内存完全连续）
            block = slice_data[n_idx, m_outer:limit]
            relayout_data.extend(block)
            
    return np.array(relayout_data, dtype=slice_data.dtype)

def get_op_id(filename):
    """根据文件名将 rope 子操作映射 to op 文件夹"""
    if "mul_cos" in filename: return "op0"  # 第第一个 mul
    if "mul_sin" in filename: return "op1"  # 第二个 mul
    if "add_final" in filename: return "op2" # 最终 add
    return "unknown_op"

def get_matrix_name(filename):
    """根据输入输出类型映射到硬件的端口名 A / B / D"""
    if "_in0" in filename: return "A"
    if "_in1" in filename: return "B"
    if "_out" in filename: return "D"
    return "unknown_matrix"

def save_before_relayout(before_install_dir, op_id, slice_idx, out_name, slice_data):
    """保存切分后、重排前的数据状态"""
    slice_dir = os.path.join(before_install_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    out_path = os.path.join(slice_dir, out_name)
    
    # 严格按照当前切片真实的内存顺序直接导出
    slice_data.tofile(out_path)
    convert_to_128bit_txt(
        out_path,
        rows=slice_data.shape[0],
        cols=slice_data.shape[1],
        file_dtype=slice_data.dtype,
    )

def split_op1_matrix_b_slices(matrix_2d, inferred_slices):
    """
    op1/matrix_B 专用：
    物理数据为 (128, 32)，在物理 N 维度（行）上均匀拆成等份，后续广播。
    """
    return list(np.array_split(matrix_2d, inferred_slices, axis=0))

def infer_rope_params(group_files, target_prefix):
    """
    从文件名精准提取物理参数：
    如果是 128x7x32x1 这种多头结构：
    - total_n = 128
    - num_heads = 文件中的 head 轴
    - tile_m = 32
    如果是 128x32x1x1 这种单头拉平结构：
    - total_n = 128
    - num_heads = 1
    - tile_m = 32
    """
    for fp in group_files:
        filename = os.path.basename(fp)
        if not filename.startswith(target_prefix):
            continue
            
        m = re.search(r"_shape([\dx]+)_dtype", filename)
        if m:
            # 过滤除去除末尾或外围为1的轴
            dims = [int(x) for x in m.group(1).split('x') if int(x) != 1]
            
            if len(dims) == 3:
                # 对应 128x7x32 的标准结构
                return dims[0], dims[2], dims[1]  # total_n, tile_m, num_heads
                
            elif len(dims) == 2:
                # 对应 128x32 的无独立头平面结构（即单头 head=1）
                return dims[0], dims[1], 1
                
    return (
        MODEL_PARAMS["head_dim"],
        MODEL_PARAMS["sequence_length"],
        MODEL_PARAMS["num_attention_heads"],
    )


def process_rope_tensors(input_dir, output_dir):
    """
    处理 rope 生成的所有 sub_op .bin 文件。
    """
    print(f"🚀 Starting RoPE tensor relayout in: {input_dir}")
    
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    valid_files = [f for f in bin_files if "_subop-" in os.path.basename(f) and "rope" in os.path.basename(f)]
    if not valid_files:
        print("❌ No valid rope subop .bin files found in the directory.")
        return

    prefixes = sorted(list(set([os.path.basename(f).split("_subop-")[0] for f in valid_files])))

    for target_prefix in prefixes:
        group_files = [f for f in valid_files if os.path.basename(f).startswith(target_prefix)]
        if not group_files:
            continue

        total_n, tile_m, inferred_heads = infer_rope_params(group_files, target_prefix)
        
        # 当 inferred_heads == 1 时，属于无独立 head 轴数据，
        # 按照设计：强行虚拟为 num_attention_heads 组。
        is_head_one_broadcasting = (inferred_heads == 1)
        current_num_heads = MODEL_PARAMS["num_attention_heads"] if is_head_one_broadcasting else inferred_heads
        
        slices_per_head = MODEL_PARAMS["slice_per_head"]
        
        print(f"🎯 Processing instance group: '{target_prefix}'")
        print(f"  🧩 [LOCAL PARAM] N={total_n}, M={tile_m}, LogicHeads={current_num_heads}, SlicesPerHead={slices_per_head} (IsBroadcasting={is_head_one_broadcasting})")

        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        for filepath in group_files:
            filename = os.path.basename(filepath)

            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match:
                continue
                
            shape_dims = [int(x) for x in match.group(1).split('x') if int(x) != 1]
            if len(shape_dims) == 1:
                shape_dims = (shape_dims[0], 1)
            else:
                shape_dims = tuple(shape_dims)

            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if op_id == "unknown_op" or matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            
            # 严格根据 F-style 排布恢复出完整形状视图
            file_dtype = dtype_from_filename(filename)
            data = np.fromfile(filepath, dtype=file_dtype).reshape(shape_dims, order='F')
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | F-view Shape: {data.shape}")

            # ----------------------------------------------------
            # 分支 A：原文件带有独立 head 轴的数据
            # ----------------------------------------------------
            if not is_head_one_broadcasting:
                # 形状为 (N=128, H=7, M=32)
                for head_idx in range(current_num_heads):
                    # 抽取特定 Head 的 2D 物理视图，尺寸为 (128, 32)，即 (物理N, 物理M)
                    head_data = data[:, head_idx, :]
                    
                    # 在物理 N 轴（128维度，即切分轴）上均匀分成 4 份
                    slices_list = np.array_split(head_data, slices_per_head, axis=0)
                    
                    for slice_in_group_idx, slice_data in enumerate(slices_list):
                        global_slice_idx = head_idx * slices_per_head + slice_in_group_idx

                        save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                        slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        
                        relayout_data = relayout_slice_M8_N(slice_data)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1], file_dtype=relayout_data.dtype)

            # ----------------------------------------------------
            # 分支 B：单头无独立轴平面数据，执行 num_attention_heads x slice_per_head 广播复制
            # ----------------------------------------------------
            else:
                # 此时 data 的基本 2D 尺寸即为 (128, 32)，也就是 (物理N, 物理M)
                data_2d = data.squeeze()
                
                # 规则：如果它是 op0, op2 全阵，或者是 op1 的 A/D 端口（具备完整的128全长序列信息）
                if (op_id in ("op0", "op2")) or (op_id == "op1" and matrix_id in ("A", "D")):
                    # 先在物理 N 维度（128轴）上均匀切分成 4 个基础分片
                    base_slices = np.array_split(data_2d, slices_per_head, axis=0)
                    
                    # 广播机制：外层循环 7 个虚拟头，生成总计 28 个一模一样的数据分片并完成重排
                    for head_idx in range(current_num_heads):
                        for slice_in_group_idx, slice_data in enumerate(base_slices):
                            global_slice_idx = head_idx * slices_per_head + slice_in_group_idx
                            
                            save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            
                            relayout_data = relayout_slice_M8_N(slice_data)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1], file_dtype=relayout_data.dtype)

                # 规则：如果是 op1 专属的控制参数矩阵 B （平面大小同样为 128x32）
                elif op_id == "op1" and matrix_id == "B":
                    matrix_b_slices = split_op1_matrix_b_slices(data_2d, slices_per_head)
                    
                    for head_idx in range(current_num_heads):
                        for slice_in_group_idx, slice_data in enumerate(matrix_b_slices):
                            global_slice_idx = head_idx * slices_per_head + slice_in_group_idx
                            
                            save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            
                            relayout_data = relayout_slice_M8_N(slice_data)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1], file_dtype=relayout_data.dtype)

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All RoPE groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "rope"))
    process_rope_tensors(input_dir, output_dir)
