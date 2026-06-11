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

def float16_to_bin(f):
    """将单个 float16 转换为 16 位二进制字符串"""
    return bin(struct.unpack('<H', struct.pack('<e', np.float16(f)))[0])[2:].zfill(16)

def float32_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', np.float32(f)))[0])[2:].zfill(32)

def dtype_from_filename(filepath):
    match = re.search(r"_dtype_(f16|f32|float16|float32)", os.path.basename(filepath).lower())
    if not match:
        raise ValueError(f"Cannot determine dtype from filename: {filepath}")
    return np.float16 if match.group(1) in ("f16", "float16") else np.float32

def convert_to_decimal_txt(bin_path, rows=None, cols=None, dtype=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）；对于 relayout 后的数据一维展开"""
    if dtype is None:
        dtype = dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=dtype)
    
    if "beforerelayout" not in bin_path:
        txt_path = bin_path.replace('.bin', '_decimal_1d.txt')
        with open(txt_path, 'w') as f:
            f.write("\n".join(f"{float(v):.17g}" for v in data) + "\n")
        return

    if rows is None or cols is None:
        rows, cols = data.size, 1
    if rows * cols != data.size:
        print(f"  ⚠️ Decimal reshape mismatch: {bin_path}, fallback to Nx1")
        rows, cols = data.size, 1

    matrix = data.reshape((rows, cols), order='F')
    txt_path = bin_path.replace('.bin', f'_decimal_{rows}x{cols}.txt')
    with open(txt_path, 'w') as f:
        for r in range(rows):
            f.write(",".join(f"{float(v):.17g}" for v in matrix[r]))
            f.write("\n")

def convert_to_128bit_txt(bin_path, rows=None, cols=None, dtype=None):
    """读取 bin 文件并输出为每行 128-bit (8个float16 或 4个float32) 的 txt 文件(二进制格式)"""
    if dtype is None:
        dtype = dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=dtype)

    if dtype == np.float16:
        remainder = len(data) % 8
        if remainder != 0:
            data = np.concatenate((data, np.zeros(8 - remainder, dtype=dtype)))

        txt_path = bin_path.replace('.bin', '.txt')
        with open(txt_path, 'w') as f:
            for i in range(0, len(data), 8):
                # 保持与原先一致的高位在左：最后一个元素拼在最左侧
                bins = [float16_to_bin(data[i + j]) for j in range(8)]
                f.write("".join(reversed(bins)) + "\n")
    else:  # np.float32
        remainder = len(data) % 4
        if remainder != 0:
            data = np.concatenate((data, np.zeros(4 - remainder, dtype=dtype)))

        txt_path = bin_path.replace('.bin', '.txt')
        with open(txt_path, 'w') as f:
            for i in range(0, len(data), 4):
                bins = [float32_to_bin(data[i + j]) for j in range(4)]
                f.write("".join(reversed(bins)) + "\n")
            
    convert_to_decimal_txt(bin_path, rows=rows, cols=cols, dtype=dtype)

# ==============================================================================
# 🚀 【核心修正】按照“左侧变化最快(最内层)，右侧变化最慢(最外层)”重建的重排函数
# ==============================================================================

def relayout_in0_N8M2N4(slice_data):
    """
    in0 硬件新规: N8M2N4 (从右往左看：外层 N4 -> 中外层 M2 -> 内层 N8变化最快)
    结合分块边界保护：
      - 最外层 n_outer 以 8*4=32 为步长遍历 N 轴 (对应 N4)
      - 次外层 m_outer 以 2 为步长遍历 M 轴 (对应 M2)
      - 最内层 n_inner 连续取出 8 个变化最快的 N 轴元素 (对应 N8)
    """
    phys_N, phys_M = slice_data.shape
    relayout_data = []

    # 最外层：N 方向大步长 (N4 组合级，每大步包含 4 个 N8，即步长 32)
    for n_outer in range(0, phys_N, 32):
        # 中层：M 方向步长 2 (M2)
        for m_outer in range(0, phys_M, 2):
            # 展开 N4 内部的 4 个子块
            for n_sub_idx in range(4):
                n_block_start = n_outer + n_sub_idx * 8
                
                # 遍历 M2 内部的 2 个位置
                for m_offset in range(2):
                    m_idx = m_outer + m_offset
                    
                    if m_idx < phys_M:
                        # 最内层：N 轴变化最快，连续取出 8 个元素 (N8)
                        for n_offset in range(8):
                            n_idx = n_block_start + n_offset
                            if n_idx < phys_N:
                                relayout_data.append(slice_data[n_idx, m_idx])
                            else:
                                relayout_data.append(0.0) # 边界对齐补零
                    else:
                        # M轴越界补零
                        relayout_data.extend([0.0] * 8)
                        
    return np.array(relayout_data, dtype=slice_data.dtype)


def relayout_in1_M8N2M4(slice_data):
    """
    in1 硬件新规: M8N2M4 (从右往左看：外层 M4 -> 中外层 N2 -> 内层 M8变化最快)
    结构对称：
      - 最外层 m_outer 以 8*4=32 为步长遍历 M 轴 (对应 M4)
      - 次外层 n_outer 以 2 为步长遍历 N 轴 (对应 N2)
      - 最内层 m_inner 连续取出 8 个变化最快的 M 轴元素 (对应 M8)
    """
    phys_N, phys_M = slice_data.shape
    relayout_data = []

    # 最外层：M 方向大步长 (M4 组合级，每大步包含 4 个 M8，即步长 32)
    for m_outer in range(0, phys_M, 32):
        # 中层：N 方向步长 2 (N2)
        for n_outer in range(0, phys_N, 2):
            # 展开 M4 内部的 4 个子块
            for m_sub_idx in range(4):
                m_block_start = m_outer + m_sub_idx * 8
                
                # 遍历 N2 内部的 2 个位置
                for n_offset in range(2):
                    n_idx = n_outer + n_offset
                    
                    if n_idx < phys_N:
                        # 最内层：M 轴变化最快，连续取出 8 个元素 (M8)
                        for m_offset in range(8):
                            m_idx = m_block_start + m_offset
                            if m_idx < phys_M:
                                relayout_data.append(slice_data[n_idx, m_idx])
                            else:
                                relayout_data.append(0.0)
                    else:
                        relayout_data.extend([0.0] * 8)

    return np.array(relayout_data, dtype=slice_data.dtype)


def relayout_out_M8N8M4N4(slice_data):
    """
    out 硬件新规: M8N8M4N4 (从右往左看：外层 N4 -> 次外层 M4 -> 中层 N8 -> 内层 M8变化最快)
    循环嵌套严格对应（从外到内）：
      - 物理 N 轴大步长 8*4=32 (对应 N4)
      - 物理 M 轴大步长 8*4=32 (对应 M4)
      - 物理 N 轴中步长 8 (对应 N8)
      - 物理 M 轴内步长 8 (对应 M8，内层最快)
    """
    phys_N, phys_M = slice_data.shape
    relayout_data = []

    # 最外层：N4 (步长 32)
    for n_outer32 in range(0, phys_N, 32):
        # 次外层：M4 (步长 32)
        for m_outer32 in range(0, phys_M, 32):
            
            # 展开 N4 内部的 4 个 N8 子块
            for n_sub_idx in range(4):
                n_block_start = n_outer32 + n_sub_idx * 8
                
                # 展开 M4 内部的 4 个 M8 子块
                for m_sub_idx in range(4):
                    m_block_start = m_outer32 + m_sub_idx * 8
                    
                    # 进入各子块内部细化排布：先遍历 N8，最内层为变化最快的 M8
                    for n_offset in range(8):
                        n_idx = n_block_start + n_offset
                        
                        if n_idx < phys_N:
                            # 最内层：M 轴变化最快，连续抽出 8 个连续元素
                            for m_offset in range(8):
                                m_idx = m_block_start + m_offset
                                if m_idx < phys_M:
                                    relayout_data.append(slice_data[n_idx, m_idx])
                                else:
                                    relayout_data.append(0.0)
                        else:
                            relayout_data.extend([0.0] * 8)

    return np.array(relayout_data, dtype=slice_data.dtype)

# ==============================================================================

def get_op_id(filename):
    return "op0"

def get_matrix_name(filename):
    # 交换 in0 和 in1 对应的矩阵名字
    if "_in0" in filename: return "B"
    if "_in1" in filename: return "A"
    if "_out" in filename: return "D"
    return "unknown_matrix"

def save_before_relayout(before_install_dir, op_id, slice_idx, out_name, matrix_2d, dtype=None):
    if dtype is None:
        dtype = matrix_2d.dtype
    matrix_2d = np.asarray(matrix_2d, dtype=dtype)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)
    slice_dir = os.path.join(before_install_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    out_path = os.path.join(slice_dir, out_name)
    matrix_2d.reshape(-1, order='C').tofile(out_path)
    convert_to_128bit_txt(out_path, rows=matrix_2d.shape[0], cols=matrix_2d.shape[1], dtype=dtype)

def infer_gemm_local_params(group_files, target_prefix):
    is_attn_scores = "attn_scores" in target_prefix
    K, L, H = None, None, None

    for fp in group_files:
        fn = os.path.basename(fp)
        m = re.search(r"_shape([\dx]+)_dtype", fn)
        if not m:
            continue
        dims = [int(x) for x in m.group(1).split('x')]

        if "_in1" in fn and len(dims) >= 3:
            H = dims[2]

        if "_in0" in fn and len(dims) >= 2:
            if is_attn_scores:
                K, L = dims[0], dims[1]
            else:
                L, K = dims[0], dims[1]

    return K, L, H

def process_gemm_local_tensors(input_dir, output_dir):
    print(f"🚀 Starting GEMM-Local tensor relayout in: {input_dir}")
    
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    
    valid_files = [f for f in bin_files if ("gemm_local" in os.path.basename(f) or 
                                             "attn_scores" in os.path.basename(f) or 
                                             "attn_out" in os.path.basename(f)) and 
                                            ("_in" in os.path.basename(f) or "_out" in os.path.basename(f))]
    if not valid_files:
        print("❌ No valid gemm_local .bin files found in the directory.")
        return
    
    print(f"✓ Found {len(valid_files)} valid gemm_local files")

    def extract_prefix(f):
        base = os.path.basename(f)
        match = re.match(r'^(.+?)(_in\d+|_out)', base)
        if match: return match.group(1)
        return re.split(r'_shape', base)[0]

    prefixes = sorted(list(set([extract_prefix(f) for f in valid_files])))

    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")
        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        group_files = [f for f in valid_files if extract_prefix(f) == target_prefix]
        if not group_files: continue

        K, L, H = infer_gemm_local_params(group_files, target_prefix)
        if K is None or L is None or H is None:
            continue

        is_attn_scores_group = "attn_scores" in target_prefix
        slices_per_head = MODEL_PARAMS["slice_per_head"]
        num_slices = slices_per_head * H
        if K % slices_per_head != 0:
            print(f"  ⚠️ Skip {target_prefix}: K={K} is not divisible by slice_per_head={slices_per_head}")
            continue
        slice_k = K // slices_per_head
        print(f"  🧩 [LOCAL PARAM] K={K}, L={L}, H={H}, SlicesPerHead={slices_per_head}")

        # 删除专用 remote_sum 块，直接跑下面的常规解析流程

        for filepath in group_files:
            filename = os.path.basename(filepath)
            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match: continue
            
            shape = tuple(map(int, match.group(1).split('x')))
            
            # 严格读取文件名中的 dtype，不再根据算子组或 in/out 猜测。
            file_dtype = dtype_from_filename(filename)
            data = np.fromfile(filepath, dtype=file_dtype).reshape(shape, order='F')
            
            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if matrix_id == "unknown_matrix": continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | Shape: {shape}")

            data_2d = data.squeeze()
            if data_2d.ndim == 0: data_2d = data_2d.reshape(1, 1)
            elif data_2d.ndim == 1: data_2d = data_2d.reshape(-1, 1)

            # ==================== attn_scores 分支 ====================
            if is_attn_scores_group:
                # B 端口（原A，对应 _in0）：无 H 轴广播复制给全 4xH=28 个 slice
                if matrix_id == "B" and data_2d.shape[:2] == (K, L):
                    for i in range(slices_per_head):
                        k_start = i * slice_k
                        slice_data = data_2d[k_start:k_start + slice_k, :]
                        
                        # 调用已修正的 N8M2N4 规则
                        # 注意：对于 attn_scores 组，in0 的切片是按 K 切出的 (Kslice, L)；
                        # 在 relayout_in0_N8M2N4 中，函数内部把“切片的第二维”当作 M（即 relayout 期望 M=切片长度(K)），
                        # 因此此处需把 slice_data 转置后传入 relayout（不改变保存到 before_relayout 的原始数据方向）。
                        relayout_data = relayout_in0_N8M2N4(slice_data.T)
                        
                        for h_idx in range(H):
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, slice_data, dtype=file_dtype)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1], dtype=file_dtype)

                # A 端口（原B，对应 _in1）
                elif matrix_id == "A":
                    data_3d = data.reshape((K, L, H), order='F')
                    for h_idx in range(H):
                        for i in range(slices_per_head):
                            k_start = i * slice_k
                            slice_2d = data_3d[k_start:k_start + slice_k, :, h_idx]
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, slice_2d, dtype=file_dtype)

                            # 调用已修正的 M8N2M4 规则
                            relayout_data = relayout_in1_M8N2M4(slice_2d)
                            
                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_2d.shape[0], cols=slice_2d.shape[1], dtype=file_dtype)

                elif matrix_id == "D":
                    # attn_scores 的 out 数据类型现已被识别并提取为 fp32
                    data_3d = data.reshape((K, L, H), order='F')
                    for h_idx in range(H):
                        for i in range(slices_per_head):
                            k_start = i * slice_k
                            slice_2d = data_3d[k_start:k_start + slice_k, :, h_idx]
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, slice_2d, dtype=file_dtype)

                            relayout_data = relayout_out_M8N8M4N4(slice_2d)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_2d.shape[0], cols=slice_2d.shape[1], dtype=file_dtype)

            # ==================== attn_out 分支 ====================
            else:
                # B 端口（原A，对应 _in0）：无 H 轴广播复制给全 4xH=28 个 slice
                if matrix_id == "B" and data_2d.shape[:2] == (L, K):
                    for i in range(slices_per_head):
                        k_start = i * slice_k
                        slice_data = data_2d[:, k_start:k_start + slice_k]
                        
                        # 调用已修正的 N8M2N4 规则
                        relayout_data = relayout_in0_N8M2N4(slice_data.T)
                        
                        for h_idx in range(H):
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, slice_data, dtype=file_dtype)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1], dtype=file_dtype)

                # A 端口（原B，对应 _in1）
                elif matrix_id == "A":
                    data_3d = data.reshape((L, L, H), order='F')
                    for h_idx in range(H):
                        head_in1 = data_3d[:, :, h_idx]
                        # 调用已修正的 M8N2M4 规则
                        relayout_head = relayout_in1_M8N2M4(head_in1)
                        for i in range(slices_per_head):
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, head_in1, dtype=file_dtype)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_head.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=L, cols=L, dtype=file_dtype)

                elif matrix_id == "D":
                    data_3d = data.reshape((K, L, H), order='F')
                    for h_idx in range(H):
                        for i in range(slices_per_head):
                            k_start = i * slice_k
                            slice_2d = data_3d[k_start:k_start + slice_k, :, h_idx]
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, slice_2d, dtype=file_dtype)

                            # 调用已修正的 M8N8M4N4 规则
                            relayout_data = relayout_out_M8N8M4N4(slice_2d)
                            
                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_2d.shape[0], cols=slice_2d.shape[1], dtype=file_dtype)

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All GEMM-Local groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "gemm_local"))
    process_gemm_local_tensors(input_dir, output_dir)
