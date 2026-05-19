import numpy as np
import os
import glob
import re
import struct

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def convert_to_decimal_txt(bin_path, rows=None, cols=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）；对于 relayout 后的数据一维展开"""
    data = np.fromfile(bin_path, dtype=np.float32)
    
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

    # 统一按 F-style 解释二维形状
    matrix = data.reshape((rows, cols), order='F')
    txt_path = bin_path.replace('.bin', f'_decimal_{rows}x{cols}.txt')
    with open(txt_path, 'w') as f:
        for r in range(rows):
            f.write(",".join(f"{float(v):.10g}" for v in matrix[r]))
            f.write("\n")

def convert_to_128bit_txt(bin_path, rows=None, cols=None):
    """读取 bin 文件并输出为每行 128-bit (4个float32) 的 txt 文件(二进制格式)"""
    data = np.fromfile(bin_path, dtype=np.float32)

    remainder = len(data) % 4
    if remainder != 0:
        data = np.concatenate((data, np.zeros(4 - remainder, dtype=np.float32)))

    txt_path = bin_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as f:
        for i in range(0, len(data), 4):
            str_float0 = float_to_bin(data[i])
            str_float1 = float_to_bin(data[i+1])
            str_float2 = float_to_bin(data[i+2])
            str_float3 = float_to_bin(data[i+3])
            f.write(f"{str_float3}{str_float2}{str_float1}{str_float0}\n")
            
    # 新增调用：生成十进制对照文本
    convert_to_decimal_txt(bin_path, rows=rows, cols=cols)

def relayout_slice_M8_N(slice_data):
    """
    对一个 MxN 的 slice 进行 M8_N 重排。
    """
    M, N = slice_data.shape
    relayout_data = []
    for m_outer in range(0, M, 8):
        limit = min(m_outer + 8, M)
        for n_idx in range(N):
            block = slice_data[m_outer:limit, n_idx]
            relayout_data.extend(block)
    return np.array(relayout_data, dtype=slice_data.dtype)

def get_op_id(filename):
    """Regular 算子一般只是单一算子操作，统一放在 op0 下"""
    return "op0"

def get_matrix_name(filename):
    """根据输入输出类型映射到硬件的端口名 A / B / D"""
    if "_in0" in filename: return "A"
    if "_in1" in filename: return "B"
    if "_out" in filename: return "D"
    return "unknown_matrix"

def save_before_relayout(before_install_dir, op_id, slice_idx, out_name, matrix_2d):
    matrix_2d = np.asarray(matrix_2d, dtype=np.float32)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)
    slice_dir = os.path.join(before_install_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    out_path = os.path.join(slice_dir, out_name)
    matrix_2d.reshape(-1, order='C').tofile(out_path)
    convert_to_128bit_txt(out_path, rows=matrix_2d.shape[0], cols=matrix_2d.shape[1])

def process_regular_tensors(input_dir, output_dir):
    """
    处理通用生成的 sub_op .bin 文件。
    自动从文件名读取数据尺寸，默认按 N 维度把数据裁剪分给 28 个 slice 并采用 M8_N 排列。
    """
    print(f"🚀 Starting Regular tensor relayout in: {input_dir}")
    
    num_slices = 28
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))

    # 移除 _subop- 限制，只要包含 _shape 即可被认为是张量数据
    # 并且跳过指定的几个算子
    skip_keywords = ["rmsnorm", "rms_norm", "softmax", "soft_max", "rope", "mul_mat"]
    
    valid_files = []
    for f in bin_files:
        basename = os.path.basename(f)
        if "_shape" not in basename:
            continue
        if any(kw in basename.lower() for kw in skip_keywords):
            continue
        valid_files.append(f)
        
    if not valid_files:
        print("❌ No valid .bin files found in the directory.")
        return

    # 提取前缀：严格到 _inX 或 _out 或 _shape 之前
    def extract_prefix(f):
        base = os.path.basename(f)
        return re.split(r'_in\d+|_out|_shape', base)[0]

    prefixes = sorted(list(set([extract_prefix(f) for f in valid_files])))
    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")
        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        # 匹配该前缀名下的所有 in0 / in1 / out 文件
        group_files = [f for f in valid_files if extract_prefix(f) == target_prefix]
        if not group_files:
            continue

        for filepath in group_files:
            filename = os.path.basename(filepath)

            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match:
                continue
            shape = tuple(map(int, match.group(1).split('x')))

            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            
            # --- 修改：根据文件名中是否包含 f16 或 float16 决定读取的类型 ---
            file_dtype = np.float32
            if "f16" in filename.lower() or "float16" in filename.lower():
                file_dtype = np.float16

            # 采用 order='F' 还原原始算子的数据读入，统一向上转为 float32 交给后续处理
            data = np.fromfile(filepath, dtype=file_dtype).astype(np.float32).reshape(shape, order='F')
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | Shape: {shape}")

            data_2d = data.squeeze()
            if data_2d.ndim == 0:
                data_2d = data_2d.reshape(1, 1)
            elif data_2d.ndim == 1:
                data_2d = data_2d.reshape(-1, 1)

            M, N = data_2d.shape

            # 通用切分与分发策略
            slices_to_distribute = []
            if N % num_slices == 0:
                # 规则 1：按 N 维度 (通常对应特征维或者 batch) 均分 28 份
                n_per_slice = N // num_slices
                for i in range(num_slices):
                    slices_to_distribute.append(data_2d[:, i * n_per_slice : (i + 1) * n_per_slice])
            elif M % num_slices == 0:
                # 规则 2：无法切 N 维，尝试按 M 维度切分
                m_per_slice = M // num_slices
                for i in range(num_slices):
                    slices_to_distribute.append(data_2d[i * m_per_slice : (i + 1) * m_per_slice, :])
            elif N % 4 == 0:
                # --- 新增规则：按 N 维分 4 份，然后复制后面 6 组相同的给 28 个 slice ---
                n_per_slice = N // 4
                base_slices = [data_2d[:, i * n_per_slice : (i + 1) * n_per_slice] for i in range(4)]
                slices_to_distribute = base_slices * 7  # 4 * 7 = 28
                print(f"  ⚠️ Cannot slice shape {M}x{N} to 28 slices. Sliced N to 4 and copied to 28.")
            elif M % 4 == 0:
                # --- 新增规则：按 M 维分 4 份，然后复制后面 6 组相同的给 28 个 slice ---
                m_per_slice = M // 4
                base_slices = [data_2d[i * m_per_slice : (i + 1) * m_per_slice, :] for i in range(4)]
                slices_to_distribute = base_slices * 7  # 4 * 7 = 28
                print(f"  ⚠️ Cannot slice shape {M}x{N} to 28 slices. Sliced M to 4 and copied to 28.")
            else:
                # 规则 5：无法切分维度（例如偏置/向量），将自身作为全量数据广播到 28 个 slice
                print(f"  ⚠️ Cannot slice shape {M}x{N} to 28 or 4 slices. Broadcasting to all slices.")
                for i in range(num_slices):
                    slices_to_distribute.append(data_2d)

            # 遍历分配后的 slice 数据执行 relayout & 落盘
            for slice_idx, slice_data in enumerate(slices_to_distribute):
                save_before_relayout(before_install_dir, op_id, slice_idx, out_name, slice_data)

                slice_dir = os.path.join(install_dir, op_id, f"slice{slice_idx:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, out_name)
                
                relayout_data = relayout_slice_M8_N(slice_data)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All Regular groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 将输入目录指向上一级的 python_golden，而不是 sub_ops
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "regular"))
    process_regular_tensors(input_dir, output_dir)
