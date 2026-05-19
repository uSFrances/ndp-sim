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

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols)

def relayout_slice_M8_N(slice_data):
    """
    对一个 MxN 的 slice 进行重排。
    规则：M8 先取，然后遍历 N 维度，最后再遍历剩下的 M 维度。
    等价于块大小：M=8, N=1
    对应外层 M 循环 -> 内层 N 循环 -> 最内层 M 循环(8次)
    """
    M, N = slice_data.shape
        
    relayout_data = []
    # 外层 M 循环
    for m_outer in range(0, M, 8):
        limit = min(m_outer + 8, M)
        # 内层 N 循环
        for n_idx in range(N):
            # 取出最多 8x1 的小块并追加
            block = slice_data[m_outer:limit, n_idx]
            relayout_data.extend(block)
            
    return np.array(relayout_data, dtype=slice_data.dtype)

def get_op_id(filename):
    """根据文件名和 rmsnorm.json 将子操作映射到对应的 op 文件夹"""
    if "sum_mac" in filename: return "op0"
    if "remote_sum" in filename: return "op1"
    if "mac_SFU" in filename: return "op2"
    if "mul_MN_M" in filename: return "op3"
    return "unknown_op"

def get_matrix_name(filename):
    """根据输入输出类型映射到硬件的端口名 A / B / D"""
    if "in0" in filename: return "A"
    if "in1" in filename: return "B"
    if "out" in filename: return "D"
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

def infer_group_params_from_filename(group_files):
    """
    从文件名中的 _shapeNxM_dtype 推导:
    - tile_m: 每个 slice 的 M
    - num_slices: slice 数
    优先从 sum_mac (大矩阵) 或 mul_MN_M 的文件名推导。
    """
    # 1) 优先从 sum_mac 或 mul_MN_M 的文件名中寻找 (N, M) 形状
    for fp in group_files:
        fn = os.path.basename(fp)
        if "sum_mac" not in fn and "mul_MN_M" not in fn:
            continue
            
        m = re.search(r"_shape([\dx]+)_dtype", fn)
        if not m:
            continue
        
        dims = [d for d in map(int, m.group(1).split('x')) if d > 1]
        if len(dims) >= 2:
            n, mm = max(dims), min(dims)
            if n > mm and n % mm == 0:
                return mm, n // mm

    # 2) 退化：从其他文件中寻找 (M, S) 形状（比如 sum_mac_out）
    for fp in group_files:
        fn = os.path.basename(fp)
        m = re.search(r"_shape([\dx]+)_dtype", fn)
        if not m:
            continue
        
        dims = [d for d in map(int, m.group(1).split('x')) if d > 1]
        if len(dims) >= 2:
            # 假设 M, S 两个维度都不为 1
            return min(dims), max(dims)

    return None, None

def process_rmsnorm_tensors(input_dir, output_dir):
    """
    处理 rmsnorm 生成的所有 sub_op .bin 文件。
    1. sum_mac_in0 / mul_MN_M_out 等原维度(N, M): 在 N 维度切成 num_slices 个 (N//slice, M) 的 slice 并 relayout。
    2. sum_mac_out (M, num_slices): 将 num_slices 个列分离出来。每个变成 (M, 1)，分配给 num_slices 个 slice 并 relayout。
    3. remote_sum, mac_SFU 等 (M, 1): 以副本形式发送给 num_slices 个 slice。
    """
    print(f"🚀 Starting RMSNorm tensor relayout in: {input_dir}")

    num_slices = 28
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))

    valid_files = [f for f in bin_files if "rms_norm" in os.path.basename(f) and "_subop" in os.path.basename(f)]
    if not valid_files:
        print("❌ No valid rms_norm .bin files found in the directory.")
        return

    prefixes = sorted(list(set([os.path.basename(f).split("_subop-")[0] for f in valid_files])))

    # 按 prefix 分组处理：每组单独输出目录，处理完一组再处理下一组
    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")

        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        group_files = [f for f in valid_files if os.path.basename(f).startswith(target_prefix)]

        # 新增：从文件名参数化推导，不再硬编码 896/32/28
        tile_m, inferred_slices = infer_group_params_from_filename(group_files)
        if tile_m is None or inferred_slices is None:
            print(f"  ⚠️ Cannot infer (tile_m, num_slices) from filenames for group: {target_prefix}, skip.")
            continue
        total_n = tile_m * inferred_slices
        print(f"  🧩 Inferred params from filename: N={total_n}, M={tile_m}, slices={inferred_slices}")

        for filepath in group_files:
            filename = os.path.basename(filepath)

            if "slice" in filename or "shared" in filename:
                continue

            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match:
                continue

            shape_str = match.group(1)
            shape = tuple(map(int, shape_str.split('x')))
            # 统一按 F-style 读取与还原原始张量形状
            data = np.fromfile(filepath, dtype=np.float32).reshape(shape, order='F')

            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if op_id == "unknown_op" or matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | Shape: {shape}")

            data_2d = data.squeeze()
            if data_2d.ndim == 0:
                data_2d = data_2d.reshape(1, 1)
            elif data_2d.ndim == 1:
                data_2d = data_2d.reshape(-1, 1)

            # 1A/1B 合并：原始维度 (N, M)，按 N 维切成 num_slices 个 (N//slice, M)
            if data_2d.shape == (total_n, tile_m):
                n_per_slice = total_n // inferred_slices
                for i in range(inferred_slices):
                    n_start = i * n_per_slice
                    slice_nxm = data_2d[n_start:n_start + n_per_slice, :]
                    save_before_relayout(before_install_dir, op_id, i, out_name, slice_nxm)

                    relayout_data = relayout_slice_M8_N(slice_nxm)
                    slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                    os.makedirs(slice_dir, exist_ok=True)

                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=slice_nxm.shape[0], cols=slice_nxm.shape[1])

            # 2. sum 输出路径：(M, num_slices) -> 每列一个 (M,1)
            elif data_2d.ndim >= 2 and data_2d.shape == (tile_m, inferred_slices):
                for i in range(inferred_slices):
                    slice_mx1 = data_2d[:, i:i+1]
                    save_before_relayout(before_install_dir, op_id, i, out_name, slice_mx1)

                    relayout_data = relayout_slice_M8_N(slice_mx1)
                    slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                    os.makedirs(slice_dir, exist_ok=True)

                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=slice_mx1.shape[0], cols=slice_mx1.shape[1])

            # 3. 汇总向量：(M,1) 复制到每个 slice
            elif data_2d.ndim >= 2 and data_2d.shape[1] == 1 and data_2d.shape[0] == tile_m:
                relayout_data = relayout_slice_M8_N(data_2d)
                for i in range(inferred_slices):
                    save_before_relayout(before_install_dir, op_id, i, out_name, data_2d)
                    slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                    os.makedirs(slice_dir, exist_ok=True)

                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=data_2d.shape[0], cols=data_2d.shape[1])

            else:
                print(f"  ⚠️ Skipping unrecognized shape pattern: {data_2d.shape}")

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All RMS-Norm groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "rmsnorm"))
    process_rmsnorm_tensors(input_dir, output_dir)
