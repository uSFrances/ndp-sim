import numpy as np
import os
import glob
import re
import struct

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def convert_to_decimal_txt(bin_path, rows=None, cols=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）"""
    data = np.fromfile(bin_path, dtype=np.float32)
    if rows is None or cols is None:
        rows, cols = data.size, 1
    if rows * cols != data.size:
        print(f"  ⚠️ Decimal reshape mismatch: {bin_path}, fallback to Nx1")
        rows, cols = data.size, 1

    matrix = data.reshape((rows, cols), order='C')
    txt_path = bin_path.replace('.bin', '_decimal.txt')
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

def process_rmsnorm_tensors(input_dir, output_dir):
    """
    处理 rmsnorm 生成的所有 sub_op .bin 文件。
    1. sum_mac_in0 / mul_MN_M_out 等原维度(896, 32): 在 M 维度切成 28 个 32x32 的 slice 并 relayout。
    2. sum_mac_out (32, 28): 将28个列分离出来。每个变成 (32, 1)，分配给 28 个 slice 并 relayout。
    3. remote_sum, mac_SFU 等 (32, 1): 以副本形式发送给 28 个 slice。
    """
    print(f"🚀 Starting RMSNorm tensor relayout in: {input_dir}")
    
    install_dir = os.path.join(output_dir, "install")
    before_install_dir = os.path.join(output_dir, "install_beforerelayout")
    num_slices = 28

    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    
    # --- 新增：提取并锁死唯一的算子实例前缀，防止多次调用互相覆盖串台 ---
    valid_files = [f for f in bin_files if "rms_norm" in f and "_subop" in f]
    if not valid_files:
        print("❌ No valid rms_norm .bin files found in the directory.")
        return
    prefixes = sorted(list(set([os.path.basename(f).split("_subop-")[0] for f in valid_files])))
    target_prefix = prefixes[0]
    print(f"🎯 Locking to specific instance: '{target_prefix}' to prevent overwrite mismatch")
    # ------------------------------------------------------------------

    for filepath in bin_files:
        filename = os.path.basename(filepath)
        
        # 避免处理已经切片过的文件或是其他垃圾文件，检查实例锁
        if "slice" in filename or "shared" in filename or "rms_norm" not in filename: 
            continue
        if target_prefix and not filename.startswith(target_prefix):
            continue
        
        match = re.search(r"_shape([\dx]+)_dtype", filename)
        if not match: continue
        
        shape_str = match.group(1)
        shape = tuple(map(int, shape_str.split('x')))
        data = np.fromfile(filepath, dtype=np.float32).reshape(shape, order='F')
        
        op_id = get_op_id(filename)
        matrix_id = get_matrix_name(filename)
        if op_id == "unknown_op" or matrix_id == "unknown_matrix": continue

        out_name = f"matrix_{matrix_id}_linearized_128bit.bin"

        print(f"📦 Processing: {filename} -> {op_id}/{out_name} | Shape: {shape}")
        
        # 将无用的维度挤掉
        data_2d = data.squeeze()
        # 兼容退化为 0 维标量或者 1 维向量的极端情况 (如 1x1x1x1 -> 0D)
        if data_2d.ndim == 0:
            data_2d = data_2d.reshape(1, 1)
        elif data_2d.ndim == 1:
            data_2d = data_2d.reshape(-1, 1)

        # matrix_A（例如 sum_mac_in0）先转置：896x32 -> 32x896
        if matrix_id == "A" and data_2d.shape == (896, 32):
            data_2d = data_2d.T

        # 1A. 专门处理转置后的 matrix_A：32x896，按列切 28 份 32x32
        if matrix_id == "A" and data_2d.shape == (32, 896):
            for i in range(num_slices):
                c_start = i * 32
                slice_32x32 = data_2d[:, c_start:c_start+32]
                save_before_relayout(before_install_dir, op_id, i, out_name, slice_32x32)

                relayout_data = relayout_slice_M8_N(slice_32x32)

                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)

                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=slice_32x32.shape[0], cols=slice_32x32.shape[1])

        # 1B. 原有大张量路径（如 mul_MN_M_out 等仍为 896x32）
        elif data_2d.shape[0] == 896:
            for i in range(num_slices):
                m_start = i * 32
                slice_32x32 = data_2d[m_start:m_start+32, :]
                save_before_relayout(before_install_dir, op_id, i, out_name, slice_32x32)

                relayout_data = relayout_slice_M8_N(slice_32x32)

                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)

                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=slice_32x32.shape[0], cols=slice_32x32.shape[1])

        # 2. 拦截分片结果 (sum_mac_out)
        elif data_2d.ndim >= 2 and data_2d.shape[1] == 28:
            for i in range(num_slices):
                # 提取属于该 slice 的局部求和序列，变成 (32, 1)
                slice_32x1 = data_2d[:, i:i+1]
                save_before_relayout(before_install_dir, op_id, i, out_name, slice_32x1)

                relayout_data = relayout_slice_M8_N(slice_32x1)

                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)

                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=slice_32x1.shape[0], cols=slice_32x1.shape[1])

        # 3. 处理 remote_sum, mac_SFU 等基于 N=32 或 N=1 输出的一维汇总结果
        elif data_2d.ndim >= 2 and data_2d.shape[1] == 1:
            relayout_data = relayout_slice_M8_N(data_2d)
            for i in range(num_slices):
                save_before_relayout(before_install_dir, op_id, i, out_name, data_2d)
                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)

                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=data_2d.shape[0], cols=data_2d.shape[1])

        else:
            print(f"  ⚠️ Skipping unrecognized shape pattern: {data_2d.shape}")

    print(f"\n✅ All RMS-Norm tensors sliced and saved under: {install_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "rmsnorm"))
    process_rmsnorm_tensors(input_dir, output_dir)
