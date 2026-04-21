import numpy as np
import os
import glob
import re
import struct

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def convert_to_128bit_txt(bin_path):
    """读取 bin 文件并输出为每行 128-bit (4个float32) 的 txt 文件(二进制格式)"""
    data = np.fromfile(bin_path, dtype=np.float32)
    
    # 如果数据量不是 4 的倍数，进行零填充
    remainder = len(data) % 4
    if remainder != 0:
        data = np.concatenate((data, np.zeros(4 - remainder, dtype=np.float32)))
        
    # 直接将 .bin 替换为 .txt (因为 bin 文件名已经包含 _linearized_128bit)
    txt_path = bin_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as f:
        # 每 4 个 float32 构成一行 128-bit
        for i in range(0, len(data), 4):
            str_float0 = float_to_bin(data[i])
            str_float1 = float_to_bin(data[i+1])
            str_float2 = float_to_bin(data[i+2])
            str_float3 = float_to_bin(data[i+3])
            # 拼接成 128 个二进制字符的数据
            f.write(f"{str_float3}{str_float2}{str_float1}{str_float0}\n")

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

def process_rmsnorm_tensors(input_dir, output_dir):
    """
    处理 rmsnorm 生成的所有 sub_op .bin 文件。
    1. sum_mac_in0 / mul_MN_M_out 等原维度(896, 32): 在 M 维度切成 28 个 32x32 的 slice 并 relayout。
    2. sum_mac_out (32, 28): 将28个列分离出来。每个变成 (32, 1)，分配给 28 个 slice 并 relayout。
    3. remote_sum, mac_SFU 等 (32, 1): 以副本形式发送给 28 个 slice。
    """
    print(f"🚀 Starting RMSNorm tensor relayout in: {input_dir}")
    
    install_dir = os.path.join(output_dir, "install")
    num_slices = 28

    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    if not bin_files:
        print("❌ No .bin files found in the directory.")
        return

    for filepath in bin_files:
        filename = os.path.basename(filepath)
        
        # 避免处理已经切片过的文件或是其他垃圾文件，仅处理 rms_norm 相关的算子
        if "slice" in filename or "shared" in filename or "rms_norm" not in filename: 
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

        # 1. 拦截大张量 896x32 或 896xN (sum_mac_in, mul_MN_M_out)
        if data_2d.shape[0] == 896:
            for i in range(num_slices):
                m_start = i * 32
                slice_32x32 = data_2d[m_start:m_start+32, :]
                relayout_data = relayout_slice_M8_N(slice_32x32)
                
                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                
                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path)
                
        # 2. 拦截分片结果 (sum_mac_out)
        elif data_2d.ndim >= 2 and data_2d.shape[1] == 28:
            for i in range(num_slices):
                # 提取属于该 slice 的局部求和序列，变成 (32, 1)
                slice_32x1 = data_2d[:, i:i+1]
                # 依然做 M8 relayout 确保内存格式彻底统一
                relayout_data = relayout_slice_M8_N(slice_32x1)
                
                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                
                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path)
                
        # 3. 处理 remote_sum, mac_SFU 等基于 N=32 或 N=1 输出的一维汇总结果
        elif data_2d.ndim >= 2 and data_2d.shape[1] == 1:
            relayout_data = relayout_slice_M8_N(data_2d)
            for i in range(num_slices):
                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                
                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path)
                
        else:
            print(f"  ⚠️ Skipping unrecognized shape pattern: {data_2d.shape}")

    print(f"\n✅ All RMS-Norm tensors sliced and saved under: {install_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "rmsnorm"))
    process_rmsnorm_tensors(input_dir, output_dir)
