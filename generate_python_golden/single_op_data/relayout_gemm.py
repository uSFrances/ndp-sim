import numpy as np
import os
import glob
import re
import struct

def float16_to_bin(f):
    """将单个 float16 转换为 16 位二进制字符串"""
    # 使用 numpy 直接提取其二进制表示
    return bin(np.float16(f).view(np.uint16))[2:].zfill(16)

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def convert_to_decimal_txt(bin_path, rows=None, cols=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）"""
    data = np.fromfile(bin_path, dtype=np.float16)
    if rows is None or cols is None:
        rows, cols = data.size, 1
    if rows * cols != data.size:
        print(f"  ⚠️ Decimal reshape mismatch: {bin_path}, fallback to Nx1")
        rows, cols = data.size, 1

    matrix = data.reshape((rows, cols), order='C')
    txt_path = bin_path.replace('.bin', f'_{rows}x{cols}_decimal.txt')
    with open(txt_path, 'w') as f:
        for r in range(rows):
            f.write(",".join(f"{float(v):.10g}" for v in matrix[r]))
            f.write("\n")

def convert_to_128bit_txt(bin_path, rows=None, cols=None):
    """读取 bin 文件并输出为每行 128-bit (8个float16) 的 txt 文件(二进制格式)"""
    data = np.fromfile(bin_path, dtype=np.float16)

    remainder = len(data) % 8
    if remainder != 0:
        data = np.concatenate((data, np.zeros(8 - remainder, dtype=np.float16)))

    txt_path = bin_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as f:
        for i in range(0, len(data), 8):
            strs = [float16_to_bin(data[i+j]) for j in range(8)]
            # 倒序拼接保证低地址对应底层比特
            f.write("".join(reversed(strs)) + "\n")

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols)

def relayout_in0_N8K2N4K(slice_data):
    """
    in0 张量 relayout: (K=896, N=64) -> N8K2N4K
    假设维度拆分为: K=(448, 2), N=(2, 4, 8)
    """
    # 调整 reshape 结构：K=448*2, N=2*4*8
    reshaped = slice_data.reshape(448, 2, 2, 4, 8)
    # 按从外层到内层或者指定排布规则进行transpose。这里给出一个示例排布：
    # 对应维度索引：K448_idx=0, K2_idx=1, N2_idx=2, N4_idx=3, N8_idx=4
    # 若规则 N8,K2,N4,K(448),N2 -> 对应索引: 4, 1, 3, 0, 2
    relayout_data = reshaped.transpose(4, 1, 3, 0, 2).flatten()
    return relayout_data

def reorder_in0_slice_by_ring(slice_data, slice_idx):
    """
    针对 in0 的每个 slice，按环形顺序重排 K 维度 (896 拆为 28 个 32)
    使得每个 slice 优先计算与自己 local tile (slice_idx) 对应的数据。
    """
    ring_order = [0, 12, 13, 15, 17, 19, 21, 23, 25, 27, 26, 10, 11, 9, 8, 24, 22, 20, 18, 16, 14, 2, 4, 6, 7, 5, 3, 1]
    
    # 找到当前 slice_idx 在环中的起始位置
    start_pos = ring_order.index(slice_idx)
    # 将环旋转，当前 slice 置于首位
    rotated_order = ring_order[start_pos:] + ring_order[:start_pos]
    
    # 将 K=896 切分为 28 个连续的 sub_block，每个 sub_block 大小为 K32 x N64
    blocks = np.split(slice_data, 28, axis=0) # [28, (32, 64)]
    
    # 按照按重新组合的顺序拼接
    reordered_blocks = [blocks[idx] for idx in rotated_order]
    reordered_slice = np.concatenate(reordered_blocks, axis=0)
    
    return reordered_slice

def relayout_in1_L8K2L4K(slice_data):
    """
    in1 张量 relayout: (K=32, L=32) -> L8K2L4K
    假设维度拆分为: K=(16, 2), L=(1, 4, 8)
    """
    # 调整 reshape 结构：K=16*2, L=1*4*8
    reshaped = slice_data.reshape(16, 2, 1, 4, 8)
    # 对应维度索引：K16=0, K2=1, L1=2, L4=3, L8=4
    # 若规则 L8,K2,L4,K(16),L(1) -> 对应索引: 4, 1, 3, 0, 2
    relayout_data = reshaped.transpose(4, 1, 3, 0, 2).flatten()
    return relayout_data

def relayout_out_L8N8L4N4N2L1(slice_data):
    """
    out 张量 relayout: (N=64, L=32) -> L8N8L4N4N2L1
    维度拆分逻辑:
    N (64) -> N8(8) * N4(4) * N2(2)
    L (32) -> L8(8) * L4(4) * L1(1)
    """
    # 1. Reshape: 将 (64, 32) 拆解为对应的子维度
    # N=64 拆为 (8, 4, 2) 对应 (N8, N4, N2)
    # L=32 拆为 (8, 4, 1) 对应 (L8, L4, L1)
    reshaped = slice_data.reshape(8, 4, 2, 8, 4, 1)
    
    # 维度索引对照表:
    # idx 0: N8 (8)
    # idx 1: N4 (4)
    # idx 2: N2 (2)
    # idx 3: L8 (8)
    # idx 4: L4 (4)
    # idx 5: L1 (1)
    
    # 2. Transpose: 按照 L8, N8, L4, N4, N2, L1 的顺序重排
    # 对应的索引顺序为: 3, 0, 4, 1, 2, 5
    relayout_data = reshaped.transpose(3, 0, 4, 1, 2, 5).flatten()
    
    return relayout_data

def relayout_slice_default(slice_data):
    """
    默认的重排函数，作为框架占位使用，后续可根据需求补充具体的重排逻辑
    """
    return slice_data.flatten()

def save_slice(output_dir, op_id, slice_idx, out_name, matrix_2d):
    """将切割后的 slice 保存并调用格式转换函数"""
    matrix_2d = np.asarray(matrix_2d, dtype=np.float16)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)
        
    slice_dir = os.path.join(output_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    
    out_path = os.path.join(slice_dir, out_name)
    matrix_2d.reshape(-1, order='C').tofile(out_path)
    convert_to_128bit_txt(out_path, rows=matrix_2d.shape[0], cols=matrix_2d.shape[1])

def process_gemm_tensors(input_dir, output_dir):
    print(f"🚀 Starting GEMM tensor split and relayout in: {input_dir}")
    install_dir = os.path.join(output_dir, "install")
    before_install_dir = os.path.join(output_dir, "install_beforerelayout")
    num_slices = 28

    target_files = {
        "in0": "blk.0_ffn_gate-0_op-mul_mat_in0_shape896x1792x1x1_dtype_f16.bin",
        "in1": "blk.0_ffn_gate-0_op-mul_mat_in1_shape896x32x1x1_dtype_f16.bin",
        "out": "blk.0_ffn_gate-0_op-mul_mat_out_shape1792x32x1x1_dtype_f16.bin"
    }

    for key, filename in target_files.items():
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue

        # 从文件名解析 shape 和 dtype
        match = re.search(r"_shape([\dx]+)_dtype_([a-z0-9]+)", filename)
        if not match:
            print(f"⚠️ Could not parse shape/dtype from filename: {filename}")
            continue
            
        shape_str = match.group(1)
        shape_tuple = tuple(map(int, shape_str.split('x')))
        dtype_str = match.group(2)
        
        print(f"📦 Processing: {filename} | Parsed Shape: {shape_tuple} | Dtype: {dtype_str}")
        
        # 假设当前均为 float16 数据
        data = np.fromfile(filepath, dtype=np.float16).reshape(shape_tuple, order='F')
        
        # 挤掉无用维度（去除H1等大小为1的维度）
        data_2d = data.squeeze()
        if data_2d.ndim == 0:
            data_2d = data_2d.reshape(1, 1)
        elif data_2d.ndim == 1:
            data_2d = data_2d.reshape(-1, 1)
        
        op_id = "gemm"
        
        if key == "in0":
            # in0 是 weight: (K896, N1792) -> KxN = 896x1792
            # 按 N 维度平均分给 28 个 slice，每个 slice K896 x N64
            out_name = "matrix_in0_linearized_128bit.bin"
            for i in range(num_slices):
                n_start = i * 64
                slice_data = data_2d[:, n_start:n_start+64]
                # 保存 relayout 前的矩阵
                save_slice(before_install_dir, op_id, i, out_name, slice_data)
                
                # 环形对角线重排 K 块
                reordered_slice = reorder_in0_slice_by_ring(slice_data, i)
                
                # N8K2N4K specific relayout
                relayout_data = relayout_in0_N8K2N4K(reordered_slice) 
                
                # 保存 relayout 后的结果。由于此处重排仅为 flatten，需保存原 shape 用于写入 txt
                relayout_matrix = relayout_data.reshape(slice_data.shape)
                save_slice(install_dir, op_id, i, out_name, relayout_matrix)
                
        elif key == "in1":
            # in1 是输入数据: (K896, L32) -> KxL = 896x32
            # 按 K 维度平均分给 28 个 slice，每个 slice K32 x L32
            out_name = "matrix_in1_linearized_128bit.bin"
            for i in range(num_slices):
                k_start = i * 32
                slice_data = data_2d[k_start:k_start+32, :]
                save_slice(before_install_dir, op_id, i, out_name, slice_data)
                
                # L8K2L4K specific relayout
                relayout_data = relayout_in1_L8K2L4K(slice_data)
                relayout_matrix = relayout_data.reshape(slice_data.shape)
                save_slice(install_dir, op_id, i, out_name, relayout_matrix)
                
        elif key == "out":
            # out 是输出数据: (N1792, L32) -> NxL = 1792x32
            out_name = "matrix_out_linearized_128bit.bin"
            for i in range(num_slices):
                n_start = i * 64
                slice_data = data_2d[n_start:n_start+64, :]
                save_slice(before_install_dir, op_id, i, out_name, slice_data)
                
                # L8N8L4N4N2L1 specific relayout
                relayout_data = relayout_out_L8N8L4N4N2L1(slice_data)
                
                relayout_matrix = relayout_data.reshape(slice_data.shape)
                save_slice(install_dir, op_id, i, out_name, relayout_matrix)

    print(f"\n✅ All GEMM tensors split and saved under: {install_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    # 输出文件夹参考 rmsnorm，存入 ndp-sim/model_execplan/data/gemm
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "gemm"))
    process_gemm_tensors(input_dir, output_dir)
