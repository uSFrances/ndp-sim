import numpy as np
import os
import glob
import re
import argparse
import struct

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def convert_to_128bit_txt(bin_path):
    """读取 bin 文件并输出为每行 128-bit (4个float32) 的 txt 文件(二进制格式)"""
    if not os.path.exists(bin_path):
        return
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
    print(f"🔎 Converted to 128-bit binary TXT: {os.path.basename(txt_path)}")

def relayout_slice_M8N2M4N(slice_data):
    """
    对一个 MxN 的 slice 进行 M8 N2 M4 N 的层级重排。
    """
    M, N = slice_data.shape
    relayout_data = []
    
    # 最外层 M=8, N=2 的宏块循环
    for m_outer in range(0, M, 8):
        limit_m = min(m_outer + 8, M)
        for n_outer in range(0, N, 2):
            limit_n = min(n_outer + 2, N)
            
            block = slice_data[m_outer:limit_m, n_outer:limit_n]
            
            # 内层 M=4, N=1
            for m4_idx in range(0, 8, 4):
                for n1_idx in range(0, 2, 1):
                    for m1_idx in range(4):
                        m_curr = m4_idx + m1_idx
                        if m_curr < block.shape[0] and n1_idx < block.shape[1]:
                            relayout_data.append(block[m_curr, n1_idx])
                            
    return np.array(relayout_data, dtype=slice_data.dtype)

def relayout_systolic_weight(input_filepath, K, N, target_dir, input_order='F', block_order='F'):
    """
    根据硬件数据流重排 GEMM 权重:
    - 7个Head并行，1个Head(Cluster)宽128
    - 1个Cluster分4个Slice，1个Slice宽32
    - 按 Slice 分别保存至 install/op0/sliceXX/
    """
    # 硬件架构常量
    CLUSTER_N = 128    # 一个 Head / Cluster 的 N 宽度
    SLICE_N = 32       # 一个 Slice 的 N 宽度
    K_MACRO = 32       # K维度的对角线调度宏块
    BK = 8             # 微块 K (匹配M8)
    BN = 2             # 微块 N (原来是8，现在最小单位切成了2)

    print(f"\n--- Relayout GEMM Hardware Weight: {os.path.basename(input_filepath)} ---")
    data = np.fromfile(input_filepath, dtype=np.float32)
    if data.size < K * N:
        raise ValueError(f"File size {data.size} is smaller than expected K*N = {K*N}")
        
    weight_matrix = data[:K*N].reshape((K, N), order=input_order)
    
    if N % CLUSTER_N != 0 or CLUSTER_N % SLICE_N != 0:
        raise ValueError("N dimensions don't match Cluster(128) and Slice(32) alignment.")
    if K % K_MACRO != 0:
        raise ValueError("K dimensions don't match K_MACRO(32) alignment.")
        
    num_slices = CLUSTER_N // SLICE_N     # 128 / 32 = 4
    num_time_steps = K_MACRO // BK        # 32 / 8 = 4
    
    total_slices = N // SLICE_N
    # 为每一个 slice 初始化一个列表用于缓存数据
    slice_data_map = {idx: [] for idx in range(total_slices)}
    
    # 1. 遍历并行的 Heads (Cluster), N维每次走128
    for head_n in range(0, N, CLUSTER_N):
        # 2. 遍历 K 维度的 32x32 宏块
        for k_macro in range(0, K, K_MACRO):
            
            # 3. 在 Cluster 内部，遍历 4 个 Slice
            for j in range(num_slices):
                # 确定当前属于全图 28 个 slice 中的哪一个
                global_slice_idx = (head_n // SLICE_N) + j
                
                # 4. 遍历计算时间步 (Time steps) 进行对角线数据请求
                for t in range(num_time_steps):
                    i = (j - t) % num_slices
                    row_start = k_macro + i * BK
                    col_start_base = head_n + j * SLICE_N
                    
                    # 5. 一个 Slice 宽 32，微块现在设为 N=2，即横向遍历16次
                    for n_sub in range(0, SLICE_N, BN):
                        col_start = col_start_base + n_sub
                        block_8x2 = weight_matrix[row_start : row_start + BK, 
                                                  col_start : col_start + BN]
                        
                        # M8_N2_M4_N 层级展平重排
                        for m4_idx in range(0, BK, 4):         
                            for n1_idx in range(0, BN, 1):     
                                for m1_idx in range(4):        
                                    m_curr = m4_idx + m1_idx
                                    if m_curr < block_8x2.shape[0] and n1_idx < block_8x2.shape[1]:
                                        # 数据追加到对应的 slice 列表中
                                        slice_data_map[global_slice_idx].append(block_8x2[m_curr, n1_idx])
                        
    # 处理完成，分别写入各 Slice 文件夹
    install_dir = os.path.join(target_dir, "install")
    for s_idx, s_data in slice_data_map.items():
        output_data = np.array(s_data, dtype=np.float32)
        
        slice_dir = os.path.join(install_dir, "op0", f"slice{s_idx:02d}")
        os.makedirs(slice_dir, exist_ok=True)
        
        # GEMM 的权重在硬件中通常作为 matrix_B 送入
        out_filepath = os.path.join(slice_dir, "matrix_B_linearized_128bit.bin")
        output_data.tofile(out_filepath)
        convert_to_128bit_txt(out_filepath)
        
    print(f"✅ Hardware relayout distributed across {total_slices} slices in: {install_dir}/op0/")

def process_gemm_directory(input_dir, output_dir, order='F'):
    """自动处理目录下的所有 GEMM 相关文件"""
    print(f"🚀 Scanning for GEMM tensors in: {input_dir}")
    bin_files = glob.glob(os.path.join(input_dir, "*mul_mat*.bin"))
    
    valid_files = [f for f in bin_files if "systolic" not in f and "linearized" not in f]
    if not valid_files:
        print("❌ No valid mul_mat .bin files found in the directory.")
        return
        
    # --- 提取并锁死完整的算子实例前缀 ---
    prefixes = sorted(list(set([re.split(r"_(in0|in1|out)", os.path.basename(f))[0] for f in valid_files if "_op-mul_mat" in f])))
    target_prefix = prefixes[0] if prefixes else ""
    if target_prefix: print(f"🎯 Locking to specific GEMM instance: '{target_prefix}'")
    # --------------------------------
    
    install_dir = os.path.join(output_dir, "install")

    for filepath in bin_files:
        filename = os.path.basename(filepath)
        if "systolic" in filename or "linearized" in filename:
            continue # 跳过已经重排过的文件
            
        if target_prefix and not filename.startswith(target_prefix):
            continue
            
        # 匹配形状提取
        match = re.search(r"_shape([\dx]+)_dtype", filename)
        if not match:
            continue
            
        dims = list(map(int, match.group(1).split('x')))
        
        # in0 是 weight (K x N), in1 是 Input, out 是 Output
        if "in0" in filename:
            K_val, N_val = dims[0], dims[1]
            print(f"📐 Detected Weight File [in0]: {filename}, K={K_val}, N={N_val}")
            # 处理并分发矩阵 B
            relayout_systolic_weight(filepath, K_val, N_val, target_dir, input_order=order, block_order=order)
            
        elif "in1" in filename:
            print(f"📐 Detected Input File [in1]: {filename}")
            data = np.fromfile(filepath, dtype=np.float32).reshape(dims, order=order).squeeze()
            if data.ndim == 1: data = data.reshape(-1, 1)
            
            # Input 为 [K, L], 广播给所有 28 个 slice 作为 matrix_A
            relayout_data = relayout_slice_M8N2M4N(data)
            for i in range(28):
                slice_dir = os.path.join(install_dir, "op0", f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, "matrix_A_linearized_128bit.bin")
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path)
                
        elif "out" in filename:
            print(f"📐 Detected Output File [out]: {filename}")
            data = np.fromfile(filepath, dtype=np.float32).reshape(dims, order=order).squeeze()
            if data.ndim == 1: data = data.reshape(-1, 1)
            
            # Output 为 [N, L], 沿 N (dim 0) 切分为 28 份，分配给矩阵 D
            slice_width = data.shape[0] // 28
            for i in range(28):
                slice_data = data[i*slice_width : (i+1)*slice_width, :]
                relayout_data = relayout_slice_M8N2M4N(slice_data)
                
                slice_dir = os.path.join(install_dir, "op0", f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, "matrix_D_linearized_128bit.bin")
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto parse and relayout GEMM weight files.")
    
    # 默认寻找上级目录中的 python_golden 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    default_out = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "gemm"))
    
    parser.add_argument('--dir', type=str, default=default_in, help="Directory containing the .bin files")
    parser.add_argument('--order', type=str, choices=['C', 'F'], default='F', help="Matrix layout in .bin")
    
    args = parser.parse_args()
    process_gemm_directory(args.dir, default_out, order=args.order)
