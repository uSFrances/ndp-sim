import numpy as np
import os
import glob
import re
import argparse
import struct

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def convert_to_decimal_txt(bin_path, rows=None, cols=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）"""
    if not os.path.exists(bin_path):
        return
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

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols)

def save_before_relayout(before_install_dir, op_id, slice_idx, out_name, matrix_2d):
    matrix_2d = np.asarray(matrix_2d, dtype=np.float32)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)
    slice_dir = os.path.join(before_install_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    out_path = os.path.join(slice_dir, out_name)
    matrix_2d.reshape(-1, order='C').tofile(out_path)
    convert_to_128bit_txt(out_path, rows=matrix_2d.shape[0], cols=matrix_2d.shape[1])

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

def relayout_systolic_weight(input_filepath, K, N, target_dir, input_dtype=np.float32):
    """
    根据硬件数据流重排 GEMM 权重:
    - 7个Head并行，1个Head(Cluster)宽 CLUSTER_N
    - 1个Cluster分4个Slice，1个Slice宽 SLICE_N
    - 按 Slice 分别保存至 install/op0/sliceXX/
    """
    # 硬件架构常量 - 动态计算以适配任何 N（固定28个slice，7个head）
    CLUSTER_N = N // 7         # 一个 Head / Cluster 的 N 宽度
    SLICE_N = N // 28          # 一个 Slice 的 N 宽度
    K_MACRO = 32               # K维度的对角线调度宏块
    BK = 8                     # 微块 K (匹配M8)
    BN = 2                     # 微块 N (原来是8，现在最小单位切成了2)

    print(f"\n--- Relayout GEMM Hardware Weight: {os.path.basename(input_filepath)} ---")
    data = np.fromfile(input_filepath, dtype=input_dtype).astype(np.float32, copy=False)
    if data.size < K * N:
        raise ValueError(f"File size {data.size} is smaller than expected K*N = {K*N}")
        
    weight_matrix = data[:K*N].reshape((K, N), order='F')
    
    if N % CLUSTER_N != 0 or CLUSTER_N % SLICE_N != 0:
        raise ValueError("N dimensions don't match Cluster and Slice alignment.")
    if K % K_MACRO != 0:
        raise ValueError("K dimensions don't match K_MACRO(32) alignment.")
        
    num_slices = CLUSTER_N // SLICE_N     # 每个cluster 4个slice
    num_time_steps = K_MACRO // BK        # 32 / 8 = 4
    
    total_slices = 28 # 永远为28个物理slice
    # 为每一个 slice 初始化一个列表用于缓存数据
    slice_data_map = {idx: [] for idx in range(total_slices)}
    
    # 1. 遍历并行的 Heads (Cluster)
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
                    
                    # 5. 一个 Slice 宽 SLICE_N，微块设为 N=2
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
    before_install_dir = os.path.join(target_dir, "install_beforerelayout")

    for s_idx, s_data in slice_data_map.items():
        output_data = np.array(s_data, dtype=np.float32)

        # 先保存未 relayout 权重切片（K x SLICE_N）
        col_start = s_idx * SLICE_N
        col_end = col_start + SLICE_N
        before_matrix = weight_matrix[:, col_start:col_end]
        save_before_relayout(before_install_dir, "op0", s_idx, "matrix_B_linearized_128bit.bin", before_matrix)

        slice_dir = os.path.join(install_dir, f"slice{s_idx:02d}")
        os.makedirs(slice_dir, exist_ok=True)
        
        # GEMM 的权重在硬件中通常作为 matrix_B 送入
        out_filepath = os.path.join(slice_dir, "matrix_B_linearized_128bit.bin")
        output_data.tofile(out_filepath)
        convert_to_128bit_txt(out_filepath, rows=K, cols=SLICE_N)
        
    print(f"✅ Hardware relayout distributed across {total_slices} slices in: {install_dir}/")

def build_systolic_input_a_by_slice(input_a, total_slices=28, slices_per_group=4, k_macro=32, bk=8):
    """
    按 GEMM 权重同样的对角线调度规则，为每个 slice 生成 A(KxL)：
    - t=0 取本地 slice 对应行块
    - t=1..3 取相邻 slice 对应行块（轮转）
    """
    K, L = input_a.shape
    if K % k_macro != 0:
        raise ValueError(f"K={K} 必须能被 k_macro={k_macro} 整除")
    if total_slices % slices_per_group != 0:
        raise ValueError("total_slices 必须能被 slices_per_group 整除")

    num_groups = total_slices // slices_per_group
    result = {}

    for g in range(num_groups):
        for j in range(slices_per_group):
            global_slice_idx = g * slices_per_group + j
            blocks = []
            for k0 in range(0, K, k_macro):
                for t in range(slices_per_group):
                    i = (j - t) % slices_per_group
                    row_start = k0 + i * bk
                    blocks.append(input_a[row_start:row_start + bk, :])
            result[global_slice_idx] = np.vstack(blocks).astype(np.float32)

    return result

def _parse_mul_mat_shape(filename):
    m = re.search(r"_shape([\dx]+)_dtype", filename)
    if not m:
        return None
    return tuple(map(int, m.group(1).split("x")))

def _parse_mul_mat_dtype(filename):
    m = re.search(r"_dtype_(f16|f32|i32)\.bin$", filename)
    if not m:
        return np.float32
    return {"f16": np.float16, "f32": np.float32, "i32": np.int32}[m.group(1)]

def _group_mul_mat_triplets(bin_files):
    groups = {}
    for filepath in bin_files:
        filename = os.path.basename(filepath)
        if "systolic" in filename or "linearized" in filename or "_op-mul_mat" not in filename:
            continue
        m = re.search(r"(?P<prefix>.+?)_(?P<io>in0|in1|out)_shape", filename)
        if not m:
            continue
        groups.setdefault(m.group("prefix"), {})[m.group("io")] = filepath
    return groups

def _safe_prefix_dir(prefix: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", prefix)

def process_gemm_directory(input_dir, output_dir, order='F'):
    """
    自动处理目录下的所有 GEMM 相关文件。

    读取目录：上一级 python_golden
    分组方式：按文件名前缀把每个 mul_mat 的 in0 / in1 / out 作为一组处理，
    避免 q/k 等不同算子串读。
    """
    print(f"🚀 Scanning for GEMM tensors in: {input_dir}")
    bin_files = glob.glob(os.path.join(input_dir, "*mul_mat*.bin"))
    valid_files = [f for f in bin_files if "systolic" not in f and "linearized" not in f]
    if not valid_files:
        print("❌ No valid mul_mat .bin files found in the directory.")
        return

    triplets = _group_mul_mat_triplets(valid_files)
    if not triplets:
        print("❌ No valid mul_mat triplets found in the directory.")
        return

    for gemm_idx, (prefix, files) in enumerate(sorted(triplets.items())):
        # ======== 锁定目标文件 ========
        if prefix != "blk.0_node_0_q_op-mul_mat":
            continue
        # ============================

        if not {"in0", "in1", "out"} <= set(files):
            continue

        # 将目录直接命名为 gemm_前缀
        safe_prefix = _safe_prefix_dir(prefix)
        group_dir = f"gemm_{safe_prefix}"
        group_root = os.path.join(output_dir, group_dir)
        install_dir = os.path.join(group_root, "install")
        before_install_dir = os.path.join(group_root, "install_beforerelayout")
        os.makedirs(install_dir, exist_ok=True)
        os.makedirs(before_install_dir, exist_ok=True)

        in0_path = files["in0"]
        in1_path = files["in1"]
        out_path = files["out"]

        in0_dims = _parse_mul_mat_shape(os.path.basename(in0_path))
        in1_dims = _parse_mul_mat_shape(os.path.basename(in1_path))
        out_dims = _parse_mul_mat_shape(os.path.basename(out_path))
        if not in0_dims or not in1_dims or not out_dims:
            continue

        in0_dtype = _parse_mul_mat_dtype(os.path.basename(in0_path))
        in1_dtype = _parse_mul_mat_dtype(os.path.basename(in1_path))
        out_dtype = _parse_mul_mat_dtype(os.path.basename(out_path))

        K_val, N_val = in0_dims[0], in0_dims[1]
        active_slices = 28 # 永远固定28个切片
        print(f"🎯 Processing mul_mat prefix: '{prefix}' -> '{group_dir}' | active slices={active_slices}")

        # in0 -> matrix_B
        relayout_systolic_weight(
            in0_path, K_val, N_val, group_root, input_dtype=in0_dtype
        )

        # in1 -> matrix_A: 固定 K x L，并为全硬件 28 个 slice 准备输入
        K_a, L_a = in1_dims[0], in1_dims[1]
        data_a = np.fromfile(in1_path, dtype=in1_dtype).astype(np.float32, copy=False)
        data_a = data_a[:K_a * L_a].reshape((K_a, L_a), order=order)
        
        # 始终生成 28 份 A 的切片数据
        a_by_slice = build_systolic_input_a_by_slice(data_a, total_slices=28, slices_per_group=4, k_macro=32, bk=8)
        for i in range(28):
            slice_a = a_by_slice[i]
            save_before_relayout(before_install_dir, "op0", i, "matrix_A_linearized_128bit.bin", slice_a)
            slice_dir = os.path.join(install_dir, f"slice{i:02d}")
            os.makedirs(slice_dir, exist_ok=True)
            out_a_path = os.path.join(slice_dir, "matrix_A_linearized_128bit.bin")
            relayout_data = relayout_slice_M8N2M4N(slice_a)
            relayout_data.tofile(out_a_path)
            convert_to_128bit_txt(out_a_path, rows=slice_a.shape[0], cols=slice_a.shape[1])

        # out -> matrix_D: N x L
        N_d, L_d = out_dims[0], out_dims[1]
        data_d = np.fromfile(out_path, dtype=out_dtype).astype(np.float32, copy=False)
        data_d = data_d[:N_d * L_d].reshape((N_d, L_d), order=order)
        
        slice_width = N_d // 28
        for i in range(28):
            slice_data = data_d[i * slice_width:(i + 1) * slice_width, :]
            save_before_relayout(before_install_dir, "op0", i, "matrix_D_linearized_128bit.bin", slice_data)
            slice_dir = os.path.join(install_dir, f"slice{i:02d}")
            os.makedirs(slice_dir, exist_ok=True)
            out_d_path = os.path.join(slice_dir, "matrix_D_linearized_128bit.bin")
            relayout_data = relayout_slice_M8N2M4N(slice_data)
            relayout_data.tofile(out_d_path)
            convert_to_128bit_txt(out_d_path, rows=slice_data.shape[0], cols=slice_data.shape[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto parse and relayout GEMM weight files.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    default_out = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "gemm"))

    parser.add_argument('--dir', type=str, default=default_in, help="Directory containing the .bin files")
    parser.add_argument('--order', type=str, choices=['C', 'F'], default='F', help="Matrix layout in .bin")

    args = parser.parse_args()
    process_gemm_directory(args.dir, default_out, order=args.order)
