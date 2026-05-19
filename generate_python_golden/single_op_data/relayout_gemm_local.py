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
    """根据文件名将 gemm_local 子操作映射到 op 文件夹"""
    return "op0"  # gemm_local 只有一个 op

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

def infer_gemm_local_params(group_files, target_prefix):
    """按 group 类型直接推 K/L/H，避免把 attn_scores / attn_out 搞反。"""
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
                K, L = dims[0], dims[1]   # in0(KxL)
            else:
                L, K = dims[0], dims[1]   # in0(LxK)

    return K, L, H

def process_gemm_local_tensors(input_dir, output_dir):
    """
    处理 gemm_local 生成的所有 .bin 文件。
    attn_scores: in0(KxL) 分为 (K//slices//H x L)，in1(KxLxH) 分为 (K//slices//H x L x H)，
                 out(LxLxH) 广播到每 slices//H 个 slice 一组，H 组
    attn_out: in0(LxK) 分为 (L x K//slices//H)，in1(LxLxH) 广播到每 slices//H 个 slice 一组，H 组，
              out(KxLxH) 分为 (K//slices//H x L x H)
    """
    print(f"🚀 Starting GEMM-Local tensor relayout in: {input_dir}")
    
    num_slices = 28
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    
    # 提取合法的 gemm_local 相关文件（包含 "gemm_local"、"attn_scores" 或 "attn_out" 且包含 "_in" 或 "_out"）
    valid_files = [f for f in bin_files if ("gemm_local" in os.path.basename(f) or 
                                             "attn_scores" in os.path.basename(f) or 
                                             "attn_out" in os.path.basename(f)) and 
                                            ("_in" in os.path.basename(f) or "_out" in os.path.basename(f))]
    if not valid_files:
        print("❌ No valid gemm_local .bin files found in the directory.")
        print(f"   Debug: Total .bin files found: {len(bin_files)}")
        if bin_files:
            print(f"   Sample filenames: {[os.path.basename(f) for f in bin_files[:3]]}")
        return
    
    print(f"✓ Found {len(valid_files)} valid gemm_local files")

    # 按前缀分组：提取到第一个 _in 或 _out 之前的部分作为前缀
    def extract_prefix(f):
        base = os.path.basename(f)
        # 尽可能提取最长的前缀，但要在 _inX 或 _out 之前停止
        match = re.match(r'^(.+?)(_in\d+|_out)', base)
        if match:
            return match.group(1)
        # fallback：按 _shape 分割
        return re.split(r'_shape', base)[0]

    prefixes = sorted(list(set([extract_prefix(f) for f in valid_files])))
    print(f"✓ Identified {len(prefixes)} instance groups")

    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")
        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        # 筛选当前前缀的文件
        group_files = [f for f in valid_files if extract_prefix(f) == target_prefix]
        if not group_files:
            continue

        print(f"  Files in group: {len(group_files)}")

        # 动态推导 K, L, H
        K, L, H = infer_gemm_local_params(group_files, target_prefix)
        if K is None or L is None or H is None:
            print(f"  ⚠️ Cannot infer K, L, H from filenames for group: {target_prefix}")
            print(f"     Inferred: K={K}, L={L}, H={H}")
            # 尝试从文件中推断
            for fp in group_files:
                fn = os.path.basename(fp)
                m = re.search(r"_shape([\dx]+)_dtype", fn)
                if m:
                    print(f"     File: {fn}, Shape: {m.group(1)}")
            continue

        is_attn_scores_group = "attn_scores" in target_prefix
        slices_per_head = num_slices // H
        slice_k = K // slices_per_head  # 128 -> 32, 每组 4 个 slice
        print(f"  🧩 [LOCAL PARAM] K={K}, L={L}, H={H}, SlicesPerHead={slices_per_head}")

        for filepath in group_files:
            filename = os.path.basename(filepath)
            
            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match:
                continue
            
            shape = tuple(map(int, match.group(1).split('x')))
            
            # 支持 fp16 读取
            file_dtype = np.float32
            if "f16" in filename.lower() or "float16" in filename.lower():
                file_dtype = np.float16
            data = np.fromfile(filepath, dtype=file_dtype).astype(np.float32).reshape(shape, order='F')
            
            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | Shape: {shape}")

            data_2d = data.squeeze()
            if data_2d.ndim == 0:
                data_2d = data_2d.reshape(1, 1)
            elif data_2d.ndim == 1:
                data_2d = data_2d.reshape(-1, 1)

            # ==================== attn_scores ====================
            if is_attn_scores_group:
                # in0: (KxL) -> (K//slices//H x L) == (32x32)
                if matrix_id == "A" and data_2d.shape[:2] == (K, L):
                    for i in range(slices_per_head):
                        k_start = i * slice_k
                        slice_data = data_2d[k_start:k_start + slice_k, :]
                        save_before_relayout(before_install_dir, op_id, i, out_name, slice_data)

                        relayout_data = relayout_slice_M8_N(slice_data)
                        slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])

                # in1: (KxLxH) -> 每个 head 内再按 4 slice 切，每个 slice 仍保持 32x32
                elif matrix_id == "B":
                    data_3d = data.reshape((K, L, H), order='F')
                    for h_idx in range(H):
                        for i in range(slices_per_head):
                            k_start = i * slice_k
                            slice_2d = data_3d[k_start:k_start + slice_k, :, h_idx]
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, slice_2d)

                            relayout_data = relayout_slice_M8_N(slice_2d)
                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_2d.shape[0], cols=slice_2d.shape[1])

                # out: (LxLxH) -> 每 slices_per_head 个 slice 一组，H 组；每个 slice 都是 32x32
                elif matrix_id == "D":
                    data_3d = data.reshape((L, L, H), order='F')
                    for h_idx in range(H):
                        head_out = data_3d[:, :, h_idx]
                        relayout_head = relayout_slice_M8_N(head_out)
                        for i in range(slices_per_head):
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, head_out)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_head.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=L, cols=L)

            # ==================== attn_out ====================
            else:
                # in0: (LxK) -> (L x K//slices//H) == (32x32)
                if matrix_id == "A" and data_2d.shape[:2] == (L, K):
                    for i in range(slices_per_head):
                        k_start = i * slice_k
                        slice_data = data_2d[:, k_start:k_start + slice_k]
                        save_before_relayout(before_install_dir, op_id, i, out_name, slice_data)

                        relayout_data = relayout_slice_M8_N(slice_data)
                        slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])

                # in1: (LxLxH) -> 每个 head 内广播到 4 个 slice；每个 slice 保持 32x32
                elif matrix_id == "B":
                    data_3d = data.reshape((L, L, H), order='F')
                    for h_idx in range(H):
                        head_in1 = data_3d[:, :, h_idx]
                        relayout_head = relayout_slice_M8_N(head_in1)
                        for i in range(slices_per_head):
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, head_in1)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_head.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=L, cols=L)

                # out: (KxLxH) -> 每个 head 内按 4 slice 切成 32x32
                elif matrix_id == "D":
                    data_3d = data.reshape((K, L, H), order='F')
                    for h_idx in range(H):
                        for i in range(slices_per_head):
                            k_start = i * slice_k
                            slice_2d = data_3d[k_start:k_start + slice_k, :, h_idx]
                            global_idx = h_idx * slices_per_head + i
                            save_before_relayout(before_install_dir, op_id, global_idx, out_name, slice_2d)

                            relayout_data = relayout_slice_M8_N(slice_2d)
                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_2d.shape[0], cols=slice_2d.shape[1])

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All GEMM-Local groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 改为指向 python_golden 目录（而不是 sub_ops）
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "gemm_local"))
    process_gemm_local_tensors(input_dir, output_dir)