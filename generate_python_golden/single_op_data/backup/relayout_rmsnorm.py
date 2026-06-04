import numpy as np
import os
import glob
import re
import struct

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def convert_to_decimal_txt(bin_path, rows=None, cols=None):
    """读取 bin 文件并输出十进制矩阵 txt；对于 relayout 后的数据一维展开"""
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

    # 严格按照固有的 F-style（列优先）还原物理算子原本的二维矩阵
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
    对一个子切片进行硬件 M8_N 重排。
    【重要修正】：这里的输入维度直接对应物理算子轴，即 (物理N维, 物理M维)。
    因为数据是 F-style（列优先）存储，所以第二维（物理M维）才是内存中变化最快的连续轴。
    
    重排规则：外层按步长 8 遍历变化最快的物理 M 轴，中层遍历物理 N 轴，内层连续读取 8 个连续的 M 轴元素。
    """
    phys_N, phys_M = slice_data.shape
    relayout_data = []
    
    # 外层按步长 8 遍历变化最快的物理 M 轴
    for m_outer in range(0, phys_M, 8):
        limit = min(m_outer + 8, phys_M)
        # 中层遍历物理 N 轴
        for n_idx in range(phys_N):
            # 提取连续的 M 轴切片块（对 F-style 而言，此块在底层内存绝对连续）
            block = slice_data[n_idx, m_outer:limit]
            relayout_data.extend(block)
            
    return np.array(relayout_data, dtype=slice_data.dtype)

def get_op_id(filename):
    """根据文件名将子操作映射到对应的 op 文件夹"""
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

def save_before_relayout(before_install_dir, op_id, slice_idx, out_name, slice_data):
    """保存切分后、重排前的数据状态"""
    slice_dir = os.path.join(before_install_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    out_path = os.path.join(slice_dir, out_name)
    
    # 严格按照当前切片真实的内存排布直接导出
    slice_data.tofile(out_path)
    # 传入真实的物理维度形状用于生成 decimal 文件
    convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])

def infer_group_params_from_filename(group_files):
    """
    从文件名中的 _shapeNxM_dtype 推导真实物理维度的 N 和 M 以及切片数
    """
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

    for fp in group_files:
        fn = os.path.basename(fp)
        m = re.search(r"_shape([\dx]+)_dtype", fn)
        if not m:
            continue
        
        dims = [d for d in map(int, m.group(1).split('x')) if d > 1]
        if len(dims) >= 2:
            return min(dims), max(dims)

    return None, None

def next_power_of_two(x: int) -> int:
	"""向上取整到最近的 2 的幂（最小为1）"""
	if x <= 1:
		return 1
	return 1 << (x - 1).bit_length()

def process_rmsnorm_tensors(
    input_dir,
    output_dir,
    kv_replace_enabled=True,
    kv_sum_bin_name="kv_caseA_sum_32x28_fstyle.bin",
    kv_replace_filenames=None,
    kv_caseA_preferred_source_key="sum_mac_in0",
):
    """
    处理 rmsnorm 生成的所有 sub_op .bin 文件。
    """
    print(f"🚀 Starting RMSNorm tensor relayout in: {input_dir}")

    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    valid_files = [f for f in bin_files if "rms_norm" in os.path.basename(f) and "_subop" in os.path.basename(f)]
    if not valid_files:
        print("❌ No valid rms_norm .bin files found in the directory.")
        return

    prefixes = sorted(list(set([os.path.basename(f).split("_subop-")[0] for f in valid_files])))

    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")

        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        group_files = [f for f in valid_files if os.path.basename(f).startswith(target_prefix)]

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
            # 解析出的尺寸直接对应文件名物理轴，剔除1
            shape = tuple(d for d in map(int, shape_str.split('x')) if d > 1)
            if not shape:
                shape = (1, 1)
            elif len(shape) == 1:
                shape = (shape[0], 1)

            # 严格使用 F-style 还原出直接对应文件名的二维物理视图 (物理N, 物理M)
            data_2d = np.fromfile(filepath, dtype=np.float32).reshape(shape, order='F')

            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if op_id == "unknown_op" or matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | Physical Shape: {data_2d.shape}")

            # 1. 原始大维度数据模式：形状为 (total_n, tile_m)，按 N 维均匀切片
            if data_2d.shape == (total_n, tile_m):
                n_per_slice = total_n // inferred_slices
                for i in range(inferred_slices):
                    n_start = i * n_per_slice
                    # 得到的切片物理形状为 (slice_N, tile_M)，第二维 M 是变化最快的轴
                    slice_nxm = data_2d[n_start:n_start + n_per_slice, :]
                    
                    save_before_relayout(before_install_dir, op_id, i, out_name, slice_nxm)

                    relayout_data = relayout_slice_M8_N(slice_nxm)
                    slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                    os.makedirs(slice_dir, exist_ok=True)

                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=slice_nxm.shape[0], cols=slice_nxm.shape[1])

            # 2. 统计中间量模式：形状为 (tile_m, inferred_slices)
            elif data_2d.shape == (tile_m, inferred_slices):
                for i in range(inferred_slices):
                    slice_mx1 = data_2d[:, i:i+1]
                    
                    save_before_relayout(before_install_dir, op_id, i, out_name, slice_mx1)

                    relayout_data = relayout_slice_M8_N(slice_mx1)
                    slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                    os.makedirs(slice_dir, exist_ok=True)

                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=slice_mx1.shape[0], cols=slice_mx1.shape[1])

            # 3. 归一化广播向量模式：形状为 (tile_m, 1)
            elif data_2d.shape == (tile_m, 1):
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

        # ===== 新增：对 RMS-NORM 组生成 KV 专用 relayout 输出（目录后缀 _kv） =====
        # 仅对包含 "rms_norm" 或 "op-rms_norm" 的前缀进行 KV 规则的二次处理
        if "rms_norm" not in target_prefix and "op-rms_norm" not in target_prefix:
            continue

        kv_group_output_dir = group_output_dir + "_kv"
        kv_install_dir = os.path.join(kv_group_output_dir, "install")
        kv_before_install_dir = os.path.join(kv_group_output_dir, "install_beforerelayout")
        os.makedirs(kv_install_dir, exist_ok=True)
        os.makedirs(kv_before_install_dir, exist_ok=True)

        print(f"🔁 Generating KV-style relayout for group: {target_prefix} -> {kv_group_output_dir}")

        if kv_replace_filenames is None:
            kv_replace_filenames = {
                "blk.0_norm-0_op-rms_norm_subop-sum_mac_out_shape32x28x1x1_dtype_f32.bin",
                "blk.0_norm-0_op-rms_norm_subop-remote_sum_in0_shape32x28x1x1_dtype_f32.bin",
            }

        # 固定策略：切成 4 份并广播7次（4*7=28）
        slices_per_group = 4
        heads = 7

        # 新增：基于情况A切片生成 32x28 的聚合矩阵（每个slice对 N 维求和 -> 32x1）
        kv_caseA_sum_matrix = None
        kv_caseA_sum_bin_path = os.path.join(kv_group_output_dir, kv_sum_bin_name)

        if kv_replace_enabled:
            caseA_candidates = []
            for fp in group_files:
                fn = os.path.basename(fp)
                if "slice" in fn or "shared" in fn:
                    continue
                m = re.search(r"_shape([\dx]+)_dtype", fn)
                if not m:
                    continue
                shape = tuple(d for d in map(int, m.group(1).split('x')) if d > 1)
                if not shape:
                    shape = (1, 1)
                elif len(shape) == 1:
                    shape = (shape[0], 1)
                if shape == (total_n, tile_m):
                    # 优先使用 sum_mac_in0；找不到则退化到第一个情况A文件
                    rank = 0 if kv_caseA_preferred_source_key in fn else 1
                    caseA_candidates.append((rank, fp, fn))

            if caseA_candidates:
                caseA_candidates.sort(key=lambda x: (x[0], x[2]))
                _, src_fp, src_fn = caseA_candidates[0]
                src_2d = np.fromfile(src_fp, dtype=np.float32).reshape((total_n, tile_m), order='F')

                padded_N = next_power_of_two(total_n)
                if padded_N == total_n:
                    padded = src_2d
                else:
                    padded = np.zeros((padded_N, tile_m), dtype=np.float32, order='F')
                    padded[:total_n, :] = src_2d

                slice_n = padded_N // slices_per_group
                kv_caseA_sum_matrix = np.zeros((tile_m, slices_per_group * heads), dtype=np.float32, order='F')

                for head_idx in range(heads):
                    for i in range(slices_per_group):
                        start = i * slice_n
                        end = start + slice_n
                        slice_nxm = padded[start:end, :]  # (N_slice, 32)
                        global_idx = head_idx * slices_per_group + i
                        kv_caseA_sum_matrix[:, global_idx] = np.sum(slice_nxm, axis=0, dtype=np.float32)

                kv_caseA_sum_matrix.reshape(-1, order='F').tofile(kv_caseA_sum_bin_path)
                print(f"  🧮 KV replacement source generated from CaseA: {src_fn} -> {kv_caseA_sum_bin_path}")
            else:
                print("  ⚠️ KV replacement source not generated: no CaseA (total_n, tile_m) file found.")

        for filepath in group_files:
            filename = os.path.basename(filepath)
            if "slice" in filename or "shared" in filename:
                continue
            m = re.search(r"_shape([\dx]+)_dtype", filename)
            if not m:
                continue
            shape = tuple(d for d in map(int, m.group(1).split('x')) if d > 1)
            if not shape:
                shape = (1,1)
            elif len(shape) == 1:
                shape = (shape[0], 1)

            if kv_replace_enabled and filename in kv_replace_filenames and kv_caseA_sum_matrix is not None:
                data_2d = kv_caseA_sum_matrix
                print(f"  ♻️ Replaced source for {filename} with {os.path.basename(kv_caseA_sum_bin_path)}")
            else:
                data_2d = np.fromfile(filepath, dtype=np.float32).reshape(shape, order='F')

            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if op_id == "unknown_op" or matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            print(f"  🔹 KV-processing: {filename} | Physical Shape: {data_2d.shape}")

            # 情况 A: (total_n, tile_m) -> pad N 到 2 的幂，按 N 切成 4 份，再广播 7 倍
            if data_2d.shape == (total_n, tile_m):
                orig_N = total_n
                phys_M = tile_m
                padded_N = next_power_of_two(orig_N)
                if padded_N == orig_N:
                    padded = data_2d
                else:
                    padded = np.zeros((padded_N, phys_M), dtype=np.float32, order='F')
                    padded[:orig_N, :] = data_2d

                # 按 4 份切分（padded_N 必须能被4整除，这里向下取整分块）
                slice_n = padded_N // slices_per_group
                for head_idx in range(heads):
                    for i in range(slices_per_group):
                        start = i * slice_n
                        end = start + slice_n
                        slice_nxm = padded[start:end, :]

                        global_idx = head_idx * slices_per_group + i
                        save_before_relayout(kv_before_install_dir, op_id, global_idx, out_name, slice_nxm)

                        relayout_data = relayout_slice_M8_N(slice_nxm)
                        slice_dir = os.path.join(kv_install_dir, op_id, f"slice{global_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_nxm.shape[0], cols=slice_nxm.shape[1])

            # 情况 B-special: 被替换后的 32x28 数据，按列一一映射到 28 个 slice
            elif (
                kv_replace_enabled
                and filename in kv_replace_filenames
                and data_2d.shape == (tile_m, slices_per_group * heads)
            ):
                for global_idx in range(slices_per_group * heads):
                    slice_mx1 = data_2d[:, global_idx:global_idx+1]
                    save_before_relayout(kv_before_install_dir, op_id, global_idx, out_name, slice_mx1)
                    relayout_data = relayout_slice_M8_N(slice_mx1)
                    slice_dir = os.path.join(kv_install_dir, op_id, f"slice{global_idx:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=slice_mx1.shape[0], cols=slice_mx1.shape[1])

            # 情况 B: (tile_m, inferred_slices) -> 每列作为 (M,1)，广播到 28 个 slice（保持原行为）
            elif data_2d.shape == (tile_m, inferred_slices):
                for i_col in range(inferred_slices):
                    slice_mx1 = data_2d[:, i_col:i_col+1]
                    for global_idx in range(slices_per_group * heads):
                        save_before_relayout(kv_before_install_dir, op_id, global_idx, out_name, slice_mx1)
                        relayout_data = relayout_slice_M8_N(slice_mx1)
                        slice_dir = os.path.join(kv_install_dir, op_id, f"slice{global_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_mx1.shape[0], cols=slice_mx1.shape[1])

            # 情况 C: (tile_m,1) -> 复制到 28 个 slice
            elif data_2d.shape == (tile_m, 1):
                relayout_data = relayout_slice_M8_N(data_2d)
                for global_idx in range(slices_per_group * heads):
                    save_before_relayout(kv_before_install_dir, op_id, global_idx, out_name, data_2d)
                    slice_dir = os.path.join(kv_install_dir, op_id, f"slice{global_idx:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=data_2d.shape[0], cols=data_2d.shape[1])

            else:
                print(f"    ⚠️ KV-skip unrecognized shape: {data_2d.shape}")

        print(f"✅ Finished KV instance group: {target_prefix} -> {kv_group_output_dir}")

    print(f"\n✅ All RMS-Norm groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "rmsnorm"))
    process_rmsnorm_tensors(input_dir, output_dir)