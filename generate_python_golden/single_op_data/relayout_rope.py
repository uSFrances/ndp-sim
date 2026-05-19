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
    """根据文件名将 rope 子操作映射 to op 文件夹"""
    if "mul_cos" in filename: return "op0"  # 第一个 mul
    if "mul_sin" in filename: return "op1"  # 第二个 mul
    if "add_final" in filename: return "op2" # 最终 add
    return "unknown_op"

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

def split_op1_matrix_b_slices(matrix_2d, total_n, tile_m):
    """
    op1/matrix_B 专用：
    (total_n x tile_m) 先沿行拆成多份 tile_m x tile_m，后续广播。
    """
    matrix_2d = np.asarray(matrix_2d, dtype=np.float32)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)

    if matrix_2d.shape != (total_n, tile_m):
        raise ValueError(f"op1 matrix_B 期望 {total_n}x{tile_m}, 实际得到: {matrix_2d.shape}")

    slices_per_group = total_n // tile_m
    return list(np.array_split(matrix_2d, slices_per_group, axis=0))

def extract_rope_head_data(data, head_idx, head_axis):
    if head_axis == 2:
        return data[:, :, head_idx, :].squeeze()
    if head_axis == 1:
        return data[:, head_idx, :, :].squeeze()
    raise ValueError(f"Unsupported head axis: {head_axis}")

def split_rope_32x32_slices(head_data, num_slices, total_n, tile_m):
    """
    按参切分片，兼容 (tile_m, total_n) / (total_n, tile_m) 两种排布。
    切分始终在 N 维度（行）上进行。
    """
    head_data = np.asarray(head_data, dtype=np.float32)
    if head_data.ndim == 1:
        head_data = head_data.reshape(-1, 1)

    # 假设 M=tile_m, N=total_n，沿 N 切
    if head_data.shape == (tile_m, total_n):
        n_per_slice = total_n // num_slices
        return [head_data[:, i * n_per_slice:(i + 1) * n_per_slice] for i in range(num_slices)]
    # 假设 M=total_n, N=tile_m，沿 M (行) 切
    if head_data.shape == (total_n, tile_m):
        m_per_slice = total_n // num_slices
        return [head_data[i * m_per_slice:(i + 1) * m_per_slice, :] for i in range(num_slices)]

    raise ValueError(f"Unexpected head_data shape for rope split: {head_data.shape}, exp dims related to: {total_n}, {tile_m}")


def infer_rope_params(group_files, target_prefix):
    """
    仅针对当前 target_prefix 的 group_files 内部进行参数推断。
    """
    for fp in group_files:
        filename = os.path.basename(fp)
        if not filename.startswith(target_prefix):
            continue
            
        m = re.search(r"_shape([\dx]+)_dtype", filename)
        if m:
            dims = [int(x) for x in m.group(1).split('x')]
            valid_dims = [d for d in dims if d != 1]
            
            if len(valid_dims) >= 3:
                sorted_dims = sorted(valid_dims)
                tile_m = sorted_dims[0]        # 最小的是 tile_m (如 32)
                total_n = sorted_dims[-1]      # 最大的是 total_n (如 128)
                current_heads = sorted_dims[1] # 中间的是 Head 数量
                return total_n, tile_m, current_heads
                
            elif len(valid_dims) == 2:
                # 🔴 如果只有两个有效轴，代表没有独立的 head 轴 (head 数量为 1)
                # 按照您的要求，如果 head 是 1，将其显式作为虚拟的 1 处理，交给后续逻辑去广播扩展为 7 组
                return max(valid_dims), min(valid_dims), 1
                
    return 128, 32, 1


def process_rope_tensors(input_dir, output_dir):
    """
    处理 rope 生成的所有 sub_op .bin 文件。
    """
    print(f"🚀 Starting RoPE tensor relayout in: {input_dir}")
    
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    valid_files = [f for f in bin_files if "_subop-" in os.path.basename(f) and "rope" in os.path.basename(f)]
    if not valid_files:
        print("❌ No valid rope subop .bin files found in the directory.")
        return

    prefixes = sorted(list(set([os.path.basename(f).split("_subop-")[0] for f in valid_files])))

    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")
        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        group_files = [f for f in valid_files if os.path.basename(f).startswith(target_prefix)]
        if not group_files:
            continue

        total_n, tile_m, inferred_heads = infer_rope_params(group_files, target_prefix)
        current_slices_per_group = total_n // tile_m
        
        # 🔴 核心业务设定：如果是虚拟的单头 (inferred_heads == 1) 
        # 为了生成 7组 乘以 4slice = 28 个完整 slice 的一模一样的数据，我们将执行循环数强制设为 7
        is_head_one_broadcasting = (inferred_heads == 1)
        current_num_heads = 7 if is_head_one_broadcasting else inferred_heads
        
        print(f"  🧩 [LOCAL PARAM] N={total_n}, M={tile_m}, SlicesPerGroup={current_slices_per_group}, LogicHeads={current_num_heads} (IsBroadcasting={is_head_one_broadcasting})")

        for filepath in group_files:
            filename = os.path.basename(filepath)

            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match:
                continue
            shape = tuple(map(int, match.group(1).split('x')))

            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename)
            if op_id == "unknown_op" or matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            data = np.fromfile(filepath, dtype=np.float32).reshape(shape, order='F')
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | Shape: {shape}")

            # 识别 head 维：如果原本就是多头数据，提取轴
            head_axis = None
            if not is_head_one_broadcasting:
                if data.ndim >= 3 and data.shape[2] == inferred_heads:
                    head_axis = 2
                elif data.ndim >= 2 and data.shape[1] == inferred_heads:
                    head_axis = 1

            # ----------------------------------------------------
            # 分支 A：存在真实独立 head 轴的多头标准处理
            # ----------------------------------------------------
            if head_axis is not None:
                for head_idx in range(current_num_heads):
                    head_data = extract_rope_head_data(data, head_idx, head_axis)
                    if head_data.ndim == 1:
                        head_data = head_data.reshape(-1, 1)
                    slice_datas = split_rope_32x32_slices(head_data, current_slices_per_group, total_n, tile_m)
                    for slice_in_group_idx, slice_data_32x32 in enumerate(slice_datas):
                        relayout_data = relayout_slice_M8_N(slice_data_32x32)
                        global_slice_idx = head_idx * current_slices_per_group + slice_in_group_idx

                        save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data_32x32)

                        slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_data_32x32.shape[0], cols=slice_data_32x32.shape[1])
            
            # ----------------------------------------------------
            # 分支 B：Head 为 1 时的共享广播数据平面处理 (彻底修好 op0, op1, op2 完整度)
            # ----------------------------------------------------
            else:
                data_2d = data.squeeze()
                if data_2d.ndim == 1:
                    data_2d = data_2d.reshape(-1, 1)

                # 情况一：如果是 op0, op2 全阵，或者 op1 的 A/D 矩阵（它们包含完整的 128 长度信息，一份切出 4 个 slice）
                if (op_id in ("op0", "op2")) or (op_id == "op1" and matrix_id in ("A", "D")):
                    # 先根据 total_n 和 tile_m 将当前的这一套一维平面数据切出 4 个基础切片
                    base_slices = split_rope_32x32_slices(data_2d, current_slices_per_group, total_n, tile_m)
                    
                    # 🔴 关键广播复制核心：外层循环 7 次模拟 7 个 head，计算生成一模一样的数据
                    for head_idx in range(current_num_heads):
                        for slice_in_group_idx, slice_data in enumerate(base_slices):
                            global_slice_idx = head_idx * current_slices_per_group + slice_in_group_idx
                            
                            save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data = relayout_slice_M8_N(slice_data)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])
                    continue

                # 情况二：如果是 op1 的矩阵 B （专属尺寸为 total_n x tile_m，切出 4 个 32x32）
                if op_id == "op1" and matrix_id == "B":
                    matrix_b_slices = split_op1_matrix_b_slices(data_2d, total_n, tile_m)
                    
                    # 同样外层循环 7 次分发出 28 个一模一样的重排切片
                    for head_idx in range(current_num_heads):
                        for slice_in_group_idx, slice_data in enumerate(matrix_b_slices):
                            global_slice_idx = head_idx * current_slices_per_group + slice_in_group_idx
                            
                            save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data = relayout_slice_M8_N(slice_data)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])
                    continue

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All RoPE groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "rope"))
    process_rope_tensors(input_dir, output_dir)