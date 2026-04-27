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
    """根据文件名将 rope 子操作映射到 op 文件夹"""
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

def split_op1_matrix_b_slices(matrix_2d):
    """
    op1/matrix_B 专用：
    128x32 先沿行拆成 4 份 32x32，后续广播到 7 个 head 组。
    """
    matrix_2d = np.asarray(matrix_2d, dtype=np.float32)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)

    if matrix_2d.shape != (128, 32):
        raise ValueError(f"op1 matrix_B 期望 128x32, 实际得到: {matrix_2d.shape}")

    return list(np.array_split(matrix_2d, 4, axis=0))

def extract_rope_head_data(data, head_idx, head_axis):
    if head_axis == 2:
        return data[:, :, head_idx, :].squeeze()
    if head_axis == 1:
        return data[:, head_idx, :, :].squeeze()
    raise ValueError(f"Unsupported head axis: {head_axis}")

def split_rope_32x32_slices(head_data, num_slices=4):
    """
    head_data 先裁成 2D 后，再把 L=128 这一维切成 4 份，每份 32x32。
    兼容 32x128 / 128x32 两种排布。
    """
    head_data = np.asarray(head_data, dtype=np.float32)
    if head_data.ndim == 1:
        head_data = head_data.reshape(-1, 1)

    if head_data.shape == (32, 128):
        return [head_data[:, i * 32:(i + 1) * 32] for i in range(num_slices)]
    if head_data.shape == (128, 32):
        return [head_data[i * 32:(i + 1) * 32, :] for i in range(num_slices)]

    raise ValueError(f"Unexpected head_data shape for rope split: {head_data.shape}")

def process_rope_tensors(input_dir, output_dir):
    """
    处理 rope 生成的所有 sub_op .bin 文件。
    - 28个slice分为7组，每组4个slice。
    - 每个head的数据 (128x32) 分配给对应的一组slice。
    - 在head内，x被看作4个32x32的slice。
    """
    print(f"🚀 Starting RoPE tensor relayout in: {input_dir}")
    
    install_dir = os.path.join(output_dir, "install")
    before_install_dir = os.path.join(output_dir, "install_beforerelayout")
    num_slices = 28
    num_heads = 7
    slices_per_group = num_slices // num_heads
    slice_width = 32

    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))

    # 仅处理 sub_op 文件（in1 已由上游保存，不再手工拼 rope_neox 常量）
    valid_files = [f for f in bin_files if "_subop-" in os.path.basename(f) and "rope" in os.path.basename(f)]
    if not valid_files:
        print("❌ No valid rope subop .bin files found in the directory.")
        return

    # 不再锁死单个 prefix，避免只处理到一组 rope 文件
    print(f"🎯 Found {len(valid_files)} rope subop files.")

    for filepath in valid_files:
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
        print(f"📦 Processing: {filename} -> {op_id}/{out_name} | Shape: {shape}")

        # 先识别 head 维：优先 axis=2，再 axis=1
        head_axis = None
        if data.ndim >= 3 and data.shape[2] == num_heads:
            head_axis = 2
        elif data.ndim >= 2 and data.shape[1] == num_heads:
            head_axis = 1

        if head_axis is not None:
            for head_idx in range(num_heads):
                head_data = extract_rope_head_data(data, head_idx, head_axis)
                if head_data.ndim == 1:
                    head_data = head_data.reshape(-1, 1)

                # 先按 L 维裁成 4 个 slice，每个 slice 都应是 32x32
                slice_datas = split_rope_32x32_slices(head_data, slices_per_group)

                for slice_in_group_idx, slice_data_32x32 in enumerate(slice_datas):
                    relayout_data = relayout_slice_M8_N(slice_data_32x32)
                    global_slice_idx = head_idx * slices_per_group + slice_in_group_idx

                    save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data_32x32)

                    slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=slice_data_32x32.shape[0], cols=slice_data_32x32.shape[1])
        else:
            # 对于没有 head 维度的数据，走特殊处理或兜底
            data_2d = data.squeeze()
            if data_2d.ndim == 1:
                data_2d = data_2d.reshape(-1, 1)

            # op1/matrix_A 和 op1/matrix_D：只裁切，不广播
            if op_id == "op1" and matrix_id in ("A", "D"):
                slice_datas = split_rope_32x32_slices(data_2d, slices_per_group)
                for slice_in_group_idx, slice_data in enumerate(slice_datas):
                    global_slice_idx = slice_in_group_idx
                    save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                    slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data = relayout_slice_M8_N(slice_data)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])
                continue

            # op1/matrix_B：保持原来的 4 份切片 + 7 组 head 广播
            if op_id == "op1" and matrix_id == "B":
                matrix_b_slices = split_op1_matrix_b_slices(data_2d)
                for head_idx in range(num_heads):
                    for slice_in_group_idx, slice_data in enumerate(matrix_b_slices):
                        global_slice_idx = head_idx * slices_per_group + slice_in_group_idx
                        save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                        slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data = relayout_slice_M8_N(slice_data)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])
                continue

            # 兜底：广播
            relayout_data = relayout_slice_M8_N(data_2d)
            for i in range(num_slices):
                save_before_relayout(before_install_dir, op_id, i, out_name, data_2d)
                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=data_2d.shape[0], cols=data_2d.shape[1])

    print(f"\n✅ All RoPE tensors sliced and saved under: {install_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "rope"))
    process_rope_tensors(input_dir, output_dir)
