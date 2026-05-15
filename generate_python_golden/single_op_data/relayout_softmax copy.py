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
    # 假设输入为 (N, M)，转置后为 (M, N)
    slice_data = slice_data.T


    M, N = slice_data.shape
    relayout_data = []
    for m_outer in range(0, M, 8):
        limit = min(m_outer + 8, M)
        for n_idx in range(N):
            block = slice_data[m_outer:limit, n_idx]
            relayout_data.extend(block)
    return np.array(relayout_data, dtype=slice_data.dtype)

def get_op_id(filename):
    """根据文件名将 softmax 子操作映射到 op 文件夹"""
    # 规则从最具体到最宽泛，防止错误匹配
    if "add_MN_MN" in filename: return "op0"
    if "sub_SFU" in filename: return "op2"
    if "sum_SFU" in filename: return "op3"
    if "mul_MN_M" in filename: return "op4"
    # 将 "max" 放在最后，因为它可能被其他名称包含
    if "max" in filename: return "op1"
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

def split_op1_max_slices(head_data, matrix_id):
    """
    op1(max) 专用切片规则（恢复原尺寸）：
    - 每个 head 的 4 个 slice 使用相同的完整输入/输出矩阵
    - 不做转置，不做切分
    """
    head_data = np.asarray(head_data, dtype=np.float32)
    if head_data.ndim == 1:
        head_data = head_data.reshape(-1, 1)
        
    # if matrix_id == "A":
    #     # 按你的要求：先转置 matrixA，再按这个维度切
    #     head_data = head_data.T
    #     return list(np.array_split(head_data, 1, axis=1))
    # if matrix_id == "D":
    #     return [head_data.copy() for _ in range(4)]    

    return [head_data.copy() for _ in range(4)]

def process_softmax_tensors(input_dir, output_dir):
    """
    处理 softmax 生成的所有 sub_op .bin 文件。
    - 28个slice分为7组，每组4个slice。
    - 每个head的数据 (LxL) 分配给对应的一组slice。
    """
    print(f"🚀 Starting Softmax tensor relayout in: {input_dir}")
    
    install_dir = os.path.join(output_dir, "install")
    before_install_dir = os.path.join(output_dir, "install_beforerelayout")
    num_slices = 28
    num_heads = 7
    slices_per_group = num_slices // num_heads # 28 / 7 = 4

    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    
    # --- 新增：锁死实例前缀 ---
    valid_files = [f for f in bin_files if "soft_max" in f and "_subop" in f]
    if not valid_files:
        print("❌ No valid softmax .bin files found in the directory.")
        return
    prefixes = sorted(list(set([os.path.basename(f).split("_subop-")[0] for f in valid_files])))
    target_prefix = prefixes[0]
    print(f"🎯 Locking to specific instance: '{target_prefix}'")
    # -------------------------

    for filepath in bin_files:
        filename = os.path.basename(filepath)
        
        if target_prefix and not filename.startswith(target_prefix):
            continue
            
        match = re.search(r"_shape([\dx]+)_dtype", filename)
        if not match: continue
        
        op_id = get_op_id(filename)
        matrix_id = get_matrix_name(filename)
        if op_id == "unknown_op" or matrix_id == "unknown_matrix": continue

        out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
        
        shape = tuple(map(int, match.group(1).split('x')))
        data = np.fromfile(filepath, dtype=np.float32).reshape(shape, order='F')
        
        print(f"📦 Processing: {filename} -> {op_id}/{out_name} | Shape: {shape}")
        
        if data.ndim >= 3 and data.shape[2] == num_heads:
            for head_idx in range(num_heads):
                head_data = data[:, :, head_idx, :].squeeze()
                if head_data.ndim == 1:
                    head_data = head_data.reshape(-1, 1)

                # op1(max) 恢复为每个head的4个slice放相同完整矩阵
                if op_id == "op1":
                    op1_slice_datas = split_op1_max_slices(head_data, matrix_id)
                    for slice_in_group_idx, slice_data in enumerate(op1_slice_datas):
                        relayout_data = relayout_slice_M8_N(slice_data)
                        global_slice_idx = head_idx * slices_per_group + slice_in_group_idx

                        save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                        slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)

                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1])
                    continue

                # 对这个 head 的 2D 数据进行重排
                relayout_data = relayout_slice_M8_N(head_data)

                # 计算这个 head 应该分发到哪些 slice
                start_slice_idx = head_idx * slices_per_group
                end_slice_idx = start_slice_idx + slices_per_group

                for slice_idx in range(start_slice_idx, end_slice_idx):
                    # 先保存未 relayout 数据
                    save_before_relayout(before_install_dir, op_id, slice_idx, out_name, head_data)

                    slice_dir = os.path.join(install_dir, op_id, f"slice{slice_idx:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=head_data.shape[0], cols=head_data.shape[1])
        else:
            # 对于没有 head 维度的数据 (例如 mask)，或者维度不匹配的，采用广播策略
            print(f"  ⚠️ No head dimension found or shape mismatch. Broadcasting to all slices.")
            if data.ndim > 2:
                data_2d = data.reshape(data.shape[0], -1, order='F')
            elif data.ndim == 1:
                data_2d = data.reshape(data.shape[0], 1)
            else:
                data_2d = data.squeeze()

            relayout_data = relayout_slice_M8_N(data_2d)
            for i in range(num_slices):
                # 先保存未 relayout 数据
                save_before_relayout(before_install_dir, op_id, i, out_name, data_2d)

                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=data_2d.shape[0], cols=data_2d.shape[1])

    print(f"\n✅ All Softmax tensors sliced and saved under: {install_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "softmax"))
    process_softmax_tensors(input_dir, output_dir)
