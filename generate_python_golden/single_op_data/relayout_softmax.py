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
            # 拼接成 128 个二进制字符
            f.write(f"{str_float3}{str_float2}{str_float1}{str_float0}\n")

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

def process_softmax_tensors(input_dir, output_dir):
    """
    处理 softmax 生成的所有 sub_op .bin 文件。
    - 28个slice分为7组，每组4个slice。
    - 每个head的数据 (LxL) 分配给对应的一组slice。
    """
    print(f"🚀 Starting Softmax tensor relayout in: {input_dir}")
    
    install_dir = os.path.join(output_dir, "install")
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
        
        # 根据 head 维度进行数据分发
        # 假设 head 维度是第3个维度 (shape[2])
        if data.ndim >= 3 and data.shape[2] == num_heads:
            for head_idx in range(num_heads):
                # 提取当前 head 的数据，并降维到 2D
                head_data = data[:, :, head_idx, :].squeeze()
                if head_data.ndim == 1:
                    head_data = head_data.reshape(-1, 1)

                # 对这个 head 的 2D 数据进行重排
                relayout_data = relayout_slice_M8_N(head_data)

                # 计算这个 head 应该分发到哪些 slice
                start_slice_idx = head_idx * slices_per_group
                end_slice_idx = start_slice_idx + slices_per_group

                for slice_idx in range(start_slice_idx, end_slice_idx):
                    slice_dir = os.path.join(install_dir, op_id, f"slice{slice_idx:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path)
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
                slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                
                out_path = os.path.join(slice_dir, out_name)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path)

    print(f"\n✅ All Softmax tensors sliced and saved under: {install_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "softmax"))
    process_softmax_tensors(input_dir, output_dir)
