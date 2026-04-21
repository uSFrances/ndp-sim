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

def process_rope_tensors(input_dir, output_dir):
    """
    处理 rope 生成的所有 sub_op .bin 文件。
    - 28个slice分为7组，每组4个slice。
    - 每个head的数据 (128x32) 分配给对应的一组slice。
    - 在head内，x被看作4个32x32的slice。
    - add_final_in1 的数据在生成时已按硬件需求调换。
    """
    print(f"🚀 Starting RoPE tensor relayout in: {input_dir}")
    
    install_dir = os.path.join(output_dir, "install")
    num_slices = 28
    num_heads = 7
    slices_per_group = num_slices // num_heads # 4
    slice_width = 32 # 每个slice的列宽

    # 直接在获取所有的 bin，包括手动放进来的 sin 和 cos
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))

    if not bin_files:
        print("❌ No .bin files found in the directory.")
        return

    for filepath in bin_files:
        filename = os.path.basename(filepath)
        
        # 为了能被 get_matrix_name 识别，我们给 sin/cos 文件名打上标记
        if "rope_neox_sin" in filename:
            # sin 用于 mul_sin 的 in1
            filename = "subop-mul_sin_in1_" + filename
        elif "rope_neox_cos" in filename:
            # cos 用于 mul_cos 的 in1
            filename = "subop-mul_cos_in1_" + filename

        match = re.search(r"_shape([\dx]+)_dtype", filename)
        # 对原始 sin/cos 文件名做特殊匹配
        if not match:
            sincos_match = re.search(r"ne2_(\d+)\.bin", filename)
            if sincos_match:
                # 手动构建元数据，文件本身可能很长(例如512 token)，但是我们加载时需要指定全shape
                shape = (64, int(sincos_match.group(1)), 1, 1)
            else:
                continue
        else:
            shape = tuple(map(int, match.group(1).split('x')))

        op_id = get_op_id(filename)
        matrix_id = get_matrix_name(filename)
        if op_id == "unknown_op" or matrix_id == "unknown_matrix": continue

        out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
        
        data = np.fromfile(filepath, dtype=np.float32).reshape(shape, order='F')
        
        print(f"📦 Processing: {os.path.basename(filepath)} -> {op_id}/{out_name} | Shape: {shape}")

        # RoPE 的 head 维度是 shape[1]
        if data.ndim >= 2 and data.shape[1] == num_heads:
            for head_idx in range(num_heads):
                # 提取当前 head 的完整数据 (e.g., 128, 32)
                # .squeeze() 会移除所有大小为1的维度
                head_data = data[:, head_idx, ...].squeeze()

                # 将 head_data (e.g., 128x32) 按【行 (特征维度)】切分为 4 个 32x32 的 slice
                for slice_in_group_idx in range(slices_per_group):
                    row_start = slice_in_group_idx * slice_width
                    row_end = row_start + slice_width
                    
                    # 这是硬件上一个slice实际需要处理的数据
                    slice_data_32x32 = head_data[row_start:row_end, :]

                    # 对这个 32x32 的数据块进行重排
                    relayout_data = relayout_slice_M8_N(slice_data_32x32)

                    # 计算这个数据块应该分发到哪个全局 slice
                    global_slice_idx = head_idx * slices_per_group + slice_in_group_idx
                    
                    slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path)
        else:
            # 对于没有 head 维度的数据 (例如 sin/cos)，需要特殊裁切处理
            print(f"  ⚠️ Extracting and broadcasting sin/cos sequence.")
            data_2d = data.squeeze()
            if data_2d.ndim == 1:
                data_2d = data_2d.reshape(-1, 1)

            # 专属提取 sin / cos (64, 512) -> (64, 32)
            if data_2d.shape[0] == 64:
                # 裁切获取序列长度为32的编码
                sincos_data_for_seq = data_2d[:, :32] # (64, 32)
                
                # 将其分为上下两份，x0和x1用的同一组对应的数据
                part0_sincos = sincos_data_for_seq[:32, :] # 前32个元素，用于 slice 0 和 2
                part1_sincos = sincos_data_for_seq[32:, :] # 后32个元素，用于 slice 1 和 3
                
                relayout_part0 = relayout_slice_M8_N(part0_sincos)
                relayout_part1 = relayout_slice_M8_N(part1_sincos)
                
                for head_idx in range(num_heads):
                    for slice_in_group_idx in range(slices_per_group):
                        global_slice_idx = head_idx * slices_per_group + slice_in_group_idx
                        slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        
                        # Slice 0, 2 拿 part0； Slice 1, 3 拿 part1
                        if slice_in_group_idx == 0 or slice_in_group_idx == 2:
                            relayout_part0.tofile(out_path)
                        else:
                            relayout_part1.tofile(out_path)
                        convert_to_128bit_txt(out_path)
            else:
                # 保底的通用形状重排广播
                relayout_data = relayout_slice_M8_N(data_2d)
                for i in range(num_slices):
                    slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path)

    print(f"\n✅ All RoPE tensors sliced and saved under: {install_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    process_rope_tensors(input_dir, current_dir)
