import numpy as np
import os
import struct
import math

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

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
            f.write(f"{str_float3}{str_float2}{str_float1}{str_float0}\\n")

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols)

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
            f.write("\\n")

def fp32_fma_accumulate(acc, a, b):
    """
    硬件 SUMMAC 对齐：acc = fma(a, b, acc)
    优先用 math.fma；若环境不支持则回退到乘加。
    """
    try:
        return np.float32(math.fma(float(a), float(b), float(acc)))
    except AttributeError:
        return np.float32(np.float32(a) * np.float32(b) + np.float32(acc))

def sum_mac(x):
    """
    实现 rmsnorm 的 op0 功能：sum_mac
    使用 fma 累加计算输入矩阵 x 每一列的平方和。
    x: 输入矩阵 (M, N)
    """
    M, N = x.shape
    output = np.zeros(N, dtype=np.float32)
    for j in range(N):
        col_sum = np.float32(0.0)
        for i in range(M):
            val = x[i, j]
            col_sum = fp32_fma_accumulate(col_sum, val, val)
        output[j] = col_sum
    return output

def main():
    """
    主函数，生成数据、执行计算、并按 slice 规则保存结果。
    """
    base_output_dir = "summac_data"
    install_dir = os.path.join(base_output_dir, "install")
    before_install_dir = os.path.join(base_output_dir, "install_beforerelayout")
    op_id = "op0"  # sum_mac 对应 op0
    num_slices = 28

    os.makedirs(os.path.join(install_dir, op_id), exist_ok=True)
    os.makedirs(os.path.join(before_install_dir, op_id), exist_ok=True)

    # 1. 生成 M512xN512 大小的 fp32 随机数据
    M, N = 512, 512
    np.random.seed(0)
    input_data = np.random.rand(M, N).astype(np.float32)

    # 2. 执行 sum_mac (op0)
    output_data = sum_mac(input_data)
    output_data = output_data.reshape(1, N) # 保持二维

    # 3. 将完整数据广播到 28 个 slice
    print(f"🚀 Broadcasting data to {num_slices} slices under '{base_output_dir}'...")
    for i in range(num_slices):
        slice_dir_before = os.path.join(before_install_dir, op_id, f"slice{i:02d}")
        slice_dir_install = os.path.join(install_dir, op_id, f"slice{i:02d}")
        os.makedirs(slice_dir_before, exist_ok=True)
        os.makedirs(slice_dir_install, exist_ok=True)

        # 定义输入输出文件名
        in_name = "matrix_A_linearized_128bit.bin"
        out_name = "matrix_D_linearized_128bit.bin"

        # --- 处理输入数据 ---
        in_path_before = os.path.join(slice_dir_before, in_name)
        input_data.flatten(order='C').tofile(in_path_before)
        convert_to_128bit_txt(in_path_before, rows=M, cols=N)
        
        # 模拟 relayout 后的文件，内容在这里是相同的
        in_path_install = os.path.join(slice_dir_install, in_name)
        input_data.flatten(order='C').tofile(in_path_install)
        convert_to_128bit_txt(in_path_install, rows=M, cols=N)

        # --- 处理输出数据 ---
        out_path_before = os.path.join(slice_dir_before, out_name)
        output_data.flatten(order='C').tofile(out_path_before)
        convert_to_128bit_txt(out_path_before, rows=1, cols=N)

        # 模拟 relayout 后的文件
        out_path_install = os.path.join(slice_dir_install, out_name)
        output_data.flatten(order='C').tofile(out_path_install)
        convert_to_128bit_txt(out_path_install, rows=1, cols=N)

    print(f"✅ All summac tensors sliced and saved under: {install_dir}")

if __name__ == "__main__":
    main()
