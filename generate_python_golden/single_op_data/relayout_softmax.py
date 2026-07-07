import numpy as np
import os
import glob
import re
import struct
import json

MODEL_PARAMS = {
    "hidden_size": 896,
    "intermediate_size": 1792,
    "num_attention_heads": 7,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "sequence_length": 32,
    "slice_per_head": 4,
    "used_slices": 28,
    "kv_padding": 256,
}

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

def load_model_params(config_path=CONFIG_PATH):
    params = dict(MODEL_PARAMS)
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Config not found, using defaults: {config_path}")
        return params

    for key in params:
        if key in config:
            params[key] = config[key]
    return params

MODEL_PARAMS = load_model_params()

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def float16_to_bin(f):
    return bin(struct.unpack('<H', struct.pack('<e', np.float16(f)))[0])[2:].zfill(16)

def dtype_from_filename(filepath):
    match = re.search(r"_dtype_(f16|f32|float16|float32)", os.path.basename(filepath).lower())
    if not match:
        raise ValueError(f"Cannot determine dtype from filename: {filepath}")
    return np.float16 if match.group(1) in ("f16", "float16") else np.float32

def convert_to_decimal_txt(bin_path, rows=None, cols=None, file_dtype=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）"""
    if file_dtype is None:
        file_dtype = dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=file_dtype)

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

def convert_to_128bit_txt(bin_path, rows=None, cols=None, file_dtype=None):
    """按真实 dtype 输出每行 128-bit：8个float16 或4个float32。"""
    if file_dtype is None:
        file_dtype = dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=file_dtype)
    values_per_line = 8 if file_dtype == np.float16 else 4
    remainder = len(data) % values_per_line
    if remainder:
        data = np.concatenate(
            (data, np.zeros(values_per_line - remainder, dtype=file_dtype))
        )

    txt_path = bin_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as f:
        converter = float16_to_bin if file_dtype == np.float16 else float_to_bin
        for i in range(0, len(data), values_per_line):
            bins = [converter(value) for value in data[i:i + values_per_line]]
            f.write("".join(reversed(bins)) + "\n")

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols, file_dtype=file_dtype)

def relayout_slice_M8_N(slice_data):
    """
    对一个子切片进行硬件标准的 M8_N 重排。
    输入维度直接对应物理算子轴，即 (物理N维, 物理M维)。
    哪个 N 被切分为多份，这里的第一维就是这个 N。
    """
    phys_N, phys_M = slice_data.shape
    relayout_data = []
    
    # 外层按步长 8 遍历变化最快的物理 M 轴
    for m_outer in range(0, phys_M, 8):
        limit = min(m_outer + 8, phys_M)
        # 中层遍历物理 N 轴
        for n_idx in range(phys_N):
            block = slice_data[n_idx, m_outer:limit]
            relayout_data.extend(block)
            
    return np.array(relayout_data, dtype=slice_data.dtype)

def get_op_id(filename):
    """根据文件名将 softmax 子操作映射到 op 文件夹"""
    if "add_MN_MN" in filename: return "op0"
    if "sub_SFU" in filename: return "op2"
    if "sum_SFU" in filename: return "op3"
    if "mul_MN_M" in filename: return "op4"
    if "max" in filename: return "op1"
    return "unknown_op"

def get_matrix_name(filename, op_id):
    """根据输入输出类型映射端口名；softmax op0 的 in1 使用 C。"""
    if "in0" in filename: return "A"
    if "in1" in filename: return "C" if op_id == "op0" else "B"
    if "out" in filename: return "D"
    return "unknown_matrix"

def save_before_relayout(before_install_dir, op_id, slice_idx, out_name, matrix_2d):
    matrix_2d = np.asarray(matrix_2d)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)
    slice_dir = os.path.join(before_install_dir, op_id, f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    out_path = os.path.join(slice_dir, out_name)
    matrix_2d.reshape(-1, order='C').tofile(out_path)
    convert_to_128bit_txt(out_path, rows=matrix_2d.shape[0], cols=matrix_2d.shape[1], file_dtype=matrix_2d.dtype)

def split_op1_max_slices(head_data, matrix_id, slices_per_group):
    """op1(max) 专用切片规则"""
    head_data = np.asarray(head_data)
    if head_data.ndim == 1:
        head_data = head_data.reshape(-1, 1)
    return [head_data.copy() for _ in range(slices_per_group)]

def infer_softmax_params(group_files):
    """从文件名里提取大于 1 的核心维度来推导切片参数"""
    num_heads = MODEL_PARAMS["num_attention_heads"]
    for fp in group_files:
        filename = os.path.basename(fp)
        m = re.search(r"_shape([\dx]+)_dtype", filename)
        if m:
            dims = [int(x) for x in m.group(1).split('x')]
            if len(dims) >= 3:
                num_heads = dims[2] if dims[2] > 1 else num_heads
                break
    return num_heads

def process_softmax_tensors(input_dir, output_dir):
    """
    处理 softmax 生成的所有 sub_op .bin 文件。
    """
    print(f"🚀 Starting Softmax tensor relayout in: {input_dir}")
    
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    
    # 保持您原汁原味的文件查找方式不变
    valid_files = [f for f in bin_files if "soft_max" in f and "_subop" in f]
    if not valid_files:
        print("❌ No valid softmax .bin files found in the directory.")
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

        num_heads = infer_softmax_params(group_files)
        slices_per_group = MODEL_PARAMS["slice_per_head"]
        num_slices = slices_per_group * num_heads
        print(f"  🧩 Inferred params from filename: heads={num_heads}, slices_per_group={slices_per_group}")

        for filepath in group_files:
            filename = os.path.basename(filepath)
            
            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match: continue
            
            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename, op_id)
            if op_id == "unknown_op" or matrix_id == "unknown_matrix": continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            
            # 【精确提取真实轴】：保留包含头轴的形状
            # 对于 1x32x7x1，剥离外围 1 得到 (32, 7)，直接准确对应物理 N 与 Head 轴
            shape_dims = [int(x) for x in match.group(1).split('x') if int(x) != 1]
            if len(shape_dims) == 1:
                shape = (shape_dims[0], 1)
            else:
                shape = tuple(shape_dims)
            
            file_dtype = dtype_from_filename(filename)
            data = np.fromfile(filepath, dtype=file_dtype).reshape(shape, order='F')
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | F-view Shape: {data.shape}")
            
            # ----------------------------------------------------
            # 模式 A：大矩阵带有独立 head 轴的数据 (如: 32x32x7)
            # ----------------------------------------------------
            if data.ndim == 3 and data.shape[2] == num_heads:
                for head_idx in range(num_heads):
                    head_data = data[:, :, head_idx]

                    if op_id == "op1":
                        op1_slice_datas = split_op1_max_slices(head_data, matrix_id, slices_per_group)
                        for slice_in_group_idx, slice_data in enumerate(op1_slice_datas):
                            relayout_data = relayout_slice_M8_N(slice_data)
                            global_slice_idx = head_idx * slices_per_group + slice_in_group_idx

                            save_before_relayout(before_install_dir, op_id, global_slice_idx, out_name, slice_data)

                            slice_dir = os.path.join(install_dir, op_id, f"slice{global_slice_idx:02d}")
                            os.makedirs(slice_dir, exist_ok=True)
                            out_path = os.path.join(slice_dir, out_name)
                            relayout_data.tofile(out_path)
                            convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1], file_dtype=relayout_data.dtype)
                        continue

                    relayout_data = relayout_slice_M8_N(head_data)
                    start_slice_idx = head_idx * slices_per_group
                    end_slice_idx = start_slice_idx + slices_per_group

                    for slice_idx in range(start_slice_idx, end_slice_idx):
                        save_before_relayout(before_install_dir, op_id, slice_idx, out_name, head_data)
                        slice_dir = os.path.join(install_dir, op_id, f"slice{slice_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=head_data.shape[0], cols=head_data.shape[1], file_dtype=relayout_data.dtype)
            
            # ----------------------------------------------------
            # 模式 B：【修复点】完美匹配形如 (32, 7) 的中间状态向量
            # ----------------------------------------------------
            elif data.ndim == 2 and data.shape[1] == num_heads:
                # 此时 data 形状是 (32, 7)，第一维是被切分的 32（物理N轴）
                for head_idx in range(num_heads):
                    # 提取对应 Head 的一列数据，转换为标准的 (32, 1) 二维物理物理切片
                    head_data = data[:, head_idx:head_idx+1]
                    
                    # 依据物理算子轴的 (32, 1) 进行 M8_N 重排
                    relayout_data = relayout_slice_M8_N(head_data)
                    
                    start_slice_idx = head_idx * slices_per_group
                    end_slice_idx = start_slice_idx + slices_per_group

                    # 将这组 Head 重排好的相同数据，精准发给该 Head 下所属的 4 个 slice 
                    for slice_idx in range(start_slice_idx, end_slice_idx):
                        save_before_relayout(before_install_dir, op_id, slice_idx, out_name, head_data)
                        slice_dir = os.path.join(install_dir, op_id, f"slice{slice_idx:02d}")
                        os.makedirs(slice_dir, exist_ok=True)
                        out_path = os.path.join(slice_dir, out_name)
                        relayout_data.tofile(out_path)
                        convert_to_128bit_txt(out_path, rows=head_data.shape[0], cols=head_data.shape[1], file_dtype=relayout_data.dtype)

            # ----------------------------------------------------
            # 模式 C：全局单头广播数据
            # ----------------------------------------------------
            else:
                if data.ndim > 2:
                    data_2d = data.reshape(data.shape[0], -1, order='F')
                elif data.ndim == 1:
                    data_2d = data.reshape(data.shape[0], 1)
                else:
                    data_2d = data.copy()

                relayout_data = relayout_slice_M8_N(data_2d)
                for i in range(num_slices):
                    save_before_relayout(before_install_dir, op_id, i, out_name, data_2d)
                    slice_dir = os.path.join(install_dir, op_id, f"slice{i:02d}")
                    os.makedirs(slice_dir, exist_ok=True)
                    out_path = os.path.join(slice_dir, out_name)
                    relayout_data.tofile(out_path)
                    convert_to_128bit_txt(out_path, rows=data_2d.shape[0], cols=data_2d.shape[1], file_dtype=relayout_data.dtype)

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All Softmax groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden", "sub_ops"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "softmax"))
    process_softmax_tensors(input_dir, output_dir)
