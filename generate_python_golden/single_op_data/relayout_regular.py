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

def float16_to_bin(f):
    return bin(struct.unpack('<H', struct.pack('<e', np.float16(f)))[0])[2:].zfill(16)

def int32_to_bin(i):
    return bin(struct.unpack('<I', struct.pack('<i', int(i)))[0])[2:].zfill(32)

def float_to_bin(f):
    """将单个 float32 转换为 32 位二进制字符串"""
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(32)

def _dtype_from_filename(filepath):
    """从文件名中的 _dtype_xxx 提取 numpy dtype"""
    basename = os.path.basename(filepath)
    m = re.search(r"_dtype_(f16|f32|float16|float32|i32|int32)", basename.lower())
    if m:
        tag = m.group(1)
        if tag in ("f16", "float16"):
            return np.float16
        elif tag in ("i32", "int32"):
            return np.int32
    return np.float32


def convert_to_decimal_txt(bin_path, rows=None, cols=None, file_dtype=None):
    """读取 bin 文件并输出十进制矩阵 txt（逗号分隔，按行换行）；对于 relayout 后的数据一维展开"""
    if file_dtype is None:
        file_dtype = _dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=file_dtype).astype(np.float32)
    
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
    if file_dtype is None:
        file_dtype = _dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=file_dtype)
    
    txt_path = bin_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as f:
        if file_dtype == np.float16:
            rem = len(data) % 8
            if rem:
                data = np.concatenate((data, np.zeros(8 - rem, dtype=file_dtype)))
            for i in range(0, len(data), 8):
                bins = [float16_to_bin(data[i + j]) for j in range(8)]
                f.write("".join(reversed(bins)) + "\n")
        else:
            rem = len(data) % 4
            if rem:
                data = np.concatenate((data, np.zeros(4 - rem, dtype=file_dtype)))
            for i in range(0, len(data), 4):
                if file_dtype == np.int32:
                    bins = [int32_to_bin(data[i + j]) for j in range(4)]
                else:
                    bins = [float_to_bin(data[i + j]) for j in range(4)]
                f.write("".join(reversed(bins)) + "\n")

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols, file_dtype=file_dtype)

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

def relayout_slice_N8_M(slice_data):
    """对一个 MxN 的 slice 先转置，再复用 M8_N 完成 N8_M 重排。"""
    return relayout_slice_M8_N(slice_data.T)

def is_kcur_vcur_add_tensor(filename):
    """Kcur/Vcur 的 add 张量需要按首维切 4 份，再复制 7 组"""
    return "_op-add_" in filename and ("Kcur" in filename or "Vcur" in filename)

def get_op_id(filename):
    """Regular 算子一般只是单一算子操作，统一放在 op0 下"""
    return "op0"

# 这些算子的 in0/in1 需要交换：in0→matrix_B, in1→matrix_A
SWAP_AB_PREFIXES = {
    "blk.0_Kcur-0-add_op-add",
    "blk.0_Qcur-0-add_op-add",
    "blk.0_Vcur-0-add_op-add",
    "blk.0_attn_norm-0_op-mul",
    "blk.0_ffn_norm-0_op-mul",
    "blk.0_ffn_inp-0_op-add",
    "blk.0_l",
}

def get_matrix_name(filename, prefix=""):
    """根据输入输出类型映射到硬件的端口名 A / B / D；特定前缀下交换 in0/in1"""
    swap = prefix in SWAP_AB_PREFIXES
    if "_in0" in filename: return "B" if swap else "A"
    if "_in1" in filename: return "A" if swap else "B"
    if "_out" in filename: return "D"
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

def process_regular_tensors(input_dir, output_dir):
    """
    处理通用生成的 sub_op .bin 文件。
    自动从文件名读取数据尺寸，默认按 N 维度把数据裁剪分给 28 个 slice 并采用 M8_N 排列。
    """
    print(f"🚀 Starting Regular tensor relayout in: {input_dir}")
    
    num_heads = MODEL_PARAMS["num_attention_heads"]
    slice_per_head = MODEL_PARAMS["slice_per_head"]
    head_replicated_slices = num_heads * slice_per_head
    num_slices = head_replicated_slices
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))

    # 移除 _subop- 限制，只要包含 _shape 即可被认为是张量数据
    # 并且跳过指定的几个算子
    skip_keywords = ["remote_sum", "rmsnorm", "rms_norm", "softmax", "soft_max", "rope", "mul_mat"]
    
    valid_files = []
    for f in bin_files:
        basename = os.path.basename(f)
        if "_shape" not in basename:
            continue
        if any(kw in basename.lower() for kw in skip_keywords):
            continue
        valid_files.append(f)
        
    if not valid_files:
        print("❌ No valid .bin files found in the directory.")
        return

    # 提取前缀：严格到 _inX 或 _out 或 _shape 之前
    def extract_prefix(f):
        base = os.path.basename(f)
        return re.split(r'_in\d+|_out|_shape', base)[0]

    prefixes = sorted(list(set([extract_prefix(f) for f in valid_files])))
    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")
        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        # 匹配该前缀名下的所有 in0 / in1 / out 文件
        group_files = [f for f in valid_files if extract_prefix(f) == target_prefix]
        if not group_files:
            continue

        for filepath in group_files:
            filename = os.path.basename(filepath)

            match = re.search(r"_shape([\dx]+)_dtype", filename)
            if not match:
                continue
            shape = tuple(map(int, match.group(1).split('x')))

            op_id = get_op_id(filename)
            matrix_id = get_matrix_name(filename, prefix=target_prefix)
            if matrix_id == "unknown_matrix":
                continue

            out_name = f"matrix_{matrix_id}_linearized_128bit.bin"
            
            # --- 从文件名 regex 中精确读取 dtype ---
            file_dtype = np.float32
            dtype_match = re.search(r"_dtype_(f16|f32|float16|float32|i32)", filename.lower())
            if dtype_match:
                tag = dtype_match.group(1)
                if tag in ("f16", "float16"):
                    file_dtype = np.float16
                elif tag in ("i32",):
                    file_dtype = np.int32
                # f32/float32 保持默认 float32

            # 采用 order='F' 还原原始算子的数据读入，并保持原始 dtype 进入后续处理
            print(f"[Original Input] Loading file: {os.path.basename(filepath)}, detected dtype: {file_dtype}")
            data = np.fromfile(filepath, dtype=file_dtype).reshape(shape, order='F')
            print(f"📦 Processing: {filename} -> {target_prefix}/{op_id}/{out_name} | Shape: {shape}")

            data_2d = data.squeeze()
            if data_2d.ndim == 0:
                data_2d = data_2d.reshape(1, 1)
            elif data_2d.ndim == 1:
                data_2d = data_2d.reshape(-1, 1)

            M, N = data_2d.shape

            # 通用切分与分发策略
            slices_to_distribute = []

            # 特判：Kcur/Vcur 的 add，首维是 head_dim，按首维切 slice_per_head 份，再复制 num_attention_heads 组
            if is_kcur_vcur_add_tensor(filename):
                if M % slice_per_head != 0:
                    print(f"  ⚠️ Cannot slice Kcur/Vcur add tensor shape {M}x{N} by slice_per_head={slice_per_head}.")
                    continue
                m_per_slice = M // slice_per_head
                base_slices = [data_2d[i * m_per_slice:(i + 1) * m_per_slice, :] for i in range(slice_per_head)]
                slices_to_distribute = base_slices * num_heads
                print(f"  ⚠️ Kcur/Vcur add tensor detected, sliced M to {slice_per_head} and copied to {head_replicated_slices}.")
            elif N % num_slices == 0:
                # 规则 1：按 N 维度 (通常对应特征维或者 batch) 均分 28 份
                n_per_slice = N // num_slices
                for i in range(num_slices):
                    slices_to_distribute.append(data_2d[:, i * n_per_slice : (i + 1) * n_per_slice])
            elif M % num_slices == 0:
                # 规则 2：无法切 N 维，尝试按 M 维度切分
                m_per_slice = M // num_slices
                for i in range(num_slices):
                    slices_to_distribute.append(data_2d[i * m_per_slice : (i + 1) * m_per_slice, :])
            elif N % slice_per_head == 0:
                # --- 新增规则：按 N 维分 slice_per_head 份，然后复制 num_attention_heads 组 ---
                n_per_slice = N // slice_per_head
                base_slices = [data_2d[:, i * n_per_slice : (i + 1) * n_per_slice] for i in range(slice_per_head)]
                slices_to_distribute = base_slices * num_heads
                print(f"  ⚠️ Cannot slice shape {M}x{N} to {num_slices} slices. Sliced N to {slice_per_head} and copied to {head_replicated_slices}.")
            elif M % slice_per_head == 0:
                # --- 新增规则：按 M 维分 slice_per_head 份，然后复制 num_attention_heads 组 ---
                m_per_slice = M // slice_per_head
                base_slices = [data_2d[i * m_per_slice : (i + 1) * m_per_slice, :] for i in range(slice_per_head)]
                slices_to_distribute = base_slices * num_heads
                print(f"  ⚠️ Cannot slice shape {M}x{N} to {num_slices} slices. Sliced M to {slice_per_head} and copied to {head_replicated_slices}.")
            else:
                # 规则 5：无法切分维度（例如偏置/向量），将自身作为全量数据广播到 28 个 slice
                print(f"  ⚠️ Cannot slice shape {M}x{N} to 28 or 4 slices. Broadcasting to all slices.")
                for i in range(num_slices):
                    slices_to_distribute.append(data_2d)

            # 遍历分配后的 slice 数据执行 relayout & 落盘
            for slice_idx, slice_data in enumerate(slices_to_distribute):
                save_before_relayout(before_install_dir, op_id, slice_idx, out_name, slice_data)

                slice_dir = os.path.join(install_dir, op_id, f"slice{slice_idx:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, out_name)
                
                # 0527
                slice_data = slice_data.T
                
                if (
                    target_prefix == "blk.0_Vcur-0-add_op-add"
                    and matrix_id in {"B", "D"}
                ):
                    relayout_data = relayout_slice_N8_M(slice_data)
                else:
                    relayout_data = relayout_slice_M8_N(slice_data)
                relayout_data.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=slice_data.shape[0], cols=slice_data.shape[1], file_dtype=slice_data.dtype)

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All Regular groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 将输入目录指向上一级的 python_golden，而不是 sub_ops
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "regular"))
    process_regular_tensors(input_dir, output_dir)
