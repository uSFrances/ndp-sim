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
            str_float1 = float_to_bin(data[i + 1])
            str_float2 = float_to_bin(data[i + 2])
            str_float3 = float_to_bin(data[i + 3])
            f.write(f"{str_float3}{str_float2}{str_float1}{str_float0}\n")

    convert_to_decimal_txt(bin_path, rows=rows, cols=cols)

# ==============================================================================
# 新增 remote_sum 专用 M8N 规则
# 说明：
#   - 这里把输入矩阵统一视为 (N, M)
#   - M 轴按 8 个一组优先展开
#   - 如果后续某个张量需要把第 0 维当成 M，只要传 transpose=True 即可
# ==============================================================================

def relayout_m8n(slice_data, transpose=False):
    view = slice_data.T if transpose else slice_data
    rows, cols = view.shape
    relayout_data = []

    for n_idx in range(rows):
        for m_outer in range(0, cols, 8):
            for m_idx in range(m_outer, min(m_outer + 8, cols)):
                relayout_data.append(view[n_idx, m_idx])

    return np.array(relayout_data, dtype=view.dtype)

def relayout_in0_M8N(slice_data):
    """in0: 直接按 (N, M) 解释，M 轴每 8 个元素展开"""
    return relayout_m8n(slice_data, transpose=False)

def relayout_out_M8N(slice_data):
    """out: 直接按 (N, M) 解释，和 in0 使用同一条 M8N 规则"""
    return relayout_m8n(slice_data, transpose=False)

def get_op_id(filename):
    return "op0"

def get_matrix_name(filename):
    if "_in0" in filename:
        return "A"
    if "_out" in filename:
        return "D"
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

def parse_tensor_file_info(filename):
    # 关键修复：先去掉 .bin，再按 stem 匹配
    stem = filename[:-4] if filename.lower().endswith(".bin") else filename
    m = re.match(r"^(?P<prefix>.+?)_(?P<kind>in\d+|out)_shape(?P<shape>[\dx]+)_dtype_(?P<dtype>\w+)$", stem)
    if not m:
        return None
    return {
        "prefix": m.group("prefix"),
        "kind": m.group("kind"),
        "shape": tuple(int(x) for x in m.group("shape").split("x")),
        "dtype": m.group("dtype").lower(),
    }

def process_remote_sum_tensors(input_dir, output_dir):
    print(f"🚀 Starting remote_sum tensor relayout in: {input_dir}")

    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    valid_files = []
    for fp in bin_files:
        info = parse_tensor_file_info(os.path.basename(fp))
        if not info:
            continue
        if "remote_sum" not in info["prefix"]:
            continue
        if info["kind"] not in ("in0", "out"):
            continue
        valid_files.append(fp)

    if not valid_files:
        print("❌ No valid remote_sum .bin files found in the directory.")
        print(f"  ℹ️ Scanned {len(bin_files)} .bin files under: {input_dir}")
        return

    print(f"✓ Found {len(valid_files)} valid remote_sum files")

    def extract_prefix(f):
        info = parse_tensor_file_info(os.path.basename(f))
        return info["prefix"] if info else "unknown"

    prefixes = sorted(set(extract_prefix(f) for f in valid_files))

    for target_prefix in prefixes:
        print(f"🎯 Processing instance group: '{target_prefix}'")
        group_output_dir = os.path.join(output_dir, target_prefix)
        install_dir = os.path.join(group_output_dir, "install")
        before_install_dir = os.path.join(group_output_dir, "install_beforerelayout")

        group_files = [f for f in valid_files if extract_prefix(f) == target_prefix]
        if not group_files:
            continue

        in0_fp = next((f for f in group_files if "_in0_" in os.path.basename(f)), None)
        out_fp = next((f for f in group_files if "_out_" in os.path.basename(f)), None)

        if in0_fp is None or out_fp is None:
            print(f"⚠️ Skip {target_prefix}: missing in0 or out")
            continue

        in0_info = parse_tensor_file_info(os.path.basename(in0_fp))
        out_info = parse_tensor_file_info(os.path.basename(out_fp))
        if in0_info is None or out_info is None:
            continue

        N, L, H, *_ = in0_info["shape"]
        out_N, out_L, out_H, *_ = out_info["shape"]

        if in0_N := N:
            pass

        if N % 4 != 0:
            print(f"⚠️ Skip {target_prefix}: in0 N={N} is not divisible by 4")
            continue

        slices_per_head = 4
        slice_n = N // slices_per_head

        print(f"  🧩 [REMOTE_SUM PARAM] in0=(N={N}, L={L}, H={H}), out=(N={out_N}, L={out_L}, H={out_H}), slice_n={slice_n}")

        os.makedirs(install_dir, exist_ok=True)
        os.makedirs(before_install_dir, exist_ok=True)

        # ----------------------------
        # in0: (N128, L32, H7, 1)
        # -> 每个 H 切 4 份，每份 (N//4, L)
        # ----------------------------
        in0_dtype = np.float16 if "f16" in in0_info["dtype"] else np.float32
        in0_raw = np.fromfile(in0_fp, dtype=in0_dtype).astype(np.float32)
        in0_4d = in0_raw.reshape(in0_info["shape"], order='F')

        for h_idx in range(H):
            for i in range(slices_per_head):
                global_idx = h_idx * slices_per_head + i
                k_start = i * slice_n
                slice_2d = in0_4d[k_start:k_start + slice_n, :, h_idx, 0]

                save_before_relayout(before_install_dir, "op0", global_idx, "matrix_A_linearized_128bit.bin", slice_2d)

                slice_dir = os.path.join(install_dir, "op0", f"slice{global_idx:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, "matrix_A_linearized_128bit.bin")

                rel = relayout_in0_M8N(slice_2d)
                rel.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=slice_2d.shape[0], cols=slice_2d.shape[1])

        # ----------------------------
        # out: (N32, L32, H7, 1)
        # -> 每个 H 的 (N, L) 基础块复制给 slice00-03
        # ----------------------------
        out_dtype = np.float16 if "f16" in out_info["dtype"] else np.float32
        out_raw = np.fromfile(out_fp, dtype=out_dtype).astype(np.float32)
        out_4d = out_raw.reshape(out_info["shape"], order='F')

        for h_idx in range(out_H):
            base_slice = out_4d[:, :, h_idx, 0]

            for i in range(slices_per_head):
                global_idx = h_idx * slices_per_head + i
                save_before_relayout(before_install_dir, "op0", global_idx, "matrix_D_linearized_128bit.bin", base_slice)

                slice_dir = os.path.join(install_dir, "op0", f"slice{global_idx:02d}")
                os.makedirs(slice_dir, exist_ok=True)
                out_path = os.path.join(slice_dir, "matrix_D_linearized_128bit.bin")

                rel = relayout_out_M8N(base_slice)
                rel.tofile(out_path)
                convert_to_128bit_txt(out_path, rows=base_slice.shape[0], cols=base_slice.shape[1])

        print(f"✅ Finished instance group: {target_prefix} -> {group_output_dir}")

    print(f"\n✅ All remote_sum groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "gemm_local"))
    process_remote_sum_tensors(input_dir, output_dir)
