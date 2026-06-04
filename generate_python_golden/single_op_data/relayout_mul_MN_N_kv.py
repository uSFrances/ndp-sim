import os
import re
import glob
import struct
import numpy as np
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
    if file_dtype is None:
        file_dtype = _dtype_from_filename(bin_path)
    data = np.fromfile(bin_path, dtype=file_dtype)

    if "beforerelayout" not in bin_path:
        txt_path = bin_path.replace('.bin', '_decimal_1d.txt')
        with open(txt_path, 'w') as f:
            f.write("\n".join(f"{float(v):.10g}" for v in data) + "\n")
        return

    if rows is None or cols is None or rows * cols != data.size:
        rows, cols = data.size, 1

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

def relayout_m8n(slice_2d):
    """
    M8N: 输入按 (N, M) 解释，M 轴每 8 个优先展开。
    循环次序：N 外层，M(8-pack) 内层。
    """
    n_dim, m_dim = slice_2d.shape
    out = []
    for n in range(n_dim):
        for m0 in range(0, m_dim, 8):
            for m in range(m0, min(m0 + 8, m_dim)):
                out.append(slice_2d[n, m])
    return np.asarray(out, dtype=slice_2d.dtype)

def relayout_in0_M8N(slice_2d):
    return relayout_m8n(slice_2d)

def relayout_in1_M8N(slice_2d):
    return relayout_m8n(slice_2d)

def relayout_out_M8N(slice_2d):
    return relayout_m8n(slice_2d)

def parse_tensor_file_info(filename):
    stem = filename[:-4] if filename.lower().endswith(".bin") else filename
    m = re.match(
        r"^(?P<prefix>.+?)_(?P<kind>in\d+|out)_shape(?P<shape>[\dx]+)_dtype_(?P<dtype>\w+)$",
        stem
    )
    if not m:
        return None
    return {
        "prefix": m.group("prefix"),
        "kind": m.group("kind"),
        "shape": tuple(int(x) for x in m.group("shape").split("x")),
        "dtype": m.group("dtype").lower(),
    }

def dtype_from_tag(tag):
    """从文件名中 _dtype_xxx 的 tag 精确映射 numpy dtype"""
    tag = tag.lower()
    if tag in ("f16", "float16"):
        return np.float16
    if tag in ("i32", "int32"):
        return np.int32
    # f32 / float32 / 其他默认 float32
    return np.float32

def load_tensor_fstyle(file_path, info):
    dt = dtype_from_tag(info["dtype"])
    print(f"[Original Input] Loading file: {os.path.basename(file_path)}, detected dtype: {dt}")
    raw = np.fromfile(file_path, dtype=dt)
    expect = int(np.prod(info["shape"]))
    if raw.size != expect:
        raise ValueError(f"size mismatch: {os.path.basename(file_path)} raw={raw.size}, expect={expect}")
    return raw.reshape(info["shape"], order='F')

def derive_kv_padded_n(model_params):
    """KV norm/mul tensors are split into slice_per_head chunks of kv_padding rows."""
    return model_params["kv_padding"] * model_params["slice_per_head"]

def save_before_relayout(before_install_dir, slice_idx, out_name, matrix_2d):
    matrix_2d = np.asarray(matrix_2d)
    if matrix_2d.ndim == 1:
        matrix_2d = matrix_2d.reshape(-1, 1)

    slice_dir = os.path.join(before_install_dir, "op0", f"slice{slice_idx:02d}")
    os.makedirs(slice_dir, exist_ok=True)
    out_path = os.path.join(slice_dir, out_name)

    matrix_2d.reshape(-1, order='C').tofile(out_path)
    convert_to_128bit_txt(out_path, rows=matrix_2d.shape[0], cols=matrix_2d.shape[1], file_dtype=matrix_2d.dtype)

def ensure_4d_shape(shape):
    if len(shape) >= 4:
        return shape[0], shape[1], shape[2], shape[3]
    if len(shape) == 3:
        return shape[0], shape[1], shape[2], 1
    if len(shape) == 2:
        return shape[0], shape[1], 1, 1
    if len(shape) == 1:
        return shape[0], 1, 1, 1
    raise ValueError("invalid shape")

def build_target_basenames(model_params):
    hidden = model_params["hidden_size"]
    seq = model_params["sequence_length"]
    return {
        f"blk.0_attn_norm-0_op-mul_in0_shape{hidden}x{seq}x1x1_dtype_f32.bin",
        f"blk.0_attn_norm-0_op-mul_in1_shape{hidden}x1x1x1_dtype_f32.bin",
        f"blk.0_attn_norm-0_op-mul_out_shape{hidden}x{seq}x1x1_dtype_f16.bin",
    }

def process_mul_mn_n_kv(input_dir, output_dir):
    print(f"🚀 Starting mul_MN_N_kv relayout in: {input_dir}")

    # 这些算子的 in0/in1 需要交换：in0→matrix_B, in1→matrix_A
    SWAP_AB_PREFIXES = {
        "blk.0_Kcur-0-add_op-add",
        "blk.0_Qcur-0-add_op-add",
        "blk.0_Vcur-0-add_op-add",
        "blk.0_attn_norm-0_op-mul",
        "blk.0_ffn_norm-0_op-mul",
    }

    target_basenames = build_target_basenames(MODEL_PARAMS)
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    valid_files = []
    seen = set()

    for fp in bin_files:
        fn = os.path.basename(fp)

        # 关键：严格只收三指定文件
        if fn not in target_basenames:
            continue

        info = parse_tensor_file_info(fn)
        if not info:
            continue
        if info["kind"] not in ("in0", "in1", "out"):
            continue

        valid_files.append(fp)
        seen.add(fn)

    missing = sorted(target_basenames - seen)
    if missing:
        print("⚠️ Missing target files:")
        for m in missing:
            print(f"  - {m}")

    if not valid_files:
        print("❌ No valid target .bin files found.")
        return

    def prefix_of(fp):
        info = parse_tensor_file_info(os.path.basename(fp))
        return info["prefix"] if info else "unknown"

    prefixes = sorted(set(prefix_of(f) for f in valid_files))
    H_REPL = MODEL_PARAMS["num_attention_heads"]
    SLICES_PER_HEAD = MODEL_PARAMS["slice_per_head"]
    KV_PADDING = MODEL_PARAMS["kv_padding"]
    N_PAD_CONFIG = derive_kv_padded_n(MODEL_PARAMS)

    for prefix in prefixes:
        group_files = [f for f in valid_files if prefix_of(f) == prefix]
        in0_fp = next((f for f in group_files if "_in0_" in os.path.basename(f)), None)
        in1_fp = next((f for f in group_files if "_in1_" in os.path.basename(f)), None)
        out_fp = next((f for f in group_files if "_out_" in os.path.basename(f)), None)

        if not (in0_fp and in1_fp and out_fp):
            print(f"⚠️ Skip {prefix}: missing one of in0/in1/out")
            continue

        print(f"🎯 Processing group: {prefix}")
        group_out = os.path.join(output_dir, prefix)
        install_dir = os.path.join(group_out, "install")
        before_dir = os.path.join(group_out, "install_beforerelayout")
        os.makedirs(install_dir, exist_ok=True)
        os.makedirs(before_dir, exist_ok=True)

        in0_info = parse_tensor_file_info(os.path.basename(in0_fp))
        in1_info = parse_tensor_file_info(os.path.basename(in1_fp))
        out_info = parse_tensor_file_info(os.path.basename(out_fp))

        N0, L0, H0, _ = ensure_4d_shape(in0_info["shape"])
        N1, L1, H1, _ = ensure_4d_shape(in1_info["shape"])
        NO, LO, HO, _ = ensure_4d_shape(out_info["shape"])

        N_pad = N_PAD_CONFIG
        max_raw_n = max(N0, N1, NO)
        if N_pad < max_raw_n:
            print(f"⚠️ Skip {prefix}: kv_padding*slice_per_head={N_pad} is smaller than raw N={max_raw_n}")
            continue
        if N_pad % SLICES_PER_HEAD != 0:
            print(f"⚠️ Skip {prefix}: kv_padding*slice_per_head={N_pad} not divisible by slice_per_head={SLICES_PER_HEAD}")
            continue
        slice_n = N_pad // SLICES_PER_HEAD

        print(
            f"  🧩 N={N0} -> N_pad={N_pad}, kv_padding={KV_PADDING}, slice_n={slice_n}, "
            f"L(in0/out)=({L0}/{LO}), H_repl={H_REPL}"
        )

        in0_4d = load_tensor_fstyle(in0_fp, in0_info)
        in1_4d = load_tensor_fstyle(in1_fp, in1_info)
        out_4d = load_tensor_fstyle(out_fp, out_info)

        # 交换前缀：in0→B, in1→A
        do_swap = prefix in SWAP_AB_PREFIXES
        name_in0 = "matrix_B" if do_swap else "matrix_A"
        name_in1 = "matrix_A" if do_swap else "matrix_B"

        # in0
        for hrep in range(H_REPL):
            src_h = hrep % max(H0, 1)
            pad2d = np.zeros((N_pad, L0), dtype=in0_4d.dtype)
            pad2d[:N0, :] = in0_4d[:N0, :L0, src_h, 0]
            for i in range(SLICES_PER_HEAD):
                gidx = hrep * SLICES_PER_HEAD + i
                s2d = pad2d[i * slice_n:(i + 1) * slice_n, :]
                fname = f"{name_in0}_linearized_128bit.bin"

                save_before_relayout(before_dir, gidx, fname, s2d)
                rel = relayout_in0_M8N(s2d)
                sdir = os.path.join(install_dir, "op0", f"slice{gidx:02d}")
                os.makedirs(sdir, exist_ok=True)
                opath = os.path.join(sdir, fname)
                rel.tofile(opath)
                convert_to_128bit_txt(opath, rows=s2d.shape[0], cols=s2d.shape[1], file_dtype=s2d.dtype)

        # in1
        for hrep in range(H_REPL):
            src_h = hrep % max(H1, 1)
            pad2d = np.zeros((N_pad, L1), dtype=in1_4d.dtype)
            pad2d[:N1, :] = in1_4d[:N1, :L1, src_h, 0]
            for i in range(SLICES_PER_HEAD):
                gidx = hrep * SLICES_PER_HEAD + i
                s2d = pad2d[i * slice_n:(i + 1) * slice_n, :]
                fname = f"{name_in1}_linearized_128bit.bin"

                save_before_relayout(before_dir, gidx, fname, s2d)
                rel = relayout_in1_M8N(s2d)
                sdir = os.path.join(install_dir, "op0", f"slice{gidx:02d}")
                os.makedirs(sdir, exist_ok=True)
                opath = os.path.join(sdir, fname)
                rel.tofile(opath)
                convert_to_128bit_txt(opath, rows=s2d.shape[0], cols=s2d.shape[1], file_dtype=s2d.dtype)

        # out: zero padding
        for hrep in range(H_REPL):
            src_h = hrep % max(HO, 1)
            pad2d = np.zeros((N_pad, LO), dtype=out_4d.dtype)  # padding = 0
            pad2d[:NO, :] = out_4d[:NO, :LO, src_h, 0]
            for i in range(SLICES_PER_HEAD):
                gidx = hrep * SLICES_PER_HEAD + i
                s2d = pad2d[i * slice_n:(i + 1) * slice_n, :]

                save_before_relayout(before_dir, gidx, "matrix_D_linearized_128bit.bin", s2d)
                rel = relayout_out_M8N(s2d)

                sdir = os.path.join(install_dir, "op0", f"slice{gidx:02d}")
                os.makedirs(sdir, exist_ok=True)
                opath = os.path.join(sdir, "matrix_D_linearized_128bit.bin")
                rel.tofile(opath)
                convert_to_128bit_txt(opath, rows=s2d.shape[0], cols=s2d.shape[1], file_dtype=s2d.dtype)

        print(f"✅ Finished: {prefix} -> {group_out}")

    print(f"\n✅ All mul_MN_N_kv groups processed under: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "python_golden"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "model_execplan", "data", "mul_MN_N_kv"))
    process_mul_mn_n_kv(input_dir, output_dir)
