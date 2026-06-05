import numpy as np
import os
import re
import time
from tqdm import tqdm


def load_bin_as_array(path, dtype, shape):
    total_elements = shape[0] * shape[1] * shape[2] * shape[3]
    data = np.fromfile(path, dtype=dtype, count=total_elements)
    return data.reshape(shape, order='F')

# Placeholder operator implementations
def rms_norm(x):
    nsamples    = x.shape[3]  # ne03
    nchannels   = x.shape[2]  # ne02
    nrows       = x.shape[1]  # ne01
    ncols       = x.shape[0]  # ne00
    dst = np.zeros_like(x)
    # print(ncols, nrows, nchannels, nsamples)
    for sample in range(nsamples):
        for channel in range(nchannels):
            tmp = np.array(0, dtype=np.float32)
            for row in range(nrows):
                x_slice = x[:, row, channel, sample]
                tmp=0
                for col in range(ncols):
                    tmp += np.float32(x_slice[col] * x_slice[col])
                mean = tmp / ncols
                scale = 1.0 / np.sqrt(mean + 0.000001)
                dst[:, row, channel, sample] =  x_slice * scale
    return dst
def mul(x, y): return x * y
def matmul_2d(src0, src1, m, n ,k):
    dst_py = np.zeros((m,n), dtype=np.float32) 
  
    for i in range(m):          
        for j in range(n):      
            sum_val = np.float32(0.0)       
            for l in range(k):  
                a_val = src0[i, l].astype(np.float16)
                b_val = src1[l, j].astype(np.float16)
                tmp = a_val.astype(np.float16) * b_val.astype(np.float16)
                sum_val += tmp
            dst_py[i, j] += sum_val.astype(np.float32)  

    return dst_py

def mul_mat(src0, src1):
    # print("src0.shape:", src0.shape)
    # print("src1.shape:", src1.shape)

    m = src0.shape[1]
    n = src1.shape[1]
    if src0.shape[0] == src1.shape[0]:
        k = src0.shape[0]
    else:
        print("tensor shape error: no same dimension")
        return None

    ne03 = src0.shape[2]
    ne04 = src0.shape[3]
    ne13 = src1.shape[2]
    ne14 = src1.shape[3]

    if (ne03 == ne13) and (ne04 == ne14) and (ne03 == 1) and (ne04 == 1):#二维张量处理，包含对向量处理

        src0_re = src0.reshape((k,m), order='F')
        src1_re = src1.reshape((k,n), order='F')

        src0_re_T = src0_re.T

        dst_py = matmul_2d(src0_re_T, src1_re, m, n, k)

        return dst_py.reshape((m,n,1,1), order='F')

    elif (ne04 == ne14) and (ne04 == 1):#批次为1的三维张量处理

        scale = int(ne13 / ne03)#ne3的广播次数
        if scale < 1 :
            print("tensor shape error: ne03 is larger than ne13")
            return None

        #dst_py = np.zeros((1,ne13,n,m), dtype=np.float32) 
        dst_py = np.zeros((m,n,ne13,1), dtype=np.float32) 
        for j in range(ne03):
            for i in range(scale):
                '''
                src0_T = np.transpose(src0,axes=(3, 2, 1, 0))
                ex_src0 = src0_T[0,j,:,:]
                print(ex_src0.shape)
                src1_T = np.transpose(src1,axes=(3, 2, 1, 0))
                ex_src1 = src1_T[0,j+i*ne03,:,:]
                print(ex_src1.shape)
                ex_src1_T = ex_src1.T
                dst_py_si = matmul_2d(ex_src0, ex_src1_T, m, n, k)
                dst_py_si_T = dst_py_si.T
                dst_py[0,j+i*ne03,:,:] = dst_py_si_T
                dst_py_T = np.transpose(dst_py,axes=(3, 2, 1, 0))
                print(dst_py_T.shape)
                '''

                ex_src0 = src0[:,:,j,0]
                ex_src1 = src1[:,:,j*scale+i,0]
                ex_src0_T = ex_src0.T
                dst = matmul_2d(ex_src0_T, ex_src1, m, n, k)
                dst_py[:,:,j*scale+i,0] = dst

        return dst_py

    else:
        print("tensor shape error: ne4 is more than 1")
        return None
def add(x, y): return x + y
def rope(x):
    ne2 = x.shape[2]
    ne1 = x.shape[1]
    ne0 = x.shape[0]   
    dst = np.zeros_like(x)
    sincos_shape = (ne0//2, ne2,1,1)
    # sin  = load_bin_as_array(f"/cluster/home/liudy/workspace/CGRA_SIM/cgra_python/op_lib/LLM_golden/math_op/rope_fp32/rope_neox_sin_float32_ne2_512.bin", np.float32, sincos_shape)
    # cos  = load_bin_as_array(f"/cluster/home/liudy/workspace/CGRA_SIM/cgra_python/op_lib/LLM_golden/math_op/rope_fp32/rope_neox_cos_float32_ne2_512.bin", np.float32, sincos_shape)
    sin  = load_bin_as_array(f"/Users/jielu/Desktop/CGRA mapping/configuration/LLM_python_golden/ndp-sim/generate_python_golden/rope_fp32/rope_neox_sin_float32_ne2_512.bin", np.float32, sincos_shape)
    cos  = load_bin_as_array(f"/Users/jielu/Desktop/CGRA mapping/configuration/LLM_python_golden/ndp-sim/generate_python_golden/rope_fp32/rope_neox_cos_float32_ne2_512.bin", np.float32, sincos_shape)
    for k in range(0, ne2, 1):
        for j in range(0, ne1, 1):
            for i in range(0, ne0 // 2, 1):
                cos_theta = cos[i,k]
                sin_theta = sin[i,k]
            
                x0 = x[i,  j,  k]
                x1 = x[i+ne0//2,  j,  k]
                dst[i,  j,  k]       = x0 * cos_theta - x1 * sin_theta
                dst[i+ne0//2,  j,  k]    = x0 * sin_theta + x1 * cos_theta
    return dst
def soft_max(x, mask=None):
    """
    x: 4D tensor, column-major (Fortran-order)
    mask: optional 4D tensor, also column-major
    """
    src0_shape = x.shape
    ncols = src0_shape[0]
    nrows_x = np.prod(src0_shape[1:])
    nrows_y = src0_shape[1] if len(src0_shape) >= 2 else 1

    base_dir = "/Users/jielu/Desktop/CGRA mapping/configuration/LLM_python_golden/ndp-sim/generate_python_golden"
    scale_path = os.path.join(base_dir, "softmax_scale.bin")
    with open(scale_path, "rb") as f:
        scale = np.frombuffer(f.read(), dtype=np.float32)[0]

    src0_flat = x.flatten(order='F')
    dst_python = np.zeros_like(src0_flat)
    mask_flat = mask.flatten(order='F') if mask is not None else None

    def soft_max_f32_simple(x_flat, mask_flat, dst_flat, ncols, nrows_x, nrows_y, scale):
        for rowx in range(nrows_x):
            rowy = rowx % nrows_y
            start = rowx * ncols
            mask_start = rowy * ncols

            row = x_flat[start:start+ncols] * scale
            if mask_flat is not None:
                row += mask_flat[mask_start:mask_start+ncols]

            row = np.exp(row - np.max(row))
            dst_flat[start:start+ncols] = row / np.sum(row)
        return dst_flat

    soft_max_f32_simple(src0_flat, mask_flat, dst_python, ncols, nrows_x, nrows_y, scale)

    dst_python = dst_python.reshape(src0_shape, order='F')
    return dst_python

def unary(x): return x / (1 + np.exp(-x))
def get_rows(src0: np.ndarray, src1: np.ndarray) -> np.ndarray:
    """
    src0: 源 tensor，4D，列优先
    src1: 索引 tensor，4D，列优先
    返回：根据索引抽取后的 dst tensor，列优先
    """
    assert src0.ndim == 4
    assert src1.ndim == 4
    assert src0.dtype == np.float32
    assert src1.dtype == np.int32

    D0, D1_src, D2, D3 = src0.shape
    _, D1_dst, D2_dst, D3_dst = src1.shape
    assert (D1_src >= np.max(src1) + 1), "src1索引超出src0范围"

    dst = np.zeros((D0, D1_dst, D2_dst, D3_dst), dtype=np.float32, order='F')

    for i1 in range(D1_dst):
        for i2 in range(D2_dst):
            for i3 in range(D3_dst):
                idx = src1[i1, i2, i3]
                # 加 .reshape(-1)，确保是 (1536,)
                dst[:, i1, i2, i3] = src0[:, idx, i2, i3].reshape(-1)

    return dst
def convert(x, dtype='fp16'):
    if dtype == 'fp16':
        return x.astype(np.float16)
    elif dtype == 'fp32':
        return x.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype for convert: {dtype}")

def get_nearest_32_multiple(token_num):
    """返回大于等于 token_num 的最近的 32 的倍数"""
    return (token_num + 31) // 32 * 32

# def init_kv_cache(store, max_seq_len):
#     """初始化 KV Cache，分配足够的内存空间"""
#     for layer_id in range(28):  # 假设有 28 层
#         store.set(f"cache_k_l{layer_id}", np.zeros((128, 2, max_seq_len, 1), dtype=np.float16, order='F'))  # 列优先布局
#         store.set(f"cache_v_l{layer_id}", np.zeros((max_seq_len, 256, 1, 1), dtype=np.float16, order='F'))  # 列优先布局

def get_kv_for_attention(store, layer_id, token_num):
    key_name = f"cache_k_l{layer_id}"
    value_name = f"cache_v_l{layer_id}"

    k_cache = store.get(key_name)  # (128, seq_len, 2, 1)
    v_cache = store.get(value_name)  # (seq_len, 256, 1, 1)

    num_tokens_to_take = get_nearest_32_multiple(token_num)
    num_tokens_in_cache = k_cache.shape[1]

    if num_tokens_in_cache < num_tokens_to_take:
        k_selected = np.zeros((128, num_tokens_to_take, 2, 1), dtype=np.float16, order='F')
        k_selected[:, :num_tokens_in_cache, :, :] = k_cache[:, :num_tokens_in_cache, :, :]

        v_selected = np.zeros((num_tokens_to_take, 128, 2, 1), dtype=np.float16, order='F')
        v_selected[:num_tokens_in_cache, :, :, :] = v_cache[:num_tokens_in_cache, :, :, :]
    else:
        k_selected = k_cache[:, -num_tokens_to_take:, :, :]
        v_selected = v_cache[-num_tokens_to_take:, :, :, :]

    return k_selected, v_selected


# 保留 update_kv_cache 并在首次 set 时初始化
# update_kv_cache() 中添加 V 的 reshape + transpose（写入前）
def update_kv_cache(store, layer_id, new_K, new_V):
    key_name = f"cache_k_l{layer_id}"
    value_name = f"cache_v_l{layer_id}"

    new_K = convert(new_K, dtype='fp16')
    new_K = new_K.transpose(0, 2, 1, 3).copy(order='F')   # (128, 2, seq_len, 1) → (128, seq_len, 2, 1)
    
    # v：(128, 2, token_num, 1)->(token_num, 128, 2, 1)
    new_V = convert(new_V, dtype='fp16')
    new_V = new_V.transpose(2, 0, 1, 3).copy(order='F')

    if key_name not in store.store:
        store.set(key_name, new_K)
        store.set(value_name, new_V)
    else:
        old_K = store.get(key_name) # (128, total_T, 2, 1)
        old_V = store.get(value_name) # (total_T, 128, 2, 1)
        concat_K = np.concatenate([old_K, new_K], axis=1)
        concat_V = np.concatenate([old_V, new_V], axis=0)
        store.set(key_name, concat_K)
        store.set(value_name, concat_V)


class TensorStore:
    def __init__(self):
        self.store = {}
        # self.debug_save_dir = "/cluster/home/liudy/workspace/CGRA_SIM/cgra_python/op_lib/LLM_golden/prefill_token8_bs1/python_golden"
        self.debug_save_dir = "/Users/jielu/Desktop/CGRA mapping/configuration/LLM_python_golden/ndp-sim/generate_python_golden/python_golden"
        os.makedirs(self.debug_save_dir, exist_ok=True)
    def get(self, name):
        if name not in self.store:
            raise KeyError(f"Tensor '{name}' not found")
        return self.store[name]
    def set(self, name, value):
        self.store[name] = value
    def load_from_bin_folder(self, folder_path):
        for fname in os.listdir(folder_path):
            if not fname.endswith(".bin"):
                continue

            full_path = os.path.join(folder_path, fname)
            match = re.match(r"(?P<name>.+?)_shape(?P<shape>[\dx]+)_dtype_(?P<dtype>f32|f16|i32).bin", fname)
            if not match:
                print(f"[Skip] Unrecognized file name format: {fname}")
                continue

            name = match.group("name")
            shape_str = match.group("shape")
            dtype_str = match.group("dtype")

            shape = tuple(map(int, shape_str.split("x")))

            if dtype_str == "f32":
                dtype = np.float32
                itemsize = 4
            elif dtype_str == "f16":
                dtype = np.float16
                itemsize = 2
            elif dtype_str == "i32":
                dtype = np.int32
                itemsize = 4
            else:
                print(f"[Skip] Unsupported dtype: {dtype_str}")
                continue

            num_elements = np.prod(shape)
            with open(full_path, "rb") as f:
                data = np.frombuffer(f.read(num_elements * itemsize), dtype=dtype)

            tensor = data.reshape(shape, order="F")  # 保持与 llama.cpp 一致的列优先布局
            self.set(name, tensor)
            # print(f"[Loaded] {name} with shape {shape}, dtype {dtype_str}")
    
    def set_debug(self, name, value):
        """设置张量并将其保存为.bin文件（列优先）"""
        self.set(name, value)  # 正常 set 进 store
        shape_str = "x".join(str(s) for s in value.shape)
        dtype_str = {
            np.float32: "f32",
            np.float16: "f16",
            np.int32: "i32"
        }.get(value.dtype.type, "unknown")

        if dtype_str == "unknown":
            raise ValueError(f"Unsupported dtype for debug export: {value.dtype}")

        filename = f"{name}_shape{shape_str}_dtype_{dtype_str}.bin"
        full_path = os.path.join(self.debug_save_dir, filename)
        value.flatten(order='F').tofile(full_path)
        print(f"[Debug Saved] {name} → {filename}")

    def summary(self):
        print("\n=== TensorStore Summary ===")
        for name, tensor in self.store.items():
            print(f"- {name:<40} | dtype: {tensor.dtype} | shape: {tensor.shape}")
        print("===========================\n")


def run_transformer_layer(store: TensorStore, layer_id: int, token_num: int):
    def timed_op(name: str, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"  ⏱️ {name:<30} took {duration:.4f} seconds")
        store.set_debug(name, result)
    lid = f"blk.{layer_id}"
    print(f"🟡 Running transformer layer {layer_id}...")
    start_time = time.time()

    input_key = "inp_embd" if layer_id == 0 else f"l_out-{layer_id - 1}"
    timed_op(f"norm-{layer_id}", rms_norm, store.get(input_key))
    timed_op(f"attn_norm-{layer_id}", mul, store.get(f"norm-{layer_id}"), store.get(f"{lid}.attn_norm.weight"))
    timed_op(f"node_{layer_id}_q", mul_mat, store.get(f"{lid}.attn_q.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Qcur-{layer_id}", add, store.get(f"node_{layer_id}_q"), store.get(f"{lid}.attn_q.bias"))
    tmp = store.get(f"Qcur-{layer_id}").reshape((128, 12, token_num, 1),  order='F')
    timed_op(f"Qcur-{layer_id}", rope, tmp)
    timed_op(f"node_{layer_id}_k", mul_mat, store.get(f"{lid}.attn_k.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Kcur-{layer_id}", add, store.get(f"node_{layer_id}_k"), store.get(f"{lid}.attn_k.bias"))
    tmp = store.get(f"Kcur-{layer_id}").reshape((128, 2, token_num, 1), order='F')
    timed_op(f"Kcur-{layer_id}", rope, tmp)
    timed_op(f"node_{layer_id}_v", mul_mat, store.get(f"{lid}.attn_v.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Vcur-{layer_id}", add, store.get(f"node_{layer_id}_v"), store.get(f"{lid}.attn_v.bias"))
    tmp = store.get(f"Vcur-{layer_id}").reshape((128, 2, token_num, 1), order='F')
    store.set(f"Vcur-{layer_id}", tmp)

    update_kv_cache(store, layer_id, store.get(f"Kcur-{layer_id}"), store.get(f"Vcur-{layer_id}"))
    k_selected, v_selected = get_kv_for_attention(store, layer_id, token_num)
    store.set(f"cache_k_l{layer_id}", k_selected)
    store.set(f"cache_v_l{layer_id}", v_selected)

    tmp = store.get(f"Qcur-{layer_id}").transpose(0, 2, 1, 3)
    store.set(f"Qcur-{layer_id}-permute", tmp)
    timed_op(f"node_{layer_id}_attn_scores", mul_mat, k_selected, tmp)
    timed_op(f"node_{layer_id}_attn_probs", soft_max, store.get(f"node_{layer_id}_attn_scores"), mask=store.get("leaf_12"))
    timed_op(f"node_{layer_id}_attn_out", mul_mat, v_selected, store.get(f"node_{layer_id}_attn_probs"))

    tmp = store.get(f"node_{layer_id}_attn_out").transpose(0, 2, 1, 3).reshape((1536, token_num, 1, 1), order='F')
    store.set(f"kqv_out-{layer_id}", tmp)
    timed_op(f"node_{layer_id}_attn_final", mul_mat, store.get(f"{lid}.attn_output.weight"), store.get(f"kqv_out-{layer_id}"))
    timed_op(f"ffn_inp-{layer_id}", add, store.get(f"node_{layer_id}_attn_final"), store.get(input_key))
    timed_op(f"norm_ffn-{layer_id}", rms_norm, store.get(f"ffn_inp-{layer_id}"))
    timed_op(f"ffn_norm-{layer_id}", mul, store.get(f"norm_ffn-{layer_id}"), store.get(f"{lid}.ffn_norm.weight"))
    timed_op(f"ffn_gate-{layer_id}", mul_mat, store.get(f"{lid}.ffn_gate.weight"), store.get(f"ffn_norm-{layer_id}"))
    timed_op(f"ffn_silu-{layer_id}", unary, store.get(f"ffn_gate-{layer_id}"))
    timed_op(f"ffn_up-{layer_id}", mul_mat, store.get(f"{lid}.ffn_up.weight"), store.get(f"ffn_norm-{layer_id}"))
    timed_op(f"ffn_gate_par-{layer_id}", mul, store.get(f"ffn_silu-{layer_id}"), store.get(f"ffn_up-{layer_id}"))
    timed_op(f"ffn_out-{layer_id}", mul_mat, store.get(f"{lid}.ffn_down.weight"), store.get(f"ffn_gate_par-{layer_id}"))
    timed_op(f"l_out-{layer_id}", add, store.get(f"ffn_out-{layer_id}"), store.get(f"ffn_inp-{layer_id}"))
    store.set_debug(f"l_out-{layer_id}", store.get(f"l_out-{layer_id}"))

    total_duration = time.time() - start_time
    print(f"✅ Finished layer {layer_id} in {total_duration:.3f} seconds | Output: l_out-{layer_id} shape = {store.get(f'l_out-{layer_id}').shape}")

def run_final_layer(store: TensorStore, token_num: int):
    layer_id = 27
    lid = f"blk.{layer_id}"
    print(f"🟡 Running transformer layer {layer_id}...")
    start_time = time.time()

    # Q/K/V 分支（与前面相同）
    store.set_debug(f"norm-{layer_id}", rms_norm(store.get(f"l_out-{layer_id - 1}")))
    store.set_debug(f"attn_norm-{layer_id}", mul(store.get(f"norm-{layer_id}"), store.get(f"{lid}.attn_norm.weight")))
    store.set_debug(f"node_{layer_id}_q", mul_mat(store.get(f"{lid}.attn_q.weight"), store.get(f"attn_norm-{layer_id}")))
    store.set_debug(f"Qcur-{layer_id}", add(store.get(f"node_{layer_id}_q"), store.get(f"{lid}.attn_q.bias")))
    store.set_debug(f"Qcur-{layer_id}", rope(store.get(f"Qcur-{layer_id}").reshape((128, 12, token_num, 1),  order='F')))
    store.set_debug(f"node_{layer_id}_k", mul_mat(store.get(f"{lid}.attn_k.weight"), store.get(f"attn_norm-{layer_id}")))
    store.set_debug(f"Kcur-{layer_id}", add(store.get(f"node_{layer_id}_k"), store.get(f"{lid}.attn_k.bias")))
    store.set_debug(f"Kcur-{layer_id}", rope(store.get(f"Kcur-{layer_id}").reshape((128, 2, token_num, 1), order='F')))
    store.set_debug(f"node_{layer_id}_v", mul_mat(store.get(f"{lid}.attn_v.weight"), store.get(f"attn_norm-{layer_id}")))
    store.set_debug(f"Vcur-{layer_id}", add(store.get(f"node_{layer_id}_v"), store.get(f"{lid}.attn_v.bias")))
    store.set_debug(f"Vcur-{layer_id}", store.get(f"Vcur-{layer_id}").reshape((128, 2, token_num, 1), order='F'))
    
    # 更新 KV Cache
    update_kv_cache(store, layer_id, store.get(f"Kcur-{layer_id}"), store.get(f"Vcur-{layer_id}"))
    
    # 获取 KV Cache 中最近的 32 倍数的 token
    k_selected, v_selected = get_kv_for_attention(store, layer_id, token_num)
    store.set_debug(f"cache_k_l{layer_id}", k_selected)
    store.set_debug(f"cache_v_l{layer_id}", v_selected)
    # 计算注意力
    store.set_debug(f"Qcur-{layer_id}-permute", store.get(f"Qcur-{layer_id}").transpose(0, 2, 1, 3))
    store.set_debug(f"node_{layer_id}_attn_scores", mul_mat(k_selected, store.get(f"Qcur-{layer_id}-permute")))
    # store.set(f"node_{layer_id}_attn_scores", mul_mat(store.get(f"cache_k_l{layer_id}_view"), store.get(f"Qcur-{layer_id}-permute")))
    store.set_debug(f"node_{layer_id}_attn_probs", soft_max(store.get(f"node_{layer_id}_attn_scores"), mask=store.get("leaf_12")))
    # # store.set_debug(f"node_{layer_id}_attn_probs", soft_max(store.get("node_21"), mask=store.get("leaf_12")))
    
    store.set_debug(f"node_{layer_id}_attn_out", mul_mat(v_selected, store.get(f"node_{layer_id}_attn_probs")))
    # store.set(f"node_{layer_id}_attn_out", mul_mat(store.get(f"cache_v_l{layer_id}_view"), store.get(f"node_{layer_id}_attn_probs")))
    # store.set(f"node_{layer_id}_attn_out", mul_mat(store.get("cache_v_l0__view"), store.get("node_22")))
    store.set_debug(f"kqv_out-{layer_id}", store.get(f"node_{layer_id}_attn_out").transpose(0, 2, 1, 3).reshape((1536, token_num, 1, 1), order='F'))
    store.set_debug(f"node_{layer_id}_attn_final", mul_mat(store.get(f"{lid}.attn_output.weight"), store.get(f"kqv_out-{layer_id}")))

    # get_rows 取出第 token_index 个（默认 0）进行 residual add
    store.set_debug(f"node_attn-{layer_id}", get_rows(store.get(f"node_{layer_id}_attn_final"), store.get("leaf_395")))
    store.set_debug(f"l_out_prev-{layer_id}", get_rows(store.get(f"l_out-{layer_id - 1}"), store.get("leaf_395")))
    store.set_debug(f"ffn_inp-{layer_id}", add(store.get(f"node_attn-{layer_id}"), store.get(f"l_out_prev-{layer_id}")))
    store.set_debug(f"norm_ffn-{layer_id}", rms_norm(store.get(f"ffn_inp-{layer_id}")))
    store.set_debug(f"ffn_norm-{layer_id}", mul(store.get(f"norm_ffn-{layer_id}"), store.get(f"{lid}.ffn_norm.weight")))
    store.set_debug(f"ffn_gate-{layer_id}", mul_mat(store.get(f"{lid}.ffn_gate.weight"), store.get(f"ffn_norm-{layer_id}")))
    store.set_debug(f"ffn_silu-{layer_id}", unary(store.get(f"ffn_gate-{layer_id}")))
    store.set_debug(f"ffn_up-{layer_id}", mul_mat(store.get(f"{lid}.ffn_up.weight"), store.get(f"ffn_norm-{layer_id}")))
    store.set_debug(f"ffn_gate_par-{layer_id}", mul(store.get(f"ffn_silu-{layer_id}"), store.get(f"ffn_up-{layer_id}")))
    store.set_debug(f"ffn_out-{layer_id}", mul_mat(store.get(f"{lid}.ffn_down.weight"), store.get(f"ffn_gate_par-{layer_id}")))
    store.set_debug(f"l_out-{layer_id}", add(store.get(f"ffn_out-{layer_id}"), store.get(f"ffn_inp-{layer_id}")))

    # Final norm and output projection
    store.set_debug("norm", rms_norm(store.get(f"l_out-{layer_id}")))
    store.set_debug("result_norm", mul(store.get("norm"), store.get("output_norm.weight")))
    store.set_debug("result_output", mul_mat(store.get("output.weight"), store.get("result_norm")))
    duration = time.time() - start_time
    print(f"✅ Finished layer {layer_id} in {duration:.3f} seconds | Output: l_out-{layer_id} shape = {store.get(f'l_out-{layer_id}').shape}")

def run_transformer(store: TensorStore, token_num: int):
    """运行完整的 Transformer 模型（带 tqdm 进度条）"""
    num_layers = 28  # 最后第 28 层是 final_layer
    print(f"\n🚀 Starting transformer forward pass for {num_layers + 1} layers...")

    for layer_id in tqdm(range(num_layers), desc="🧠 Transformer Layers", unit="layer"):
        start_time = time.time()
        run_transformer_layer(store, layer_id, token_num)
        duration = time.time() - start_time
        print(f"  ⏱️ Layer {layer_id} finished in {duration:.3f} s | l_out-{layer_id} shape = {store.get(f'l_out-{layer_id}').shape}")

    print("🔚 Running final transformer layer...")
    run_final_layer(store, token_num)

    return store.get("result_output")

# 示例用法
if __name__ == "__main__":
    # 创建 TensorStore 并初始化输入
    store = TensorStore()
    # 加载模型权重（非输入）
    # weight_folder = "/cluster/home/liudy/SACA/LP6/model_cache/model--deepseep-1.5b/extracted_weights/DeepSeek-R1-Distill-Qwen-1.5B-f16"
    weight_folder = "/Users/jielu/Desktop/CGRA mapping/configuration/LLM_python_golden/ndp-sim/generate_python_golden/model_weights_full"
    store.load_from_bin_folder(weight_folder)

    # 加载输入张量（inp_embd, leaf_12, leaf_395）
    # input_folder = "/cluster/home/liudy/workspace/CGRA_SIM/cgra_python/op_lib/LLM_golden/prefill_token8_bs1/llamacpp_cpu/inputs"
    input_folder = "/Users/jielu/Desktop/CGRA mapping/configuration/LLM_python_golden/ndp-sim/generate_python_golden/inputs_full"
    store.load_from_bin_folder(input_folder)

    # 加载llamacpp_cpu的kvcache
    # input_folder = "/nvme1/liudongyan/workspace/CGRA_SIM/cgra_python/op_lib/LLM_golden/prefill_token8_bs1/llamacpp_cpu/kvcache"
    # store.load_from_bin_folder(input_folder)

    # 打印所有张量信息（名称、shape、dtype）
    # store.summary()

    # 运行 Transformer 模型
    token_num = 8  # 假设输入有 10 个 token
    output = run_transformer(store, token_num)
    print("Transformer 输出形状:", output.shape)