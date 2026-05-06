import numpy as np
import os
import re
import time
import functools
from tqdm import tqdm
import json
import math

try:
    FLOAT_ACCUM = np.float128  # Prefer highest precision available
except AttributeError:  # pragma: no cover - fallback on platforms without float128
    FLOAT_ACCUM = np.longdouble

# --- Model Configuration Loader ---
def load_config(config_path):
    """从 JSON 文件加载模型配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("✅ Model configuration loaded:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    return config

# 加载配置
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)

# 从配置中获取参数
HIDDEN_SIZE = config['hidden_size']
INTERMEDIATE_SIZE = config['intermediate_size']
NUM_Q_HEADS = config['num_attention_heads']
NUM_KV_HEADS = config['num_key_value_heads']
HEAD_DIM = config['head_dim']
NUM_LAYERS = config['num_hidden_layers']
SEQUENCE_LENGTH = config['sequence_length']
# ------------------------------------


def load_bin_as_array(path, dtype, shape):
    total_elements = shape[0] * shape[1] * shape[2] * shape[3]
    data = np.fromfile(path, dtype=dtype, count=total_elements)
    return data.reshape(shape, order='F')

# ================= 节点IO存储机制 =================
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "python_golden")
SUB_OP_DIR = os.path.join(GOLDEN_DIR, "sub_ops")
os.makedirs(GOLDEN_DIR, exist_ok=True)
os.makedirs(SUB_OP_DIR, exist_ok=True)

# 追踪当前运行的算子名称前缀
CURRENT_NODE_PREFIX = ""

def save_io_tensor(name, tensor, is_sub_op=False):
    """根据节点名称，保存其输入/输出张量"""
    if not isinstance(tensor, np.ndarray): return
    shape_str = "x".join(str(s) for s in tensor.shape)
    dtype_str = {
        np.float32: "f32",
        np.float16: "f16",
        np.float64: "f64",
        np.int32: "i32"
    }.get(tensor.dtype.type, "unknown")
    
    if dtype_str != "unknown":
        folder = SUB_OP_DIR if is_sub_op else GOLDEN_DIR
        filepath = os.path.join(folder, f"{name}_shape{shape_str}_dtype_{dtype_str}.bin")
        tensor.flatten(order='F').tofile(filepath)

def log_op(func):
    """已废弃：留空装饰器防止因删除导致大面积报错"""
    return func

def fp32_fma_accumulate(acc, a, b):
    """
    硬件 SUMMAC 对齐：acc = fma(a, b, acc)
    优先用 math.fma；若环境不支持则回退到乘加。
    """
    try:
        return np.float32(math.fma(float(a), float(b), float(acc)))
    except AttributeError:
        return np.float32(np.float32(a) * np.float32(b) + np.float32(acc))

# =================================================

# 为你关心的算子加上 @log_op 装饰器，它们将自动将 IO 存入 python_golden 文件夹
@log_op
def rms_norm(x):
    nsamples    = x.shape[3]  # ne03
    nchannels   = x.shape[2]  # ne02
    nrows       = x.shape[1]  # ne01 (32)
    ncols       = x.shape[0]  # ne00 (896)
    dst = np.zeros_like(x)
    
    # 追踪子操作输出，完全不影响您的算法计算结果
    # 按照 nrows=32 (M), num_slices=28 (N) 设置追踪容器，以匹配后续重排 (32, 28)
    sum_mac_out = np.zeros((nrows, 28, nchannels, nsamples), dtype=np.float32)
    remote_sum_out = np.zeros((nrows, 1, nchannels, nsamples), dtype=np.float32)
    mac_SFU_out = np.zeros((nrows, 1, nchannels, nsamples), dtype=np.float32)
    
    # 完全保留您原本的算法逻辑
    for sample in range(nsamples):
        for channel in range(nchannels):
            tmp = np.float32(0.0)
            for row in range(nrows):
                x_slice = x[:, row, channel, sample]
                tmp = np.float32(0.0)

                # 原始实现（保留）：
                # tmp = 0
                # for col in range(ncols):
                #     tmp += np.float32(x_slice[col] * x_slice[col])

                # 按硬件 SUMMAC 语义改为 fma 累加：
                for col in range(ncols):
                    tmp = fp32_fma_accumulate(tmp, x_slice[col], x_slice[col])

                mean = tmp / ncols
                scale = 1.0 / np.sqrt(mean + 0.000001)
                dst[:, row, channel, sample] =  x_slice * scale
                
                # --- 仅用于旁路追踪 Sub-Op 数据 (提取 28 组 slice 平方和) ---
                if CURRENT_NODE_PREFIX:
                    for s_idx in range(28):
                        s_data = x_slice[s_idx*32 : (s_idx+1)*32]

                        # 原始实现（保留）：
                        # s_sum = 0.0
                        # for c in range(32):
                        #     s_sum += np.float32(s_data[c] * s_data[c])

                        # 按硬件 SUMMAC 语义改为 fma 累加：
                        s_sum = np.float32(0.0)
                        for c in range(32):
                            s_sum = fp32_fma_accumulate(s_sum, s_data[c], s_data[c])

                        sum_mac_out[row, s_idx, channel, sample] = s_sum
                    
                    # 记录一维结果
                    remote_sum_out[row, 0, channel, sample] = tmp
                    mac_SFU_out[row, 0, channel, sample] = scale
                # -----------------------------------------------------------

    if CURRENT_NODE_PREFIX:
        # 保存输入
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-sum_mac_in0", x, is_sub_op=True)
        # 保存输出与后续流转
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-sum_mac_out", sum_mac_out, is_sub_op=True)
        
        # --- 补充：各子操作中间步骤的输入提取 ---
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-remote_sum_in0", sum_mac_out, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mac_SFU_in0", remote_sum_out, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_MN_M_in0", x, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_MN_M_in1", mac_SFU_out, is_sub_op=True)
        # --------------------------------------

        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-remote_sum_out", remote_sum_out, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mac_SFU_out", mac_SFU_out, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_MN_M_out", dst, is_sub_op=True)

    return dst

@log_op
def ensure_fp32(x):
    if isinstance(x, np.ndarray) and x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x

@log_op
def mul(x, y):
    return ensure_fp32(x) * ensure_fp32(y)

@log_op
def matmul_2d(src0, src1, m, n ,k, out_dtype=np.float32):
    dst_py = np.zeros((m,n), dtype=np.float32) 
  
    for i in range(m):          
        for j in range(n):      
            sum_val = FLOAT_ACCUM(0.0)       
            for l in range(k):  
                a_val = src0[i, l].astype(FLOAT_ACCUM)
                b_val = src1[l, j].astype(FLOAT_ACCUM)
                tmp = a_val * b_val
                sum_val += tmp
            dst_py[i, j] = np.float32(sum_val)  

    return dst_py.astype(out_dtype, copy=False)

@log_op
def mul_mat(src0, src1, out_dtype=np.float32):
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

        dst_py = matmul_2d(src0_re_T, src1_re, m, n, k, out_dtype=out_dtype)

        return dst_py.reshape((m,n,1,1), order='F').astype(out_dtype, copy=False)

    elif (ne04 == ne14) and (ne04 == 1):#批次为1的三维张量处理

        scale = int(ne13 / ne03)#ne3的广播次数
        if scale < 1 :
            print("tensor shape error: ne03 is larger than ne13")
            return None

        #dst_py = np.zeros((1,ne13,n,m), dtype=np.float32) 
        dst_py = np.zeros((m,n,ne13,1), dtype=out_dtype) 
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
                dst = matmul_2d(ex_src0_T, ex_src1, m, n, k, out_dtype=out_dtype)
                dst_py[:,:,j*scale+i,0] = dst

        return dst_py

    else:
        print("tensor shape error: ne4 is more than 1")
        return None

@log_op
def add(x, y):
    return ensure_fp32(x) + ensure_fp32(y)

@log_op
def rope(x):
    ne2 = x.shape[2]
    ne1 = x.shape[1]
    ne0 = x.shape[0]   
    dst = np.zeros_like(x)
    sincos_shape = (ne0//2, ne2,1,1)
    
    # 追踪子操作，严格按照硬件流程
    # mul1: cos 乘法
    mul1_in0 = np.zeros_like(x) # x0, x1
    mul1_in1 = np.zeros_like(x) # cos, cos
    mul1_out = np.zeros_like(x) # x0*cos, x1*cos
    
    # mul2: sin 乘法
    mul2_in0 = np.zeros_like(x) # x0, x1
    mul2_in1 = np.zeros_like(x) # sin, -sin
    mul2_out = np.zeros_like(x) # x0*sin, -x1*sin
    
    # add: 最终加法
    add_out = np.zeros_like(x)
    
    base_dir = os.path.dirname(__file__)
    sin_path = os.path.join(base_dir, "rope_fp32", "rope_neox_sin_float32_ne2_512.bin")
    cos_path = os.path.join(base_dir, "rope_fp32", "rope_neox_cos_float32_ne2_512.bin")
    sin  = load_bin_as_array(sin_path, np.float32, sincos_shape)
    cos  = load_bin_as_array(cos_path, np.float32, sincos_shape)
    
    for k in range(0, ne2, 1):
        for j in range(0, ne1, 1):
            for i in range(0, ne0 // 2, 1):
                cos_theta = cos[i,k]
                sin_theta = sin[i,k]
            
                x0 = x[i,  j,  k]
                x1 = x[i+ne0//2,  j,  k]
                
                # --- 模拟硬件计算流程 ---
                # 第一个 mul: 计算 cos 相关项
                mul1_in0[i, j, k] = x0
                mul1_in0[i+ne0//2, j, k] = x1
                mul1_in1[i, j, k] = cos_theta
                mul1_in1[i+ne0//2, j, k] = cos_theta
                
                x0_cos = x0 * cos_theta
                x1_cos = x1 * cos_theta
                mul1_out[i, j, k] = x0_cos
                mul1_out[i+ne0//2, j, k] = x1_cos

                # 第二个 mul: 保存给文件的 in1 统一为 +sin
                # 负号仅保留在计算路径，避免后续 relayout/硬件侧重复取负
                mul2_in0[i, j, k] = x0
                mul2_in0[i+ne0//2, j, k] = x1
                mul2_in1[i, j, k] = sin_theta
                mul2_in1[i+ne0//2, j, k] = -sin_theta  # 改：保存时不带负号

                x0_sin = x0 * sin_theta
                x1_neg_sin = -x1 * sin_theta  # 负号仅在计算中体现
                mul2_out[i, j, k] = x0_sin
                mul2_out[i+ne0//2, j, k] = x1_neg_sin

                # 最终加法
                # y0 = x0*cos - x1*sin  ->  x0_cos + (-x1*sin)
                # y1 = x0*sin + x1*cos
                # 我们的 mul2_out 已经是 x0_sin 和 -x1_sin 了
                # 所以 y0 = mul1_out[x0] + mul2_out[x1]  -> 错误
                # y1 = mul2_out[x0] + mul1_out[x1]  -> 正确
                
                res1 = x0_cos + x1_neg_sin # y0
                res2 = x0_sin + x1_cos     # y1
                
                add_out[i, j, k] = res1
                add_out[i+ne0//2, j, k] = res2
                
                dst[i,  j,  k]       = res1
                dst[i+ne0//2,  j,  k]    = res2
                
    if CURRENT_NODE_PREFIX:
        # 保存第一个 mul (cos) 的输入输出
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_cos_in0", mul1_in0, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_cos_in1", mul1_in1, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_cos_out", mul1_out, is_sub_op=True)
        
        # 保存第二个 mul (sin) 的输入输出
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_sin_in0", mul2_in0, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_sin_in1", mul2_in1, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_sin_out", mul2_out, is_sub_op=True)

        # 保存最终 add 的输入输出
        # add 的输入是 mul1_out 和 mul2_out 的特定组合，这里为了清晰分开保存
        add_in0 = np.zeros_like(x)
        add_in1 = np.zeros_like(x)
        add_in0[:,:,:,:] = mul1_out # x0_cos, x1_cos
        add_in1[0:ne0//2,:,:,:] = mul2_out[ne0//2:,:,:,:] # -x1_sin
        add_in1[ne0//2:,:,:,:] = mul2_out[0:ne0//2,:,:,:] # x0_sin
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-add_final_in0", add_in0, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-add_final_in1", add_in1, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-add_final_out", add_out, is_sub_op=True)
        
    return dst

@log_op
def soft_max(x, mask=None):
    """
    x: 4D tensor, column-major (Fortran-order)
    mask: optional 4D tensor, also column-major
    """
    src0_shape = x.shape
    ncols = src0_shape[0]
    nrows_x = np.prod(src0_shape[1:])
    nrows_y = src0_shape[1] if len(src0_shape) >= 2 else 1

    base_dir = os.path.dirname(__file__)
    scale_path = os.path.join(base_dir, "softmax_scale.bin")
    # 这里读取 softmax_scale.bin 的单个 fp32 缩放系数，用于 QK 分数预缩放
    with open(scale_path, "rb") as f:
        scale = np.frombuffer(f.read(), dtype=np.float32)[0]

    src0_flat = x.flatten(order='F')
    dst_python = np.zeros_like(src0_flat)
    mask_flat = mask.flatten(order='F') if mask is not None else None
    
    # 追踪 sub_ops
    scaled_x = np.zeros_like(src0_flat)
    scaled_masked_x = np.zeros_like(src0_flat)
    max_out = np.zeros(nrows_x, dtype=np.float32)
    sub_SFU_out = np.zeros_like(src0_flat)
    sum_SFU_out = np.zeros(nrows_x, dtype=np.float32)

    def soft_max_f32_simple(x_flat, mask_flat, dst_flat, ncols, nrows_x, nrows_y, scale):
        for rowx in range(nrows_x):
            rowy = rowx % nrows_y
            start = rowx * ncols
            mask_start = rowy * ncols

            row = x_flat[start:start+ncols] * scale
            scaled_x[start:start+ncols] = row # 保存 scale 后的结果
            
            if mask_flat is not None:
                row += mask_flat[mask_start:mask_start+ncols]

            scaled_masked_x[start:start+ncols] = row
            
            r_max = np.max(row)
            max_out[rowx] = r_max
            
            r_sub = np.exp(row - r_max)
            sub_SFU_out[start:start+ncols] = r_sub
            
            r_sum = np.sum(r_sub)
            # op3: sum_SFU 计算 1 / sum(exp(...))
            sum_SFU_out[rowx] = np.float32(1.0) / r_sum
            
            dst_flat[start:start+ncols] = r_sub * sum_SFU_out[rowx]
        return dst_flat

    soft_max_f32_simple(src0_flat, mask_flat, dst_python, ncols, nrows_x, nrows_y, scale)

    dst_python = dst_python.reshape(src0_shape, order='F')
    
    if CURRENT_NODE_PREFIX:
        # 重塑并保存子图操作
        scaled_x_re = scaled_x.reshape(src0_shape, order='F')
        scaled_masked_x_re = scaled_masked_x.reshape(src0_shape, order='F')
        max_out_re = max_out.reshape((1, *src0_shape[1:]), order='F')
        sum_SFU_out_re = sum_SFU_out.reshape((1, *src0_shape[1:]), order='F')
        sub_SFU_out_re = sub_SFU_out.reshape(src0_shape, order='F')
        
        # 保存输入
        # add_MN_MN (masking)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-add_MN_MN_in0", scaled_x_re, is_sub_op=True)
        if mask is not None:
            save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-add_MN_MN_in1", mask, is_sub_op=True)
        
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-max_in0", scaled_masked_x_re, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-sub_SFU_in0", scaled_masked_x_re, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-sub_SFU_in1", max_out_re, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-sum_SFU_in0", sub_SFU_out_re, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_MN_M_in0", sub_SFU_out_re, is_sub_op=True)
        # op3 输出：sum_SFU_out 是 1/sum(exp(...))，也是 op4 的输入
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_MN_M_in1", sum_SFU_out_re, is_sub_op=True)
        
        # 保存输出
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-add_MN_MN_out", scaled_masked_x_re, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-max_out", max_out_re, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-sub_SFU_out", sub_SFU_out_re, is_sub_op=True)
        # op3 输出：sum_SFU 的结果是倒数
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-sum_SFU_out", sum_SFU_out_re, is_sub_op=True)
        save_io_tensor(f"{CURRENT_NODE_PREFIX}_subop-mul_MN_M_out", dst_python, is_sub_op=True)

    return dst_python

@log_op
def unary(x):
    x_fp32 = ensure_fp32(x)
    return x_fp32 / (1 + np.exp(-x_fp32))

@log_op
def get_rows(src0: np.ndarray, src1: np.ndarray) -> np.ndarray:
    """
    src0: 源 tensor，4D，列优先
    src1: 索引 tensor，4D，列优先
    返回：根据索引抽取后的 dst tensor，列优先
    """
    assert src0.ndim == 4
    assert src1.ndim == 4
    assert src0.dtype in (np.float16, np.float32)
    assert src1.dtype == np.int32

    D0, D1_src, D2, D3 = src0.shape
    _, D1_dst, D2_dst, D3_dst = src1.shape
    assert (D1_src >= np.max(src1) + 1), "src1索引超出src0范围"

    dst = np.zeros((D0, D1_dst, D2_dst, D3_dst), dtype=src0.dtype, order='F')

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
#     for layer_id in range(NUM_LAYERS):  # 假设有 28 层
#         store.set(f"cache_k_l{layer_id}", np.zeros((HEAD_DIM, 2, max_seq_len, 1), dtype=np.float16, order='F'))  # 列优先布局
#         store.set(f"cache_v_l{layer_id}", np.zeros((max_seq_len, HEAD_DIM * NUM_KV_HEADS, 1, 1), dtype=np.float16, order='F'))  # 列优先布局

def get_kv_for_attention(store, layer_id, token_num):
    key_name = f"cache_k_l{layer_id}"
    value_name = f"cache_v_l{layer_id}"

    k_cache = store.get(key_name)  # (128, seq_len, 2, 1)
    v_cache = store.get(value_name)  # (seq_len, 256, 1, 1)

    num_tokens_to_take = get_nearest_32_multiple(token_num)
    num_tokens_in_cache = k_cache.shape[1]

    if num_tokens_in_cache < num_tokens_to_take:
        k_selected = np.zeros((HEAD_DIM, num_tokens_to_take, NUM_KV_HEADS, 1), dtype=np.float16, order='F')
        k_selected[:, :num_tokens_in_cache, :, :] = k_cache[:, :num_tokens_in_cache, :, :]

        v_selected = np.zeros((num_tokens_to_take, HEAD_DIM, NUM_KV_HEADS, 1), dtype=np.float16, order='F')
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
        self.debug_save_dir = os.path.join(os.path.dirname(__file__), "python_golden_debug")
        os.makedirs(self.debug_save_dir, exist_ok=True)
        # 取消之前在这里创建 python_golden 的代码，由最上方的装饰器处理
        
    def get(self, name):
        if name not in self.store:
            raise KeyError(f"Tensor '{name}' not found")
        return self.store[name]
        
    def set(self, name, value):
        # 还原为其最原始纯净的功能，仅保存在内存中
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
            
            # 使用直接赋值，避免在加载网络权重和输入时也将它们又保存到 python_golden 中
            self.store[name] = tensor
            # print(f"[Loaded] {name} with shape {shape}, dtype {dtype_str}")
    
    def set_debug(self, name, value):
        """设置张量并将其保存为.bin文件（列优先）"""
        self.store[name] = value  # 这里也改成直接赋值，防止互相触发双重保存
        shape_str = "x".join(str(s) for s in value.shape)
        dtype_str = {
            np.float32: "f32",
            np.float16: "f16",
            np.float64: "f64",
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
    def timed_op(name: str, func, *args, store_dtype=None, **kwargs):
        global CURRENT_NODE_PREFIX
        op_name = func.__name__
        # 拼接：层数 + 节点名 + 算子名
        prefix = f"blk.{layer_id}_{name}_op-{op_name}"
        
        # 激活全局节点追踪以进行 sub_op 保存
        CURRENT_NODE_PREFIX = prefix
        
        # 1. 自动保存带有名称的输入节点张量
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                save_io_tensor(f"{prefix}_in{i}", arg)
                
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"  ⏱️ {name:<30} took {duration:.4f} seconds")

        if store_dtype is not None:
            result = convert(result, dtype=store_dtype)
        
        # 2. 自动保存带有名称的输出节点张量
        if isinstance(result, np.ndarray):
            save_io_tensor(f"{prefix}_out", result)
            
        store.set(name, result)
        CURRENT_NODE_PREFIX = ""

    def timed_matmul_fp16(name: str, lhs: np.ndarray, rhs: np.ndarray):
        timed_op(
            name,
            mul_mat,
            lhs,
            rhs,
            out_dtype=np.float32,
            store_dtype='fp16',
        )
    def timed_matmul_fp32(name: str, lhs: np.ndarray, rhs: np.ndarray):
        timed_op(
            name,
            mul_mat,
            lhs,
            rhs,
            out_dtype=np.float32
        )

    def store_fp16(name: str, value: np.ndarray):
        store.set(name, convert(value, dtype='fp16'))
        
    lid = f"blk.{layer_id}"
    print(f"🟡 Running transformer layer {layer_id}...")
    start_time = time.time()

    input_key = "inp_embd" if layer_id == 0 else f"l_out-{layer_id - 1}"
    timed_op(f"norm-{layer_id}", rms_norm, store.get(input_key))
    timed_op(
        f"attn_norm-{layer_id}",
        mul,
        store.get(f"norm-{layer_id}"),
        store.get(f"{lid}.attn_norm.weight"),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"node_{layer_id}_q", store.get(f"{lid}.attn_q.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Qcur-{layer_id}-add", add, store.get(f"node_{layer_id}_q"), store.get(f"{lid}.attn_q.bias"))
    timed_op(
        f"Qcur-{layer_id}",
        rope,
        store.get(f"Qcur-{layer_id}-add").reshape((HEAD_DIM, NUM_Q_HEADS, token_num, 1),  order='F'),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"node_{layer_id}_k", store.get(f"{lid}.attn_k.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Kcur-{layer_id}-add", add, store.get(f"node_{layer_id}_k"), store.get(f"{lid}.attn_k.bias"))
    timed_op(
        f"Kcur-{layer_id}",
        rope,
        store.get(f"Kcur-{layer_id}-add").reshape((HEAD_DIM, NUM_KV_HEADS, token_num, 1), order='F'),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"node_{layer_id}_v", store.get(f"{lid}.attn_v.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Vcur-{layer_id}-add", add, store.get(f"node_{layer_id}_v"), store.get(f"{lid}.attn_v.bias"))
    store_fp16(
        f"Vcur-{layer_id}",
        store.get(f"Vcur-{layer_id}-add").reshape((HEAD_DIM, NUM_KV_HEADS, token_num, 1), order='F'),
    )
    
    update_kv_cache(store, layer_id, store.get(f"Kcur-{layer_id}"), store.get(f"Vcur-{layer_id}"))
    
    k_selected, v_selected = get_kv_for_attention(store, layer_id, token_num)
    store.set(f"cache_k_l{layer_id}", k_selected)
    store.set(f"cache_v_l{layer_id}", v_selected)
    
    store_fp16(f"Qcur-{layer_id}-permute", store.get(f"Qcur-{layer_id}").transpose(0, 2, 1, 3))
    timed_matmul_fp32(f"node_{layer_id}_attn_scores", k_selected, store.get(f"Qcur-{layer_id}-permute"))
    timed_op(
        f"node_{layer_id}_attn_probs",
        soft_max,
        store.get(f"node_{layer_id}_attn_scores"),
        mask=store.get("leaf_12"),
        store_dtype='fp16',
    )
    
    timed_matmul_fp16(f"node_{layer_id}_attn_out", v_selected, store.get(f"node_{layer_id}_attn_probs"))
    store_fp16(
        f"kqv_out-{layer_id}",
        store.get(f"node_{layer_id}_attn_out").transpose(0, 2, 1, 3).reshape((HIDDEN_SIZE, token_num, 1, 1), order='F'),
    )
    timed_matmul_fp16(f"node_{layer_id}_attn_final", store.get(f"{lid}.attn_output.weight"), store.get(f"kqv_out-{layer_id}"))

    # 单层情形：如果模型只有一层，跳过 get_rows（避免把序列 L 收缩为 1）
    if NUM_LAYERS == 1:
        # 直接用完整序列 residual 进入 FFN
        timed_op(f"ffn_inp-{layer_id}", add, store.get(f"node_{layer_id}_attn_final"), store.get(input_key))
    else:
        # 多层情形保留原有的 token gather 行为
        timed_op(f"node_attn-{layer_id}", get_rows, store.get(f"node_{layer_id}_attn_final"), store.get("leaf_395"))
        timed_op(f"l_out_prev-{layer_id}", get_rows, store.get(input_key), store.get("leaf_395"))
        timed_op(f"ffn_inp-{layer_id}", add, store.get(f"node_attn-{layer_id}"), store.get(f"l_out_prev-{layer_id}"))

    timed_op(f"norm_ffn-{layer_id}", rms_norm, store.get(f"ffn_inp-{layer_id}"))
    timed_op(
        f"ffn_norm-{layer_id}",
        mul,
        store.get(f"norm_ffn-{layer_id}"),
        store.get(f"{lid}.ffn_norm.weight"),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"ffn_gate-{layer_id}", store.get(f"{lid}.ffn_gate.weight"), store.get(f"ffn_norm-{layer_id}"))
    timed_op(f"ffn_silu-{layer_id}", unary, store.get(f"ffn_gate-{layer_id}"))
    timed_matmul_fp16(f"ffn_up-{layer_id}", store.get(f"{lid}.ffn_up.weight"), store.get(f"ffn_norm-{layer_id}"))
    timed_op(
        f"ffn_gate_par-{layer_id}",
        mul,
        store.get(f"ffn_silu-{layer_id}"),
        store.get(f"ffn_up-{layer_id}"),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"ffn_out-{layer_id}", store.get(f"{lid}.ffn_down.weight"), store.get(f"ffn_gate_par-{layer_id}"))
    timed_op(f"l_out-{layer_id}", add, store.get(f"ffn_out-{layer_id}"), store.get(f"ffn_inp-{layer_id}"))
    
    store.set_debug(f"l_out-{layer_id}", store.get(f"l_out-{layer_id}"))

    total_duration = time.time() - start_time
    print(f"✅ Finished layer {layer_id} in {total_duration:.3f} seconds | Output: l_out-{layer_id} shape = {store.get(f'l_out-{layer_id}').shape}")

def run_final_layer(store: TensorStore, token_num: int):
    layer_id = NUM_LAYERS - 1
    lid = f"blk.{layer_id}"
    print(f"🟡 Running transformer layer {layer_id}...")
    start_time = time.time()

    # 此处同样植入带跟踪的 timed_op，替换原来的 store.set
    def timed_op(name: str, func, *args, store_dtype=None, **kwargs):
        global CURRENT_NODE_PREFIX
        op_name = func.__name__
        prefix = f"blk.{layer_id}_{name}_op-{op_name}"
        
        CURRENT_NODE_PREFIX = prefix
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray): save_io_tensor(f"{prefix}_in{i}", arg)
        result = func(*args, **kwargs)
        if store_dtype is not None:
            result = convert(result, dtype=store_dtype)
        if isinstance(result, np.ndarray): save_io_tensor(f"{prefix}_out", result)
        store.set(name, result)
        CURRENT_NODE_PREFIX = ""

    def timed_matmul_fp16(name: str, lhs: np.ndarray, rhs: np.ndarray):
        timed_op(
            name,
            mul_mat,
            lhs,
            rhs,
            out_dtype=np.float32,
            store_dtype='fp16',
        )
    def timed_matmul_fp32(name: str, lhs: np.ndarray, rhs: np.ndarray):
        timed_op(
            name,
            mul_mat,
            lhs,
            rhs,
            out_dtype=np.float32
        )

    def store_fp16(name: str, value: np.ndarray):
        store.set(name, convert(value, dtype='fp16'))

    # 处理单层情况：如果是第一层(layer_id=0)，输入为inp_embd
    input_key = "inp_embd" if layer_id == 0 else f"l_out-{layer_id - 1}"

    # Q/K/V 分支
    timed_op(f"norm-{layer_id}", rms_norm, store.get(input_key))
    timed_op(
        f"attn_norm-{layer_id}",
        mul,
        store.get(f"norm-{layer_id}"),
        store.get(f"{lid}.attn_norm.weight"),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"node_{layer_id}_q", store.get(f"{lid}.attn_q.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Qcur-{layer_id}-add", add, store.get(f"node_{layer_id}_q"), store.get(f"{lid}.attn_q.bias"))
    timed_op(
        f"Qcur-{layer_id}",
        rope,
        store.get(f"Qcur-{layer_id}-add").reshape((HEAD_DIM, NUM_Q_HEADS, token_num, 1),  order='F'),
        store_dtype='fp16',
    )
    
    timed_matmul_fp16(f"node_{layer_id}_k", store.get(f"{lid}.attn_k.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Kcur-{layer_id}-add", add, store.get(f"node_{layer_id}_k"), store.get(f"{lid}.attn_k.bias"))
    timed_op(
        f"Kcur-{layer_id}",
        rope,
        store.get(f"Kcur-{layer_id}-add").reshape((HEAD_DIM, NUM_KV_HEADS, token_num, 1), order='F'),
        store_dtype='fp16',
    )
    
    timed_matmul_fp16(f"node_{layer_id}_v", store.get(f"{lid}.attn_v.weight"), store.get(f"attn_norm-{layer_id}"))
    timed_op(f"Vcur-{layer_id}-add", add, store.get(f"node_{layer_id}_v"), store.get(f"{lid}.attn_v.bias"))
    store_fp16(
        f"Vcur-{layer_id}",
        store.get(f"Vcur-{layer_id}-add").reshape((HEAD_DIM, NUM_KV_HEADS, token_num, 1), order='F'),
    )
    
    # 更新 KV Cache
    update_kv_cache(store, layer_id, store.get(f"Kcur-{layer_id}"), store.get(f"Vcur-{layer_id}"))
    
    # 获取 KV Cache 中最近的 32 倍数的 token
    k_selected, v_selected = get_kv_for_attention(store, layer_id, token_num)
    store.set(f"cache_k_l{layer_id}", k_selected)
    store.set(f"cache_v_l{layer_id}", v_selected)
    
    # 计算注意力
    store_fp16(f"Qcur-{layer_id}-permute", store.get(f"Qcur-{layer_id}").transpose(0, 2, 1, 3))
    timed_matmul_fp32(f"node_{layer_id}_attn_scores", k_selected, store.get(f"Qcur-{layer_id}-permute"))
    timed_op(
        f"node_{layer_id}_attn_probs",
        soft_max,
        store.get(f"node_{layer_id}_attn_scores"),
        mask=store.get("leaf_12"),
        store_dtype='fp16',
    )
    
    timed_matmul_fp16(f"node_{layer_id}_attn_out", v_selected, store.get(f"node_{layer_id}_attn_probs"))
    store_fp16(
        f"kqv_out-{layer_id}",
        store.get(f"node_{layer_id}_attn_out").transpose(0, 2, 1, 3).reshape((HIDDEN_SIZE, token_num, 1, 1), order='F'),
    )
    timed_matmul_fp16(f"node_{layer_id}_attn_final", store.get(f"{lid}.attn_output.weight"), store.get(f"kqv_out-{layer_id}"))

    # 单层情形：如果模型只有一层，跳过 get_rows（避免把序列 L 收缩为 1）
    if NUM_LAYERS == 1:
        # 直接用完整序列 residual 进入 FFN
        timed_op(f"ffn_inp-{layer_id}", add, store.get(f"node_{layer_id}_attn_final"), store.get(input_key))
    else:
        # 多层情形保留原有的 token gather 行为
        timed_op(f"node_attn-{layer_id}", get_rows, store.get(f"node_{layer_id}_attn_final"), store.get("leaf_395"))
        timed_op(f"l_out_prev-{layer_id}", get_rows, store.get(input_key), store.get("leaf_395"))
        timed_op(f"ffn_inp-{layer_id}", add, store.get(f"node_attn-{layer_id}"), store.get(f"l_out_prev-{layer_id}"))
    
    timed_op(f"norm_ffn-{layer_id}", rms_norm, store.get(f"ffn_inp-{layer_id}"))
    timed_op(
        f"ffn_norm-{layer_id}",
        mul,
        store.get(f"norm_ffn-{layer_id}"),
        store.get(f"{lid}.ffn_norm.weight"),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"ffn_gate-{layer_id}", store.get(f"{lid}.ffn_gate.weight"), store.get(f"ffn_norm-{layer_id}"))
    timed_op(f"ffn_silu-{layer_id}", unary, store.get(f"ffn_gate-{layer_id}"))
    timed_matmul_fp16(f"ffn_up-{layer_id}", store.get(f"{lid}.ffn_up.weight"), store.get(f"ffn_norm-{layer_id}"))
    timed_op(
        f"ffn_gate_par-{layer_id}",
        mul,
        store.get(f"ffn_silu-{layer_id}"),
        store.get(f"ffn_up-{layer_id}"),
        store_dtype='fp16',
    )
    timed_matmul_fp16(f"ffn_out-{layer_id}", store.get(f"{lid}.ffn_down.weight"), store.get(f"ffn_gate_par-{layer_id}"))
    timed_op(f"l_out-{layer_id}", add, store.get(f"ffn_out-{layer_id}"), store.get(f"ffn_inp-{layer_id}"))
    
    store.set_debug(f"l_out-{layer_id}", store.get(f"l_out-{layer_id}"))

    # Final norm and output projection 
    # (纯Python计算 L=32 时的词表投影极慢，且调试单层特征时通常不需要这部分，故注释跳过)
    # timed_op("norm", rms_norm, store.get(f"l_out-{layer_id}"))
    # timed_op("result_norm", mul, store.get("norm"), store.get("output_norm.weight"))
    # timed_op("result_output", mul_mat, store.get("output.weight"), store.get("result_norm"))
    # store.set_debug("result_output", store.get("result_output"))
    
    duration = time.time() - start_time
    print(f"✅ Finished layer {layer_id} in {duration:.3f} seconds | Output: l_out-{layer_id} shape = {store.get(f'l_out-{layer_id}').shape}")

def run_transformer(store: TensorStore, token_num: int):
    """运行完整的 Transformer 模型（带 tqdm 进度条）"""
    num_layers = NUM_LAYERS - 1  # 循环到倒数第二层，最后一层由 run_final_layer 处理
    print(f"\n🚀 Starting transformer forward pass for {num_layers + 1} layers...")

    for layer_id in tqdm(range(num_layers), desc="🧠 Transformer Layers", unit="layer"):
        start_time = time.time()
        run_transformer_layer(store, layer_id, token_num)
        duration = time.time() - start_time
        print(f"  ⏱️ Layer {layer_id} finished in {duration:.3f} s | l_out-{layer_id} shape = {store.get(f'l_out-{layer_id}').shape}")

    print("🔚 Running final transformer layer...")
    # 取消这里的注释！必须让最后一层运行！
    run_final_layer(store, token_num)

    # 兼容注释掉 output 投影的情况，获取不到 result_output 就返回最后一层的 l_out
    return store.store.get("result_output", store.get(f"l_out-{NUM_LAYERS - 1}"))

# 示例用法
if __name__ == "__main__":
    # 创建 TensorStore 并初始化输入
    store = TensorStore()

    # 获取当前脚本所在目录
    base_dir = os.path.dirname(__file__)

    # 加载模型权重（非输入）
    # weight_folder = "/mnt/139_nvme2/liudongyan/workspace/CGRA_SIM/cgra_python/op_lib/LLM_golden/prefill_token8_bs1/llamacpp_cpu/model_weights"
    weight_folder = os.path.join(base_dir, "model_weights_small")
    print(f"✅ Loading sliced weights from: {weight_folder}")
    if not os.path.exists(weight_folder):
        print(f"❌ Error: Weight folder '{weight_folder}' not found.")
        print("Please run 'python weight_gen.py' first to generate the sliced weights.")
    else:
        store.load_from_bin_folder(weight_folder)

    # 加载输入张量（inp_embd, leaf_12, leaf_395）
    input_folder = "/Users/jielu/Desktop/CGRA mapping/configuration/LLM_python_golden/0316_python_golden/inputs"
    print(f"✅ Loading inputs from local path: {input_folder}")
    store.load_from_bin_folder(input_folder)

    # 加载llamacpp_cpu的kvcache
    # input_folder = "/nvme1/liudongyan/workspace/CGRA_SIM/cgra_python/op_lib/LLM_golden/prefill_token8_bs1/llamacpp_cpu/kvcache"
    # store.load_from_bin_folder(input_folder)

    # 打印所有张量信息（名称、shape、dtype）
    # store.summary()

    # 运行 Transformer 模型
    if 'output.weight' in store.store: # 仅在权重加载成功时运行
        output = run_transformer(store, SEQUENCE_LENGTH)
        print("Transformer 输出形状:", output.shape)
    else:
        print("Skipping transformer run due to missing weights.")
