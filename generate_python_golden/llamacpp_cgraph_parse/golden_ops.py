import os
import math
import numpy as np

""" oplib: 执行算子运算，并输出 input/output golden 数据"""
class oplib:
    def __init__(self, golden_dir, golden_subop_dir, rope_fp32_dir):
        self.golden_dir = golden_dir                # Golden 输出目录
        self.golden_subop_dir = golden_subop_dir    # Golden subop 输出目录
        self.rope_fp32_dir = rope_fp32_dir          # ROPE cos/sin 查找表
        self.current_node_prefix = ""
        self.current_node_store_dtype = None
        try:
            self.FLOAT_ACCUM = np.float128  # Prefer highest precision available
        except AttributeError:  # pragma: no cover - fallback on platforms without float128
            self.FLOAT_ACCUM = np.longdouble

        # Create output directory
        os.makedirs(self.golden_dir, exist_ok=True)
        os.makedirs(self.golden_subop_dir, exist_ok=True)

    def save_io_tensor(self, name, tensor, is_sub_op=False):
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
            folder = self.golden_subop_dir if is_sub_op else self.golden_dir
            filepath = os.path.join(folder, f"{name}_shape{shape_str}_dtype_{dtype_str}.bin")
            tensor.flatten(order='F').tofile(filepath)

    def load_bin_as_array(self, path, dtype, shape):
        total_elements = shape[0] * shape[1] * shape[2] * shape[3]
        data = np.fromfile(path, dtype=dtype, count=total_elements)
        return data.reshape(shape, order='F')

    def fp32_fma_accumulate(self, acc, a, b):
        """
        硬件 SUMMAC 对齐：acc = fma(a, b, acc)
        优先用 math.fma；若环境不支持则回退到乘加。
        """
        if hasattr(math, "fma"):
            return np.float32(math.fma(float(a), float(b), float(acc))) # type: ignore
        else:
            return np.float32(np.float32(a) * np.float32(b) + np.float32(acc))

    def ensure_fp32(self, x):
        if isinstance(x, np.ndarray) and x.dtype != np.float32:
            return x.astype(np.float32, copy=False)
        return x
    
    def add(self, x, y):
        dst = self.ensure_fp32(x) + self.ensure_fp32(y)
        # 新增：非 subop 输入与输出
        self.save_io_tensor(f"{self.current_node_prefix}_in0", x, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_in1", y, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)
        return dst
    
    def mul(self, x, y):
        dst = self.ensure_fp32(x) * self.ensure_fp32(y)
        # 新增：非 subop 输入与输出
        self.save_io_tensor(f"{self.current_node_prefix}_in0", x, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_in1", y, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)
        return dst
    
    def unary(self, x):
        x_fp32 = self.ensure_fp32(x)
        dst = x_fp32 / (1 + np.exp(-x_fp32))
        # 新增：非 subop 输入与输出
        self.save_io_tensor(f"{self.current_node_prefix}_in0", x, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)
        return dst
    
    def get_rows(self, src0: np.ndarray, src1: np.ndarray) -> np.ndarray:
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

        # 新增：非 subop 输入与输出
        self.save_io_tensor(f"{self.current_node_prefix}_in0", src0, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_in1", src1, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)
        return dst
    
    def matmul_2d(self, src0, src1, m, n ,k, out_dtype=np.float32):
        dst_py = np.zeros((m,n), dtype=np.float32) 
    
        for i in range(m):          
            for j in range(n):      
                sum_val = self.FLOAT_ACCUM(0.0)       
                for l in range(k):  
                    # 修正：防止越界
                    if l >= src0.shape[1] or l >= src1.shape[0]:
                        continue
                    a_val = src0[i, l].astype(self.FLOAT_ACCUM)
                    b_val = src1[l, j].astype(self.FLOAT_ACCUM)
                    tmp = a_val * b_val
                    sum_val += tmp
                dst_py[i, j] = np.float32(sum_val)  

        dst = dst_py.astype(out_dtype, copy=False)
    
        # 新增：非 subop 输入与输出
        self.save_io_tensor(f"{self.current_node_prefix}_in0", src0, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_in1", src1, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)
        return dst

    def mul_mat(self, src0, src1, out_dtype=np.float32):
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

            dst_py = self.matmul_2d(src0_re_T, src1_re, m, n, k, out_dtype=out_dtype)

            dst = dst_py.reshape((m,n,1,1), order='F').astype(out_dtype, copy=False)

        elif (ne04 == ne14) and (ne04 == 1):#批次为1的三维张量处理

            scale = int(ne13 / ne03)#ne3的广播次数
            if scale < 1 :
                print("tensor shape error: ne03 is larger than ne13")
                return None

            #dst_py = np.zeros((1,ne13,n,m), dtype=np.float32) 
            dst_py = np.zeros((m,n,ne13,1), dtype=out_dtype) 
            for j in range(ne03):
                for i in range(scale):
                    ex_src0 = src0[:,:,j,0]
                    ex_src1 = src1[:,:,j*scale+i,0]
                    ex_src0_T = ex_src0.T
                    dst = self.matmul_2d(ex_src0_T, ex_src1, m, n, k, out_dtype=out_dtype)
                    dst_py[:,:,j*scale+i,0] = dst

            dst = dst_py

        else:
            print("tensor shape error: ne4 is more than 1")
            return None
        
        # 新增：非 subop 输入与输出
        self.save_io_tensor(f"{self.current_node_prefix}_in0", src0, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_in1", src1, is_sub_op=False)
        self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)
        return dst


    def rms_norm(self, x):
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
                        tmp = self.fp32_fma_accumulate(tmp, x_slice[col], x_slice[col])

                    mean = tmp / ncols
                    # TODO: 可配置的eps?
                    scale = 1.0 / np.sqrt(mean + 0.000001)
                    dst[:, row, channel, sample] =  x_slice * scale
                    
                    # --- 仅用于旁路追踪 Sub-Op 数据 (提取 28 组 slice 平方和) ---
                    if self.current_node_prefix:
                        for s_idx in range(28):
                            s_data = x_slice[s_idx*32 : (s_idx+1)*32]

                            # 原始实现（保留）：
                            # s_sum = 0.0
                            # for c in range(32):
                            #     s_sum += np.float32(s_data[c] * s_data[c])

                            # 按硬件 SUMMAC 语义改为 fma 累加：
                            s_sum = np.float32(0.0)
                            for c in range(32):
                                s_sum = self.fp32_fma_accumulate(s_sum, s_data[c], s_data[c])

                            sum_mac_out[row, s_idx, channel, sample] = s_sum
                        
                        # 记录一维结果
                        remote_sum_out[row, 0, channel, sample] = tmp
                        mac_SFU_out[row, 0, channel, sample] = scale
                    # -----------------------------------------------------------

        if self.current_node_prefix:
            # 新增：非 subop 输入与输出
            self.save_io_tensor(f"{self.current_node_prefix}_in0", x, is_sub_op=False)
            self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)

            # 保存输入
            self.save_io_tensor(f"{self.current_node_prefix}_subop-sum_mac_in0", x, is_sub_op=True)
            # 保存输出与后续流转
            self.save_io_tensor(f"{self.current_node_prefix}_subop-sum_mac_out", sum_mac_out, is_sub_op=True)
            
            # --- 补充：各子操作中间步骤的输入提取 ---
            self.save_io_tensor(f"{self.current_node_prefix}_subop-remote_sum_in0", sum_mac_out, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mac_SFU_in0", remote_sum_out, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_MN_M_in0", x, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_MN_M_in1", mac_SFU_out, is_sub_op=True)
            # --------------------------------------

            self.save_io_tensor(f"{self.current_node_prefix}_subop-remote_sum_out", remote_sum_out, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mac_SFU_out", mac_SFU_out, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_MN_M_out", dst, is_sub_op=True)

        return dst

    def rope(self, x):
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
        
        sin_path = os.path.join(self.rope_fp32_dir, "rope_neox_sin_float32_ne2_512.bin")
        cos_path = os.path.join(self.rope_fp32_dir, "rope_neox_cos_float32_ne2_512.bin")
        sin  = self.load_bin_as_array(sin_path, np.float32, sincos_shape)
        cos  = self.load_bin_as_array(cos_path, np.float32, sincos_shape)
        
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
                    
        if self.current_node_prefix:
            # 新增：非 subop 输入与输出
            self.save_io_tensor(f"{self.current_node_prefix}_in0", x, is_sub_op=False)
            self.save_io_tensor(f"{self.current_node_prefix}_out", dst, is_sub_op=False)

            # 保存第一个 mul (cos) 的输入输出
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_cos_in0", mul1_in0, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_cos_in1", mul1_in1, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_cos_out", mul1_out, is_sub_op=True)
            
            # 保存第二个 mul (sin) 的输入输出
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_sin_in0", mul2_in0, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_sin_in1", mul2_in1, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_sin_out", mul2_out, is_sub_op=True)

            # 保存最终 add 的输入输出
            # add 的输入是 mul1_out 和 mul2_out 的特定组合，这里为了清晰分开保存
            add_in0 = np.zeros_like(x)
            add_in1 = np.zeros_like(x)
            add_in0[:,:,:,:] = mul1_out # x0_cos, x1_cos
            add_in1[0:ne0//2,:,:,:] = mul2_out[ne0//2:,:,:,:] # -x1_sin
            add_in1[ne0//2:,:,:,:] = mul2_out[0:ne0//2,:,:,:] # x0_sin
            self.save_io_tensor(f"{self.current_node_prefix}_subop-add_final_in0", add_in0, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-add_final_in1", add_in1, is_sub_op=True)
            add_final_out = add_out
            if self.current_node_store_dtype == "f16":
                add_final_out = add_out.astype(np.float16)
            elif self.current_node_store_dtype == "f32":
                add_final_out = add_out.astype(np.float32)
            self.save_io_tensor(
                f"{self.current_node_prefix}_subop-add_final_out",
                add_final_out,
                is_sub_op=True,
            )
            
        return dst

    def soft_max(self, x, scale, mask=None):
        """
        x: 4D tensor, column-major (Fortran-order)
        mask: optional 4D tensor, also column-major
        """
        src0_shape = x.shape
        ncols = src0_shape[0]
        nrows_x = np.prod(src0_shape[1:])
        nrows_y = src0_shape[1] if len(src0_shape) >= 2 else 1

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
        
        if self.current_node_prefix:
            # 新增：非 subop 输入与输出
            self.save_io_tensor(f"{self.current_node_prefix}_in0", x, is_sub_op=False)
            self.save_io_tensor(f"{self.current_node_prefix}_out", dst_python, is_sub_op=False)

            # 重塑并保存子图操作
            scaled_x_re = scaled_x.reshape(src0_shape, order='F')
            scaled_masked_x_re = scaled_masked_x.reshape(src0_shape, order='F')
            max_out_re = max_out.reshape((1, *src0_shape[1:]), order='F')
            sum_SFU_out_re = sum_SFU_out.reshape((1, *src0_shape[1:]), order='F')
            sub_SFU_out_re = sub_SFU_out.reshape(src0_shape, order='F')
            
            # 保存输入
            # add_MN_MN (masking)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-add_MN_MN_in0", scaled_x_re, is_sub_op=True)
            if mask is not None:
                self.save_io_tensor(f"{self.current_node_prefix}_subop-add_MN_MN_in1", mask, is_sub_op=True)
            
            self.save_io_tensor(f"{self.current_node_prefix}_subop-max_in0", scaled_masked_x_re, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-sub_SFU_in0", scaled_masked_x_re, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-sub_SFU_in1", max_out_re, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-sum_SFU_in0", sub_SFU_out_re, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_MN_M_in0", sub_SFU_out_re, is_sub_op=True)
            # op3 输出：sum_SFU_out 是 1/sum(exp(...))，也是 op4 的输入
            self.save_io_tensor(f"{self.current_node_prefix}_subop-mul_MN_M_in1", sum_SFU_out_re, is_sub_op=True)
            
            # 保存输出
            self.save_io_tensor(f"{self.current_node_prefix}_subop-add_MN_MN_out", scaled_masked_x_re, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-max_out", max_out_re, is_sub_op=True)
            self.save_io_tensor(f"{self.current_node_prefix}_subop-sub_SFU_out", sub_SFU_out_re, is_sub_op=True)
            # op3 输出：sum_SFU 的结果是倒数
            self.save_io_tensor(f"{self.current_node_prefix}_subop-sum_SFU_out", sum_SFU_out_re, is_sub_op=True)
            mul_out = dst_python
            if self.current_node_store_dtype == "f16":
                mul_out = dst_python.astype(np.float16)
            elif self.current_node_store_dtype == "f32":
                mul_out = dst_python.astype(np.float32)
            self.save_io_tensor(
                f"{self.current_node_prefix}_subop-mul_MN_M_out",
                mul_out,
                is_sub_op=True,
            )

        return dst_python




