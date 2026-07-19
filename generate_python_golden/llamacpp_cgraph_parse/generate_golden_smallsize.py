import argparse
import os
import json
import numpy as np

import golden_ops
    

class TensorStore:
    def __init__(self):
        self.store = {}
        self.debug_save_dir = os.path.join(os.path.dirname(__file__), "python_golden_debug")
        os.makedirs(self.debug_save_dir, exist_ok=True)
        
    def get(self, name):
        if name not in self.store:
            raise KeyError(f"Tensor '{name}' not found")
        return self.store[name]
        
    def set(self, name, value):
        self.store[name] = value

    def set_debug(self, name, value):
        """设置张量并将其保存为.bin文件（列优先）"""
        self.store[name] = value  # 这里也改成直接赋值，防止互相触发双重保存
        shape_str = "x".join(str(s) for s in value.shape)
        dtype_str = {
            np.float32: "f32",
            np.float16: "f16",
            np.float64: "f64",
            np.int32: "i32",
            np.int64: "i64"
        }.get(value.dtype.type, "unknown")

        if dtype_str == "unknown":
            raise ValueError(f"Unsupported dtype for debug export: {value.dtype}")

        filename = f"{name}_shape{shape_str}_dtype_{dtype_str}.bin"
        full_path = os.path.join(self.debug_save_dir, filename)
        value.flatten(order='F').tofile(full_path)
        print(f"[Debug Saved] {name} → {filename}")

    def load_from_cgraph_json(self, cgraph, input_dir):
        """从 cgraph.json 中读取权重信息和数据"""
        # parse cgraph
        n_leafs = cgraph.get("n_leafs", 0)
        leafs = cgraph.get("leafs", [])
        for leaf in leafs:
            id = leaf.get("id")
            name = leaf.get("name")
            shape = leaf.get("ne", [])
            dtype_str = leaf.get("dtype")
            data_path = leaf.get("data_path", "")

            # Get dtype
            dtype = np.float32
            if dtype_str == "f32":
                dtype = np.float32
            elif dtype_str == "f16":
                dtype = np.float16
            elif dtype_str == "i32":
                dtype = np.int32
            elif dtype_str == "i64":
                dtype = np.int64
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")
            
            # Load data from binary file
            num_elements = np.prod(shape)
            full_path = os.path.join(input_dir, data_path)
            with open(full_path, "rb") as f:
                data = np.frombuffer(f.read(num_elements * np.dtype(dtype).itemsize), dtype=dtype)
            tensor = data.reshape(shape, order="F")  # 保持与 llama.cpp 一致的列优先布局
            
            # 使用直接赋值，避免在加载网络权重和输入时也将它们又保存到 python_golden 中
            self.store[id] = tensor
            # print(f"[Loaded] {name} with shape {shape}, dtype {dtype_str}")

    def summary(self):
        print("\n=== TensorStore Summary ===")
        for name, tensor in self.store.items():
            print(f"- {name:<40} | dtype: {tensor.dtype} | shape: {tensor.shape}")
        print("===========================\n")

# TODO: move this to oplib
def set_rows(src0: np.ndarray, src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """
    src0: 目标 tensor，4D，列优先
    src1: 源 tensor，4D，列优先
    src2: 索引 tensor，shape[0] 代表了行数
    src1.shape[-2] == src2.shape[-1]
    返回：根据索引写入后的 src0 的 view，列优先
    """
    #print(f"[set_rows]: src0.shape={src0.shape}, src1.shape={src1.shape}, src2.shape={src2.shape}")
    assert src0.ndim == 4
    assert src1.ndim == 4
    assert src2.ndim == 4
    assert src0.dtype in (np.float16, np.float32)
    assert src1.dtype in (np.float16, np.float32)
    assert src2.dtype in (np.int32, np.int64)

    D0_src0, D1_src0, D2_src0, D3_src0 = src0.shape
    D0_src1, D1_src1, D2_src1, D3_src1 = src1.shape
    D0_src2, D1_src2, D2_src2, _ = src2.shape
    assert (D0_src0 == D0_src1), f"set_rows函数中n_embeddings不匹配, {D0_src0} vs {D0_src1}"
    assert (D1_src1 == D0_src2), f"set_rows函数中n_rows不匹配, {D1_src1} vs {D0_src2}"
    n_rows = D0_src2
    assert (D1_src0 > np.max(src2)), f"src2索引值超出src0范围, {np.max(src2)} should in {D1_src0}"

    for i3 in range(D2_src2):
        for i2 in range(D1_src2):
            for row_idx in range(n_rows):
                idx = src2[row_idx, i2, i3]
                # 加 .reshape(-1)，确保是 (1536,)
                src0[:, idx, i2, i3] = src1[:, row_idx, i2, i3].reshape(-1)
    return src0.view()

def run_transformer(store: TensorStore, cgraph: dict, output_dir: str) -> np.ndarray:
    # Create output directory
    output_subop_dir = os.path.join(output_dir, "sub_ops")
    # TODO: hardcode rope directory here
    rope_dir = os.path.join(os.path.dirname(__file__), "../rope_fp32") 
    # Init oplib
    oplib = golden_ops.oplib(output_dir, output_subop_dir, rope_dir)

    # Process each node
    n_nodes = cgraph.get("n_nodes", 0)
    nodes = cgraph.get("nodes", [])
    result_node = nodes[-1]
    for i in range(n_nodes):
        node = nodes[i]
        id = node.get("id")
        name = node.get("name")
        dtype_str = node.get("dtype")
        shape = node.get("ne", [])
        op = node.get("op")
        srcs = node.get("srcs", [])

        # TODO: only generate one layer now!
        if name == "norm-1":
            result_node = nodes[i-1]
            #print(f"result node id={result_node["id"]}")
            break

        # TODO: prefix 格式可能需修正
        # TODO: C or F order
        match op:
            case "RMS_NORM":
                oplib.current_node_prefix = f"{name}_op-rms_norm"
                result = oplib.rms_norm(store.get(srcs[0]))
                store.set(id, result)
            case "MUL":
                oplib.current_node_prefix = f"{name}_op-mul"
                result = oplib.mul(store.get(srcs[0]), store.get(srcs[1]))
                store.set(id, result)
            case "MUL_MAT":
                oplib.current_node_prefix = f"{name}_op-mul_mat"
                result = oplib.mul_mat(store.get(srcs[0]), store.get(srcs[1]))
                if result is not None and dtype_str == "f16":
                    result = result.astype(np.float16) 
                store.set(id, result)
            case "ADD":
                oplib.current_node_prefix = f"{name}_op-add"
                result = oplib.add(store.get(srcs[0]), store.get(srcs[1]))
                store.set(id, result)
            case "ROPE":
                oplib.current_node_prefix = f"{name}_op-rope"
                oplib.current_node_store_dtype = dtype_str
                # TODO: where is leaf 6
                result = oplib.rope(store.get(srcs[0]))
                store.set(id, result)
            case "SOFT_MAX":
                scale = node.get("scale", 1)
                oplib.current_node_prefix = f"{name}_op-soft_max"
                oplib.current_node_store_dtype = dtype_str
                result = oplib.soft_max(store.get(srcs[0]), scale, mask=store.get(srcs[1]))
                store.set(id, result)
            case "GLU":
                glu_op = node.get("glu_op")
                if (glu_op == "SWIGLU"):
                    # TODO: not match: silu vs swiglu!!
                    oplib.current_node_prefix = f"{name}_op-unary"
                    result = oplib.unary(store.get(srcs[0]))
                    store.set(id, result)
                else:
                    raise ValueError(f"Not supported glu_op: {glu_op}")
            case "GET_ROWS":
                oplib.current_node_prefix = f"{name}_op-get_rows"
                result = oplib.get_rows(store.get(srcs[0]), store.get(srcs[1]))
                store.set(id, result)
            # Special ops for llama.cpp
            case "RESHAPE":     
                src = store.get(srcs[0])
                store.set(id, np.reshape(src, shape, order="F"))
            case "VIEW":
                src = store.get(srcs[0])
                # reshape naturally return a view
                store.set(id, np.reshape(src, shape, order="F"))
            case "PERMUTE":
                axis = node.get("axis", [])
                src = store.get(srcs[0])
                store.set(id, np.transpose(src, axis))
            case "CONT":
                src = store.get(srcs[0])
                store.set(id, np.reshape(src, shape, order="F"))
            case "SET_ROWS":
                # The order of srcs follows llama.cpp, may seem weird
                src = srcs[0]
                idx = srcs[1]
                dst = srcs[2]
                #print(f"[set_rows]: src0.id={dst}, src1.id={src}, src2.id={idx}")
                # TODO: need correct row index data
                #result = set_rows(store.get(dst), store.get(src), store.get(idx))
                #store.set(id, result)
                store.set(id, store.get(dst).view())
            case _:
                raise ValueError(f"Unsupported op: {op}")
            
    # Get result
    result_id = result_node.get("id")
    result = store.get(result_id)
    return result


if __name__ == "__main__":
    # Default arguments
    output_default = os.path.join(os.path.dirname(__file__), "../python_golden")
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the cgraph json file.")
    parser.add_argument("-o", "--output", default=output_default, type=str, required=True, help="Output data directory.")
    args = parser.parse_args()

    # Parse input path
    cgraph_json_path = args.input
    input_dir = os.path.dirname(cgraph_json_path)

    # Load cgraph.json
    cgraph = {}
    with open(args.input, 'r') as f:
        cgraph = json.load(f)
    # Init TensorStore
    store = TensorStore()
    store.load_from_cgraph_json(cgraph, input_dir)

    # 运行 Transformer 模型
    output = run_transformer(store, cgraph, args.output)
    print("Transformer 输出形状:", output.shape)
