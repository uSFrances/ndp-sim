import numpy as np
import os
import json
import re

def load_original_weights(folder_path):
    """加载原始权重文件，并从文件名解析元数据"""
    weights = {}
    print(f"🔍 Loading original weights from: {folder_path}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Original weights folder not found: {folder_path}")
        
    # 修改正则表达式以匹配原始文件命名格式 "__dtype=f16__shape=1536x256.bin"
    pattern = re.compile(r"(?P<name>.+?)__dtype=(?P<dtype>f32|f16|i32)__shape=(?P<shape>[\dx]+)\.bin")

    for fname in os.listdir(folder_path):
        match = pattern.match(fname)
        if not match:
            # 放宽一点，如果文件明明是.bin却没有匹配上，打印出来方便排查
            if fname.endswith(".bin"):
                print(f"  [Skip] Filename does not match expected pattern: {fname}")
            continue

        meta = match.groupdict()
        name = meta['name']
        shape = tuple(map(int, meta['shape'].split('x')))
        dtype_str = meta['dtype']
        
        dtype = {'f16': np.float16, 'f32': np.float32, 'i32': np.int32}.get(dtype_str)
        if not dtype: continue

        full_path = os.path.join(folder_path, fname)
        tensor_data = np.fromfile(full_path, dtype=dtype)
        
        if tensor_data.size != np.prod(shape):
            print(f"  [Error] Shape mismatch for {name}. Prod: {np.prod(shape)}, but got {tensor_data.size}.")
            continue
            
        weights[name] = tensor_data.reshape(shape, order='F')
        print(f"  - Loaded {name} with shape {shape}")
        
    return weights

def save_new_weight(folder, name, tensor):
    """保存新的权重张量"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    tensor = tensor.astype(np.float32)
    shape_str = "x".join(map(str, tensor.shape))
    dtype_str = "f32" 
    filename = f"{name}_shape{shape_str}_dtype_{dtype_str}.bin"
    filepath = os.path.join(folder, filename)
    tensor.flatten(order='F').tofile(filepath)
    print(f"  -> Saved new weight: {filename}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    
    target_config_path = os.path.join(base_dir, 'config.json')
    with open(target_config_path, 'r') as f:
        target_config = json.load(f)
    
    H_TGT = target_config['hidden_size']
    I_TGT = target_config['intermediate_size']
    HEAD_DIM_TGT = target_config['head_dim']
    NUM_KV_HEADS_TGT = target_config['num_key_value_heads']

    orig_weights_folder = os.path.join(base_dir, "DeepSeek-R1-Distill-Qwen-1.5B-f16")
    original_weights = load_original_weights(orig_weights_folder)
    new_weights_folder = os.path.join(base_dir, "model_weights_small")
    
    if not os.path.exists(new_weights_folder):
        os.makedirs(new_weights_folder)

    print("\n🔥 Generating and slicing weights...")

    for name, orig_tensor in original_weights.items():
        orig_tensor_sq = np.squeeze(orig_tensor)
        new_tensor = orig_tensor_sq
        
        # 修正了所有权重的裁剪轴向 (Axis)
        if "attn_norm.weight" in name or "ffn_norm.weight" in name or "output_norm.weight" in name:
            new_tensor = orig_tensor_sq[:H_TGT]
        elif "attn_q.weight" in name or "attn_output.weight" in name:
            # 原始 (1536, 1536) -> 目标 (128, 128)
            new_tensor = orig_tensor_sq[:H_TGT, :H_TGT]
        elif "attn_k.weight" in name or "attn_v.weight" in name:
            # 原始 (1536, 256) -> 目标 (H_TGT, HEAD_DIM_TGT * NUM_KV_HEADS_TGT)
            new_d1 = HEAD_DIM_TGT * NUM_KV_HEADS_TGT
            new_tensor = orig_tensor_sq[:H_TGT, :new_d1]
        elif "ffn_gate.weight" in name or "ffn_up.weight" in name:
            # 原始 (1536, 8960) -> 目标 (H_TGT, I_TGT)
            new_tensor = orig_tensor_sq[:H_TGT, :I_TGT]
        elif "ffn_down.weight" in name:
            # 原始 (8960, 1536) -> 目标 (I_TGT, H_TGT)
            new_tensor = orig_tensor_sq[:I_TGT, :H_TGT]
        elif ".bias" in name:
            if "attn_q.bias" in name:
                new_tensor = orig_tensor_sq[:H_TGT]
            elif "attn_k.bias" in name or "attn_v.bias" in name:
                new_d0 = HEAD_DIM_TGT * NUM_KV_HEADS_TGT
                new_tensor = orig_tensor_sq[:new_d0]
        elif "output.weight" in name or "tok_embeddings.weight" in name:
            # 假设原始形状是 (vocab_size, 1536) 或相反，取决于实际情况
            # 为了安全起见这里根据真实形状的第一个维度来切
            if orig_tensor_sq.shape[0] == target_config.get('vocab_size', 91392):
                 new_tensor = orig_tensor_sq[:, :H_TGT]
            else:
                 new_tensor = orig_tensor_sq[:H_TGT, :]

        if new_tensor.ndim == 1:
            new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, 1, 1, order='F')
        elif new_tensor.ndim == 2:
            new_tensor = new_tensor.reshape(new_tensor.shape[0], new_tensor.shape[1], 1, 1, order='F')
        
        save_new_weight(new_weights_folder, name, new_tensor)

    print("\n✅ Weight generation complete.")

