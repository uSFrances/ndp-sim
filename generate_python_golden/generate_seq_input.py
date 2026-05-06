import numpy as np
import os
import re
import math

def duplicate_tokens_to_seq_length(input_bin_path, output_dir, original_seq_len=8, target_seq_len=32):
    """
    Reads an existing bin file with a base sequence length (e.g., 8 tokens),
    and duplicates/tiles the token data to match the target sequence length.
    """
    if not os.path.exists(input_bin_path):
        print(f"❌ File not found: {input_bin_path}")
        return

    filename = os.path.basename(input_bin_path)
    match = re.search(r"_shape([\dx]+)_dtype_([a-z0-9]+)", filename)
    if not match:
        print(f"⚠️ Could not parse shape/dtype from filename: {filename}")
        return

    shape_str = match.group(1)
    shape_tuple = tuple(map(int, shape_str.split('x')))
    
    # Notice: do not hardcode the new shape here before loading original data
    # Use the original shape_tuple for loading
    
    # 注意：如果输入文件是 dtype_f32.bin，必须使用 dtype=np.float32
    # 动态确定 dtype
    dtype = np.float32 if "dtype_f32" in filename else np.float16
    data = np.fromfile(input_bin_path, dtype=dtype)
    
    expected_size = math.prod(shape_tuple)
    if data.size != expected_size:
        print(f"⚠️ Size mismatch for {filename}: expected {expected_size}, got {data.size}. Skipping.")
        return
        
    data = data.reshape(shape_tuple, order='F')
    
    # Assuming shape is (K, L, 1, 1) or similar, where L is original_seq_len
    # Find the dimension that matches original_seq_len
    seq_dim = None
    for i, dim in enumerate(shape_tuple):
        if dim == original_seq_len:
            seq_dim = i
            break
            
    if seq_dim is None:
        print(f"⚠️ Original sequence length {original_seq_len} not found in shape {shape_tuple}. Falling back to dimension 1.")
        seq_dim = 1

    # Calculate how many times to repeat the tokens
    repeats = int(np.ceil(target_seq_len / original_seq_len))
    
    # Repeat data along the sequence dimension
    repeated_data = np.repeat(data, repeats, axis=seq_dim)
    
    # Slice to exact target length in case target_seq_len is not a perfect multiple
    slices = [slice(None)] * len(shape_tuple)
    slices[seq_dim] = slice(0, target_seq_len)
    final_data = repeated_data[tuple(slices)]
    
    # Construct new filename with updated shape
    new_shape_list = list(shape_tuple)
    new_shape_list[seq_dim] = target_seq_len
    new_shape_str = "x".join(map(str, new_shape_list))
    
    new_filename = filename.replace(f"_shape{shape_str}", f"_shape{new_shape_str}")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, new_filename)
    
    final_data.flatten('F').tofile(out_path)
    print(f"✅ Successfully created {new_filename} with sequence length {target_seq_len} in {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "inputs_good")
    output_dir = os.path.join(current_dir, "python_golden_custom_seq")
    
    # Process all bin files in the inputs_good directory
    if os.path.exists(input_dir):
        token_files = [f for f in os.listdir(input_dir) if f.endswith('.bin')]
    else:
        print(f"❌ Input directory not found: {input_dir}")
        token_files = []
    
    for f in token_files:
        duplicate_tokens_to_seq_length(
            input_bin_path=os.path.join(input_dir, f),
            output_dir=output_dir,
            original_seq_len=8,
            target_seq_len=32
        )
