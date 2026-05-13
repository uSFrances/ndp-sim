import numpy as np
import os
import json

def create_dummy_bin_file(folder, name, shape, dtype):
    """创建具有随机数据的虚拟二进制文件"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 从数据类型获取numpy dtype
    if dtype == 'f32':
        np_dtype = np.float32
    elif dtype == 'i32':
        np_dtype = np.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # 生成随机数据
    if np.issubdtype(np_dtype, np.integer):
        # 对于索引，通常用0或特定值，这里用0
        data = np.zeros(shape, dtype=np_dtype) 
    else:
        # 对于浮点数，生成范围在[-1, 1)的随机数
        data = (np.random.rand(*shape) * 2.0 - 1.0).astype(np_dtype)

    # 构建文件名
    shape_str = "x".join(map(str, shape))
    filename = f"{name}_shape{shape_str}_dtype_{dtype}.bin"
    filepath = os.path.join(folder, filename)

    # 以列优先顺序（'F' order）写入文件
    data.flatten(order='F').tofile(filepath)
    print(f"Created dummy input: {filepath}")

if __name__ == "__main__":
    # 设置固定的随机种子，确保每次生成的输入都一样
    np.random.seed(0)
    
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(__file__)

    # 加载配置以获取维度
    config_path = os.path.join(base_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    HIDDEN_SIZE = config['hidden_size']
    TOKEN_NUM = config['sequence_length']
    
    # 定义输入文件夹
    input_folder = os.path.join(base_dir, 'inputs')
    
    # 清理旧的输入文件，防止不同尺寸的文件冲突
    if os.path.exists(input_folder):
        print(f"🧹 Clearing old input files in {input_folder}...")
        for old_file in os.listdir(input_folder):
            if old_file.endswith(".bin"):
                os.remove(os.path.join(input_folder, old_file))
    else:
        os.makedirs(input_folder)

    # 根据token_num计算mask的维度 (与soft_max逻辑相关)
    # get_nearest_32_multiple(token_num) -> (e.g. 8+31)//32*32 = 32
    mask_dim0 = (TOKEN_NUM + 31) // 32 * 32

    # 创建虚拟输入文件
    print("--- Creating dummy input files ---")
    create_dummy_bin_file(input_folder, "inp_embd", (HIDDEN_SIZE, TOKEN_NUM, 1, 1), 'f32')
    create_dummy_bin_file(input_folder, "leaf_12", (mask_dim0, TOKEN_NUM, 1, 1), 'f32')
    create_dummy_bin_file(input_folder, "leaf_395", (1, 1, 1, 1), 'i32')
    print("----------------------------------")
