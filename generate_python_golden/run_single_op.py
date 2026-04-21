import json
import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, 'config.json')
    
    # 1. 载入 config.json
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    target_op = config.get("target_op", "")
    if not target_op:
        print("⚠️ No 'target_op' specified in config.json. Skipping single op generation.")
        return
        
    # 2. 判断需要运行哪些算子脚本
    if target_op == "all":
        ops_to_run = ["gemm", "rmsnorm", "rope", "softmax"]
    else:
        # 移除下划线以匹配实际的文件名（比如 rms_norm -> rmsnorm）
        ops_to_run = [target_op.replace("_", "")]
        
    # 3. 循环执行对应的脚本
    for op in ops_to_run:
        script_path = os.path.join(base_dir, "single_op_data", f"relayout_{op}.py")
        
        if not os.path.exists(script_path):
            print(f"❌ Error: Script for operator '{op}' not found at {script_path}")
            continue
            
        print(f"\n🚀 Running single op script for '{op}': {script_path}")
        try:
            subprocess.run([sys.executable, script_path], check=True)
            print(f"✅ Finished generating sliced golden data for '{op}'")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run {script_path}. Error code: {e.returncode}")

if __name__ == "__main__":
    main()
