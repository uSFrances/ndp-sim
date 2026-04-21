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
        
    execplan_main_py = os.path.abspath(os.path.join(base_dir, "..", "model_execplan", "main.py"))
    # 将原来的 "examples" 替换为 "op_json"
    execplan_json_dir = os.path.abspath(os.path.join(base_dir, "..", "model_execplan", "op_json"))
        
    # 3. 循环执行对应的脚本及编译指令
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
            continue
            
        # 4. 执行 model_execplan 的 main.py 生成指令
        op_json_path = os.path.join(execplan_json_dir, f"{op}.json")
        if not os.path.exists(op_json_path) and not os.path.exists(execplan_main_py):
            print(f"⚠️ Warning: {op_json_path} or execplan main.py not found. Skipping instruction generation.")
            continue
            
        print(f"⚙️  Running execution plan generator for '{op}'...")
        try:
            # 运行命令: python main.py examples/xx.json
            subprocess.run([sys.executable, execplan_main_py, op_json_path], check=True)
            print(f"✅ Finished execution plan generation for '{op}'")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed execution plan generation for '{op}'. Error code: {e.returncode}")

if __name__ == "__main__":
    main()
