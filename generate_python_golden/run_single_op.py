import json
import os
import sys
import subprocess

ADDRESS_REMAPPING_SUPPORTED_OPS = {
    "rmsnorm": "fill-remapping",
    "softmax": "fill-remapping",
}

def run_address_remapping(repo_root, op, op_json_path):
    """Run address_remapping before model_execplan when the operator is supported."""
    command = ADDRESS_REMAPPING_SUPPORTED_OPS.get(op)
    if command is None:
        return op_json_path

    address_remapping_src = os.path.join(repo_root, "address_remapping", "src")
    if not os.path.isdir(address_remapping_src):
        print(f"[warn] address_remapping not found at {address_remapping_src}. Skipping remapping.")
        return op_json_path

    remapped_dir = os.path.join(os.path.dirname(op_json_path), "remapped")
    os.makedirs(remapped_dir, exist_ok=True)
    remapped_json_path = os.path.join(remapped_dir, os.path.basename(op_json_path))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        address_remapping_src
        if not existing_pythonpath
        else os.pathsep.join([address_remapping_src, existing_pythonpath])
    )

    print(f"[address-remapping] Running address remapping for '{op}'...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "address_remapping.cli",
                command,
                op_json_path,
                "--output",
                remapped_json_path,
            ],
            check=True,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout[-1200:])
        if e.stderr:
            print(e.stderr[-1200:])
        raise
    print(f"[address-remapping] Remapped JSON written to: {remapped_json_path}")
    return remapped_json_path

def main():
    base_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))
    config_path = os.path.join(base_dir, 'config.json')
    
    # 1. 载入 config.json
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    target_op = config.get("target_op", "")
    if not target_op:
        print("[warn] No 'target_op' specified in config.json. Skipping single op generation.")
        return
        
    # 2. 判断需要运行哪些算子脚本
    if target_op == "all":
        ops_to_run = ["gemm", "rmsnorm", "rope", "softmax"]
    else:
        # 移除下划线以匹配实际的文件名（比如 rms_norm -> rmsnorm）
        ops_to_run = [target_op.replace("_", "")]
        
    execplan_main_py = os.path.join(repo_root, "model_execplan", "main.py")
    # 将原来的 "examples" 替换为 "op_json"
    execplan_json_dir = os.path.join(repo_root, "model_execplan", "op_json")
        
    # 3. 循环执行对应的脚本及编译指令
    for op in ops_to_run:
        script_path = os.path.join(base_dir, "single_op_data", f"relayout_{op}.py")
        
        if not os.path.exists(script_path):
            print(f"[error] Script for operator '{op}' not found at {script_path}")
            continue
            
        print(f"\n[single-op] Running single op script for '{op}': {script_path}")
        try:
            subprocess.run([sys.executable, script_path], check=True)
            print(f"[single-op] Finished generating sliced golden data for '{op}'")
        except subprocess.CalledProcessError as e:
            print(f"[error] Failed to run {script_path}. Error code: {e.returncode}")
            continue
            
        # 4. 执行 model_execplan 的 main.py 生成指令
        op_json_path = os.path.join(execplan_json_dir, f"{op}.json")
        if not os.path.exists(op_json_path) or not os.path.exists(execplan_main_py):
            print(f"[warn] {op_json_path} or execplan main.py not found. Skipping instruction generation.")
            continue

        try:
            op_json_path = run_address_remapping(repo_root, op, op_json_path)
        except subprocess.CalledProcessError as e:
            print(f"[error] Failed address remapping for '{op}'. Error code: {e.returncode}")
            continue
            
        print(f"[execplan] Running execution plan generator for '{op}'...")
        try:
            # 运行命令: python main.py examples/xx.json
            subprocess.run([sys.executable, execplan_main_py, op_json_path], check=True)
            print(f"[execplan] Finished execution plan generation for '{op}'")
        except subprocess.CalledProcessError as e:
            print(f"[error] Failed execution plan generation for '{op}'. Error code: {e.returncode}")

if __name__ == "__main__":
    main()
