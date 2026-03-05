import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_name: str, description: str):
    """运行脚本"""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    start_time = time.time()
    
    try:
        exec(open(script_path).read())
        elapsed_time = time.time() - start_time
        print(f"\n{description} completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error running {description}: {e}")
        return False
    
    return True

def main():
    print("Starting full training and evaluation pipeline...")
    print("This will run:")
    print("1. Data building")
    print("2. SFT training (using GSM8K dataset)")
    print("3. GRPO training (using MATH500 dataset)")
    print("4. Model evaluation")
    
    # 运行数据构建
    if not run_script("build_data.py", "Data Building"):
        print("Pipeline stopped due to data building failure")
        return
    
    # 运行SFT训练
    if not run_script("run_sft.py", "SFT Training"):
        print("Pipeline stopped due to SFT training failure")
        return
    
    # 运行GRPO训练
    if not run_script("run_grpo.py", "GRPO Training"):
        print("Pipeline stopped due to GRPO training failure")
        return
    
    # 运行评估
    if not run_script("run_eval.py", "Model Evaluation"):
        print("Pipeline stopped due to evaluation failure")
        return
    
    print("\n{'='*60}")
    print("Full pipeline completed successfully!")
    print("{'='*60}")

if __name__ == "__main__":
    main()