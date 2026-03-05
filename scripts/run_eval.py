import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.eval_config import EvalConfig
from src.eval.evaluator import Evaluator

def main():
    # 初始化配置
    config = EvalConfig()
    
    # 检查模型是否存在
    if not os.path.exists(config.eval_model_name):
        print(f"Error: Model not found at {config.eval_model_name}")
        print("Please run GRPO training first.")
        sys.exit(1)
    
    # 创建评估器
    evaluator = Evaluator(config)
    
    # 开始评估
    print("Starting evaluation...")
    results = evaluator.evaluate()
    
    # 打印评估结果
    print("\nEvaluation Results:")
    for dataset, result in results.items():
        print(f"{dataset}: Accuracy = {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()