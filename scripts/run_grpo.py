import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.grpo_config import GRPOConfig
from src.train.grpo_trainer import GRPOTrainer

def main():
    # 初始化配置
    config = GRPOConfig()
    
    # 检查SFT模型是否存在
    if not os.path.exists(config.grpo_model_name):
        print(f"Error: SFT model not found at {config.grpo_model_name}")
        print("Please run SFT training first.")
        sys.exit(1)
    
    # 创建GRPO训练器
    trainer = GRPOTrainer(config)
    
    # 开始训练
    print("Starting GRPO training...")
    trainer.train()
    
    print("GRPO training completed successfully!")

if __name__ == "__main__":
    main()