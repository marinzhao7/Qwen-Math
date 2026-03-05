import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.sft_config import SFTConfig
from src.train.sft_trainer import SFTTrainer

def main():
    # 初始化配置
    config = SFTConfig()
    
    # 创建SFT训练器
    trainer = SFTTrainer(config)
    
    # 开始训练
    print("Starting SFT training...")
    trainer.train()
    
    print("SFT training completed successfully!")

if __name__ == "__main__":
    main()