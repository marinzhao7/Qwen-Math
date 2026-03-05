import sys
import os
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.grpo_config import GRPOConfig


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run GRPO training')
    parser.add_argument(
        '--use-swift',
        action='store_true',
        help='Use ms-swift GRPO implementation'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the model to train (overrides config)'
    )
    args = parser.parse_args()
    
    # 初始化配置
    config = GRPOConfig()
    
    # 如果指定了模型路径，覆盖配置
    if args.model_path:
        config.grpo_model_name = args.model_path
    
    # 检查模型是否存在
    if not os.path.exists(config.grpo_model_name):
        print(f"Warning: Model not found at {config.grpo_model_name}")
        print("Will attempt to load from HuggingFace Hub...")
    
    # 根据参数选择训练器
    if args.use_swift:
        print("Using ms-swift GRPO implementation...")
        from src.train.grpo_trainer_swift import GRPOTrainerSwift
        trainer = GRPOTrainerSwift(config)
    else:
        print("Using custom GRPO implementation...")
        from src.train.grpo_trainer import GRPOTrainer
        trainer = GRPOTrainer(config)
    
    # 开始训练
    print("Starting GRPO training...")
    trainer.train()
    
    print("GRPO training completed successfully!")


if __name__ == "__main__":
    main()
