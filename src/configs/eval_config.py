from src.configs.base_config import BaseConfig

class EvalConfig(BaseConfig):
    # 评估配置
    eval_output_dir = "output/eval"
    eval_log_dir = "logs/eval"
    
    # 模型配置
    eval_model_name = "output/grpo/best_model"  # 使用GRPO阶段的最佳模型
    eval_max_length = 8192
    
    # 数据集配置
    eval_datasets = [
        "gsm8k",
        "math",
        "aime"
    ]
    
    # OpenCompass配置
    opencompass_config = {
        "datasets": {
            "gsm8k": {
                "type": "GSM8K",
                "path": "data/processed/gsm8k_test.json"
            },
            "math": {
                "type": "MATH",
                "path": "data/processed/math_test.json"
            }
        },
        "metrics": {
            "accuracy": {
                "type": "Accuracy"
            },
            "exact_match": {
                "type": "ExactMatch"
            }
        }
    }
    
    # 评估参数
    eval_batch_size = 8
    eval_max_new_tokens = 2048
    eval_temperature = 0.7
    eval_top_p = 0.95