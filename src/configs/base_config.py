# 基础配置

class BaseConfig:
    # 模型配置
    model_name = "Qwen/Qwen3.5-2B-Base"
    max_length = 8192  # 8K推理长度约束
    
    # 数据配置
    data_dir = "data"
    gsm8k_train_path = "data/gsm8k/train.jsonl"
    math_train_path = "data/math/train.jsonl"
    use_filtered_data = False  # 是否使用质量过滤后的数据
    
    # 训练配置
    batch_size = 8
    learning_rate = 1e-5
    epochs = 3
    
    # LoRA配置
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1
    
    # 输出配置
    output_dir = "output"
    log_dir = "logs"
    
    # 评估配置
    eval_interval = 100
    save_interval = 500
    
    # GRPO配置
    grpo_beta = 0.1
    grpo_tau = 0.95
    
    # GPU配置
    gpu_ids = None  # None表示使用所有可用GPU，或指定列表如[0, 1, 2, 3]
    use_multi_gpu = False  # 是否使用多GPU训练
    local_rank = -1  # 分布式训练本地rank