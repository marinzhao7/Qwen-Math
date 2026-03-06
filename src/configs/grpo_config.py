from src.configs.base_config import BaseConfig

class GRPOConfig(BaseConfig):
    # GRPO训练配置
    grpo_output_dir = "output/grpo"
    grpo_log_dir = "logs/grpo"
    
    # 训练参数
    grpo_batch_size = 2  # prompt batch size
    grpo_learning_rate = 1e-6
    grpo_epochs = 3
    grpo_warmup_steps = 50
    grpo_weight_decay = 0.01
    
    # LoRA配置
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # GRPO特定参数
    grpo_beta = 0.1  # 优势函数的权重
    grpo_kl_coef = 0.01  # KL 惩罚系数
    grpo_clip_epsilon = 0.2  # PPO-style clipping
    
    # 生成参数
    grpo_num_generations = 4  # G in GRPO：每组生成样本数
    grpo_max_prompt_length = 512
    grpo_max_new_tokens = 512
    grpo_temperature = 1.0
    grpo_top_p = 0.9
    
    # 数据配置
    grpo_train_data_path = "data/processed/math_train.json"  # 使用处理过的MATH500数据集
    grpo_val_data_path = "data/processed/math_val.json"
    
    # 模型配置
    grpo_model_name = "output/sft/best_model"  # 使用SFT阶段的最佳模型
    
    # 训练设置
    grpo_max_steps = -1  # -1表示根据epochs计算
    grpo_save_steps = 500
    grpo_eval_steps = 100
    grpo_log_steps = 10
    
    # 其他配置
    grpo_use_lora = True
    grpo_use_gradient_checkpointing = True
    grpo_use_fp16 = False
    grpo_use_bf16 = True
    
    # 多GPU训练配置
    grpo_use_multi_gpu = False  # 是否使用多GPU训练
    grpo_gpu_ids = None  # None表示使用所有可用GPU，或指定列表如[0, 1]
    grpo_ddp_find_unused_parameters = False  # DDP是否查找未使用参数