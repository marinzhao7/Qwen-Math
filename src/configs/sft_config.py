from src.configs.base_config import BaseConfig

class SFTConfig(BaseConfig):
    # SFT训练配置
    sft_output_dir = "output/sft"
    sft_log_dir = "logs/sft"
    
    # 训练参数
    sft_batch_size = 8
    sft_learning_rate = 1e-5
    sft_epochs = 3
    sft_warmup_steps = 100
    sft_weight_decay = 0.01
    
    # LoRA配置
    sft_use_lora = True
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # 数据配置
    sft_train_data_path = "data/processed/gsm8k_train.json"  # 使用处理过的GSM8K数据集
    sft_val_data_path = "data/processed/gsm8k_val.json"
    
    # 训练设置
    sft_max_steps = -1  # -1表示根据epochs计算
    sft_save_steps = 500
    sft_eval_steps = 100
    sft_log_steps = 10
    
    # 模型配置
    sft_model_name = "Qwen/Qwen3-1.7B-Base"
    sft_max_length = 8192
    
    # 其他配置
    sft_use_gradient_checkpointing = True
    sft_use_fp16 = False
    sft_use_bf16 = True
    
    # 多GPU训练配置
    sft_use_multi_gpu = False  # 是否使用多GPU训练
    sft_gpu_ids = [1,]  # None表示使用所有可用GPU，或指定列表如[0, 1]
    sft_ddp_find_unused_parameters = False  # DDP是否查找未使用参数