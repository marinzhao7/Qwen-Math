import os
import json
import torch
from typing import List, Dict, Any, Callable

from transformers import AutoTokenizer
from swift import Swift, LoRAConfig, SwiftModel
from swift.trainers import Trainer, TrainingArguments
from swift.trainers.rlhf_arguments import GRPOConfig


class GRPOTrainerSwift:
    """基于ms-swift的GRPO训练器"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config.grpo_output_dir
        self.log_dir = config.grpo_log_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置GPU
        self._setup_gpu()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.grpo_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = self._load_model()
        
        # 配置LoRA
        if config.grpo_use_lora:
            lora_config = LoRAConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none"
            )
            self.model = SwiftModel(self.model, lora_config)
            print("LoRA applied to model")
    
    def _setup_gpu(self):
        """设置GPU环境"""
        if self.config.grpo_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.grpo_gpu_ids))
            print(f"Using specified GPUs: {self.config.grpo_gpu_ids}")
        
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def _load_model(self):
        """加载模型"""
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.grpo_model_name,
            torch_dtype=torch.bfloat16 if self.config.grpo_use_bf16 else torch.float16,
            trust_remote_code=True
        )
        model = model.to(self.device)
        print(f"Model loaded on GPU: {self.device}")
        return model
    
    def _reward_function(self, generated_texts: List[str], reference_answers: List[str]) -> List[float]:
        """奖励函数：评估生成答案的质量"""
        rewards = []
        
        for gen_text, ref_text in zip(generated_texts, reference_answers):
            # 格式奖励
            format_score = self._check_format(gen_text)
            
            # 准确性奖励
            accuracy_score = self._check_accuracy(gen_text, ref_text)
            
            # 组合奖励
            total_reward = format_score * 0.3 + accuracy_score * 0.7
            rewards.append(total_reward)
        
        return rewards
    
    def _check_format(self, text: str) -> float:
        """检查答案格式"""
        score = 0.0
        
        if any(marker in text for marker in ['Step', '步骤', '解析', '过程']):
            score += 0.4
            
        if any(op in text for op in ['=', '+', '-', '×', '÷', '*', '/']):
            score += 0.3
            
        if any(marker in text for marker in ['答案', 'answer', '因此', '所以']):
            score += 0.3
            
        return min(score, 1.0)
    
    def _check_accuracy(self, generated: str, reference: str) -> float:
        """检查答案准确性 - 提取数字比较"""
        import re
        
        # 提取数字
        gen_numbers = re.findall(r'-?\d+\.?\d*', generated)
        ref_numbers = re.findall(r'-?\d+\.?\d*', reference)
        
        if not ref_numbers:
            return 0.5
            
        try:
            # 检查最后一个数字（通常是最终答案）
            if gen_numbers and abs(float(gen_numbers[-1]) - float(ref_numbers[-1])) < 1e-3:
                return 1.0
            elif gen_numbers and any(abs(float(g) - float(ref_numbers[-1])) < 1e-3 for g in gen_numbers):
                return 0.5
            return 0.0
        except ValueError:
            return 0.0
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载训练数据"""
        data = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if 'question' in item and 'answer' in item:
                        data.append(item)
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data
    
    def train(self):
        """执行GRPO训练"""
        # 加载数据
        train_data = self.load_data(self.config.grpo_train_data_path)
        val_data = self.load_data(self.config.grpo_val_data_path)
        
        print(f"Training data: {len(train_data)} samples")
        print(f"Validation data: {len(val_data)} samples")
        
        # 配置GRPO训练参数
        grpo_config = GRPOConfig(
            output_dir=self.output_dir,
            num_generations=self.config.grpo_num_generations,
            temperature=self.config.grpo_temperature,
            top_p=self.config.grpo_top_p,
            beta=self.config.grpo_beta,
            learning_rate=self.config.grpo_learning_rate,
            per_device_train_batch_size=self.config.grpo_batch_size,
            num_train_epochs=self.config.grpo_epochs,
            logging_dir=self.log_dir,
            logging_steps=self.config.grpo_log_steps,
            save_steps=self.config.grpo_save_steps,
            eval_steps=self.config.grpo_eval_steps,
            warmup_steps=self.config.grpo_warmup_steps,
            weight_decay=self.config.grpo_weight_decay,
            bf16=self.config.grpo_use_bf16,
            fp16=self.config.grpo_use_fp16,
            gradient_accumulation_steps=self.config.grpo_gradient_accumulation_steps,
            max_grad_norm=self.config.grpo_max_grad_norm,
        )
        
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.grpo_batch_size,
            per_device_eval_batch_size=self.config.grpo_batch_size,
            num_train_epochs=self.config.grpo_epochs,
            learning_rate=self.config.grpo_learning_rate,
            weight_decay=self.config.grpo_weight_decay,
            warmup_steps=self.config.grpo_warmup_steps,
            logging_dir=self.log_dir,
            logging_steps=self.config.grpo_log_steps,
            eval_steps=self.config.grpo_eval_steps,
            save_steps=self.config.grpo_save_steps,
            bf16=self.config.grpo_use_bf16,
            fp16=self.config.grpo_use_fp16,
            gradient_accumulation_steps=self.config.grpo_gradient_accumulation_steps,
            max_grad_norm=self.config.grpo_max_grad_norm,
            remove_unused_columns=False,
        )
        
        # 准备数据集
        def prepare_dataset(data):
            """准备数据集格式"""
            formatted_data = []
            for item in data:
                formatted_data.append({
                    'prompt': f"Question: {item['question']}\nAnswer:",
                    'completion': item['answer'],
                    'reference': item['answer']
                })
            return formatted_data
        
        train_dataset = prepare_dataset(train_data)
        val_dataset = prepare_dataset(val_data)
        
        # 创建奖励函数
        def reward_func(prompts: List[str], completions: List[str], references: List[str]) -> List[float]:
            return self._reward_function(completions, references)
        
        # 创建ms-swift训练器
        try:
            from swift.trainers import RLHFTrainer
            
            trainer = RLHFTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                reward_funcs=reward_func,
                rlhf_config=grpo_config,
            )
            
            # 开始训练
            print("Starting GRPO training with ms-swift...")
            trainer.train()
            
            # 保存最终模型
            final_save_path = os.path.join(self.output_dir, "final_model")
            trainer.save_model(final_save_path)
            self.tokenizer.save_pretrained(final_save_path)
            print(f"Final model saved to {final_save_path}")
            
        except ImportError as e:
            print(f"Error importing RLHFTrainer from ms-swift: {e}")
            print("Falling back to custom GRPO implementation...")
            # 如果ms-swift的RLHFTrainer不可用，使用自定义实现
            self._train_custom(train_dataset, val_dataset, reward_func)
    
    def _train_custom(self, train_dataset, val_dataset, reward_func):
        """自定义GRPO训练实现（当ms-swift的RLHFTrainer不可用时使用）"""
        print("Using custom GRPO implementation...")
        
        # 这里可以调用原有的GRPOTrainer实现
        # 或者实现一个简化的版本
        from .grpo_trainer import GRPOTrainer as CustomGRPOTrainer
        
        custom_trainer = CustomGRPOTrainer(self.config)
        custom_trainer.train()


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.configs.grpo_config import GRPOConfig
    
    config = GRPOConfig()
    trainer = GRPOTrainerSwift(config)
    trainer.train()
