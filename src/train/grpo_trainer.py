import os
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from swift import Swift, LoraConfig, SwiftModel

from .trainer_utils import (
    setup_gpu, load_model, setup_distributed_training, 
    apply_lora, create_data_loader, get_model_to_save, log_message, load_data
)

class GRPOTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.grpo_output_dir
        self.log_dir = config.grpo_log_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志文件
        self.log_file = os.path.join(self.log_dir, f"grpo_training_{int(time.time())}.log")
        
        # 初始化数据收集
        self.train_metrics = {
            'steps': [],
            'loss': [],
            'policy_loss': [],
            'kl_div': [],
            'mean_reward': [],
            'std_reward': []
        }
        self.val_metrics = {
            'steps': [],
            'loss': []
        }
        
        # 设置GPU
        self.device = setup_gpu(self.config.grpo_gpu_ids)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.grpo_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载策略模型（当前模型）
        self.model = load_model(
            self.config.grpo_model_name,
            self.config.grpo_use_multi_gpu,
            self.device,
            self.config.grpo_use_fp16,
            self.config.grpo_use_bf16
        )
        print(f"Policy model loaded")
        
        # 配置LoRA
        self.model = apply_lora(self.model, self.config, self.config.grpo_use_lora)   
        
        # 加载参考模型（冻结，用于 KL 散度）
        self.reference_model = load_model(
            self.config.grpo_model_name,
            self.config.grpo_use_multi_gpu,
            self.device,
            self.config.grpo_use_fp16,
            self.config.grpo_use_bf16
        )
        print(f"Reference model loaded")
        
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        # 设置多GPU训练
        self.is_distributed, self.local_rank, self.world_size = setup_distributed_training(
            self.config.grpo_use_multi_gpu,
            self.config.grpo_gpu_ids,
            self.config.grpo_ddp_find_unused_parameters
        )
        
        if self.is_distributed:
            # 包装策略模型为DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.grpo_ddp_find_unused_parameters
            )
            
            # 包装参考模型为DDP
            self.reference_model = DDP(
                self.reference_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.grpo_ddp_find_unused_parameters
            )
            print(f"GRPO distributed training initialized: rank {self.local_rank}/{self.world_size}")
    

    
    def tokenize_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        GRPO 分词：只编码 prompt，答案由模型生成
        
        Returns:
            {
                'input_ids': [batch_size, prompt_len],
                'attention_mask': [batch_size, prompt_len],
                'answers': List[str],  # 标准答案，用于奖励计算
                'questions': List[str]
            }
        """
        prompts = [f"Question: {item['question']}\nAnswer:" for item in batch]
        answers = [item['answer'] for item in batch]
        questions = [item['question'] for item in batch]
        
        # 只编码 prompt
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            padding_side='left',
            truncation=True,
            max_length=self.config.grpo_max_prompt_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized.input_ids,
            'attention_mask': tokenized.attention_mask,
            'answers': answers,
            'questions': questions,
            'prompts': prompts
        }
    
    def generate_group_samples(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为每个 prompt 生成 G 个样本（Group Sampling）
        
        Returns:
            generated_ids: [batch_size * G, seq_len]
            old_log_probs: [batch_size * G, gen_len] 每个生成 token 的 log prob
        """
        batch_size = input_ids.shape[0]
        G = self.config.grpo_num_generations
        device = input_ids.device
        
        # 扩展输入以生成 G 个样本
        input_ids_expanded = input_ids.repeat_interleave(G, dim=0)
        attention_mask_expanded = attention_mask.repeat_interleave(G, dim=0)
        
        # 生成样本
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids_expanded,
                attention_mask=attention_mask_expanded,
                max_new_tokens=self.config.grpo_max_new_tokens,
                do_sample=True,
                temperature=self.config.grpo_temperature,
                top_p=self.config.grpo_top_p,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = outputs.sequences  # [batch_size*G, prompt_len + gen_len]
        
        # 计算生成过程中每个 token 的 log probability
        # 构建完整的 attention mask 覆盖整个生成序列
        gen_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        old_log_probs = self._compute_log_probs(
            generated_ids, 
            gen_attention_mask,
            prompt_len=input_ids.shape[1]
        )
        
        return generated_ids, old_log_probs
    
    def _compute_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                          prompt_len: int = None) -> torch.Tensor:
        """
        计算序列中每个生成位置的 log probability
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            prompt_len: prompt 的长度，用于只返回生成部分的 log probs
        
        Returns:
            log_probs: [batch_size, gen_len] 或 [batch_size, seq_len-1]
        """
        # 前向传播
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 计算 log softmax
        log_probs_all = F.log_softmax(logits, dim=-1)
        
        # 获取实际 token 的 log prob（shift by 1）
        target_log_probs = log_probs_all[:, :-1, :].gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        if prompt_len is not None:
            # 只返回生成部分的 log probs
            return target_log_probs[:, prompt_len-1:]
        
        return target_log_probs
    
    def compute_rewards(self, generated_texts: List[str], reference_answers: List[str]) -> torch.Tensor:
        """
        计算奖励分数
        
        Args:
            generated_texts: 模型生成的文本列表 [batch_size * G]
            reference_answers: 标准答案列表 [batch_size]（需要扩展）
        
        Returns:
            rewards: [batch_size * G]
        """
        G = self.config.grpo_num_generations
        rewards = []
        
        # 扩展参考答案以匹配生成的样本数
        expanded_refs = []
        for ans in reference_answers:
            expanded_refs.extend([ans] * G)
        
        for gen_text, ref_text in zip(generated_texts, expanded_refs):
            # 格式奖励
            format_score = self._check_format(gen_text)
            
            # 准确性奖励（对比标准答案）
            accuracy_score = self._check_accuracy(gen_text, ref_text)
            
            # 组合奖励
            total_reward = format_score * 0.3 + accuracy_score * 0.7
            rewards.append(total_reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
    
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
    
    def grpo_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 GRPO 损失
        
        GRPO 核心：
        1. 对每个 prompt 生成 G 个样本
        2. 计算组内相对优势（减去组内均值，除以组内标准差）
        3. 使用 PPO-clip 更新策略
        4. 添加 KL 散度约束
        """
        input_ids = batch['input_ids']  # [batch_size, prompt_len]
        attention_mask = batch['attention_mask']
        reference_answers = batch['answers']
        
        batch_size = input_ids.shape[0]
        G = self.config.grpo_num_generations
        device = input_ids.device
        
        # ========== 1. 生成 Group Samples ==========
        with torch.no_grad():
            generated_ids, old_log_probs = self.generate_group_samples(input_ids, attention_mask)
            # generated_ids: [batch_size*G, total_len]
            # old_log_probs: [batch_size*G, gen_len]
        
        prompt_len = input_ids.shape[1]
        gen_len = generated_ids.shape[1] - prompt_len
        
        # ========== 2. 解码生成文本并计算奖励 ==========
        # 只解码生成部分
        generated_texts = self.tokenizer.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=True
        )
        
        rewards = self.compute_rewards(generated_texts, reference_answers)  # [batch_size*G]
        
        # ========== 3. 计算优势函数（组内相对奖励）==========
        # reshape 为 [batch_size, G]
        rewards_grouped = rewards.view(batch_size, G)
        
        # 计算组内均值和标准差
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)  # [batch_size, 1]
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8  # [batch_size, 1]
        
        # 优势函数 = (r - mean) / std
        advantages = ((rewards_grouped - mean_rewards) / std_rewards).view(-1)  # [batch_size*G]
        
        # ========== 4. 重新计算当前策略的 log probs（用于梯度更新）==========
        # 构建完整的 attention mask
        gen_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        
        # 前向传播计算当前 log probs
        current_log_probs = self._compute_log_probs(
            generated_ids, 
            gen_attention_mask,
            prompt_len=prompt_len
        )  # [batch_size*G, gen_len]
        
        # ========== 5. 计算参考模型的 log probs（用于 KL 约束）==========
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=generated_ids,
                attention_mask=gen_attention_mask
            )
            ref_logits = ref_outputs.logits
            ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_probs_all[:, prompt_len-1:-1, :].gather(
                dim=-1,
                index=generated_ids[:, prompt_len:].unsqueeze(-1)
            ).squeeze(-1)  # [batch_size*G, gen_len]
        
        # ========== 6. 计算损失 ==========
        # 创建 mask（只考虑非 padding 的生成 token）
        gen_mask = gen_attention_mask[:, prompt_len:].float()  # [batch_size*G, gen_len]
        
        # PPO-clip 风格的目标
        # ratio = exp(current_log_prob - old_log_prob)
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # 裁剪
        clipped_ratio = torch.clamp(
            ratio, 
            1 - self.config.grpo_clip_epsilon,
            1 + self.config.grpo_clip_epsilon
        )
        
        # 扩展 advantages 到每个 token
        advantages_expanded = advantages.unsqueeze(1).expand_as(current_log_probs)
        
        # 策略损失
        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded
        policy_loss = -torch.min(surr1, surr2) * gen_mask
        policy_loss = policy_loss.sum() / gen_mask.sum()
        
        # KL 散度（防止策略偏离参考模型太远）
        # KL = sum(pi_theta * (log pi_theta - log pi_ref))
        kl_div = (current_log_probs - ref_log_probs) * gen_mask
        kl_div = kl_div.sum() / gen_mask.sum()
        
        # 总损失
        total_loss = policy_loss + self.config.grpo_kl_coef * kl_div
        
        # 记录指标
        metrics = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'mean_reward': rewards.mean().item(),
            'std_reward': rewards.std().item()
        }
        
        return total_loss, metrics
    

    
    def train(self):
        """执行GRPO训练"""
        # 加载数据
        train_data = load_data(self.config.grpo_train_data_path)
        val_data = load_data(self.config.grpo_val_data_path)
        
        # 创建数据加载器
        train_loader = create_data_loader(
            train_data, 
            self.config.grpo_batch_size, 
            shuffle=True, 
            is_distributed=self.is_distributed, 
            collate_fn=self.tokenize_batch
        )
        val_loader = create_data_loader(
            val_data, 
            self.config.grpo_batch_size, 
            shuffle=False, 
            is_distributed=self.is_distributed, 
            collate_fn=self.tokenize_batch
        )
        
        # 配置训练参数
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.grpo_learning_rate,
            weight_decay=self.config.grpo_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.grpo_warmup_steps
        )
        
        # 训练循环
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.grpo_epochs):
            self._log(f"\nEpoch {epoch + 1}/{self.config.grpo_epochs}")
            
            # 训练阶段
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                # 移动到GPU
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 计算GRPO损失
                loss, metrics = self.grpo_loss(batch)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                global_step += 1
                
                # 日志 - 只在主进程输出
                if global_step % self.config.grpo_log_steps == 0 and (not self.is_distributed or self.local_rank == 0):
                    self._log(f"Step {global_step} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"Policy: {metrics['policy_loss']:.4f} | "
                          f"KL: {metrics['kl_div']:.4f} | "
                          f"Reward: {metrics['mean_reward']:.3f}±{metrics['std_reward']:.3f}")
                    # 记录训练指标
                    self.train_metrics['steps'].append(global_step)
                    self.train_metrics['loss'].append(metrics['loss'])
                    self.train_metrics['policy_loss'].append(metrics['policy_loss'])
                    self.train_metrics['kl_div'].append(metrics['kl_div'])
                    self.train_metrics['mean_reward'].append(metrics['mean_reward'])
                    self.train_metrics['std_reward'].append(metrics['std_reward'])
                    total_loss = 0
                
                # 保存模型 - 只在主进程执行
                if global_step % self.config.grpo_save_steps == 0 and (not self.is_distributed or self.local_rank == 0):
                    save_path = os.path.join(self.output_dir, f"checkpoint_{global_step}")
                    # 获取底层模型（处理DDP包装的情况）
                    model_to_save = get_model_to_save(self.model, self.is_distributed)
                    model_to_save.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    self._log(f"Model saved to {save_path}")
                
                # 评估 - 只在主进程执行
                if global_step % self.config.grpo_eval_steps == 0 and (not self.is_distributed or self.local_rank == 0):
                    val_loss = self.evaluate(val_loader)
                    self._log(f"Validation Loss: {val_loss:.4f}")
                    # 记录验证指标
                    self.val_metrics['steps'].append(global_step)
                    self.val_metrics['loss'].append(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_save_path = os.path.join(self.output_dir, "best_model")
                        # 获取底层模型（处理DDP包装的情况）
                        model_to_save = get_model_to_save(self.model, self.is_distributed)
                        model_to_save.save_pretrained(best_save_path)
                        self.tokenizer.save_pretrained(best_save_path)
                        self._log(f"Best model saved to {best_save_path}")
        
        # 保存最终模型 - 只在主进程执行
        if not self.is_distributed or self.local_rank == 0:
            final_save_path = os.path.join(self.output_dir, "final_model")
            # 获取底层模型（处理DDP包装的情况）
            model_to_save = get_model_to_save(self.model, self.is_distributed)
            model_to_save.save_pretrained(final_save_path)
            self.tokenizer.save_pretrained(final_save_path)
            self._log(f"Final model saved to {final_save_path}")
            
            # 生成并保存训练指标图表
            self._generate_plots()
    
    def _generate_plots(self):
        """生成训练指标图表"""
        # 创建图表保存目录
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 生成损失和策略损失图表
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_metrics['steps'], self.train_metrics['loss'], label='Total Loss')
        plt.plot(self.train_metrics['steps'], self.train_metrics['policy_loss'], label='Policy Loss')
        plt.plot(self.val_metrics['steps'], self.val_metrics['loss'], label='Validation Loss', linestyle='--')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('GRPO Training Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(plots_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        self._log(f"Loss plot saved to {loss_plot_path}")
        
        # 生成KL散度图表
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_metrics['steps'], self.train_metrics['kl_div'], label='KL Divergence')
        plt.xlabel('Steps')
        plt.ylabel('KL Divergence')
        plt.title('GRPO KL Divergence')
        plt.legend()
        plt.grid(True)
        kl_plot_path = os.path.join(plots_dir, 'kl_div_plot.png')
        plt.savefig(kl_plot_path)
        plt.close()
        self._log(f"KL divergence plot saved to {kl_plot_path}")
        
        # 生成奖励图表
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_metrics['steps'], self.train_metrics['mean_reward'], label='Mean Reward')
        plt.fill_between(
            self.train_metrics['steps'],
            np.array(self.train_metrics['mean_reward']) - np.array(self.train_metrics['std_reward']),
            np.array(self.train_metrics['mean_reward']) + np.array(self.train_metrics['std_reward']),
            alpha=0.2,
            label='Reward Std Dev'
        )
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('GRPO Reward')
        plt.legend()
        plt.grid(True)
        reward_plot_path = os.path.join(plots_dir, 'reward_plot.png')
        plt.savefig(reward_plot_path)
        plt.close()
        self._log(f"Reward plot saved to {reward_plot_path}")
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                loss, _ = self.grpo_loss(batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def _log(self, message):
        """记录日志到文件和控制台"""
        log_message(message, self.log_file)