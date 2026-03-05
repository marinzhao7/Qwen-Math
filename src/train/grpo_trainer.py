import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from ms_swift import Swift, LoraConfig, SwiftModel

class GRPOTrainer:
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
        self.tokenizer = AutoTokenizer.from_pretrained(config.grpo_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载策略模型（当前模型）
        self.model = self._load_model()
        
        # 配置LoRA
        if config.grpo_use_lora:
            self.lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none"
            )
            
            # 应用LoRA到策略模型
            self.model = SwiftModel(self.model, self.lora_config)   
        
        # 加载参考模型（冻结，用于 KL 散度）
        self.reference_model = self._load_reference_model()
        
        # 设置多GPU训练
        self._setup_distributed_training()
    
    def _setup_gpu(self):
        """设置GPU环境"""
        if self.config.grpo_gpu_ids is not None:
            # 指定GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.grpo_gpu_ids))
            print(f"Using specified GPUs: {self.config.grpo_gpu_ids}")
        
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def _load_model(self):
        """加载策略模型"""
        if self.config.grpo_use_multi_gpu and torch.cuda.device_count() > 1:
            # 多GPU训练，不使用device_map
            model = AutoModelForCausalLM.from_pretrained(
                self.config.grpo_model_name,
                torch_dtype=torch.bfloat16 if self.config.grpo_use_bf16 else torch.float16 if self.config.grpo_use_fp16 else torch.float32,
            )
            model = model.to(self.device)
            print(f"Policy model loaded on {torch.cuda.device_count()} GPUs")
        else:
            # 单GPU或CPU，使用device_map自动分配
            model = AutoModelForCausalLM.from_pretrained(
                self.config.grpo_model_name,
                torch_dtype=torch.bfloat16 if self.config.grpo_use_bf16 else torch.float16 if self.config.grpo_use_fp16 else torch.float32,
                device_map="auto" if not self.config.grpo_use_multi_gpu else None
            )
            if not self.config.grpo_use_multi_gpu:
                print("Policy model loaded with device_map='auto'")
        
        return model
    
    def _load_reference_model(self):
        """加载参考模型"""
        if self.config.grpo_use_multi_gpu and torch.cuda.device_count() > 1:
            # 多GPU训练
            model = AutoModelForCausalLM.from_pretrained(
                self.config.grpo_model_name,
                torch_dtype=torch.bfloat16 if self.config.grpo_use_bf16 else torch.float16 if self.config.grpo_use_fp16 else torch.float32,
            )
            model = model.to(self.device)
            print(f"Reference model loaded on {torch.cuda.device_count()} GPUs")
        else:
            # 单GPU或CPU
            model = AutoModelForCausalLM.from_pretrained(
                self.config.grpo_model_name,
                torch_dtype=torch.bfloat16 if self.config.grpo_use_bf16 else torch.float16 if self.config.grpo_use_fp16 else torch.float32,
                device_map="auto" if not self.config.grpo_use_multi_gpu else None
            )
            if not self.config.grpo_use_multi_gpu:
                print("Reference model loaded with device_map='auto'")
        
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        return model
    
    def _setup_distributed_training(self):
        """设置分布式训练"""
        self.is_distributed = False
        self.local_rank = -1
        
        if self.config.grpo_use_multi_gpu and torch.cuda.device_count() > 1:
            # 初始化分布式训练
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.local_rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
            else:
                self.local_rank = 0
                self.world_size = torch.cuda.device_count()
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12356'
            
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://')
            
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.local_rank)
            
            # 包装策略模型为DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.grpo_ddp_find_unused_parameters
            )
            
            self.is_distributed = True
            print(f"GRPO distributed training initialized: rank {self.local_rank}/{self.world_size}")
        elif torch.cuda.device_count() == 1:
            print("Single GPU training")
        else:
            print("CPU training")
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载训练数据"""
        data = []
        if data_path.endswith('.jsonl'):
            # 加载JSONL格式文件
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # 适配GSM8K格式
                    if 'question' in item and 'answer' in item:
                        data.append(item)
                    elif 'problem' in item and 'solution' in item:
                        # 适配MATH格式
                        data.append({
                            'question': item['problem'],
                            'answer': item['solution']
                        })
        elif data_path.endswith('.json'):
            # 加载JSON格式文件
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data
    
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
        old_log_probs = self._compute_log_probs(
            generated_ids, 
            attention_mask_expanded,
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
    
    def create_data_loader(self, data: List[Dict[str, Any]], batch_size: int, shuffle: bool = True):
        """创建数据加载器"""
        class MathDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = MathDataset(data)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenize_batch,
            num_workers=0
        )
    
    def train(self):
        """执行GRPO训练"""
        # 加载数据
        train_data = self.load_data(self.config.grpo_train_data_path)
        val_data = self.load_data(self.config.grpo_val_data_path)
        
        # 创建数据加载器
        train_loader = self.create_data_loader(train_data, self.config.grpo_batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_data, self.config.grpo_batch_size, shuffle=False)
        
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
            print(f"\nEpoch {epoch + 1}/{self.config.grpo_epochs}")
            
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
                
                # 日志
                if global_step % self.config.grpo_log_steps == 0:
                    print(f"Step {global_step} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"Policy: {metrics['policy_loss']:.4f} | "
                          f"KL: {metrics['kl_div']:.4f} | "
                          f"Reward: {metrics['mean_reward']:.3f}±{metrics['std_reward']:.3f}")
                    total_loss = 0
                
                # 保存模型
                if global_step % self.config.grpo_save_steps == 0:
                    save_path = os.path.join(self.output_dir, f"checkpoint_{global_step}")
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    print(f"Model saved to {save_path}")
                
                # 评估
                if global_step % self.config.grpo_eval_steps == 0:
                    val_loss = self.evaluate(val_loader)
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_save_path = os.path.join(self.output_dir, "best_model")
                        self.model.save_pretrained(best_save_path)
                        self.tokenizer.save_pretrained(best_save_path)
                        print(f"Best model saved to {best_save_path}")
        
        # 保存最终模型
        final_save_path = os.path.join(self.output_dir, "final_model")
        self.model.save_pretrained(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)
        print(f"Final model saved to {final_save_path}")
    
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