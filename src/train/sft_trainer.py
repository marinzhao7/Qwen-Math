import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from swift import Swift, LoraConfig, SwiftModel

from .trainer_utils import (
    setup_gpu, load_model, setup_distributed_training, 
    apply_lora, create_data_loader, get_model_to_save, log_message, load_data
)

class SFTTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.sft_output_dir
        self.log_dir = config.sft_log_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志文件
        self.log_file = os.path.join(self.log_dir, f"sft_training_{int(time.time())}.log")
        
        # 初始化数据收集
        self.train_metrics = {
            'steps': [],
            'loss': []
        }
        self.val_metrics = {
            'steps': [],
            'loss': []
        }
        
        # 设置GPU
        self.device = setup_gpu(self.config.sft_gpu_ids)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = load_model(
            self.config.sft_model_name,
            self.config.sft_use_multi_gpu,
            self.device,
            self.config.sft_use_fp16,
            self.config.sft_use_bf16
        )
        
        # 配置LoRA
        self.model = apply_lora(self.model, self.config, self.config.sft_use_lora)
        
        # 设置多GPU训练
        self.is_distributed, self.local_rank, self.world_size = setup_distributed_training(
            self.config.sft_use_multi_gpu,
            self.config.sft_gpu_ids,
            self.config.sft_ddp_find_unused_parameters
        )
        
        if self.is_distributed:
            # 包装模型为DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.sft_ddp_find_unused_parameters
            )
        

    
    def tokenize_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """对批次数据进行分词 - 只学习生成答案部分"""
        full_texts = []
        prompt_texts = []
        
        for item in batch:
            prompt = f"Question: {item['question']}\nAnswer:"
            full_text = f"{prompt}{item['answer']}"
            
            full_texts.append(full_text)
            prompt_texts.append(prompt)
        
        # 分词完整文本（问题+答案）
        tokenized = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.sft_max_length,
            return_tensors="pt"
        )
        
        # 分词问题部分（用于确定mask边界）
        prompt_tokenized = self.tokenizer(
            prompt_texts,
            padding=False,  # 不padding，用长度计算
            truncation=True,
            max_length=self.config.sft_max_length
        )
        
        # 构建标签：问题部分设为-100，只学习答案
        labels = tokenized.input_ids.clone()
        
        for i, prompt_len in enumerate([len(t) for t in prompt_tokenized.input_ids]):
            # 将问题部分（包括padding前的问题token）设为-100
            labels[i, :prompt_len] = -100
        
        # 同时mask掉所有的pad_token
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': tokenized.input_ids,
            'attention_mask': tokenized.attention_mask,
            'labels': labels
        }
    

    
    def train(self):
        """执行SFT训练"""
        # 加载数据
        train_data = load_data(self.config.sft_train_data_path)
        val_data = load_data(self.config.sft_val_data_path)
        
        # 创建数据加载器
        train_loader = create_data_loader(
            train_data, 
            self.config.sft_batch_size, 
            shuffle=True, 
            is_distributed=self.is_distributed, 
            collate_fn=self.tokenize_batch
        )
        val_loader = create_data_loader(
            val_data, 
            self.config.sft_batch_size, 
            shuffle=False, 
            is_distributed=self.is_distributed, 
            collate_fn=self.tokenize_batch
        )
        
        # 配置训练参数
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.sft_learning_rate,
            weight_decay=self.config.sft_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.sft_warmup_steps
        )
        
        # 训练循环
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.sft_epochs):

            self._log(f"Epoch {epoch + 1}/{self.config.sft_epochs}")
            
            # 训练阶段
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                # 移动到GPU
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                global_step += 1
                
                # 日志 - 只在主进程输出
                if global_step % self.config.sft_log_steps == 0 and (not self.is_distributed or self.local_rank == 0):
                    avg_loss = total_loss / self.config.sft_log_steps
                    self._log(f"Step {global_step}, Loss: {avg_loss:.4f}")
                    # 记录训练指标
                    self.train_metrics['steps'].append(global_step)
                    self.train_metrics['loss'].append(avg_loss)
                    total_loss = 0
                
                # 保存模型 - 只在主进程执行
                if global_step % self.config.sft_save_steps == 0 and (not self.is_distributed or self.local_rank == 0):
                    save_path = os.path.join(self.output_dir, f"checkpoint_{global_step}")
                    # 获取底层模型（处理DDP包装的情况）
                    model_to_save = get_model_to_save(self.model, self.is_distributed)
                    model_to_save.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    self._log(f"Model saved to {save_path}")
                
                # 评估 - 只在主进程执行
                if global_step % self.config.sft_eval_steps == 0 and (not self.is_distributed or self.local_rank == 0):
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
        
        # 生成损失图表
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_metrics['steps'], self.train_metrics['loss'], label='Training Loss')
        plt.plot(self.val_metrics['steps'], self.val_metrics['loss'], label='Validation Loss', linestyle='--')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('SFT Training Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(plots_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        self._log(f"Loss plot saved to {loss_plot_path}")
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def _log(self, message):
        """记录日志到文件和控制台"""
        log_message(message, self.log_file)