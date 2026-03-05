import os
import json
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from ms_swift import Swift, LoraConfig, SwiftModel

class SFTTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.sft_output_dir
        self.log_dir = config.sft_log_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置GPU
        self._setup_gpu()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = self._load_model()
        
        # 配置LoRA
        self.lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none"
        )
        
        # 应用LoRA
        self.model = SwiftModel(self.model, self.lora_config)
        
        # 设置多GPU训练
        self._setup_distributed_training()
    
    def _setup_gpu(self):
        """设置GPU环境"""
        if self.config.sft_gpu_ids is not None:
            # 指定GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.sft_gpu_ids))
            print(f"Using specified GPUs: {self.config.sft_gpu_ids}")
        
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def _load_model(self):
        """加载模型"""
        # 判断是否使用多GPU
        if self.config.sft_use_multi_gpu and torch.cuda.device_count() > 1:
            # 多GPU训练，不使用device_map
            model = AutoModelForCausalLM.from_pretrained(
                self.config.sft_model_name,
                torch_dtype=torch.float16 if self.config.sft_use_fp16 else torch.bfloat16 if self.config.sft_use_bf16 else torch.float32,
            )
            model = model.to(self.device)
            print(f"Model loaded on {torch.cuda.device_count()} GPUs")
        else:
            # 单GPU或CPU，使用device_map自动分配
            model = AutoModelForCausalLM.from_pretrained(
                self.config.sft_model_name,
                torch_dtype=torch.float16 if self.config.sft_use_fp16 else torch.bfloat16 if self.config.sft_use_bf16 else torch.float32,
                device_map="auto" if not self.config.sft_use_multi_gpu else None
            )
            if not self.config.sft_use_multi_gpu:
                print("Model loaded with device_map='auto'")
        
        return model
    
    def _setup_distributed_training(self):
        """设置分布式训练"""
        self.is_distributed = False
        self.local_rank = -1
        
        if self.config.sft_use_multi_gpu and torch.cuda.device_count() > 1:
            # 初始化分布式训练
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.local_rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
            else:
                self.local_rank = 0
                self.world_size = torch.cuda.device_count()
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
            
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://')
            
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.local_rank)
            
            # 包装模型为DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.sft_ddp_find_unused_parameters
            )
            
            self.is_distributed = True
            print(f"Distributed training initialized: rank {self.local_rank}/{self.world_size}")
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
    
    def create_data_loader(self, data: List[Dict[str, Any]], batch_size: int, shuffle: bool = True):
        """创建数据加载器"""
        from torch.utils.data import DataLoader, Dataset
        
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
            collate_fn=self.tokenize_batch
        )
    
    def train(self):
        """执行SFT训练"""
        # 加载数据
        train_data = self.load_data(self.config.sft_train_data_path)
        val_data = self.load_data(self.config.sft_val_data_path)
        
        # 创建数据加载器
        train_loader = self.create_data_loader(train_data, self.config.sft_batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_data, self.config.sft_batch_size, shuffle=False)
        
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
            print(f"Epoch {epoch + 1}/{self.config.sft_epochs}")
            
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
                
                # 日志
                if global_step % self.config.sft_log_steps == 0:
                    avg_loss = total_loss / self.config.sft_log_steps
                    print(f"Step {global_step}, Loss: {avg_loss:.4f}")
                    total_loss = 0
                
                # 保存模型
                if global_step % self.config.sft_save_steps == 0:
                    save_path = os.path.join(self.output_dir, f"checkpoint_{global_step}")
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    print(f"Model saved to {save_path}")
                
                # 评估
                if global_step % self.config.sft_eval_steps == 0:
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
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss