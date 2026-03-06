import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM
from swift import SwiftModel, LoraConfig
from typing import List, Dict, Any, Tuple


def setup_gpu(gpu_ids=None):
    """设置GPU环境"""
    if gpu_ids is not None:
        # 指定GPU
        torch.cuda.set_device(gpu_ids[0])
        print(f"Using specified GPUs: {gpu_ids}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    return device


def load_model(model_name, use_multi_gpu, device, use_fp16=False, use_bf16=False):
    """加载模型"""
    if use_multi_gpu and torch.cuda.device_count() > 1:
        # 多GPU训练，不使用device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.bfloat16 if use_bf16 else torch.float32,
        )
        model = model.to(device)
        print(f"Model loaded on {torch.cuda.device_count()} GPUs")
    else:
        # 单GPU训练，显式指定GPU设备
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.bfloat16 if use_bf16 else torch.float32,
        )
        model = model.to(device)
        print(f"Model loaded on GPU: {device}")
    
    return model


def setup_distributed_training(use_multi_gpu, gpu_ids=None, ddp_find_unused_parameters=False):
    """设置分布式训练"""
    is_distributed = False
    local_rank = -1
    world_size = 1
    
    if use_multi_gpu and torch.cuda.device_count() > 1:
        # 初始化分布式训练
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            local_rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            local_rank = 0
            world_size = torch.cuda.device_count()
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        
        is_distributed = True
        print(f"Distributed training initialized: rank {local_rank}/{world_size}")
    elif torch.cuda.device_count() == 1 or len(gpu_ids) == 1:
        print("Single GPU training")
    else:
        print("CPU training")
    
    return is_distributed, local_rank, world_size


def apply_lora(model, config, use_lora):
    """应用LoRA配置"""
    if use_lora:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none"
        )
        
        # 应用LoRA
        model = SwiftModel(model, lora_config)
        print("LoRA applied successfully")
    else:
        print("LoRA not enabled, using full parameter training")
    
    return model


def create_data_loader(data, batch_size, shuffle=True, is_distributed=False, collate_fn=None):
    """创建数据加载器"""
    class MathDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = MathDataset(data)
    
    # 在分布式训练中使用DistributedSampler
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )


def get_model_to_save(model, is_distributed):
    """获取底层模型（处理DDP包装的情况）"""
    return model.module if is_distributed else model


def log_message(message, log_file=None):
    """记录日志到文件和控制台"""
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """加载训练数据"""
    import json
    from typing import List, Dict, Any
    
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
