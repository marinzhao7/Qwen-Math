import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from src.configs.base_config import BaseConfig
from src.data.data_builder import DataBuilder
from src.data.data_utils import DataUtils

def main():
    # 初始化配置
    config = BaseConfig()
    
    # 创建数据目录
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, 'gsm8k'), exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, 'math'), exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, 'custom'), exist_ok=True)
    
    # 下载GSM8K数据集
    print("Downloading GSM8K dataset...")
    gsm8k_path = config.gsm8k_train_path
    if not os.path.exists(gsm8k_path):
        # 使用datasets库下载GSM8K数据集
        ds = load_dataset("openai/gsm8k", "main")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(gsm8k_path), exist_ok=True)
        
        # 保存为JSONL格式
        with open(gsm8k_path, 'w', encoding='utf-8') as f:
            for item in ds['train']:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"GSM8K dataset saved to {gsm8k_path}")
    else:
        print(f"GSM8K dataset already exists at {gsm8k_path}")
    
    # 下载MATH500数据集
    print("Downloading MATH500 dataset...")
    math_path = config.math_train_path
    if not os.path.exists(math_path):
        # 使用datasets库下载MATH-500数据集
        ds = load_dataset("HuggingFaceH4/MATH-500", "main")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(math_path), exist_ok=True)
        
        # 保存为JSONL格式
        with open(math_path, 'w', encoding='utf-8') as f:
            for item in ds['train']:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"MATH500 dataset saved to {math_path}")
    else:
        print(f"MATH500 dataset already exists at {math_path}")
    
    # 跳过创建自定义竞赛级数据
    print("Skipping custom competition data creation...")
    
    # 构建数据集
    print("Building dataset...")
    data_builder = DataBuilder(config)
    split_data = data_builder.build_dataset()
    
    # 验证数据
    print("Validating data...")
    # 验证合并后的数据集
    if 'all' in split_data:
        print("All dataset:")
        for split, data in split_data['all'].items():
            print(f"  {split}: {len(data)} items")
    # 验证GSM8K数据集
    if 'gsm8k' in split_data:
        print("GSM8K dataset:")
        for split, data in split_data['gsm8k'].items():
            print(f"  {split}: {len(data)} items")
    # 验证MATH数据集
    if 'math' in split_data:
        print("MATH dataset:")
        for split, data in split_data['math'].items():
            print(f"  {split}: {len(data)} items")
    
    # 验证处理后的数据
    processed_dir = os.path.join(config.data_dir, 'processed')
    # 验证合并后的数据集
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(processed_dir, f'{split}.json')
        if os.path.exists(file_path):
            print(f"Validating {split} data...")
            DataUtils.validate_data(file_path)
    # 验证GSM8K数据集
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(processed_dir, f'gsm8k_{split}.json')
        if os.path.exists(file_path):
            print(f"Validating gsm8k_{split} data...")
            DataUtils.validate_data(file_path)
    # 验证MATH数据集
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(processed_dir, f'math_{split}.json')
        if os.path.exists(file_path):
            print(f"Validating math_{split} data...")
            DataUtils.validate_data(file_path)
    
    print("Data building completed successfully!")

if __name__ == "__main__":
    main()