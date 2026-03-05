import os
import json
import requests
from tqdm import tqdm

class DataUtils:
    @staticmethod
    def download_file(url: str, save_path: str):
        """下载文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    
    @staticmethod
    def convert_jsonl_to_json(jsonl_path: str, json_path: str):
        """将jsonl文件转换为json文件"""
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def sample_data(input_path: str, output_path: str, sample_size: int):
        """从数据集中采样指定数量的样本"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        import random
        sampled_data = random.sample(data, min(sample_size, len(data)))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def validate_data(data_path: str):
        """验证数据格式"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                required_fields = ['question', 'answer']
                for field in required_fields:
                    if field not in item:
                        print(f"Error: Item {i} missing field: {field}")
                        return False
                
                if not isinstance(item['question'], str) or not isinstance(item['answer'], str):
                    print(f"Error: Item {i} has invalid field types")
                    return False
            
            print(f"Validation successful: {len(data)} items")
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False