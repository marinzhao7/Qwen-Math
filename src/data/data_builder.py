import json
from logging import config
import os
from typing import List, Dict, Any

class DataBuilder:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        
    def load_gsm8k(self, file_path: str) -> List[Dict[str, Any]]:
        """加载GSM8K数据集，根据解题步骤数进行难度分级，并根据难度平均采样"""
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        answer = item.get('answer', '')
                        
                        # 计算解题步骤数
                        steps = self._count_gsm8k_steps(answer)
                        
                        # 根据步骤数确定难度等级
                        difficulty = self._classify_gsm8k_difficulty(steps)
                        
                        data.append({
                            'question': item.get('question', ''),
                            'answer': answer,
                            'steps': steps,
                            'difficulty': difficulty,
                            'source': 'gsm8k'
                        })
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                        continue
                    except KeyError as e:
                        print(f"Warning: Missing key {e} in line {line_num} of {file_path}")
                        continue
        
        # 根据难度平均采样，确保各难度等级的分布均匀
        sampled_data = self._sample_by_difficulty(data, sample_size=300)
        return sampled_data
    
    def _count_gsm8k_steps(self, answer: str) -> int:
        """计算GSM8K题目的解题步骤数"""
        if not answer:
            return 1
        
        # 方法1：计算 <<...>> 计算标记的数量
        import re
        calculation_steps = len(re.findall(r'<<.*?>>', answer))
        
        # 方法2：计算换行符数量（每个换行通常代表一个步骤）
        newline_steps = answer.count('\n')
        
        # 取两种方法的最大值，确保步骤数准确
        steps = max(calculation_steps, newline_steps)
        
        # 确保至少有1步
        return max(steps, 1)
    
    def _classify_gsm8k_difficulty(self, steps: int) -> str:
        """根据步骤数分类GSM8K题目难度"""
        if steps <= 3:
            return 'easy'
        elif steps <= 5:
            return 'medium'
        else:
            return 'hard'
    
    def load_math(self, file_path: str) -> List[Dict[str, Any]]:
        """加载MATH500数据集，根据level和subject平均采样"""
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        # 合并solution和answer部分
                        answer = item.get('solution', '') + '\n' + item.get('answer', '')
                        data.append({
                            'question': item.get('problem', ''),
                            'answer': answer,
                            'level': item.get('level', ''),
                            'subject': item.get('subject', ''),
                            'source': 'math500'
                        })
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                        continue
                    except KeyError as e:
                        print(f"Warning: Missing key {e} in line {line_num} of {file_path}")
                        continue
        
        # 根据level和subject平均采样，确保各层级和主题的分布均匀
        sampled_data = self._sample_by_level_and_subject(data, sample_size=300)
        return sampled_data    
    
    def _sample_by_difficulty(self, data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """根据难度平均采样GSM8K数据"""
        return self._sample_by_key(data, sample_size, key_func=lambda x: x.get('difficulty', 'unknown'))
    
    def _sample_by_level_and_subject(self, data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """根据level和subject平均采样数据"""
        return self._sample_by_key(data, sample_size, key_func=lambda x: (x.get('level', 'unknown'), x.get('subject', 'unknown')))
    
    def _sample_by_key(self, data: List[Dict[str, Any]], sample_size: int, key_func) -> List[Dict[str, Any]]:
        """根据指定的键函数进行分组采样"""
        import random
        
        if not data:
            return []
        
        # 按指定的键分组
        groups = {}
        for item in data:
            key = key_func(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # 计算每组应采样的数量
        num_groups = len(groups)
        if num_groups == 0:
            return []
        samples_per_group = sample_size // num_groups
        remaining_samples = sample_size % num_groups
        
        # 采样
        sampled_data = []
        for key, items in groups.items():
            sample_count = samples_per_group + (1 if remaining_samples > 0 else 0)
            remaining_samples -= 1
            
            if len(items) >= sample_count:
                sampled_data.extend(random.sample(items, sample_count))
            else:
                sampled_data.extend(items)
        
        # 确保总样本数不超过sample_size
        if len(sampled_data) > sample_size:
            sampled_data = random.sample(sampled_data, sample_size)
        
        return sampled_data
    
    def filter_quality_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """筛选高质量数据"""
        filtered_data = []
        for item in data:
            # 检查问题长度
            if len(item['question']) < 30 or len(item['question']) > 1000:
                continue
            # 检查答案长度
            if len(item['answer']) < 50 or len(item['answer']) > 2000:
                continue
            filtered_data.append(item)
        return filtered_data
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """预处理数据"""
        processed_data = []
        for item in data:
            # 格式化问题
            question = item['question'].strip()
            if not question.endswith('?'):
                question += '?'
            
            # 格式化答案
            answer = item['answer'].strip()
            
            processed_data.append({
                'question': question,
                'answer': answer,
                'source': item.get('source', 'others')
            })
        return processed_data
    
    def split_data(self, data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """划分数据集"""
        import random
        random.shuffle(data)
        
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def build_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """构建完整数据集"""
        # 加载所有数据
        gsm8k_data = self.load_gsm8k(self.config.gsm8k_train_path)
        math_data = self.load_math(self.config.math_train_path)
        
        # 分别处理GSM8K和MATH数据
        if self.config.use_filtered_data:
            processed_gsm8k = self.filter_quality_data(gsm8k_data)
            processed_math = self.filter_quality_data(math_data)
        else:
            processed_gsm8k = self.preprocess_data(gsm8k_data)
            processed_math = self.preprocess_data(math_data)
        
        # 划分数据集
        gsm8k_split = self.split_data(processed_gsm8k)
        math_split = self.split_data(processed_math)
        
        # 保存处理后的数据
        self.save_processed_data(gsm8k_split, prefix="gsm8k_")
        self.save_processed_data(math_split, prefix="math_")
        
        # 合并数据用于通用训练
        all_processed = processed_gsm8k + processed_math
        all_split = self.split_data(all_processed)
        self.save_processed_data(all_split)
        
        return {
            'gsm8k': gsm8k_split,
            'math': math_split,
            'all': all_split
        }
    
    def save_processed_data(self, split_data: Dict[str, List[Dict[str, Any]]], prefix: str = ""):
        """保存处理后的数据"""
        output_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        for split, data in split_data.items():
            file_path = os.path.join(output_dir, f'{prefix}{split}.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)