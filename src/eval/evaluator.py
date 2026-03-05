import os
import json
import torch
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.eval_output_dir
        self.log_dir = config.eval_log_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.eval_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.eval_model_name,
            torch_dtype=torch.float16
        )
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_test_data(self, dataset_name: str) -> List[Dict[str, Any]]:
        """加载测试数据"""
        data_path = self.config.opencompass_config["datasets"][dataset_name]["path"]
        
        if not os.path.exists(data_path):
            print(f"Test data not found at {data_path}")
            return []
        
        data = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # 适配不同数据集格式
                    if 'question' in item and 'answer' in item:
                        data.append(item)
                    elif 'problem' in item and 'solution' in item:
                        # 适配MATH格式
                        data.append({
                            'question': item['problem'],
                            'answer': item['solution']
                        })
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                for item in loaded_data:
                    # 适配不同数据集格式
                    if 'question' in item and 'answer' in item:
                        data.append(item)
                    elif 'problem' in item and 'solution' in item:
                        # 适配MATH格式
                        data.append({
                            'question': item['problem'],
                            'answer': item['solution']
                        })
        
        return data
    
    def generate_answer(self, question: str) -> str:
        """生成答案"""
        prompt = f"Question: {question}\nAnswer:"
        
        # 分词
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.eval_max_new_tokens,
            temperature=self.config.eval_temperature,
            top_p=self.config.eval_top_p,
            do_sample=True
        )
        
        # 解码
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案部分
        if "Answer:" in answer:
            answer = answer.split("Answer:")[1].strip()
        
        return answer
    
    def evaluate_gsm8k(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估GSM8K数据集"""
        correct = 0
        total = len(data)
        
        for item in data:
            question = item['question']
            ground_truth = item['answer']
            
            # 生成答案
            generated_answer = self.generate_answer(question)
            
            # 简单的答案匹配（实际项目中可能需要更复杂的评估逻辑）
            if ground_truth in generated_answer:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def evaluate_math(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估MATH数据集"""
        correct = 0
        total = len(data)
        
        for item in data:
            question = item['question']
            ground_truth = item['answer']
            
            # 生成答案
            generated_answer = self.generate_answer(question)
            
            # 简单的答案匹配
            if ground_truth in generated_answer:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def evaluate(self):
        """执行评估"""
        results = {}
        
        for dataset in self.config.eval_datasets:
            print(f"Evaluating {dataset}...")
            
            # 加载测试数据
            data = self.load_test_data(dataset)
            
            if not data:
                print(f"No test data found for {dataset}")
                results[dataset] = {"accuracy": 0, "correct": 0, "total": 0}
                continue
            
            # 执行评估
            if dataset == "gsm8k":
                result = self.evaluate_gsm8k(data)
            elif dataset == "math":
                result = self.evaluate_math(data)
            else:
                result = {"accuracy": 0, "correct": 0, "total": 0}
            
            results[dataset] = result
            print(f"{dataset} evaluation result: {result}")
        
        # 保存评估结果
        result_path = os.path.join(self.output_dir, "eval_results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation results saved to {result_path}")
        return results