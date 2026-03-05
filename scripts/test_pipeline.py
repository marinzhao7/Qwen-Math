import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.base_config import BaseConfig
from src.data.data_builder import DataBuilder
from src.data.data_utils import DataUtils

def test_data_building():
    """测试数据构建功能"""
    print("Testing data building...")
    
    # 初始化配置
    config = BaseConfig()
    
    # 创建数据目录
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, 'custom'), exist_ok=True)
    
    # 创建测试数据
    custom_path = os.path.join(config.data_dir, 'custom', 'test_competition.json')
    DataUtils.create_custom_competition_data(custom_path, count=10)
    
    # 测试数据加载
    data_builder = DataBuilder(config)
    data = data_builder.load_custom_competition(custom_path)
    
    print(f"Loaded {len(data)} test data items")
    print("Data building test passed!")

def test_model_loading():
    """测试模型加载功能"""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 测试加载基础模型
        model_name = "Qwen/Qwen-3B-Thinking"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
        
        print("Model and tokenizer loaded successfully!")
        print("Model loading test passed!")
    except Exception as e:
        print(f"Model loading test failed: {e}")

def test_pipeline():
    """测试整个 pipeline"""
    print("\nTesting pipeline functionality...")
    
    # 测试数据构建
    test_data_building()
    
    # 测试模型加载
    test_model_loading()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_pipeline()