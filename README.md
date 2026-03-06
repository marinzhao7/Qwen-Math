# Qwen-Math 项目

## 项目简介

Qwen-Math 是一个基于 Qwen3 Base 模型的数学推理能力微调项目。本项目通过 SFT（监督微调）和 GRPO（强化学习）两个阶段，提升模型在数学推理任务上的表现。

## 项目结构

```
Qwen-Math/
├── data/               # 数据集目录
├── src/                # 源代码目录
│   ├── configs/        # 配置文件
│   ├── data/           # 数据处理模块
│   ├── train/          # 训练模块
│   └── eval/           # 评估模块
├── scripts/            # 脚本目录
├── logs/               # 日志目录
├── output/             # 输出目录
├── requirements.txt    # 依赖文件
└── README.md           # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (推荐)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

项目使用以下数据集：

1. **GSM8K**：小学数学问题数据集，用于SFT训练
2. **MATH500**：高中数学竞赛题数据集，用于GRPO训练

数据处理后保存在 `data/processed/` 目录下，包含以下文件：
- `train.json` / `val.json` / `test.json`：合并数据集
- `gsm8k_train.json` / `gsm8k_val.json` / `gsm8k_test.json`：GSM8K数据集
- `math_train.json` / `math_val.json` / `math_test.json`：MATH数据集

## 使用方法

### 1. 数据构建

```bash
python scripts/build_data.py
```

此脚本会：
- 下载 GSM8K 和 MATH 数据集
- 筛选和预处理数据
- 划分训练、验证和测试集

### 2. SFT 监督微调

```bash
python scripts/run_sft.py
```

使用 ms-swift 框架和 LoRA 技术进行监督微调。

### 3. GRPO 强化训练

```bash
python scripts/run_grpo.py
```

使用 GRPO 算法进行强化训练，进一步提升模型性能。

### 4. 模型评估

```bash
python scripts/run_eval.py
```

在 GSM8K 和 MATH 数据集上评估模型性能。

### 5. 一键运行完整流程

```bash
python scripts/run_full_pipeline.py
```

此脚本会按顺序执行数据构建、SFT训练、GRPO训练和模型评估。

## 配置说明

项目的主要配置文件位于 `src/configs/` 目录：

- `base_config.py`：基础配置
- `sft_config.py`：SFT 训练配置
- `grpo_config.py`：GRPO 训练配置
- `eval_config.py`：评估配置

## 模型输出

- SFT 模型：`output/sft/`
- GRPO 模型：`output/grpo/`
- 评估结果：`output/eval/`

## 评估指标

- **准确率**：模型回答正确的比例
- **精确匹配**：模型回答与标准答案完全匹配的比例

## GPU配置与多GPU训练

项目支持灵活的GPU配置：

### 单GPU训练（默认，稳定）
```python
# 使用默认GPU
sft_use_multi_gpu = False

# 或指定特定GPU
sft_gpu_ids = [0]  # 使用第0块GPU
```

### 多GPU训练
```python
# 启用多GPU训练
sft_use_multi_gpu = True
sft_gpu_ids = [0, 1, 2, 3]  # 使用第0,1,2,3块GPU
```

### 使用torchrun启动多GPU训练
```bash
# 使用4块GPU进行SFT训练
torchrun --nproc_per_node=4 scripts/run_sft.py

# 使用2块GPU进行GRPO训练
torchrun --nproc_per_node=2 scripts/run_grpo.py
```

### 配置参数说明

在 `src/configs/sft_config.py` 和 `src/configs/grpo_config.py` 中：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `sft_use_multi_gpu` / `grpo_use_multi_gpu` | 是否启用多GPU训练 | `False` |
| `sft_gpu_ids` / `grpo_gpu_ids` | 指定使用的GPU ID列表 | `None`（使用所有可用GPU） |
| `sft_ddp_find_unused_parameters` / `grpo_ddp_find_unused_parameters` | DDP是否查找未使用参数 | `False` |

## GRPO训练器选择

项目支持两种GRPO训练器实现：

### 1. 自定义GRPO实现（默认）
```bash
# 使用自定义GRPO训练器
python scripts/run_grpo.py
```

### 2. ms-swift GRPO实现
```bash
# 使用ms-swift的GRPO训练器
python scripts/run_grpo.py --use-swift

# 指定模型路径
python scripts/run_grpo.py --use-swift --model-path output/sft/final_model
```

### 两种实现的区别

| 特性 | 自定义实现 | ms-swift实现 |
|------|-----------|-------------|
| 依赖 | 仅PyTorch | ms-swift框架 |
| 灵活性 | 高，易于修改 | 中等，遵循框架规范 |
| 功能完整性 | 基础GRPO | 完整的RLHF功能 |
| 推荐场景 | 研究/调试 | 生产环境 |

## 注意事项

1. 训练过程需要大量计算资源，推荐使用 GPU 进行训练
2. 首次运行会下载模型和数据集，可能需要较长时间
3. 可以根据硬件条件调整 batch_size 和其他训练参数
4. 多GPU训练时，建议适当增大 batch_size 以充分利用显存
5. 使用ms-swift实现时，确保已安装ms-swift包：`pip install ms-swift`

## 项目特点

- 使用 LoRA 技术减少内存占用
- 采用 GRPO 算法进行强化学习
- 支持自定义和ms-swift两种GRPO实现
- 支持多数据集评估
- 模块化设计，易于扩展

## 参考资料

- [Qwen 模型](https://github.com/QwenLM/Qwen)
- [ms-swift 框架](https://github.com/modelscope/swift)
- [OpenCompass 评估框架](https://github.com/open-compass/opencompass)
- [GRPO算法论文](https://arxiv.org/pdf/2402.03300)