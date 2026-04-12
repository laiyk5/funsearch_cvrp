# LLM配置文件

import os

# OpenAI API配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "DASHSCOPE_API_KEY")  # 优先从环境变量读取
OPENAI_MODEL = "qwen3-max-2025-09-23"  # 使用的模型

# LLM生成配置
LLM_TEMPERATURE = 0.7  # 生成温度，控制创造性
LLM_MAX_TOKENS = 2000  # 最大生成 tokens

# 迭代搜索配置
N_ITERATIONS = 10  # 迭代次数
MIN_HEURISTICS_PER_ITER = 50  # 每轮最小生成算法数
MAX_HEURISTICS_PER_ITER = 100  # 每轮最大生成算法数

# 数据集配置
DATASET_SIZES = [20, 50, 100]  # 客户数量
DATASET_SEED = 2026  # 随机种子

# 评估配置
EARLY_PRUNING_THRESHOLD = 1.5  # 早期减枝阈值
LARGE_SCALE_THRESHOLD = 4.0  # 大规模分数阈值