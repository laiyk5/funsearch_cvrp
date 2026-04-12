# FUN Search项目说明文档

## 项目概述

本项目实现了基于Fun Search的样本高效算法，用于解决容量车辆路径问题（CVRP）。通过优化权重参数来提高贪心算法的性能，并与传统基线算法进行比较，展示了Fun Search在组合优化问题中的有效性。

## 文件功能清单

### 核心文件
| 文件名 | 功能描述 |
|-------|---------|
| `cvrp_core.py` | CVRP数据模型、距离计算工具、合成基准生成、基线贪心求解器 |
| `sample_efficient_search.py` | 样本高效搜索算法，包含早期剪枝和功能去重 |
| `baselines.py` | 额外基线算法（Clarke-Wright Savings）和2-opt局部改进包装器 |
| `cvrplib_io.py` | CVRPLib `.vrp`文件加载器 |

### 运行脚本
| 文件名 | 功能描述 |
|-------|---------|
| `run_milestone.py` | 原始里程碑实验运行器 |
| `run_full_project.py` | 完整基准运行器，包含方法排名和JSON输出 |

### 报告生成
| 文件名 | 功能描述 |
|-------|---------|
| `generate_report.py` | 里程碑Markdown报告生成 |
| `generate_report_detailed.py` | 扩展里程碑Markdown报告生成 |
| `generate_final_report.py` | 从完整输出生成最终项目Markdown报告 |
| `export_report_docx.py` | Word报告导出器 |
| `export_report_detailed.py` | 详细Word报告导出器 |

### 测试
| 文件名 | 功能描述 |
|-------|---------|
| `tests/test_project.py` | 核心管道的冒烟测试 |

### 输出
| 文件名 | 功能描述 |
|-------|---------|
| `outputs/milestone_results.json` | 里程碑实验结果 |
| `outputs/search_history.json` | 里程碑搜索历史 |
| `outputs/full_project_results.json` | 完整项目结果 |
| `outputs/full_search_history.json` | 完整项目搜索历史 |

## 已实现功能

### 1. CVRP数据模型与工具
- 定义了`CVRPInstance`数据类，包含名称、容量、需求和坐标
- 实现了欧几里得距离计算
- 提供了路径距离和解决方案距离计算
- 支持合成基准生成，可生成不同规模的CVRP实例

### 2. 基线算法
- **最近邻启发式**：基于距离的贪心算法
- **Clarke-Wright Savings启发式**：基于节约的路径合并算法
- **2-opt局部改进**：对生成的路径进行优化
- **带权重的贪心启发式**：可自定义权重的贪心算法

### 3. 样本高效搜索算法
- 实现了基于权重优化的样本高效搜索
- 包含早期剪枝策略，减少计算开销
- 实现了功能去重，避免重复评估
- 支持配置搜索参数，如初始种群、迭代次数、突变标准差等

### 4. CVRPLib支持
- 支持加载CVRPLib格式的`.vrp`文件
- 可批量加载文件夹中的多个实例
- 自动处理 depot 节点和客户节点的重新索引

### 5. 完整实验流程
- 支持合成数据集和CVRPLib数据集
- 自动运行多种算法并进行比较
- 生成详细的结果报告和搜索历史
- 提供方法排名和最佳方法推荐

### 6. 报告生成
- 生成Markdown格式的实验报告
- 支持导出Word格式的报告
- 包含详细的实验结果和分析

### 7. 测试
- 提供核心功能的冒烟测试
- 确保算法能够正常运行
- 验证搜索算法能够返回有效结果

## 技术特点
- 核心算法仅使用Python标准库，无外部依赖
- 支持确定性合成基准，使用固定种子
- 模块化设计，便于扩展和测试
- 提供详细的配置选项和运行参数

## 快速开始

### 里程碑实验
```powershell
python run_milestone.py
python generate_report.py
python generate_report_detailed.py
```

### 完整项目
```powershell
python run_full_project.py --dataset synthetic
python generate_final_report.py
```

### 测试
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### CVRPLib支持
```powershell
python run_full_project.py --dataset cvrplib --cvrplib-dir "path/to/vrp_files" --limit-instances 10
```

## 输出文件
- 里程碑：
  - `outputs/milestone_results.json`
  - `outputs/search_history.json`
- 完整项目：
  - `outputs/full_project_results.json`
  - `outputs/full_search_history.json`
- 报告：
  - `milestone_report.md`
  - `milestone_report_detailed.md`
  - `final_project_report.md`

## 环境要求
- Python 3.13+
- 核心算法仅使用Python标准库
- Word导出脚本需要`python-docx`

## 项目目标
该项目实现了基于Fun Search的样本高效算法，用于解决容量车辆路径问题（CVRP），通过优化权重参数来提高贪心算法的性能，并与传统基线算法进行比较，展示了Fun Search在组合优化问题中的有效性。