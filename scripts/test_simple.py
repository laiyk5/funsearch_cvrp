#!/usr/bin/env python3
"""
简单测试脚本，只运行一轮迭代，生成少量算法
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_heuristics import run_iterative_search

if __name__ == "__main__":
    print("运行简单测试...")
    # 只运行1轮迭代，生成5个算法
    best_results = run_iterative_search(n_iterations=1, n_heuristics_per_iter=5)
    print(f"测试完成，共进行了{len(best_results)}轮迭代")
    if best_results:
        print(f"最佳算法分数: {best_results[-1]['stability']['avg_score']:.3f}")
