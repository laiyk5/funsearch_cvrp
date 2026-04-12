#!/usr/bin/env python3
"""
提取生成的启发式算法代码到独立的 .py 文件

用法:
    python scripts/extract_generated_codes.py
    python scripts/extract_generated_codes.py --output-dir generated_codes/
"""

import argparse
import json
from pathlib import Path


def extract_codes(input_file: Path, output_dir: Path):
    """从 JSON 结果文件中提取代码并保存为 .py 文件"""
    
    if not input_file.exists():
        print(f"错误: 文件不存在 {input_file}")
        return
    
    # 读取结果
    with open(input_file) as f:
        results = json.load(f)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 过滤有效结果
    valid_results = [r for r in results if r and r.get('heuristic_code')]
    
    print(f"从 {input_file.name} 中提取 {len(valid_results)} 个算法...")
    
    for result in valid_results:
        iteration = result.get('iteration', 0)
        code = result['heuristic_code']
        score = result.get('stability', {}).get('avg_score', 'N/A')
        
        # 构建文件名
        filename = f"heuristic_iter{iteration:02d}_score{score}.py"
        filepath = output_dir / filename
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f'"""\n')
            f.write(f'FunSearch 生成的启发式算法\n')
            f.write(f'来源: {input_file.name}\n')
            f.write(f'迭代: {iteration}\n')
            f.write(f'评分: {score}\n')
            f.write(f'"""\n\n')
            f.write(code)
        
        print(f"  ✓ 保存: {filename}")
    
    print(f"\n共提取 {len(valid_results)} 个算法到 {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='提取生成的启发式算法代码')
    parser.add_argument('--input', '-i', type=Path, 
                        default=Path('outputs/iterative_search_results.json'),
                        help='输入的 JSON 结果文件')
    parser.add_argument('--output-dir', '-o', type=Path,
                        default=Path('generated_codes'),
                        help='输出目录')
    args = parser.parse_args()
    
    extract_codes(args.input, args.output_dir)
    
    # 同时检查其他结果文件
    other_files = [
        Path('outputs/test_a_data_results.json'),
        Path('outputs/milestone_results.json'),
    ]
    
    for f in other_files:
        if f.exists() and f != args.input:
            print(f"\n发现其他结果文件: {f}")
            ans = input("是否也提取? (y/n): ").strip().lower()
            if ans == 'y':
                out_dir = args.output_dir / f.stem
                extract_codes(f, out_dir)


if __name__ == "__main__":
    main()
