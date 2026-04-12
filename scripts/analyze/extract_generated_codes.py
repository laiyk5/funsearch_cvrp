#!/usr/bin/env python3
"""
提取生成的启发式算法代码到独立的 .py 文件

用法:
    # 提取最新结果
    python scripts/extract_generated_codes.py
    
    # 提取指定结果文件
    python scripts/extract_generated_codes.py --input outputs/latest/iterative_search_results.json
    
    # 提取指定 commit 的所有结果
    python scripts/extract_generated_codes.py --commit abc123
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_default_output_dir(input_file: Path) -> Path:
    """根据输入文件路径确定默认输出目录"""
    # 解析路径: outputs/{commit_hash}/{timestamp}/xxx.json
    try:
        # 处理 symlink 路径
        resolved = input_file.resolve()
        parts = resolved.parts
        if 'outputs' in parts:
            outputs_idx = parts.index('outputs')
            if len(parts) > outputs_idx + 3:
                # 返回: outputs/{commit_hash}/{timestamp}/generated/{filename}/
                commit_hash = parts[outputs_idx + 1]
                timestamp = parts[outputs_idx + 2]
                return Path('outputs') / commit_hash / timestamp / 'generated' / input_file.stem
    except Exception:
        pass
    
    # 回退: 放在 input 文件所在目录的 generated/{filename}/ 子目录下
    return input_file.parent / 'generated' / input_file.stem


def extract_codes(input_file: Path, output_dir: Path = None):
    """从 JSON 结果文件中提取代码并保存为 .py 文件"""
    
    if not input_file.exists():
        print(f"错误: 文件不存在 {input_file}")
        return 0
    
    # 确定输出目录
    if output_dir is None:
        output_dir = get_default_output_dir(input_file)
    
    # 读取结果
    with open(input_file) as f:
        results = json.load(f)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 过滤有效结果
    valid_results = [r for r in results if r and r.get('heuristic_code')]
    
    print(f"从 {input_file} 中提取 {len(valid_results)} 个算法...")
    
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
            f.write(f'来源: {input_file}\n')
            f.write(f'迭代: {iteration}\n')
            f.write(f'评分: {score}\n')
            f.write(f'提取时间: {datetime.now().isoformat()}\n')
            f.write(f'"""\n\n')
            f.write(code)
        
        print(f"  ✓ 保存: {filename}")
    
    print(f"\n共提取 {len(valid_results)} 个算法到 {output_dir}/")
    return len(valid_results)


def find_results_files(commit_hash: str = None, base_dir: str = "outputs") -> list:
    """查找所有结果文件"""
    base_path = Path(base_dir)
    
    if commit_hash:
        # 查找指定 commit 的所有结果
        commit_dir = base_path / commit_hash
        if not commit_dir.exists():
            print(f"错误: Commit {commit_hash} 没有结果")
            return []
        result_files = list(commit_dir.rglob("*.json"))
    else:
        # 查找所有结果文件
        result_files = list(base_path.rglob("*.json"))
    
    # 过滤掉 run_info.json
    result_files = [f for f in result_files if f.name != "run_info.json"]
    
    return result_files


def main():
    parser = argparse.ArgumentParser(description='提取生成的启发式算法代码')
    parser.add_argument('--input', '-i', type=Path, 
                        help='输入的 JSON 结果文件')
    parser.add_argument('--commit', '-c', type=str,
                        help='指定 commit hash 提取该 commit 的所有结果')
    parser.add_argument('--output-dir', '-o', type=Path,
                        default=None,
                        help='输出目录 (默认: outputs/{commit_hash}/generated/)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出所有可用的结果')
    args = parser.parse_args()
    
    # 列出所有结果
    if args.list:
        from funsearch_cvrp.utils.output_manager import print_results_summary
        print_results_summary("outputs")
        return
    
    # 指定了输入文件
    if args.input:
        extract_codes(args.input, args.output_dir)
        return
    
    # 指定了 commit
    if args.commit:
        result_files = find_results_files(args.commit)
        if not result_files:
            return
        
        print(f"找到 {len(result_files)} 个结果文件 for commit {args.commit}")
        for f in result_files:
            # 默认: outputs/{commit}/{timestamp}/generated/
            out_dir = args.output_dir  # extract_codes 内部会处理为 None 的情况
            extract_codes(f, out_dir)
        return
    
    # 默认：提取最新结果
    latest = Path("outputs/latest")
    if latest.exists() and latest.is_symlink():
        result_files = list(latest.glob("*.json"))
        result_files = [f for f in result_files if f.name != "run_info.json"]
        
        if result_files:
            for f in result_files:
                extract_codes(f, args.output_dir)
        else:
            print("错误: 没有找到结果文件")
            print("请先运行实验或指定 --input")
    else:
        print("错误: 没有找到最新结果")
        print("提示: 使用 --list 查看所有可用的结果")


if __name__ == "__main__":
    main()
