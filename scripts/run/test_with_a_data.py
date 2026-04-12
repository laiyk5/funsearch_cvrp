#!/usr/bin/env python3
"""
测试文件：使用A文件夹的测试数据运行FUNSearch
迭代2轮，每轮生成10个算法
"""

from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cvrp.core import CVRPInstance, evaluate_heuristic, solution_distance
from src.cvrp.io import load_cvrplib_instance
from src.llm.interface import LLMInterface
from src.llm.equivalence import FunctionEquivalenceDetector

def load_sol_file(sol_file: Path) -> List[List[int]]:
    """加载.sol文件中的最优解
    
    Args:
        sol_file: .sol文件路径
    
    Returns:
        最优路由列表
    """
    try:
        text = sol_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        routes = []
        current_route = []
        
        for line in text:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Route"):
                if current_route:
                    routes.append(current_route)
                    current_route = []
                # 解析Route行中的客户索引
                parts = line.split()
                for part in parts[2:]:  # 跳过"Route"和"#1:"等前缀
                    try:
                        customer = int(part)
                        if customer != -1:  # -1表示路由结束
                            current_route.append(customer)
                    except ValueError:
                        pass
            elif line.startswith("Cost"):
                # 跳过成本行
                continue
            else:
                # 解析普通行中的客户索引
                parts = line.split()
                for part in parts:
                    try:
                        customer = int(part)
                        if customer != -1:  # -1表示路由结束
                            current_route.append(customer)
                    except ValueError:
                        pass
        
        if current_route:
            routes.append(current_route)
        
        return routes
    except Exception as e:
        print(f"加载.sol文件 {sol_file} 失败: {e}")
        return []

def is_valid_route(instance: CVRPInstance, route: List[int]) -> bool:
    """检查路由是否有效
    
    Args:
        instance: CVRP实例
        route: 路由列表
    
    Returns:
        路由是否有效
    """
    if not route:
        return False
    
    # 检查每个节点是否在有效范围内
    max_node = instance.n_customers  # 客户索引从1开始，到n_customers结束
    for node in route:
        if node < 1 or node > max_node:
            return False
    
    return True

def check_capacity_constraint(instance: CVRPInstance, routes: List[List[int]]) -> bool:
    """检查路由是否满足容量约束
    
    Args:
        instance: CVRP实例
        routes: 路由列表
    
    Returns:
        是否满足容量约束
    """
    for route in routes:
        total_demand = 0
        for customer in route:
            total_demand += instance.demands[customer]
        if total_demand > instance.capacity:
            print(f"  容量约束违反: 路由 {route} 的总需求 {total_demand} 超过车辆容量 {instance.capacity}")
            return False
    return True

def evaluate_with_optimal(instances: list[CVRPInstance], solver: Callable[[CVRPInstance], list[list[int]]], optimal_solutions: Dict[str, List[List[int]]]) -> dict:
    """评估启发式算法，并与最优解进行比较
    
    Args:
        instances: 测试实例列表
        solver: 启发式算法函数
        optimal_solutions: 最优解字典
    
    Returns:
        评估结果，包含与最优解的比较
    """
    total_distance = 0.0
    total_routes = 0
    total_optimal_distance = 0.0
    total_gap = 0.0
    per_instance: list[dict] = []

    for inst in instances:
        routes = solver(inst)
        
        # 检查容量约束
        if not check_capacity_constraint(inst, routes):
            print(f"  实例 {inst.name} 的路由违反容量约束，使用惩罚距离")
            # 违反容量约束，使用惩罚距离
            dist = float('inf')
        else:
            dist = solution_distance(inst, routes)
        
        total_distance += dist
        total_routes += len(routes)
        
        # 计算与最优解的差距
        optimal_routes = []
        # 优先使用file_stem获取最优解，其次使用name
        if hasattr(inst, 'file_stem'):
            optimal_routes = optimal_solutions.get(inst.file_stem, [])
        if not optimal_routes:
            optimal_routes = optimal_solutions.get(inst.name, [])
        
        optimal_dist = dist
        
        # 检查最优解是否有效
        if optimal_routes and all(is_valid_route(inst, route) for route in optimal_routes):
            try:
                optimal_dist = solution_distance(inst, optimal_routes)
            except Exception as e:
                optimal_dist = dist
        
        total_optimal_distance += optimal_dist
        
        gap = ((dist - optimal_dist) / optimal_dist * 100) if optimal_dist > 0 else 0
        total_gap += gap
        
        per_instance.append(
            {
                "instance": inst.name,
                "distance": round(dist, 3),
                "optimal_distance": round(optimal_dist, 3),
                "gap": round(gap, 2),
                "num_routes": len(routes),
                "optimal_num_routes": len(optimal_routes)
            }
        )

    return {
        "avg_distance": total_distance / len(instances),
        "avg_num_routes": total_routes / len(instances),
        "avg_optimal_distance": total_optimal_distance / len(instances),
        "avg_gap": total_gap / len(instances),
        "details": per_instance,
    }

def load_a_folder_data() -> Tuple[List[CVRPInstance], List[CVRPInstance], List[CVRPInstance], Dict[str, List[List[int]]]]:
    """加载A文件夹的测试数据
    
    Returns:
        小规模、中等规模、大规模测试实例列表，以及最优解字典
    """
    # 尝试不同的路径
    possible_paths = [
        Path("data/A"),
        Path("A"),
        Path("A/A"),
    ]
    
    a_folder = None
    for path in possible_paths:
        if path.exists() and any(path.glob("*.vrp")):
            a_folder = path
            break
    
    if not a_folder:
        print("错误: 找不到 CVRPLib A 数据集")
        print("期望路径: data/A/ (包含 .vrp 文件)")
        print("当前工作目录:", Path.cwd())
        return [], [], [], {}
    vrp_files = list(a_folder.glob("*.vrp"))
    
    # 按客户数量分类
    small_instances = []
    medium_instances = []
    large_instances = []
    optimal_solutions = {}
    
    for vrp_file in vrp_files:
        try:
            instance = load_cvrplib_instance(str(vrp_file))
            
            # 加载对应的.sol文件
            sol_file = vrp_file.with_suffix(".sol")
            if sol_file.exists():
                optimal_routes = load_sol_file(sol_file)
                # 使用文件路径作为键，确保能够找到对应的最优解
                optimal_solutions[vrp_file.stem] = optimal_routes
                print(f"  加载最优解: {vrp_file.stem}.sol")
            
            # 存储文件路径，以便在评估时使用
            instance.file_stem = vrp_file.stem
            
            if instance.n_customers <= 35:
                small_instances.append(instance)
            elif instance.n_customers <= 55:
                medium_instances.append(instance)
            else:
                large_instances.append(instance)
        except Exception as e:
            print(f"加载文件 {vrp_file} 失败: {e}")
    
    # 限制每个规模的实例数量，避免测试时间过长
    small_instances = small_instances[:5]
    medium_instances = medium_instances[:5]
    large_instances = large_instances[:5]
    
    print(f"加载完成：")
    print(f"  小规模实例: {len(small_instances)}个 (≤35客户)")
    print(f"  中等规模实例: {len(medium_instances)}个 (36-55客户)")
    print(f"  大规模实例: {len(large_instances)}个 (≥56客户)")
    print(f"  加载的最优解数量: {len(optimal_solutions)}")
    
    return small_instances, medium_instances, large_instances, optimal_solutions

def run_test_with_a_data() -> List[Dict]:
    """使用A文件夹的测试数据运行迭代搜索
    
    Returns:
        每轮最佳启发式算法的结果列表
    """
    # 加载A文件夹的测试数据
    print("加载A文件夹测试数据...")
    small_instances, medium_instances, large_instances, optimal_solutions = load_a_folder_data()
    
    # 确保有足够的测试数据
    if not small_instances or not medium_instances or not large_instances:
        print("测试数据不足，无法运行测试")
        return []
    
    # 初始化LLM接口
    llm = LLMInterface()
    
    # 初始化功能等价检测器
    test_instances = small_instances + medium_instances
    equivalence_detector = FunctionEquivalenceDetector(test_instances)
    
    # 存储每轮的最佳结果
    best_results = []
    
    # 生成初始算法
    print("\n生成初始启发式算法...")
    initial_code = llm.generate_heuristic()
    initial_solver = llm.load_heuristic(initial_code)
    
    if initial_solver:
        # 评估初始算法，与最优解比较
        small_metrics = evaluate_with_optimal(small_instances, initial_solver, optimal_solutions)
        medium_metrics = evaluate_with_optimal(medium_instances, initial_solver, optimal_solutions)
        large_metrics = evaluate_with_optimal(large_instances, initial_solver, optimal_solutions)
        
        # 计算综合分数
        small_score = small_metrics["avg_distance"] + 20.0 * small_metrics["avg_num_routes"]
        medium_score = medium_metrics["avg_distance"] + 20.0 * medium_metrics["avg_num_routes"]
        large_score = large_metrics["avg_distance"] + 20.0 * large_metrics["avg_num_routes"]
        avg_score = (small_score + medium_score + large_score) / 3
        
        # 存储初始结果
        initial_result = {
            "iteration": 0,
            "id": 0,
            "signature": equivalence_detector.get_behavior_signature(initial_solver),
            "heuristic_code": initial_code,
            "small_scale": {
                "avg_distance": small_metrics["avg_distance"],
                "avg_num_routes": small_metrics["avg_num_routes"],
                "score": small_score,
                "avg_optimal_distance": small_metrics["avg_optimal_distance"],
                "avg_gap": small_metrics["avg_gap"]
            },
            "medium_scale": {
                "avg_distance": medium_metrics["avg_distance"],
                "avg_num_routes": medium_metrics["avg_num_routes"],
                "score": medium_score,
                "avg_optimal_distance": medium_metrics["avg_optimal_distance"],
                "avg_gap": medium_metrics["avg_gap"]
            },
            "large_scale": {
                "avg_distance": large_metrics["avg_distance"],
                "avg_num_routes": large_metrics["avg_num_routes"],
                "score": large_score,
                "avg_optimal_distance": large_metrics["avg_optimal_distance"],
                "avg_gap": large_metrics["avg_gap"]
            },
            "stability": {
                "avg_score": avg_score
            }
        }
        print(f"初始算法评估完成，平均分数: {avg_score:.3f}, 平均差距: {((small_metrics['avg_gap'] + medium_metrics['avg_gap'] + large_metrics['avg_gap']) / 3):.2f}%")
        best_results.append(initial_result)
        print(f"初始算法评估完成，平均分数: {avg_score:.3f}")
    else:
        print("初始算法生成失败，使用默认算法")
        best_results.append(None)
    
    start_time = time.time()
    
    # 运行2轮迭代，每轮生成10个算法
    n_iterations = 2
    heuristics_per_iter = 10
    pruning_threshold = 1.5
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n=== 第{iteration}轮迭代 ===")
        print(f"生成{heuristics_per_iter}个新启发式算法...")
        
        # 存储当前轮的结果
        current_results = []
        evaluated_signatures = set()
        
        # 并行生成和评估启发式算法
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        
        def process_heuristic(i):
            print(f"生成第{i+1}个启发式算法...")
            
            # 生成启发式算法，传入上一轮的最佳算法
            previous_heuristic = best_results[-1]["heuristic_code"] if best_results and best_results[-1] else None
            heuristic_code = llm.generate_heuristic(previous_heuristic)
            
            # 验证并加载启发式算法
            if not llm.validate_heuristic(heuristic_code):
                print(f"第{i+1}个启发式算法验证失败，跳过")
                return None
            
            solver = llm.load_heuristic(heuristic_code)
            if solver is None:
                print(f"第{i+1}个启发式算法加载失败，跳过")
                return None
            
            # 生成行为签名，检测功能等价性
            signature = equivalence_detector.get_behavior_signature(solver)
            if signature is None:
                print(f"第{i+1}个启发式算法签名生成失败，跳过")
                return None
            
            # 早期减枝：先在小规模数据集上测试
            small_metrics = evaluate_with_optimal(small_instances, solver, optimal_solutions)
            small_score = small_metrics["avg_distance"] + 20.0 * small_metrics["avg_num_routes"]
            
            # 如果小规模分数太差，直接跳过
            if best_results and best_results[-1] and small_score > best_results[-1]["small_scale"]["score"] * pruning_threshold:
                print(f"第{i+1}个启发式算法小规模分数太差，跳过")
                return None
            
            # 在中等规模和大规模数据集上测试
            medium_metrics = evaluate_with_optimal(medium_instances, solver, optimal_solutions)
            large_metrics = evaluate_with_optimal(large_instances, solver, optimal_solutions)
            
            # 计算综合分数
            medium_score = medium_metrics["avg_distance"] + 20.0 * medium_metrics["avg_num_routes"]
            large_score = large_metrics["avg_distance"] + 20.0 * large_metrics["avg_num_routes"]
            avg_score = (small_score + medium_score + large_score) / 3
            avg_gap = (small_metrics["avg_gap"] + medium_metrics["avg_gap"] + large_metrics["avg_gap"]) / 3
            
            # 存储结果
            result = {
                "iteration": iteration,
                "id": i + 1,
                "signature": signature,
                "heuristic_code": heuristic_code,
                "small_scale": {
                    "avg_distance": small_metrics["avg_distance"],
                    "avg_num_routes": small_metrics["avg_num_routes"],
                    "score": small_score,
                    "avg_optimal_distance": small_metrics["avg_optimal_distance"],
                    "avg_gap": small_metrics["avg_gap"]
                },
                "medium_scale": {
                    "avg_distance": medium_metrics["avg_distance"],
                    "avg_num_routes": medium_metrics["avg_num_routes"],
                    "score": medium_score,
                    "avg_optimal_distance": medium_metrics["avg_optimal_distance"],
                    "avg_gap": medium_metrics["avg_gap"]
                },
                "large_scale": {
                    "avg_distance": large_metrics["avg_distance"],
                    "avg_num_routes": large_metrics["avg_num_routes"],
                    "score": large_score,
                    "avg_optimal_distance": large_metrics["avg_optimal_distance"],
                    "avg_gap": large_metrics["avg_gap"]
                },
                "stability": {
                    "avg_score": avg_score,
                    "avg_gap": avg_gap
                }
            }
            
            print(f"第{i+1}个启发式算法测试完成，平均分数: {avg_score:.3f}, 平均差距: {avg_gap:.2f}%")
            return result
        
        # 使用线程池并行处理
        max_workers = min(5, heuristics_per_iter)  # 限制最大线程数，避免超过API速率限制
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_heuristic, i) for i in range(heuristics_per_iter)]
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    # 检查功能等价性
                    if result["signature"] not in evaluated_signatures:
                        evaluated_signatures.add(result["signature"])
                        current_results.append(result)
        
        # 选择当前轮得分最高的算法
        if current_results:
            # 按平均分数排序
            current_results.sort(key=lambda x: x["stability"]["avg_score"])
            best_current = current_results[0]
            
            # 只有当当前轮的最佳算法比上一轮更好时，才更新
            if not best_results or not best_results[-1] or best_current["stability"]["avg_score"] < best_results[-1]["stability"]["avg_score"]:
                best_results.append(best_current)
                print(f"\n第{iteration}轮最佳算法，平均分数: {best_current['stability']['avg_score']:.3f} (优于上一轮)")
            else:
                # 否则，保留上一轮的最佳算法
                best_results.append(best_results[-1])
                print(f"\n第{iteration}轮没有找到更好的算法，保留上一轮的最佳算法")
        else:
            print(f"\n第{iteration}轮没有生成有效的启发式算法")
            best_results.append(best_results[-1] if best_results and best_results[-1] else None)
    
    end_time = time.time()
    print(f"\n测试完成，耗时{end_time - start_time:.2f}秒")
    
    return best_results

def main() -> None:
    """主函数"""
    # 运行测试
    best_results = run_test_with_a_data()
    
    # 保存结果
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每轮最佳结果
    (out_dir / "test_a_data_results.json").write_text(
        json.dumps(best_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # 过滤掉None结果
    valid_results = [r for r in best_results if r is not None]
    
    print(f"\n结果已保存到outputs/test_a_data_results.json")
    print(f"共进行了{len(valid_results)}轮有效迭代")
    
    # 打印每轮最佳算法信息
    if valid_results:
        print("\n每轮最佳算法:")
        for i, result in enumerate(valid_results):
            print(f"\n第{i}轮:")
            print(f"平均分数: {result['stability']['avg_score']:.3f}")
            print(f"平均差距: {result['stability'].get('avg_gap', 0):.2f}%")
            print(f"小规模分数: {result['small_scale']['score']:.3f}, 差距: {result['small_scale'].get('avg_gap', 0):.2f}%")
            print(f"中等规模分数: {result['medium_scale']['score']:.3f}, 差距: {result['medium_scale'].get('avg_gap', 0):.2f}%")
            print(f"大规模分数: {result['large_scale']['score']:.3f}, 差距: {result['large_scale'].get('avg_gap', 0):.2f}%")
        
        # 打印最终最佳算法
        final_best = min(valid_results, key=lambda x: x["stability"]["avg_score"])
        print(f"\n最终最佳算法 (第{final_best['iteration']}轮):")
        print(f"平均分数: {final_best['stability']['avg_score']:.3f}")
        print(f"平均差距: {final_best['stability'].get('avg_gap', 0):.2f}%")
        print(f"小规模分数: {final_best['small_scale']['score']:.3f}, 差距: {final_best['small_scale'].get('avg_gap', 0):.2f}%")
        print(f"中等规模分数: {final_best['medium_scale']['score']:.3f}, 差距: {final_best['medium_scale'].get('avg_gap', 0):.2f}%")
        print(f"大规模分数: {final_best['large_scale']['score']:.3f}, 差距: {final_best['large_scale'].get('avg_gap', 0):.2f}%")


if __name__ == "__main__":
    main()
