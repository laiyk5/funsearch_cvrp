from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cvrp.core import CVRPInstance, evaluate_heuristic, solution_distance
from src.cvrp.io import load_cvrplib_instance
from src.llm.interface import LLMInterface
from src.llm.equivalence import FunctionEquivalenceDetector


def load_sol_file(sol_file: Path) -> List[List[int]]:
    """加载CVRPLib格式的最优解文件
    
    Args:
        sol_file: .sol文件路径
    
    Returns:
        最优路由列表
    """
    routes = []
    current_route = []
    
    with open(sol_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Route'):
                # 保存当前路由（如果有）
                if current_route:
                    routes.append(current_route)
                    current_route = []
                # 解析Route行中的客户索引
                parts = line.split(':')[1].strip().split()
                for part in parts:
                    try:
                        customer = int(part)
                        if customer > 0:  # 跳过配送中心0
                            current_route.append(customer)
                    except ValueError:
                        pass
            elif line.startswith('Cost'):
                # 跳过成本行
                continue
            else:
                # 解析普通行中的客户索引
                parts = line.split()
                for part in parts:
                    try:
                        customer = int(part)
                        if customer > 0:  # 跳过配送中心0
                            current_route.append(customer)
                    except ValueError:
                        pass
    
    # 保存最后一个路由
    if current_route:
        routes.append(current_route)
    
    return routes

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

def evaluate_with_optimal(instances: list[CVRPInstance], solver: callable, optimal_solutions: Dict[str, List[List[int]]]) -> dict:
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
        小规模、中等规模、大规模实例列表，以及最优解字典
    """
    # 尝试不同的路径 (使用标准路径 data/A，这是我们 reorganized 后的位置)
    possible_paths = [
        Path("data/A"),
        Path("A"),
        Path("A/A"),
        Path("../data/A"),
        Path("../A"),
    ]
    
    a_folder = None
    for path in possible_paths:
        if path.exists() and any(path.glob("*.vrp")):
            a_folder = path
            print(f"  找到数据文件夹: {path}")
            break
    
    if not a_folder:
        print("错误: 找不到 CVRPLib A 数据集")
        print("期望路径: data/A/ (包含 .vrp 文件)")
        print("当前工作目录:", Path.cwd())
        return [], [], [], {}
    
    small_instances = []
    medium_instances = []
    large_instances = []
    optimal_solutions = {}
    
    # 加载所有vrp文件
    vrp_files = list(a_folder.glob("*.vrp"))
    
    # 加载所有sol文件
    sol_files = list(a_folder.glob("*.sol"))
    for sol_file in sol_files:
        routes = load_sol_file(sol_file)
        optimal_solutions[sol_file.stem] = routes
        print(f"  加载最优解: {sol_file.name}")
    
    # 按客户数量分类实例
    for vrp_file in vrp_files:
        try:
            instance = load_cvrplib_instance(vrp_file)
            # 添加file_stem属性，用于匹配最优解
            instance.file_stem = vrp_file.stem
            
            # 按客户数量分类
            if instance.n_customers <= 35:
                small_instances.append(instance)
            elif instance.n_customers <= 55:
                medium_instances.append(instance)
            else:
                large_instances.append(instance)
        except Exception as e:
            print(f"  加载 {vrp_file.name} 失败: {e}")
    
    # 打印加载结果
    print("加载完成：")
    print(f"  小规模实例: {len(small_instances)}个 (≤35客户)")
    print(f"  中等规模实例: {len(medium_instances)}个 (36-55客户)")
    print(f"  大规模实例: {len(large_instances)}个 (≥56客户)")
    print(f"  加载的最优解数量: {len(optimal_solutions)}")
    
    return small_instances, medium_instances, large_instances, optimal_solutions

def run_iterative_search(n_iterations: int = None, n_heuristics_per_iter: int = None, 
                         parallel: bool = True) -> List[Dict]:
    """运行迭代搜索过程
    
    Args:
        n_iterations: 迭代次数
        n_heuristics_per_iter: 每轮生成的启发式算法数量
        parallel: 是否并行生成（默认True，小批量测试可设为False）
        
    Returns:
        每轮最佳启发式算法的结果列表
    """
    # 从配置文件读取配置
    try:
        from src.utils.config import (
            N_ITERATIONS, MIN_HEURISTICS_PER_ITER, MAX_HEURISTICS_PER_ITER,
            EARLY_PRUNING_THRESHOLD
        )
        n_iterations = n_iterations or N_ITERATIONS
        # 如果指定了每轮生成的算法数量，使用指定的值
        if n_heuristics_per_iter:
            min_heuristics = n_heuristics_per_iter
            max_heuristics = n_heuristics_per_iter
        else:
            # 减少30%每轮生成的算法数量
            min_heuristics = int(MIN_HEURISTICS_PER_ITER * 0.7)
            max_heuristics = int(MAX_HEURISTICS_PER_ITER * 0.7)
        pruning_threshold = EARLY_PRUNING_THRESHOLD
    except ImportError:
        # 如果配置文件不存在，使用默认值
        n_iterations = n_iterations or 10
        # 如果指定了每轮生成的算法数量，使用指定的值
        if n_heuristics_per_iter:
            min_heuristics = n_heuristics_per_iter
            max_heuristics = n_heuristics_per_iter
        else:
            # 减少30%每轮生成的算法数量
            min_heuristics = 35  # 50 * 0.7
            max_heuristics = 70   # 100 * 0.7
        pruning_threshold = 1.5
    
    # 初始化LLM接口
    llm = LLMInterface()
    
    # 加载A文件夹的测试数据
    print("加载A文件夹测试数据...")
    small_instances, medium_instances, large_instances, optimal_solutions = load_a_folder_data()
    
    # 确保有足够的测试数据
    if not small_instances or not medium_instances or not large_instances:
        print("测试数据不足，无法运行测试")
        return []
    
    # 初始化功能等价检测器
    test_instances = small_instances + medium_instances
    equivalence_detector = FunctionEquivalenceDetector(test_instances)
    
    # 存储每轮的最佳结果
    best_results = []
    
    # 生成初始算法
    print("生成初始启发式算法...")
    initial_code = llm.generate_heuristic()
    initial_solver = llm.load_heuristic(initial_code)
    
    if initial_solver:
        # 评估初始算法
        small_metrics = evaluate_with_optimal(small_instances, initial_solver, optimal_solutions)
        medium_metrics = evaluate_with_optimal(medium_instances, initial_solver, optimal_solutions)
        large_metrics = evaluate_with_optimal(large_instances, initial_solver, optimal_solutions)
        
        # 计算综合分数
        small_score = small_metrics["avg_distance"] + 20.0 * small_metrics["avg_num_routes"]
        medium_score = medium_metrics["avg_distance"] + 20.0 * medium_metrics["avg_num_routes"]
        large_score = large_metrics["avg_distance"] + 20.0 * large_metrics["avg_num_routes"]
        avg_score = (small_score + medium_score + large_score) / 3
        
        # 计算平均差距
        avg_gap = (small_metrics["avg_gap"] + medium_metrics["avg_gap"] + large_metrics["avg_gap"]) / 3
        
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
                "gap": small_metrics["avg_gap"]
            },
            "medium_scale": {
                "avg_distance": medium_metrics["avg_distance"],
                "avg_num_routes": medium_metrics["avg_num_routes"],
                "score": medium_score,
                "gap": medium_metrics["avg_gap"]
            },
            "large_scale": {
                "avg_distance": large_metrics["avg_distance"],
                "avg_num_routes": large_metrics["avg_num_routes"],
                "score": large_score,
                "gap": large_metrics["avg_gap"]
            },
            "stability": {
                "avg_score": avg_score,
                "avg_gap": avg_gap
            }
        }
        best_results.append(initial_result)
        print(f"初始算法评估完成，平均分数: {avg_score:.3f}, 平均差距: {avg_gap:.2f}%")
    else:
        print("初始算法生成失败，使用默认算法")
        best_results.append(None)
    
    start_time = time.time()
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n=== 第{iteration}轮迭代 ===")
        
        # 生成min_heuristics-max_heuristics个新启发式算法
        current_n_heuristics = random.randint(min_heuristics, max_heuristics)
        print(f"生成{current_n_heuristics}个新启发式算法...")
        
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
            
            # 计算平均差距
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
                    "gap": small_metrics["avg_gap"]
                },
                "medium_scale": {
                    "avg_distance": medium_metrics["avg_distance"],
                    "avg_num_routes": medium_metrics["avg_num_routes"],
                    "score": medium_score,
                    "gap": medium_metrics["avg_gap"]
                },
                "large_scale": {
                    "avg_distance": large_metrics["avg_distance"],
                    "avg_num_routes": large_metrics["avg_num_routes"],
                    "score": large_score,
                    "gap": large_metrics["avg_gap"]
                },
                "stability": {
                    "avg_score": avg_score,
                    "avg_gap": avg_gap
                }
            }
            
            print(f"第{i+1}个启发式算法测试完成，平均分数: {avg_score:.3f}, 平均差距: {avg_gap:.2f}%")
            return result
        
        # 使用线程池并行处理，或单线程模式（小批量测试用）
        if parallel and current_n_heuristics > 1:
            max_workers = min(3, current_n_heuristics)  # 限制最大线程数为3，避免API阻塞
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                futures = [executor.submit(process_heuristic, i) for i in range(current_n_heuristics)]
                # 收集结果（带超时，防止卡住）
                for future in concurrent.futures.as_completed(futures, timeout=300):  # 5分钟超时
                    try:
                        result = future.result(timeout=60)  # 每个任务1分钟超时
                        if result:
                            # 检查功能等价性
                            if result["signature"] not in evaluated_signatures:
                                evaluated_signatures.add(result["signature"])
                                current_results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("  任务超时，跳过")
                        continue
                    except Exception as e:
                        print(f"  任务异常: {e}")
                        continue
        else:
            # 单线程模式（更稳定，适合小批量测试）
            print("  使用单线程模式...")
            for i in range(current_n_heuristics):
                try:
                    result = process_heuristic(i)
                    if result:
                        if result["signature"] not in evaluated_signatures:
                            evaluated_signatures.add(result["signature"])
                            current_results.append(result)
                except Exception as e:
                    print(f"  处理第{i+1}个算法时出错: {e}")
                    continue
        
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
    print(f"\n迭代搜索完成，耗时{end_time - start_time:.2f}秒")
    
    return best_results


def main() -> None:
    """主函数"""
    # 导入输出管理器
    from src.utils.output_manager import get_output_dir, save_run_info
    
    # 运行迭代搜索，设置为2轮迭代，每轮生成20个算法
    best_results = run_iterative_search(n_iterations=2, n_heuristics_per_iter=20)
    
    # 保存结果到按 commit 组织的目录
    out_dir = get_output_dir("outputs")
    print(f"\n保存结果到: {out_dir}")
    
    # 保存每轮最佳结果
    results_file = out_dir / "iterative_search_results.json"
    results_file.write_text(
        json.dumps(best_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # 保存运行信息
    save_run_info(out_dir, {
        "script": "generate_heuristics.py",
        "n_iterations": 2,
        "n_heuristics_per_iter": 20,
        "n_valid_results": len([r for r in best_results if r is not None])
    })
    
    # 过滤掉None结果
    valid_results = [r for r in best_results if r is not None]
    
    print(f"结果已保存到: {results_file}")
    print(f"共进行了{len(valid_results)}轮有效迭代")
    
    # 打印每轮最佳算法信息
    if valid_results:
        print("\n每轮最佳算法:")
        for i, result in enumerate(valid_results):
            print(f"\n第{i}轮:")
            print(f"平均分数: {result['stability']['avg_score']:.3f}")
            print(f"平均差距: {result['stability']['avg_gap']:.2f}%")
            print(f"小规模分数: {result['small_scale']['score']:.3f}, 差距: {result['small_scale']['gap']:.2f}%")
            print(f"中等规模分数: {result['medium_scale']['score']:.3f}, 差距: {result['medium_scale']['gap']:.2f}%")
            print(f"大规模分数: {result['large_scale']['score']:.3f}, 差距: {result['large_scale']['gap']:.2f}%")
        
        # 打印最终最佳算法
        final_best = min(valid_results, key=lambda x: x["stability"]["avg_score"])
        print(f"\n最终最佳算法 (第{final_best['iteration']}轮):")
        print(f"平均分数: {final_best['stability']['avg_score']:.3f}")
        print(f"平均差距: {final_best['stability']['avg_gap']:.2f}%")
        print(f"小规模分数: {final_best['small_scale']['score']:.3f}, 差距: {final_best['small_scale']['gap']:.2f}%")
        print(f"中等规模分数: {final_best['medium_scale']['score']:.3f}, 差距: {final_best['medium_scale']['gap']:.2f}%")
        print(f"大规模分数: {final_best['large_scale']['score']:.3f}, 差距: {final_best['large_scale']['gap']:.2f}%")


if __name__ == "__main__":
    main()
