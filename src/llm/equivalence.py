from __future__ import annotations

import hashlib
import json
from typing import Callable, List, Tuple, Optional

from src.cvrp.core import CVRPInstance, solution_distance


class FunctionEquivalenceDetector:
    """功能等价检测机制，使用输入输出向量哈希或行为签名"""
    
    def __init__(self, test_instances: List[CVRPInstance]):
        """初始化功能等价检测器
        
        Args:
            test_instances: 用于测试功能等价性的CVRP实例列表
        """
        self.test_instances = test_instances
    
    def get_behavior_signature(self, heuristic: Callable[[CVRPInstance], List[List[int]]]) -> Optional[str]:
        """获取启发式算法的行为签名
        
        Args:
            heuristic: 启发式算法函数
            
        Returns:
            行为签名字符串，失败返回None
        """
        try:
            # 对每个测试实例运行启发式算法，收集结果
            behavior = []
            for instance in self.test_instances:
                routes = heuristic(instance)
                # 计算路径距离和路径数量
                dist = solution_distance(instance, routes)
                num_routes = len(routes)
                # 对路径进行排序，确保顺序不影响签名
                sorted_routes = sorted([sorted(route) for route in routes])
                # 生成每个实例的行为特征
                instance_behavior = {
                    "distance": round(dist, 3),
                    "num_routes": num_routes,
                    "route_structure": sorted_routes
                }
                behavior.append(instance_behavior)
            
            # 将行为特征转换为JSON字符串
            behavior_str = json.dumps(behavior, sort_keys=True)
            # 使用SHA256生成哈希值作为签名
            signature = hashlib.sha256(behavior_str.encode()).hexdigest()
            
            return signature
        except Exception as e:
            print(f"Error generating behavior signature: {e}")
            return None
    
    def are_equivalent(self, heuristic1: Callable[[CVRPInstance], List[List[int]]], 
                       heuristic2: Callable[[CVRPInstance], List[List[int]]]) -> bool:
        """判断两个启发式算法是否功能等价
        
        Args:
            heuristic1: 第一个启发式算法
            heuristic2: 第二个启发式算法
            
        Returns:
            两个算法是否功能等价
        """
        sig1 = self.get_behavior_signature(heuristic1)
        sig2 = self.get_behavior_signature(heuristic2)
        
        if sig1 is None or sig2 is None:
            return False
        
        return sig1 == sig2
    
    def get_input_output_vector(self, heuristic: Callable[[CVRPInstance], List[List[int]]]) -> Optional[List[Tuple[float, int]]]:
        """获取启发式算法的输入输出向量
        
        Args:
            heuristic: 启发式算法函数
            
        Returns:
            输入输出向量列表，每个元素为(distance, num_routes)，失败返回None
        """
        try:
            io_vector = []
            for instance in self.test_instances:
                routes = heuristic(instance)
                dist = solution_distance(instance, routes)
                num_routes = len(routes)
                io_vector.append((round(dist, 3), num_routes))
            
            return io_vector
        except Exception as e:
            print(f"Error generating input-output vector: {e}")
            return None
    
    def get_io_vector_hash(self, heuristic: Callable[[CVRPInstance], List[List[int]]]) -> Optional[str]:
        """获取输入输出向量的哈希值
        
        Args:
            heuristic: 启发式算法函数
            
        Returns:
            输入输出向量的哈希值，失败返回None
        """
        io_vector = self.get_input_output_vector(heuristic)
        if io_vector is None:
            return None
        
        # 将向量转换为字符串并生成哈希
        vector_str = str(io_vector)
        hash_value = hashlib.sha256(vector_str.encode()).hexdigest()
        
        return hash_value
