from __future__ import annotations

import json
import random
import re
from typing import Callable, List, Tuple, Optional

from cvrp_core import CVRPInstance


class LLMInterface:
    """LLM接口，用于生成CVRP启发式算法"""
    
    def __init__(self, model: str = "gpt-4"):
        """初始化LLM接口
        
        Args:
            model: 使用的LLM模型名称
        """
        self.model = model
        self.prompt_template = """
你是一个专业的运筹学和算法专家，擅长设计CVRP（容量车辆路径问题）的启发式算法。

请为CVRP问题设计一个启发式算法，要求：
1. 函数名为`custom_heuristic`，接受一个`CVRPInstance`类型的参数`instance`，返回一个`List[List[int]]`类型的路径列表
2. 算法应该考虑车辆容量约束
3. 算法应该尽可能减少总行驶距离
4. 算法应该高效，能够处理不同规模的问题实例
5. 请提供完整的函数实现，包括必要的导入语句
6. 请确保代码能够直接运行，不要包含任何解释性文本

CVRPInstance的定义如下：
```python
@dataclass
class CVRPInstance:
    name: str
    capacity: int
    demands: list[int]
    coords: list[tuple[float, float]]
    
    @property
    def n_customers(self) -> int:
        return len(self.demands) - 1
```

距离计算函数：
```python
def euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])
```

示例启发式算法：
```python
def nearest_neighbor_heuristic(instance: CVRPInstance) -> list[list[int]]:
    unserved = set(range(1, instance.n_customers + 1))
    routes: list[list[int]] = []
    while unserved:
        cap_left = instance.capacity
        current = 0
        route: list[int] = []
        while True:
            feasible = [c for c in unserved if instance.demands[c] <= cap_left]
            if not feasible:
                break
            nxt = min(feasible, key=lambda c: euclid(instance.coords[current], instance.coords[c]))
            route.append(nxt)
            unserved.remove(nxt)
            cap_left -= instance.demands[nxt]
            current = nxt
        routes.append(route)
    return routes
```

请生成一个新的、不同的启发式算法，不要重复示例中的算法。
        """
    
    def __init__(self, api_key: str = None, model: str = None):
        """初始化LLM接口
        
        Args:
            api_key: LLM API密钥
            model: 使用的LLM模型
        """
        # 从配置文件读取配置
        try:
            from config import OPENAI_API_KEY, OPENAI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
            self.api_key = api_key or OPENAI_API_KEY
            self.model = model or OPENAI_MODEL
            self.temperature = LLM_TEMPERATURE
            self.max_tokens = LLM_MAX_TOKENS
        except ImportError:
            # 如果配置文件不存在，使用默认值
            self.api_key = api_key or "YOUR_API_KEY"
            self.model = model or "gpt-4"
            self.temperature = 0.7
            self.max_tokens = 2000
    
    def generate_heuristic(self, previous_heuristic: Optional[str] = None) -> str:
        """使用LLM生成一个CVRP启发式算法
        
        Args:
            previous_heuristic: 上一轮的最佳启发式算法代码，用于指导生成更好的算法
            
        Returns:
            生成的启发式算法代码
        """
        print("正在调用LLM生成启发式算法...")
        
        # 构建LLM提示
        prompt = self._build_llm_prompt(previous_heuristic)
        
        # 调用LLM API生成代码
        generated_code = self._call_llm_api(prompt)
        
        # 验证生成的代码
        if not generated_code:
            print("LLM生成失败，使用备用方案...")
            # 使用备用方案：从预定义模板中选择
            heuristics = [
                self._generate_nearest_neighbor_variation(),
                self._generate_savings_variation(),
                self._generate_sweep_variation(),
                self._generate_cluster_first_route_second(),
                self._generate_regret_heuristic(),
            ]
            generated_code = random.choice(heuristics)
        
        # 添加随机扰动，增加多样性
        generated_code = self._add_random_perturbation(generated_code)
        
        print("LLM生成完成")
        return generated_code
    
    def _build_llm_prompt(self, previous_heuristic: Optional[str] = None) -> str:
        """构建LLM提示
        
        Args:
            previous_heuristic: 上一轮的最佳启发式算法代码
            
        Returns:
            LLM提示字符串
        """
        prompt = """你是一个CVRP（容量车辆路径问题）启发式算法专家。请生成一个高效的CVRP启发式算法实现。

要求：
1. 算法应该接收一个CVRPInstance对象作为输入，返回一个路由列表
2. CVRPInstance对象有以下属性：
   - n_customers: 客户数量
   - capacity: 车辆容量
   - coords: 坐标字典，格式为 {customer_id: (x, y)}
   - demands: 需求字典，格式为 {customer_id: demand}
3. 算法应该考虑距离和容量约束
4. 代码应该高效，能够处理100个客户的规模
5. 不需要导入任何模块，所有功能都应该内联实现
6. 函数名必须为custom_heuristic
7. 只返回代码，不要返回任何解释或说明

"""
        
        if previous_heuristic:
            prompt += """

以下是上一轮的最佳算法，你可以参考并改进它：

%s

请生成一个更好的算法，在保持稳定性的同时提高性能。
""" % previous_heuristic
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """调用LLM API生成代码
        
        Args:
            prompt: LLM提示
            
        Returns:
            生成的代码
        """
        import time
        max_retries = 3
        retry_delay = 5  # 初始重试延迟
        
        for attempt in range(max_retries):
            try:
                # 使用阿里通义千问大模型
                import sys
                from pathlib import Path
                
                # 添加项目根目录到Python路径
                sys.path.insert(0, str(Path(__file__).parent.parent))
                
                from models import get_normal_client, ALI_TONGYI_TURBO_MODEL
                
                # 创建阿里通义千问大模型客户端
                client = get_normal_client()
                
                # 调用API
                response = client.chat.completions.create(
                    model=ALI_TONGYI_TURBO_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个CVRP启发式算法专家"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # 提取生成的代码
                generated_code = response.choices[0].message.content.strip()
                
                # 清理代码，移除可能的markdown标记
                if generated_code.startswith("```python"):
                    generated_code = generated_code[9:]
                if generated_code.endswith("```"):
                    generated_code = generated_code[:-3]
                
                # 添加延迟，避免速率限制
                time.sleep(2)  # 2秒延迟
                
                return generated_code
            except Exception as e:
                print(f"LLM API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    print("达到最大重试次数，使用备用方案")
                    return ""
    
    def _add_random_perturbation(self, code: str) -> str:
        """为生成的代码添加随机扰动，增加多样性
        
        Args:
            code: 原始代码
            
        Returns:
            添加扰动后的代码
        """
        # 随机选择扰动类型
        perturbation_type = random.choice([
            self._perturb_score_function,
            self._perturb_routing_logic,
            self._perturb_clustering_logic
        ])
        
        return perturbation_type(code)
    
    def _perturb_score_function(self, code: str) -> str:
        """扰动评分函数"""
        # 随机修改评分函数的权重
        if "def score(c):" in code:
            # 随机生成新的权重
            weight = round(random.uniform(0.05, 0.3), 2)
            # 替换权重值
            import re
            code = re.sub(r"return dist - [0-9.]+ \* demand_ratio", f"return dist - {weight} * demand_ratio", code)
        return code
    
    def _perturb_routing_logic(self, code: str) -> str:
        """扰动路由逻辑"""
        # 随机添加或修改路由逻辑
        if "while unserved:" in code:
            # 随机添加一个简单的局部优化
            if random.random() > 0.5:
                # 在路由构建后添加简单的2-opt优化
                opt_code = """
    
    # 简单的2-opt局部优化
    def two_opt(route):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 2, len(route)):
                    # 计算交换前后的距离
                    def distance(a, b):
                        dx = instance.coords[a][0] - instance.coords[b][0]
                        dy = instance.coords[a][1] - instance.coords[b][1]
                        return (dx * dx + dy * dy) ** 0.5
                    
                    old_dist = distance(route[i-1], route[i]) + distance(route[j], route[j+1] if j+1 < len(route) else 0)
                    new_dist = distance(route[i-1], route[j]) + distance(route[i], route[j+1] if j+1 < len(route) else 0)
                    
                    if new_dist < old_dist:
                        # 执行2-opt交换
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
        return route
        
            """
                code = code.replace("    return routes", "    # 应用2-opt优化\n    for i in range(len(routes)):\n        routes[i] = two_opt(routes[i])\n    return routes")
                code = code.replace("def custom_heuristic(instance):", "def custom_heuristic(instance):\n" + opt_code)
        return code
    
    def _perturb_clustering_logic(self, code: str) -> str:
        """扰动聚类逻辑"""
        # 随机修改聚类参数
        if "n_clusters = max(1, instance.n_customers // 10)" in code:
            # 随机生成新的聚类系数
            factor = random.randint(8, 12)
            code = code.replace("n_clusters = max(1, instance.n_customers // 10)", f"n_clusters = max(1, instance.n_customers // {factor})")
        return code
    
    def _generate_nearest_neighbor_variation(self) -> str:
        """生成最近邻算法的变体"""
        return """
def custom_heuristic(instance):
    unserved = set(range(1, instance.n_customers + 1))
    routes = []
    
    while unserved:
        cap_left = instance.capacity
        current = 0
        route = []
        
        while True:
            feasible = [c for c in unserved if instance.demands[c] <= cap_left]
            if not feasible:
                break
            
            # 考虑距离和需求的加权
            def score(c):
                dx = instance.coords[current][0] - instance.coords[c][0]
                dy = instance.coords[current][1] - instance.coords[c][1]
                dist = (dx * dx + dy * dy) ** 0.5
                demand_ratio = instance.demands[c] / instance.capacity
                return dist - 0.1 * demand_ratio
            
            nxt = min(feasible, key=score)
            route.append(nxt)
            unserved.remove(nxt)
            cap_left -= instance.demands[nxt]
            current = nxt
        
        routes.append(route)
    
    return routes
        """
    
    def _generate_savings_variation(self) -> str:
        """生成节约算法的变体"""
        return """
def custom_heuristic(instance):
    depot = 0
    n = instance.n_customers
    
    # 初始化为单客户路径
    routes = {i: [i] for i in range(1, n + 1)}
    route_of = {i: i for i in range(1, n + 1)}
    route_demand = {i: instance.demands[i] for i in range(1, n + 1)}
    
    # 计算节约值
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # 计算距离
            def distance(a, b):
                dx = instance.coords[a][0] - instance.coords[b][0]
                dy = instance.coords[a][1] - instance.coords[b][1]
                return (dx * dx + dy * dy) ** 0.5
            
            s = distance(i, depot) + distance(depot, j) - distance(i, j)
            savings.append((-s, i, j))  # 负值用于排序
    
    savings.sort()
    
    for _, i, j in savings:
        ri = route_of.get(i)
        rj = route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        
        if route_demand[ri] + route_demand[rj] > instance.capacity:
            continue
        
        # 合并路径
        routes[ri].extend(routes[rj])
        route_demand[ri] += route_demand[rj]
        
        for c in routes[rj]:
            route_of[c] = ri
        
        del routes[rj]
        del route_demand[rj]
    
    return list(routes.values())
        """
    
    def _generate_sweep_variation(self) -> str:
        """生成扫描算法的变体"""
        return """
def custom_heuristic(instance):
    depot = 0
    depot_coords = instance.coords[depot]
    
    # 计算每个客户相对于 depot 的极角
    customer_angles = []
    for i in range(1, instance.n_customers + 1):
        x, y = instance.coords[i]
        dx = x - depot_coords[0]
        dy = y - depot_coords[1]
        # 简单计算角度
        angle = (dy / (dx + 1e-9)) if dx != 0 else 0
        customer_angles.append((angle, i))
    
    # 按极角排序
    customer_angles.sort()
    sorted_customers = [c for _, c in customer_angles]
    
    routes = []
    current_route = []
    current_demand = 0
    
    for customer in sorted_customers:
        if current_demand + instance.demands[customer] <= instance.capacity:
            current_route.append(customer)
            current_demand += instance.demands[customer]
        else:
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_demand = instance.demands[customer]
    
    if current_route:
        routes.append(current_route)
    
    return routes
        """
    
    def _generate_cluster_first_route_second(self) -> str:
        """生成先聚类后路由的算法"""
        return """
def custom_heuristic(instance):
    # K-means聚类
    customers = list(range(1, instance.n_customers + 1))
    n_clusters = max(1, instance.n_customers // 10)  # 简单估计聚类数
    
    # 初始化聚类中心
    import random
    random.seed(42)  # 固定种子以确保可重复性
    centers = [instance.coords[c] for c in random.sample(customers, n_clusters)]
    
    # 计算距离
    def distance(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5
    
    # 迭代聚类
    for _ in range(10):  # 固定迭代次数
        clusters = [[] for _ in range(n_clusters)]
        for c in customers:
            distances = [distance(instance.coords[c], center) for center in centers]
            nearest = distances.index(min(distances))
            clusters[nearest].append(c)
        
        # 更新聚类中心
        new_centers = []
        for cluster in clusters:
            if cluster:
                x = sum(instance.coords[c][0] for c in cluster) / len(cluster)
                y = sum(instance.coords[c][1] for c in cluster) / len(cluster)
                new_centers.append((x, y))
            else:
                new_centers.append(centers[0])  # 避免空聚类
        
        centers = new_centers
    
    # 对每个聚类使用最近邻算法
    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        
        unserved = set(cluster)
        while unserved:
            cap_left = instance.capacity
            current = 0
            route = []
            
            while True:
                feasible = [c for c in unserved if instance.demands[c] <= cap_left]
                if not feasible:
                    break
                
                nxt = min(feasible, key=lambda c: distance(instance.coords[current], instance.coords[c]))
                route.append(nxt)
                unserved.remove(nxt)
                cap_left -= instance.demands[nxt]
                current = nxt
            
            routes.append(route)
    
    return routes
        """
    
    def _generate_regret_heuristic(self) -> str:
        """生成后悔值启发式算法"""
        return """
def custom_heuristic(instance):
    unserved = set(range(1, instance.n_customers + 1))
    routes = []
    
    # 计算距离
    def distance(a, b):
        dx = instance.coords[a][0] - instance.coords[b][0]
        dy = instance.coords[a][1] - instance.coords[b][1]
        return (dx * dx + dy * dy) ** 0.5
    
    # 初始化第一条路线
    if unserved:
        first_customer = min(unserved, key=lambda c: distance(0, c))
        routes.append([first_customer])
        unserved.remove(first_customer)
    
    while unserved:
        # 计算每个未服务客户的后悔值
        regrets = []
        for c in unserved:
            # 计算添加到各路线的成本
            costs = []
            for i, route in enumerate(routes):
                if sum(instance.demands[r] for r in route) + instance.demands[c] <= instance.capacity:
                    # 找到最佳插入位置
                    min_increment = float('inf')
                    for j in range(len(route) + 1):
                        if j == 0:
                            prev = 0
                        else:
                            prev = route[j-1]
                        if j == len(route):
                            next_node = 0
                        else:
                            next_node = route[j]
                        
                        increment = distance(prev, c) + distance(c, next_node) - distance(prev, next_node)
                        if increment < min_increment:
                            min_increment = increment
                    costs.append((i, min_increment))
            
            if costs:
                # 计算后悔值：第二好与最好的差
                costs.sort(key=lambda x: x[1])
                if len(costs) >= 2:
                    regret = costs[1][1] - costs[0][1]
                else:
                    regret = costs[0][1]  # 只有一条路线时，后悔值为成本本身
                regrets.append((-regret, c, costs[0][0], costs[0][1]))  # 负值用于排序
        
        if regrets:
            # 选择后悔值最大的客户
            regrets.sort()
            _, c, route_idx, _ = regrets[0]
            
            # 将客户添加到最佳路线
            route = routes[route_idx]
            best_pos = 0
            min_increment = float('inf')
            
            for j in range(len(route) + 1):
                if j == 0:
                    prev = 0
                else:
                    prev = route[j-1]
                if j == len(route):
                    next_node = 0
                else:
                    next_node = route[j]
                
                increment = distance(prev, c) + distance(c, next_node) - distance(prev, next_node)
                if increment < min_increment:
                    min_increment = increment
                    best_pos = j
            
            route.insert(best_pos, c)
            unserved.remove(c)
        else:
            # 无法添加到现有路线，创建新路线
            new_customer = min(unserved, key=lambda c: distance(0, c))
            routes.append([new_customer])
            unserved.remove(new_customer)
    
    return routes
        """
    
    def validate_heuristic(self, code: str) -> bool:
        """验证生成的启发式算法代码是否有效
        
        Args:
            code: 生成的启发式算法代码
            
        Returns:
            代码是否有效
        """
        try:
            # 检查是否包含必要的函数定义
            if "def custom_heuristic" not in code:
                return False
            
            # 检查是否返回List[List[int]]
            if "return" not in code:
                return False
            
            return True
        except:
            return False
    
    def load_heuristic(self, code: str) -> Optional[Callable[[CVRPInstance], List[List[int]]]]:
        """加载生成的启发式算法
        
        Args:
            code: 生成的启发式算法代码
            
        Returns:
            加载的启发式算法函数，失败返回None
        """
        try:
            # 确保math和random模块在全局命名空间中
            import math
            import random
            from cvrp_core import CVRPInstance
            
            # 创建一个局部命名空间，包含所有必要的依赖
            local_vars = {
                'math': math,
                'random': random,
                'CVRPInstance': CVRPInstance
            }
            
            # 执行生成的代码
            exec(code, globals(), local_vars)
            
            # 获取custom_heuristic函数
            if "custom_heuristic" in local_vars:
                # 确保函数能够访问math模块
                heuristic = local_vars["custom_heuristic"]
                return heuristic
            
            return None
        except Exception as e:
            print(f"Error loading heuristic: {e}")
            return None
