CVRP 问题描述
=============

问题定义
--------

**CVRP**（Capacitated Vehicle Routing Problem，容量限制车辆路径问题）是运筹学中的经典组合优化问题。

问题场景
^^^^^^^^

一个配送中心（仓库）需要用一组车辆为多个客户配送货物：

- 每个客户有一个需求量
- 每辆车有容量限制
- 车辆从仓库出发，服务完客户后返回仓库
- 目标：最小化总行驶距离

数学定义
--------

输入
^^^^

给定一个 CVRP 实例 :math:`I = (G, Q, d, c)`：

.. list-table:: CVRP 输入参数
   :header-rows: 1

   * - 符号
     - 含义
     - 说明
   * - :math:`G = (V, E)`
     - 无向图
     - :math:`V = \\{0, 1, ..., n\\}`，0 表示仓库，1~n 表示客户
   * - :math:`Q`
     - 车辆容量
     - 每辆车的最大载重量
   * - :math:`d_i`
     - 客户 :math:`i` 的需求量
     - 满足 :math:`d_i \\leq Q`，且 :math:`d_0 = 0`（仓库无需求）
   * - :math:`c_{ij}`
     - 距离
     - 节点 :math:`i` 到节点 :math:`j` 的欧几里得距离

约束条件
^^^^^^^^

1. **每条路线必须从仓库开始并结束**
   
   路线形式：仓库 → 客户₁ → 客户₂ → ... → 仓库

2. **容量约束**
   
   每条路线上所有客户的总需求不超过车辆容量：
   
   .. math::
   
      \\sum_{i \\in \\text{route}} d_i \\leq Q

3. **每个客户被访问恰好一次**
   
   所有客户必须被服务，且只能被一辆车服务

4. **车辆数量不限（但通常希望最少）**
   
   可以使用多辆车，目标是总距离最短

目标函数
^^^^^^^^

最小化总行驶距离：

.. math::

   \\min \\sum_{k=1}^{K} \\sum_{(i,j) \\in \\text{route}_k} c_{ij}

其中 :math:`K` 是使用的车辆数。

统一评分函数
^^^^^^^^^^^^

在本项目中，我们使用以下评分函数来综合评估算法性能：

.. math::

   \\text{score} = \\text{average distance} + 20 \\times \\text{average number of routes}

**说明：**

- **距离项**：路径总距离，越小越好
- **路线数项**：使用的车辆数，越少越好（节省资源）
- **权重 20**：平衡距离和车辆数的超参数

评分越低表示算法性能越好。

问题复杂度
----------

CVRP 是 **NP-hard** 问题：

- 随着客户数 :math:`n` 增加，可行解的数量呈指数增长
- 对于 :math:`n = 50` 的问题，精确求解可能需要数小时
- 对于大规模问题（:math:`n > 100`），通常使用启发式算法

为什么难？
^^^^^^^^^^

1. **组合爆炸**
   
   将 :math:`n` 个客户分配给 :math:`K` 辆车的方案数是指数级的

2. **两个相互冲突的目标**
   
   - 距离最短 ↔ 车辆数最少
   - 装满一辆车可能增加行驶距离

3. **约束耦合**
   
   容量约束和路径规划相互影响

经典启发式算法
--------------

1. 最近邻算法（Nearest Neighbor）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**思路**：每次选择距离当前位置最近的未服务客户

**步骤**：

1. 从仓库出发
2. 选择距离当前位置最近且能装下的客户
3. 重复直到车辆满载或无客户可服务
4. 返回仓库，开启新路线

**特点**：

- ✓ 简单快速
- ✗ 容易陷入局部最优

2. Clarke-Wright 节约算法（Savings Algorithm）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**思路**：计算合并两条路线的"节约值"，优先合并节约值大的

**节约值计算**：

.. math::

   s_{ij} = c_{i0} + c_{0j} - c_{ij}

其中 :math:`0` 是仓库。

**特点**：

- ✓ 比最近邻效果好
- ✓ 经典的构造式启发算法
- ✗ 对容量约束敏感

3. Sweep 算法
^^^^^^^^^^^^^

**思路**：按极角将客户分组，每组服务一辆车

**步骤**：

1. 以仓库为原点，计算每个客户的极角
2. 按极角排序客户
3. 按顺序分配客户到车辆，直到容量满

**特点**：

- ✓ 适合地理分布有规律的问题
- ✗ 性能受客户分布影响大

FunSearch 方法
--------------

传统方法的局限
^^^^^^^^^^^^^^

- 需要人工设计启发式规则
- 泛化能力差（在不同规模问题上表现不稳定）
- 难以找到全局最优

FunSearch 的创新
^^^^^^^^^^^^^^^^

使用 **大语言模型（LLM）+ 进化搜索** 自动生成算法：

1. **LLM 生成**：根据提示词生成候选算法代码
2. **评估筛选**：在数据集上评估性能
3. **进化优化**：选择优秀算法作为下一轮的基础
4. **迭代改进**：多轮迭代后得到高质量算法

本项目的评分体系
^^^^^^^^^^^^^^^^

.. list-table:: 评估维度
   :header-rows: 1

   * - 指标
     - 说明
     - 权重
   * - 平均距离
     - 所有测试实例的路径总距离均值
     - 1.0
   * - 路线数量
     - 使用的车辆数（越少越好）
     - 20.0
   * - 稳定性
     - 不同规模数据上的标准差
     - 优先选择
   * - Gap
     - 与最优解的差距百分比
     - 参考指标

Gap 计算公式：

.. math::

   \\text{Gap} = \\frac{\\text{算法距离} - \\text{最优解距离}}{\\text{最优解距离}} \\times 100\\%

例如：

- 算法距离 = 850
- 最优解距离 = 784
- Gap = (850-784)/784 × 100% = **8.4%**

项目中的实现
------------

核心模块
^^^^^^^^

.. code-block:: text

   src/cvrp/
   ├── core.py       # CVRPInstance 类、距离计算、评估
   ├── baselines.py  # 经典算法（最近邻、Clarke-Wright）
   ├── search.py     # FunSearch 搜索框架
   └── io.py         # 数据集加载

使用示例
^^^^^^^^

**创建实例并求解：**

.. code-block:: python

   from src.cvrp.core import CVRPInstance, nearest_neighbor_heuristic

   # 创建实例
   instance = CVRPInstance(
       name="test",
       capacity=100,
       demands=[0, 10, 20, 15],  # 仓库 + 3个客户
       coords=[(0,0), (10,0), (0,10), (10,10)]
   )

   # 使用最近邻算法求解
   routes = nearest_neighbor_heuristic(instance)
   # 输出: [[1, 2], [3]] 表示两条路线

**评估算法：**

.. code-block:: python

   from src.cvrp.core import evaluate_heuristic

   instances = [...]  # 加载多个测试实例

   metrics = evaluate_heuristic(instances, my_heuristic)
   print(f"平均距离: {metrics['avg_distance']}")
   print(f"平均路线数: {metrics['avg_num_routes']}")

相关资源
--------

经典论文
^^^^^^^^

1. **Dantzig & Ramser (1959)** - "The Truck Dispatching Problem"
   
   CVRP 问题的开创性论文

2. **Clarke & Wright (1964)** - "Scheduling of Vehicles from a Central Depot"
   
   节约算法的经典论文

3. **FunSearch Paper (2023)** - "FunSearch: Discovering Novel Algorithms"
   
   Nature 期刊，LLM 驱动算法发现的里程碑工作

在线资源
^^^^^^^^

- `CVRPLib <http://vrp.atd-lab.inf.puc-rio.br/>`_ - 标准测试数据集
- `OR-Tools <https://developers.google.com/optimization/routing>`_ - Google 的路由优化库
