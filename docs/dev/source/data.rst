数据集
=====

本项目使用 **CVRPLib** 的 **A-instances** 数据集作为标准测试基准。

什么是 CVRPLib？
---------------

**CVRPLib**（Capacitated Vehicle Routing Problem Library）是学术界公认的 CVRP **标准测试库**，类似于：

- 图像领域的 ImageNet
- NLP 领域的 GLUE 基准

研究人员使用这些数据集来**公平比较不同算法的性能**。

A-instances 数据集
------------------

A-instances 是由 Augerat 等人在 1995 年创建的经典 CVRP 测试集。

命名规则
^^^^^^^^

.. code-block:: text

   A-n32-k5
   │  │   └── 需要 5 辆车
   │  └────── 32 个节点（1个仓库 + 31个客户）
   └───────── A 类实例（Augerat 等人创建）

规模分布
^^^^^^^^

.. list-table:: 数据集规模分布
   :header-rows: 1

   * - 规模
     - 客户数
     - 文件示例
     - 用途
   * - 小规模
     - 32-35
     - A-n32-k5, A-n35-k5
     - 快速测试
   * - 中规模
     - 36-55
     - A-n44-k6, A-n55-k9
     - 标准测试
   * - 大规模
     - 56-80
     - A-n69-k9, A-n80-k10
     - 压力测试

文件格式
--------

.vrp 文件（问题定义）
^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   NAME : A-n32-k5          # 实例名称
   COMMENT : (Augerat et al, No of trucks: 5, Optimal value: 784)
   TYPE : CVRP              # 问题类型
   DIMENSION : 32           # 总节点数（含仓库）
   EDGE_WEIGHT_TYPE : EUC_2D  # 欧几里得距离
   CAPACITY : 100           # 车辆容量

   NODE_COORD_SECTION       # 节点坐标（x, y）
    1 82 76                 # 1号节点：仓库（配送中心）
    2 96 44                 # 2号节点：客户1，坐标(96,44)
    3 50 5                  # 3号节点：客户2，坐标(50,5)
    ...

   DEMAND_SECTION           # 需求量
    1 0                     # 仓库需求为0
    2 19                    # 客户1需求19
    3 21                    # 客户2需求21
    ...

.sol 文件（最优解）
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Route #1: 2 11 7        # 第1辆车服务客户2,11,7
   Route #2: 5 8 3         # 第2辆车服务客户5,8,3
   Route #3: 4 10 9        # ...
   Cost 784                 # 最优总距离

在项目中使用
------------

评估生成的算法
^^^^^^^^^^^^^^

.. code-block:: python

   from src.cvrp.io import load_cvrplib_folder
   from src.cvrp.core import solution_distance

   # 加载数据集
   instances = load_cvrplib_folder("data/A")

   # 评估算法
   for inst in instances:
       routes = my_heuristic(inst)
       distance = solution_distance(inst, routes)
       print(f"{inst.name}: 距离={distance:.1f}")

与最优解比较
^^^^^^^^^^^^

代码会计算 **Gap**（差距百分比）：

.. math::

   \text{Gap} = \frac{\text{你的算法距离} - \text{最优解距离}}{\text{最优解距离}} \times 100\%

示例：

.. code-block:: text

   你的算法距离: 850
   最优解距离: 784
   Gap = (850-784)/784 × 100% = 8.4%

渐进式测试策略
^^^^^^^^^^^^^^

.. list-table:: 测试策略
   :header-rows: 1

   * - 数据规模
     - 客户数
     - 用途
   * - 小数据
     - ≤35
     - 快速筛选算法（早期剪枝）
   * - 中数据
     - 36-55
     - 评估中等规模表现
   * - 大数据
     - ≥56
     - 最终评估算法质量

数据来源
--------

- **创建者**: Augerat et al. (1995)
- **经典论文**: *"Computational results with a branch and cut code for the capacitated vehicle routing problem"*
- **官网**: http://vrp.atd-lab.inf.puc-rio.br/

文件位置
--------

项目中的数据集文件位置：

.. code-block:: text

   data/
   └── A/
       ├── A-n32-k5.vrp    # 问题定义
       ├── A-n32-k5.sol    # 最优解
       ├── A-n33-k5.vrp
       ├── A-n33-k5.sol
       └── ...
