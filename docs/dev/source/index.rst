FunSearch CVRP 开发者文档
==========================

.. toctree::
   :maxdepth: 2
   :caption: 目录:

   data

简介
----

本项目是一个基于 **FunSearch** 方法的 **CVRP（容量限制车辆路径问题）** 启发式算法自动发现框架。

通过结合大语言模型（LLM）和进化搜索，本框架能够自动生成高性能的 CVRP 求解算法，无需人工设计启发式规则。

主要特性
--------

- **自动算法生成**：使用 LLM 迭代生成候选算法
- **样本高效搜索**：通过早期剪枝减少评估开销
- **功能等价检测**：避免重复评估语义相同的算法
- **多尺度评估**：在小/中/大规模数据集上测试
- **标准基准测试**：支持 CVRPLib 标准数据集

快速开始
--------

.. code-block:: bash

   # 安装依赖
   uv pip install -e .

   # 运行快速测试
   python scripts/test_simple.py

   # 运行完整实验
   python scripts/run_full_project.py --dataset synthetic

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
