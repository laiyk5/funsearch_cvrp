FunSearch CVRP Developer Docs
==============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   cvrp
   baselines
   funsearch
   data

Introduction
------------

This project is an automatic CVRP heuristic discovery framework based on the **FunSearch** method.
By combining Large Language Models (LLMs) with evolutionary search, the framework generates
high-performance CVRP solvers without hand-crafted heuristic rules.

Key Features
------------

- **Automatic algorithm generation**: LLM iteratively generates candidate algorithms
- **Sample-efficient search**: early pruning reduces evaluation overhead
- **Functional equivalence detection**: avoids re-evaluating semantically identical algorithms
- **Multi-scale evaluation**: tests on small / medium / large instances
- **Standard benchmarks**: supports CVRPLib benchmark datasets

Quick Start
-----------

.. code-block:: bash

   # Install dependencies
   uv pip install -e .

   # Run baselines
   uv run python scripts/run/run_baselines.py --dataset synthetic

   # Run FunSearch
   uv run python scripts/run/run_funsearch.py specifications/specification_cvrp.txt

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
