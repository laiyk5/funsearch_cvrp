Datasets
========

This project uses the **CVRPLib A-instances** as the standard benchmark.

What is CVRPLib?
----------------

**CVRPLib** (Capacitated Vehicle Routing Problem Library) is the community-standard
benchmark library for CVRP research — analogous to ImageNet for vision or GLUE for NLP.
Researchers use these datasets to compare algorithm performance on a level playing field.

A-instances
-----------

The A-instances were created by Augerat et al. (1995) and are among the most widely
used CVRP benchmarks.

Naming Convention
^^^^^^^^^^^^^^^^^

.. code-block:: text

   A-n32-k5
   │  │   └── 5 vehicles required
   │  └────── 32 nodes (1 depot + 31 customers)
   └───────── A-class instances (Augerat et al.)

Scale Distribution
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Scale
     - Customers
     - Example files
     - Use
   * - Small
     - 32–35
     - A-n32-k5, A-n35-k5
     - Quick testing / early pruning
   * - Medium
     - 36–55
     - A-n44-k6, A-n55-k9
     - Standard evaluation
   * - Large
     - 56–80
     - A-n69-k9, A-n80-k10
     - Stress testing

File Format
-----------

.vrp file (problem definition)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   NAME : A-n32-k5
   COMMENT : (Augerat et al, No of trucks: 5, Optimal value: 784)
   TYPE : CVRP
   DIMENSION : 32           # total nodes including depot
   EDGE_WEIGHT_TYPE : EUC_2D
   CAPACITY : 100

   NODE_COORD_SECTION
    1 82 76                 # node 1: depot
    2 96 44                 # node 2: customer 1
    3 50 5
    ...

   DEMAND_SECTION
    1 0                     # depot demand = 0
    2 19
    3 21
    ...

.sol file (optimal solution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Route #1: 2 11 7
   Route #2: 5 8 3
   Route #3: 4 10 9
   Cost 784

Usage
-----

Loading instances
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from funsearch_cvrp.cvrp.io import load_cvrplib_folder

   instances_and_solutions = load_cvrplib_folder("data/cvrplib/A")
   instances = [inst for inst, _ in instances_and_solutions]

Computing gap to optimal
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from funsearch_cvrp.cvrp.core import solution_distance

   for inst, opt_routes in instances_and_solutions:
       my_routes = my_heuristic(inst)
       my_dist = solution_distance(inst, my_routes)
       opt_dist = solution_distance(inst, opt_routes)
       gap = (my_dist - opt_dist) / opt_dist * 100
       print(f"{inst.name}: gap={gap:.1f}%")

Progressive evaluation strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Scale
     - Customers
     - Purpose
   * - Small
     - ≤35
     - Fast candidate screening (early pruning)
   * - Medium
     - 36–55
     - Mid-scale performance
   * - Large
     - ≥56
     - Final quality assessment

Data Location
-------------

.. code-block:: text

   data/
   └── cvrplib/
       └── A/
           ├── A-n32-k5.vrp
           ├── A-n32-k5.sol
           ├── A-n33-k5.vrp
           ├── A-n33-k5.sol
           └── ...

Source
------

- **Created by**: Augerat et al. (1995)
- **Paper**: *"Computational results with a branch and cut code for the capacitated vehicle routing problem"*
- **Website**: http://vrp.atd-lab.inf.puc-rio.br/
