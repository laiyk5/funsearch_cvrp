CVRP Problem
============

Problem Definition
------------------

The **Capacitated Vehicle Routing Problem (CVRP)** is a classic combinatorial
optimisation problem in operations research.

Scenario
^^^^^^^^

A distribution centre (depot) must deliver goods to a set of customers using a fleet of vehicles:

- Each customer has a demand quantity
- Each vehicle has a capacity limit
- Vehicles depart from the depot, serve customers, and return to the depot
- Objective: minimise total travel distance

Mathematical Definition
-----------------------

Input
^^^^^

Given a CVRP instance :math:`I = (G, Q, d, c)`:

.. list-table:: CVRP Input Parameters
   :header-rows: 1

   * - Symbol
     - Meaning
     - Notes
   * - :math:`G = (V, E)`
     - Undirected graph
     - :math:`V = \{0, 1, \ldots, n\}`, node 0 is the depot, 1–n are customers
   * - :math:`Q`
     - Vehicle capacity
     - Maximum load per vehicle
   * - :math:`d_i`
     - Demand of customer :math:`i`
     - :math:`d_i \leq Q`; depot has :math:`d_0 = 0`
   * - :math:`c_{ij}`
     - Distance
     - Euclidean distance between nodes :math:`i` and :math:`j`

Constraints
^^^^^^^^^^^

1. **Each route starts and ends at the depot**

   Route form: depot → customer₁ → customer₂ → … → depot

2. **Capacity constraint**

   Total demand on each route must not exceed vehicle capacity:

   .. math::

      \sum_{i \in \text{route}} d_i \leq Q

3. **Each customer is visited exactly once**

4. **Number of vehicles is unlimited** (but minimising vehicle count is desirable)

Objective
^^^^^^^^^

Minimise total travel distance:

.. math::

   \min \sum_{k=1}^{K} \sum_{(i,j) \in \text{route}_k} c_{ij}

where :math:`K` is the number of vehicles used.

Score Function
^^^^^^^^^^^^^^

This project uses the following composite score to evaluate solver performance:

.. math::

   \text{score} = \text{avg\_distance} + 20 \times \text{avg\_num\_routes}

- **Distance term**: average total route distance across instances — lower is better
- **Route count term**: average number of vehicles used — lower is better
- **Weight 20**: hyperparameter balancing the two objectives

A lower score indicates better performance.

Representation
--------------

Route
^^^^^

A route is an ordered list of customer indices (integers), implicitly starting and
ending at the depot (node 0). The depot is **not** included in the list.

.. code-block:: python

   route = [3, 7, 2]   # depot → 3 → 7 → 2 → depot

A solution is a list of routes — one per vehicle used:

.. code-block:: python

   solution = [[3, 7, 2], [1, 5], [4, 6, 8]]

In code, these types are aliased as:

.. code-block:: python

   type Route    = list[int]
   type Solution = tuple[list[Route], float]   # (routes, total_distance)

Distance
^^^^^^^^

All distances are **Euclidean** (straight-line) between 2-D coordinates:

.. math::

   c_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}

The distance of a single route includes the return leg to the depot:

.. math::

   \text{dist}(\text{route}) = c_{0,\, r_1} + \sum_{k=1}^{m-1} c_{r_k,\, r_{k+1}} + c_{r_m,\, 0}

The total solution distance is the sum over all routes:

.. math::

   \text{dist}(\text{solution}) = \sum_{k} \text{dist}(\text{route}_k)

What makes a good solution?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A solution is **feasible** if every customer is visited exactly once and no route
exceeds vehicle capacity. Among feasible solutions, a good one minimises the
composite score:

.. math::

   \text{score} = \text{avg\_distance} + 20 \times \text{avg\_num\_routes}

Intuitively: short total travel distance with as few vehicles as possible.

What makes a good solver?
^^^^^^^^^^^^^^^^^^^^^^^^^^

A good solver produces low-score solutions consistently across instances of varying
size. Concretely, it should:

- **Be fast** — practical solvers run in seconds, not hours
- **Generalise** — perform well on both small and large instances without tuning
- **Be stable** — low variance across different random instances
- **Approach optimality** — a Gap below ~10% is considered competitive for heuristics

Complexity
----------

CVRP is **NP-hard**:

- The number of feasible solutions grows exponentially with :math:`n`
- Exact solvers for :math:`n = 50` may take hours
- For large instances (:math:`n > 100`), heuristics are the practical choice

Why is it hard?
^^^^^^^^^^^^^^^

CVRP is a generalisation of the **Travelling Salesman Problem (TSP)**: if you set
vehicle capacity to infinity and use a single vehicle, CVRP reduces exactly to TSP.
Since TSP is already NP-hard, CVRP inherits that hardness — and adds more on top.

The difficulty compounds from two sources:

1. **Partitioning** — dividing :math:`n` customers into :math:`K` routes is a set
   partition problem. The number of ways to do this grows as :math:`O(K^n)` (Stirling
   numbers of the second kind), which is super-exponential in :math:`n`.
2. **Routing** — once customers are assigned to a vehicle, finding the optimal visit
   order within that route is itself a TSP. So you face an exponential search inside
   each partition of an already exponential partition space.
3. **Conflicting objectives** — minimising distance vs. minimising vehicle count
4. **Coupled constraints** — capacity limits and route planning interact

References
----------

1. **Dantzig & Ramser (1959)** — *"The Truck Dispatching Problem"* — founding paper for CVRP
2. **Clarke & Wright (1964)** — *"Scheduling of Vehicles from a Central Depot"* — savings algorithm
