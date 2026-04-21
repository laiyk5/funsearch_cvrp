Baseline Solvers
================

All baseline solvers are implemented in ``src/funsearch_cvrp/cvrp/baselines.py``.
Each solver takes a :class:`CVRPInstance` and returns ``list[list[int]]`` — a list of routes,
where each route is an ordered list of customer indices (depot not included).

Nearest Neighbor
----------------

Greedy construction heuristic. At each step, the vehicle moves to the nearest
unserved customer that fits within remaining capacity. When no feasible customer
exists, the route ends and a new vehicle departs from the depot.

**Time complexity**: :math:`O(n^2)` per instance.

**Weakness**: Greedy choices early in a route can force long detours later.

.. code-block:: python

   from funsearch_cvrp.cvrp.baselines import nearest_neighbor_heuristic
   routes = nearest_neighbor_heuristic(instance)

Clarke-Wright Savings
---------------------

Constructive heuristic based on the savings algorithm (Clarke & Wright, 1964).

**Idea**: Start with one dedicated vehicle per customer (depot → customer → depot).
Merging two routes :math:`(\ldots, i)` and :math:`(j, \ldots)` saves:

.. math::

   s_{ij} = c_{i,\text{depot}} + c_{\text{depot},j} - c_{ij}

Routes are merged in decreasing order of savings, subject to capacity constraints
and the requirement that the merged customers are at the ends of their respective routes
(parallel Clarke-Wright variant).

**Time complexity**: :math:`O(n^2 \log n)` (dominated by sorting savings list).

.. code-block:: python

   from funsearch_cvrp.cvrp.baselines import clarke_wright_savings_heuristic
   routes = clarke_wright_savings_heuristic(instance)

2-opt Local Search
------------------

Post-processing improvement applied independently to each route. Repeatedly reverses
sub-sequences of a route while the reversal reduces total route distance. Stops when
no improving swap exists.

.. math::

   \Delta = d(r_{i-1}, r_j) + d(r_i, r_{j+1}) - d(r_{i-1}, r_i) - d(r_j, r_{j+1})

A swap is accepted when :math:`\Delta < 0`.

**Note**: 2-opt only reorders customers within a route — it does not move customers
between routes.

.. code-block:: python

   from funsearch_cvrp.cvrp.baselines import two_opt_route, two_opt_improvement

   # Improve a single route
   better_route = two_opt_route(instance, route)

   # Improve all routes in a solution
   better_routes = two_opt_improvement(instance, routes)

Composing with ``with_two_opt``
--------------------------------

``with_two_opt`` is a decorator that wraps any solver with a 2-opt post-processing step:

.. code-block:: python

   from funsearch_cvrp.cvrp.baselines import with_two_opt, nearest_neighbor_heuristic

   solver = with_two_opt(nearest_neighbor_heuristic)
   routes = solver(instance)

Weighted Greedy
---------------

Generalisation of the nearest-neighbor heuristic parameterised by three weights
:math:`(w_1, w_2, w_3)`. At each step the next customer is chosen to maximise:

.. math::

   \text{score}(c) = w_1 \cdot (-d_{\text{cur},c}) + w_2 \cdot \frac{d_c}{Q} + w_3 \cdot (-d_{c,\text{depot}})

where :math:`d_{\text{cur},c}` is distance from current position, :math:`d_c / Q` is
the demand ratio, and :math:`d_{c,\text{depot}}` is distance from customer to depot.

Setting :math:`w_1=1, w_2=0, w_3=0` recovers the nearest-neighbor heuristic.
The weights are the search target for the sample-efficient search in
``src/funsearch_cvrp/search/``.

.. code-block:: python

   from funsearch_cvrp.cvrp.baselines import weighted_greedy_heuristic

   routes = weighted_greedy_heuristic(instance, weights=(1.0, 0.5, 0.2))

Benchmark Results
-----------------

Results on synthetic instances (sizes 20, 50, 100; seed 2026):

.. list-table::
   :header-rows: 1

   * - Solver
     - Avg Distance
     - Avg Routes
   * - nearest_neighbor
     - 1695.33
     - 12.67
   * - nearest_neighbor + 2-opt
     - 1662.33
     - 12.67
   * - clarke_wright
     - 1289.67
     - 13.00
   * - clarke_wright + 2-opt
     - 1288.67
     - 13.00
