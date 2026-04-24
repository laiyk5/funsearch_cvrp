"""FunSearch specification for CVRP.

This module contains only the function to evolve.  The full solver harness
lives in ``funsearch_cvrp.cvrp.core`` and is passed to FunSearch as
``evaluate_fn``.
"""


def priority(
  current_node: int,
  candidate: int,
  instance,
  remaining_capacity: int,
  route: list,
  route_demand: int,
  unserved: set,
) -> float:
  """Score a candidate customer for the greedy route builder.

  Higher score = more likely `candidate` is chosen as the next stop.

  Args:
    current_node: The node the vehicle is currently at.
    candidate: The customer under consideration.
    instance: CVRP instance with attributes ``coords``, ``demands``,
      ``capacity``, ``n_customers``.
    remaining_capacity: Unused capacity on the current vehicle.
    route: Customers already on the current partial route.
    route_demand: Sum of demands of customers already on ``route``.
    unserved: Set of customers not yet in any route.

  Returns:
    A float priority.  Higher is better.
  """
  import math
  d_cur = math.hypot(
    instance.coords[current_node][0] - instance.coords[candidate][0],
    instance.coords[current_node][1] - instance.coords[candidate][1],
  )
  return -d_cur
