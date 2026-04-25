"""FunSearch specification for CVRP using a savings-based construction.

This module contains the function to evolve.  The solver harness lives in
``funsearch_cvrp.cvrp.core`` and is passed to FunSearch as ``evaluate_fn``.
"""


def savings(i: int, j: int, instance) -> float:
  """Score the benefit of merging a route ending at `i` with one starting at `j`.

  Higher score = more likely the merge happens.

  Args:
    i: Last customer of the first route.
    j: First customer of the second route.
    instance: CVRP instance with attributes ``coords``, ``demands``,
      ``capacity``, ``n_customers``.

  Returns:
    A float savings value.  Higher is better.
  """
  import math

  # Calculate distances
  d_i_depot = math.hypot(
    instance.coords[i][0] - instance.coords[0][0],
    instance.coords[i][1] - instance.coords[0][1],
  )
  d_depot_j = math.hypot(
    instance.coords[0][0] - instance.coords[j][0],
    instance.coords[0][1] - instance.coords[j][1],
  )
  d_i_j = math.hypot(
    instance.coords[i][0] - instance.coords[j][0],
    instance.coords[i][1] - instance.coords[j][1],
  )

  # Guard against division by zero or identical coordinates.
  # Some instances have customers at the same location as the depot
  # or at the exact same coordinates as another customer.
  if d_i_depot < 1e-9 or d_i_j < 1e-9:
      return float('inf')

  # Classic Clarke-Wright savings
  base_savings = d_i_depot + d_depot_j - d_i_j

  return base_savings
