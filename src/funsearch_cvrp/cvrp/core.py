from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

type Coord = tuple[float, float]

@dataclass
class CVRPInstance:
    name: str
    capacity: int
    demands: list[int]
    coords: list[Coord]
    number_of_trucks: int | None = None

    @property
    def n_customers(self) -> int:
        return len(self.demands) - 1
    
    def __str__(self):
        return f"CVRPInstance {self.name}: capacity={self.capacity}, n_customers={self.n_customers}"

type Route = list[int]
type Solution = list[Route]


################################################
# EVALUATION
################################################

def euclid(a: Coord, b: Coord) -> float:
    """EUC_2D distance: rounded Euclidean distance per TSPLIB/CVRPLib convention."""
    return round(math.hypot(a[0] - b[0], a[1] - b[1]))


def route_distance(instance: CVRPInstance, route: Route) -> float:
    """
    Calculate the total distance of a single route, including returning to the depot.
    """
    if not route:
        return 0.0
    total = 0.0
    depot = 0
    prev = depot
    for node in route:
        total += euclid(instance.coords[prev], instance.coords[node])
        prev = node
    total += euclid(instance.coords[prev], instance.coords[depot])
    return total


def solution_distance(instance: CVRPInstance, solution: Solution) -> float:
    """
    Calculate the total distance of a complete solution, which is the sum of distances of all routes.
    """
    return sum(route_distance(instance, r) for r in solution)


def is_valid_solution(instance: CVRPInstance, solution: Solution) -> tuple[bool, str]:
    """Check if a CVRP solution is fully valid.

    Validates:
      - No empty routes
      - All customer indices are valid (1..n_customers)
      - Every customer is visited exactly once
      - Total demand of each route does not exceed vehicle capacity

    Returns:
        (is_valid, reason) where reason is empty if valid.
    """
    if not solution:
        return False, "Solution is empty (no routes)"

    visited: set[int] = set()

    for route in solution:
        if not route:
            return False, "Route is empty"

        route_demand = 0
        for node in route:
            if node < 1 or node > instance.n_customers:
                return False, f"Invalid customer index {node} (must be 1..{instance.n_customers})"
            if node in visited:
                return False, f"Customer {node} is visited more than once"
            visited.add(node)
            route_demand += instance.demands[node]

        if route_demand > instance.capacity:
            return False, (
                f"Capacity violation: route {route} total demand {route_demand} "
                f"exceeds capacity {instance.capacity}"
            )

    all_customers = set(range(1, instance.n_customers + 1))
    missing = all_customers - visited
    if missing:
        return False, f"Missing customers: {sorted(missing)}"

    return True, ""



def make_greedy_solver(priority_fn: Callable[..., float]) -> Callable[[CVRPInstance], Solution]:
    """Build a greedy constructive solver using `priority_fn`.

    Args:
        priority_fn: ``(current_node, candidate, instance, remaining_capacity,
                       route, route_demand, unserved) -> float``.
        Higher is better.

    Returns:
        ``solver(instance) -> Solution``
    """
    def solver(instance: CVRPInstance) -> Solution:
        n_customers = instance.n_customers
        capacity = instance.capacity
        demands = instance.demands
        unserved = set(range(1, n_customers + 1))
        routes: Solution = []

        while unserved:
            route: Route = []
            current = 0
            cap_left = capacity
            route_demand = 0

            while True:
                feasible = [c for c in unserved if demands[c] <= cap_left]
                if not feasible:
                    break

                best = max(
                    feasible,
                    key=lambda c: priority_fn(
                        current_node=current,
                        candidate=c,
                        instance=instance,
                        remaining_capacity=cap_left,
                        route=route,
                        route_demand=route_demand,
                        unserved=unserved,
                    ) or 0.0,
                )
                route.append(best)
                unserved.remove(best)
                cap_left -= demands[best]
                route_demand += demands[best]
                current = best

            if not route:
                break  # infeasible (demand > capacity)

            routes.append(route)

        return routes

    return solver


def gap_score(distance: float, optimal: float) -> float:
    """Return percentage gap: (distance - optimal) / optimal."""
    return (distance - optimal) / optimal


def evaluate_solver(instances: list[CVRPInstance], solver: Callable[[CVRPInstance], Solution]) -> dict:
    """
    Evaluate a solver on a list of instances, returning average distance and average number of routes.
    """
    total_distance = 0.0
    total_routes = 0
    per_instance: list[dict] = []

    is_valid_solver = True
    invalid_cases: list[dict] = []
    for inst in instances:
        routes = solver(inst)

        _valid, reason = is_valid_solution(inst, routes)
        if not _valid:
            print(f"Invalid solution: solution {routes} is invalid for instance {inst.name}")
            print(f"Reason: {reason}")
            is_valid_solver = False
            invalid_cases.append({
                "instance": inst.name,
                "solution": routes,
                "reason": reason,
            })

        dist = solution_distance(inst, routes)
        total_distance += dist
        total_routes += len(routes)
        per_instance.append(
            {
                "instance": inst.name,
                "distance": round(dist, 3),
                "num_routes": len(routes),
            }
        )

    return {
        "is_valid_solver": is_valid_solver,
        "invalid_cases": invalid_cases,
        "avg_distance": total_distance / len(instances),
        "avg_num_routes": total_routes / len(instances),
        "details": per_instance,
    }

