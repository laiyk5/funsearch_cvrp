from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable


@dataclass
class CVRPInstance:
    name: str
    capacity: int
    demands: list[int]
    coords: list[tuple[float, float]]

    @property
    def n_customers(self) -> int:
        return len(self.demands) - 1

type Route = list[int]
type Solution = tuple[list[Route], float]  # (routes, total_distance)
type Coord = tuple[float, float]

def euclid(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def solution_distance(instance: CVRPInstance, routes: list[Route]) -> float:
    """
    Calculate the total distance of a complete solution, which is the sum of distances of all routes.
    """
    return sum(route_distance(instance, r) for r in routes)



def evaluate_heuristic(instances: list[CVRPInstance], solver: Callable[[CVRPInstance], list[Route]]) -> dict:
    """
    Evaluate a heuristic solver on a list of instances, returning average distance and average number of routes.
    """
    total_distance = 0.0
    total_routes = 0
    per_instance: list[dict] = []

    for inst in instances:
        routes = solver(inst)
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
        "avg_distance": total_distance / len(instances),
        "avg_num_routes": total_routes / len(instances),
        "details": per_instance,
    }


def generate_synthetic_benchmarks(seed: int = 2026, sizes: list[int] = [20, 50, 100]) -> list[CVRPInstance]:
    """
    Generate synthetic CVRP instances with random coordinates and demands.
    """

    rng = random.Random(seed)
    instances: list[CVRPInstance] = []

    for idx, n in enumerate(sizes, start=1):
        coords = [(50.0, 50.0)]
        demands = [0]
        for _ in range(n):
            coords.append((rng.uniform(0, 100), rng.uniform(0, 100)))
            demands.append(rng.randint(1, 10))

        capacity = max(25, int(sum(demands) / (n / 4.0)))
        instances.append(
            CVRPInstance(
                name=f"SYN-{idx:02d}-N{n}",
                capacity=capacity,
                demands=demands,
                coords=coords,
            )
        )

    return instances


def check_capacity_constraint(instance: CVRPInstance, routes: list[Route]) -> bool:
    """检查路由是否满足容量约束
    
    Args:
        instance: CVRP实例
        routes: 路由列表
    
    Returns:
        是否满足容量约束
    """
    for route in routes:
        total_demand = 0
        for customer in route:
            total_demand += instance.demands[customer]
        if total_demand > instance.capacity:
            print(f"  容量约束违反: 路由 {route} 的总需求 {total_demand} 超过车辆容量 {instance.capacity}")
            return False
    return True