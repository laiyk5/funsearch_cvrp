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


def euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_distance(instance: CVRPInstance, route: list[int]) -> float:
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


def solution_distance(instance: CVRPInstance, routes: list[list[int]]) -> float:
    return sum(route_distance(instance, r) for r in routes)


def nearest_neighbor_heuristic(instance: CVRPInstance) -> list[list[int]]:
    unserved = set(range(1, instance.n_customers + 1))
    routes: list[list[int]] = []
    while unserved:
        cap_left = instance.capacity
        current = 0
        route: list[int] = []
        while True:
            feasible = [c for c in unserved if instance.demands[c] <= cap_left]
            if not feasible:
                break
            nxt = min(feasible, key=lambda c: euclid(instance.coords[current], instance.coords[c]))
            route.append(nxt)
            unserved.remove(nxt)
            cap_left -= instance.demands[nxt]
            current = nxt
        routes.append(route)
    return routes


def weighted_greedy_heuristic(instance: CVRPInstance, weights: tuple[float, float, float]) -> list[list[int]]:
    # score = w1 * (-distance from current) + w2 * (demand ratio) + w3 * (-distance to depot)
    w1, w2, w3 = weights
    unserved = set(range(1, instance.n_customers + 1))
    routes: list[list[int]] = []

    while unserved:
        cap_left = instance.capacity
        current = 0
        route: list[int] = []

        while True:
            feasible = [c for c in unserved if instance.demands[c] <= cap_left]
            if not feasible:
                break

            def score(c: int) -> float:
                d_cur = euclid(instance.coords[current], instance.coords[c])
                d_dep = euclid(instance.coords[c], instance.coords[0])
                demand_ratio = instance.demands[c] / instance.capacity
                return w1 * (-d_cur) + w2 * demand_ratio + w3 * (-d_dep)

            nxt = max(feasible, key=score)
            route.append(nxt)
            unserved.remove(nxt)
            cap_left -= instance.demands[nxt]
            current = nxt

        routes.append(route)

    return routes


def evaluate_heuristic(instances: list[CVRPInstance], solver: Callable[[CVRPInstance], list[list[int]]]) -> dict:
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


def generate_synthetic_benchmarks(seed: int = 2026) -> list[CVRPInstance]:
    rng = random.Random(seed)
    instances: list[CVRPInstance] = []

    for idx, n in enumerate([20, 24, 28, 32, 36], start=1):
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
