from __future__ import annotations

from typing import Callable

from cvrp_core import CVRPInstance, euclid, route_distance


def clarke_wright_savings_heuristic(instance: CVRPInstance) -> list[list[int]]:
    """Construct routes using a simple parallel Clarke-Wright savings heuristic."""
    depot = 0
    n = instance.n_customers

    # Start with one-customer routes.
    routes: dict[int, list[int]] = {i: [i] for i in range(1, n + 1)}
    route_of: dict[int, int] = {i: i for i in range(1, n + 1)}
    route_demand: dict[int, int] = {i: instance.demands[i] for i in range(1, n + 1)}

    savings: list[tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = (
                euclid(instance.coords[i], instance.coords[depot])
                + euclid(instance.coords[depot], instance.coords[j])
                - euclid(instance.coords[i], instance.coords[j])
            )
            savings.append((s, i, j))

    savings.sort(reverse=True, key=lambda x: x[0])

    for _, i, j in savings:
        ri = route_of.get(i)
        rj = route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Merge only when i and j are route ends.
        i_is_head = route_i[0] == i
        i_is_tail = route_i[-1] == i
        j_is_head = route_j[0] == j
        j_is_tail = route_j[-1] == j

        if not (i_is_head or i_is_tail) or not (j_is_head or j_is_tail):
            continue

        if route_demand[ri] + route_demand[rj] > instance.capacity:
            continue

        if i_is_head:
            route_i = list(reversed(route_i))
        if j_is_tail:
            route_j = list(reversed(route_j))

        merged = route_i + route_j

        routes[ri] = merged
        route_demand[ri] = route_demand[ri] + route_demand[rj]

        for c in route_j:
            route_of[c] = ri

        del routes[rj]
        del route_demand[rj]

    return list(routes.values())


def two_opt_route(instance: CVRPInstance, route: list[int]) -> list[int]:
    """Run 2-opt on a single route sequence (without changing customer set)."""
    if len(route) < 4:
        return route

    best = route[:]
    best_dist = route_distance(instance, best)
    improved = True

    while improved:
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                cand_dist = route_distance(instance, candidate)
                if cand_dist + 1e-9 < best_dist:
                    best = candidate
                    best_dist = cand_dist
                    improved = True
                    break
            if improved:
                break

    return best


def two_opt_improvement(instance: CVRPInstance, routes: list[list[int]]) -> list[list[int]]:
    """Apply 2-opt independently to each route."""
    return [two_opt_route(instance, r) for r in routes]


def with_two_opt(
    base_solver: Callable[[CVRPInstance], list[list[int]]],
) -> Callable[[CVRPInstance], list[list[int]]]:
    """Compose a solver with route-level 2-opt improvement."""

    def solver(instance: CVRPInstance) -> list[list[int]]:
        return two_opt_improvement(instance, base_solver(instance))

    return solver
