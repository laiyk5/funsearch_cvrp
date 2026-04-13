from __future__ import annotations

from typing import Callable

from .core import CVRPInstance, Route, euclid, route_distance


def clarke_wright_savings_heuristic(instance: CVRPInstance) -> list[Route]:
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


def two_opt_route(instance: CVRPInstance, route: Route) -> Route:
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


def two_opt_improvement(instance: CVRPInstance, routes: list[Route]) -> list[Route]:
    """Apply 2-opt independently to each route."""
    return [two_opt_route(instance, r) for r in routes]


def with_two_opt(
    base_solver: Callable[[CVRPInstance], list[Route]],
) -> Callable[[CVRPInstance], list[Route]]:
    """Compose a solver with route-level 2-opt improvement."""

    def solver(instance: CVRPInstance) -> list[Route]:
        return two_opt_improvement(instance, base_solver(instance))

    return solver



def nearest_neighbor_heuristic(instance: CVRPInstance) -> list[Route]:
    """
    Solve the CVRP using the nearest neighbor heuristic.
    """
    unserved = set(range(1, instance.n_customers + 1))
    routes: list[Route] = []
    while unserved:
        cap_left = instance.capacity
        current = 0
        route: Route = []
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



def weighted_greedy_heuristic(instance: CVRPInstance, weights: tuple[float, float, float]) -> list[Route]:
    # score = w1 * (-distance from current) + w2 * (demand ratio) + w3 * (-distance to depot)
    w1, w2, w3 = weights
    unserved = set(range(1, instance.n_customers + 1))
    routes: list[Route] = []

    while unserved:
        cap_left = instance.capacity
        current = 0
        route: Route = []

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
