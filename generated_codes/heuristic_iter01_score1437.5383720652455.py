"""
FunSearch 生成的启发式算法
来源: iterative_search_results.json
迭代: 1
评分: 1437.5383720652455
"""

def custom_heuristic(instance):
    n_customers = instance.n_customers
    capacity = instance.capacity
    coords = instance.coords
    demands = instance.demands

    def euclidean_distance(i, j):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    depot = 0
    unvisited = set(range(1, n_customers + 1))
    routes = []

    # Precompute distances from depot to all customers
    depot_distances = {}
    for c in range(1, n_customers + 1):
        depot_distances[c] = euclidean_distance(depot, c)

    # Sort customers by distance from depot (farthest first)
    sorted_customers = sorted(depot_distances.keys(), key=lambda x: -depot_distances[x])

    for customer in sorted_customers:
        if customer not in unvisited:
            continue

        route = [depot]
        current_load = 0
        current = depot

        # Add the farthest unvisited customer as seed
        route.append(customer)
        current_load += demands[customer]
        unvisited.remove(customer)
        current = customer

        # Nearest neighbor insertion with capacity constraint
        while True:
            best_customer = None
            best_cost = float('inf')

            for c in unvisited:
                if current_load + demands[c] > capacity:
                    continue
                dist = euclidean_distance(current, c)
                if dist < best_cost:
                    best_cost = dist
                    best_customer = c

            if best_customer is None:
                break

            route.append(best_customer)
            current_load += demands[best_customer]
            unvisited.remove(best_customer)
            current = best_customer

        route.append(depot)
        routes.append(route)

    return routes