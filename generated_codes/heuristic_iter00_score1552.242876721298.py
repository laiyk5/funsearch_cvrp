"""
FunSearch 生成的启发式算法
来源: iterative_search_results.json
迭代: 0
评分: 1552.242876721298
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

    while unvisited:
        route = [depot]
        current_load = 0
        current = depot

        while True:
            best_customer = None
            best_cost = float('inf')

            for customer in unvisited:
                if current_load + demands[customer] > capacity:
                    continue
                dist = euclidean_distance(current, customer)
                if dist < best_cost:
                    best_cost = dist
                    best_customer = customer

            if best_customer is None:
                break

            route.append(best_customer)
            current_load += demands[best_customer]
            unvisited.remove(best_customer)
            current = best_customer

        route.append(depot)
        routes.append(route)

    return routes