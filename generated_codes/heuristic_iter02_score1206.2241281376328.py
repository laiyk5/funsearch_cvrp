"""
FunSearch 生成的启发式算法
来源: iterative_search_results.json
迭代: 2
评分: 1206.2241281376328
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

    # Precompute all pairwise distances for efficiency
    dist_cache = {}
    for i in range(n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            d = euclidean_distance(i, j)
            dist_cache[(i, j)] = d
            dist_cache[(j, i)] = d

    def get_dist(i, j):
        if i == j:
            return 0.0
        return dist_cache.get((i, j), euclidean_distance(i, j))

    # Savings heuristic (Clarke and Wright)
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = get_dist(depot, i) + get_dist(depot, j) - get_dist(i, j)
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Initialize each customer as its own route
    route_of = {}
    route_load = {}
    route_seq = {}
    for c in range(1, n_customers + 1):
        route_of[c] = c
        route_load[c] = demands[c]
        route_seq[c] = [depot, c, depot]

    for s_val, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        if route_load[ri] + route_load[rj] > capacity:
            continue

        # Determine merge feasibility: i must be last in ri, j must be first in rj (or vice versa)
        seq_i = route_seq[ri]
        seq_j = route_seq[rj]
        feasible = False
        new_seq = None

        # Case 1: i at end of ri, j at start of rj
        if seq_i[-2] == i and seq_j[1] == j:
            new_seq = seq_i[:-1] + seq_j[1:]
            feasible = True
        # Case 2: j at end of rj, i at start of ri
        elif seq_j[-2] == j and seq_i[1] == i:
            new_seq = seq_j[:-1] + seq_i[1:]
            feasible = True

        if feasible:
            # Merge rj into ri
            new_load = route_load[ri] + route_load[rj]
            route_seq[ri] = new_seq
            route_load[ri] = new_load
            # Update route_of for all nodes in rj
            for node in seq_j[1:-1]:
                route_of[node] = ri
            # Remove rj
            del route_seq[rj]
            del route_load[rj]

    # Collect final routes
    added_routes = set()
    for c in range(1, n_customers + 1):
        r = route_of[c]
        if r not in added_routes:
            routes.append(route_seq[r])
            added_routes.add(r)

    return routes