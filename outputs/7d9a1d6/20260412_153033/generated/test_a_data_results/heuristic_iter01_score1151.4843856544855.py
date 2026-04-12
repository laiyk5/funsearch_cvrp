"""
FunSearch 生成的启发式算法
来源: outputs/latest/test_a_data_results.json
迭代: 1
评分: 1151.4843856544855
提取时间: 2026-04-12T15:42:54.201461
"""

def custom_heuristic(instance):
    n_customers = instance.n_customers
    capacity = instance.capacity
    coords = instance.coords
    demands = instance.demands
    
    depot = 0
    depot_x, depot_y = coords[depot]
    
    def dist(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        return ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
    
    unvisited = set(range(1, n_customers + 1))
    routes = []
    
    # Precompute savings using Clarke-Wright formula
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist(depot, i) + dist(depot, j) - dist(i, j)
            savings.append((s, i, j))
    savings.sort(reverse=True)
    
    # Initialize each customer as a separate route
    route_of = {}
    route_load = {}
    route_seq = {}
    for i in range(1, n_customers + 1):
        route_of[i] = i
        route_load[i] = demands[i]
        route_seq[i] = [depot, i, depot]
    
    # Try to merge routes based on savings
    for s, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        if route_load[ri] + route_load[rj] > capacity:
            continue
        
        # Check if i is last in ri and j is first in rj (or vice versa)
        seq_i = route_seq[ri]
        seq_j = route_seq[rj]
        
        # i must be adjacent to depot in its route
        if seq_i[-2] != i and seq_i[1] != i:
            continue
        if seq_j[-2] != j and seq_j[1] != j:
            continue
        
        # Determine orientation
        new_seq = None
        if seq_i[-2] == i and seq_j[1] == j:
            new_seq = seq_i[:-1] + seq_j[1:]
        elif seq_i[1] == i and seq_j[-2] == j:
            new_seq = seq_j[:-1] + seq_i[1:]
        elif seq_i[-2] == i and seq_j[-2] == j:
            new_seq = seq_i[:-1] + seq_j[-2::-1] + [depot]
        elif seq_i[1] == i and seq_j[1] == j:
            new_seq = [depot] + seq_j[1:][::-1] + seq_i[1:]
        
        if new_seq is not None:
            # Merge route rj into ri
            new_load = route_load[ri] + route_load[rj]
            route_seq[ri] = new_seq
            route_load[ri] = new_load
            for cust in route_seq[rj]:
                if cust != depot:
                    route_of[cust] = ri
            del route_seq[rj]
            del route_load[rj]
    
    # Collect final routes
    used_routes = set(route_of.values())
    for r in used_routes:
        routes.append(route_seq[r])
    
    # Handle any remaining unmerged customers (should not happen but safe)
    for cust in unvisited:
        if cust not in route_of:
            routes.append([depot, cust, depot])
    
    return routes