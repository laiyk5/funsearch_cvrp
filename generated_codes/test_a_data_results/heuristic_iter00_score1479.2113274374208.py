"""
FunSearch 生成的启发式算法
来源: test_a_data_results.json
迭代: 0
评分: 1479.2113274374208
"""

def custom_heuristic(instance):
    n_customers = instance.n_customers
    capacity = instance.capacity
    coords = instance.coords
    demands = instance.demands
    
    # Precompute distances from depot (customer 0) to all customers
    depot = 0
    depot_x, depot_y = coords[depot]
    
    def dist(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        return ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
    
    # Create list of unvisited customers (excluding depot)
    unvisited = set(range(1, n_customers + 1))
    routes = []
    
    while unvisited:
        route = [depot]
        current_load = 0
        current_pos = depot
        
        while True:
            # Find the nearest feasible customer
            best_customer = None
            best_dist = float('inf')
            
            for cust in unvisited:
                if current_load + demands[cust] <= capacity:
                    d = dist(current_pos, cust)
                    if d < best_dist:
                        best_dist = d
                        best_customer = cust
            
            if best_customer is None:
                break
                
            route.append(best_customer)
            current_load += demands[best_customer]
            current_pos = best_customer
            unvisited.remove(best_customer)
        
        route.append(depot)
        routes.append(route)
    
    return routes