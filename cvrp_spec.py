"""CVRP specification for FunSearch."""
import funsearch
import math

@funsearch.evolve  # LLM will improve this function

def priority(customer: int, current_node: int, capacity_left: int,
             demands: list, coords: list) -> float:
    """Calculate priority score for selecting next customer.
    
    Higher score = more likely to be selected next.
    
    Args:
        customer: Index of candidate customer
        current_node: Index of current location (0 = depot)
        capacity_left: Remaining vehicle capacity
        demands: List of customer demands (index 0 is depot with demand 0)
        coords: List of (x, y) coordinates
    
    Returns:
        Priority score (higher = better)
    """
    # Initial simple version: prefer closer customers
    x1, y1 = coords[current_node]
    x2, y2 = coords[customer]
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    # Closer = higher priority
    return -distance


@funsearch.run  # This evaluates each generated priority function
def evaluate(instances: list) -> float:
    """Solve CVRP using priority function and return total distance.
    
    Lower score = better solution.
    
    Args:
        instances: List of CVRP instances with attributes:
            - n_customers: number of customers
            - capacity: vehicle capacity
            - demands: list of demands (index 0 is depot)
            - coords: list of (x, y) coordinates
    
    Returns:
        Total distance across all instances (lower is better)
    """
    total_distance = 0.0
    
    for inst in instances:
        # Greedy routing using priority function
        unserved = set(range(1, inst.n_customers + 1))
        routes = []
        
        while unserved:
            route = []
            current = 0  # Start at depot
            cap_left = inst.capacity
            
            while True:
                # Find feasible customers
                feasible = [c for c in unserved 
                           if inst.demands[c] <= cap_left]
                
                if not feasible:
                    break
                
                # Use priority function to select next customer
                best_customer = max(feasible,
                    key=lambda c: priority(
                        c, current, cap_left,
                        inst.demands, inst.coords
                    ))
                
                route.append(best_customer)
                unserved.remove(best_customer)
                cap_left -= inst.demands[best_customer]
                current = best_customer
            
            routes.append(route)
        
        # Calculate total distance for this instance
        inst_distance = 0.0
        for route in routes:
            if not route:
                continue
            # Depot -> first customer
            prev = 0
            for c in route:
                x1, y1 = inst.coords[prev]
                x2, y2 = inst.coords[c]
                inst_distance += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                prev = c
            # Last customer -> depot
            x1, y1 = inst.coords[prev]
            x2, y2 = inst.coords[0]
            inst_distance += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        total_distance += inst_distance
    
    return total_distance
