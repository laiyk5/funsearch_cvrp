import logging


def check_load_cvrplib(path: str):
    """test loading CVRPLib dataset, and print basic info of the first few instances"""
    from funsearch_cvrp.cvrp.io import load_cvrplib_folder
    try:
        instance = load_cvrplib_folder(path)
        logging.info(f"Successfully Loaded CVRPLib instances, count: {len(instance)}")
        for i, (inst, sol, cost) in enumerate(instance[:3]):  # Print basic info of the first 3 instances
            logging.info(f"Instance {i+1}: {inst}, Solution cost: {cost}")
    except Exception as e:
        logging.error(f"Failed to load instances: {e}")
        raise e


def check_cvrplib_solution_cost(path: str, limit: int | None = 3):
    """test loading CVRPLib solutions and check if the costs match"""
    from funsearch_cvrp.cvrp.core import solution_distance
    from funsearch_cvrp.cvrp.io import load_cvrplib_folder
    try:
        instances = load_cvrplib_folder(path)
        limit = limit if limit is not None else len(instances)
        for i, (inst, sol, cost) in enumerate(instances[:limit]):  # Check the first `limit` instances
            computed_cost = solution_distance(inst, sol)
            if abs(computed_cost - cost) > 1e-6:
                logging.error(f"Cost mismatch for instance {inst.name}: computed {computed_cost}, expected {cost}")
            else:
                logging.info(f"Cost match for instance {inst.name}: {cost}")
    except Exception as e:
        logging.error(f"Failed to check solutions: {e}")
        raise e