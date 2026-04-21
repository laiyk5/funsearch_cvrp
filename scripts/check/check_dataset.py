import logging
logging.basicConfig(level=logging.INFO)

from funsearch_cvrp.utils.check_dataset import check_load_cvrplib, check_cvrplib_solution_cost

from funsearch_cvrp.config import config


dataset_folder = config.get("CVRP", "dataset_folder")

try:
    check_load_cvrplib(dataset_folder)
    check_cvrplib_solution_cost(dataset_folder, limit=None)
except Exception as e:
    from traceback import format_exc
    print(format_exc())
    print(f"Error occurred while checking dataset: {e}")
