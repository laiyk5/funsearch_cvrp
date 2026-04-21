from __future__ import annotations

import logging
from pathlib import Path

import vrplib   #type: ignore

from .core import CVRPInstance, Route, Solution


def load_cvrplib_instance(file_path: str | Path) -> CVRPInstance:
    """Load a CVRPLib .vrp file using vrplib."""
    path = Path(file_path)
    inst = vrplib.read_instance(path, compute_edge_weights=False)

    name = inst.get("name", path.stem)
    capacity = int(inst["capacity"])
    demands = [int(d) for d in inst["demand"]]
    coords = [(float(x), float(y)) for x, y in inst["node_coord"]]
    
    # load number of trucks from the name if it follows the format "X-nXX-kYY" where YY is the number of trucks
    number_of_trucks = None
    if "-k" in name:
        try:
            number_of_trucks = int(name.split("-k")[-1])
        except ValueError:
            logging.warning(f"Could not parse number of trucks from instance name '{name}'")

    return CVRPInstance(
        name=name,
        capacity=capacity,
        demands=demands,
        coords=coords,
        number_of_trucks=number_of_trucks,
    )

def load_cvrplib_solution(file_path: str | Path) -> tuple[Solution, float]:
    """Load a CVRPLib solution file using vrplib.

    Routes are returned as 1-indexed customer lists, matching our internal format
    where depot is node 0.
    """
    path = Path(file_path)
    sol = vrplib.read_solution(path)

    routes: list[Route] = [[int(node) for node in route] for route in sol["routes"]]
    cost = float(sol["cost"])

    return routes, cost


def load_cvrplib_folder(folder_path: str | Path, limit: int | None = None) -> list[tuple[CVRPInstance, Solution, float]]:
    """Load all .vrp instances and their corresponding .sol solutions from a folder."""
    folder = Path(folder_path)
    instances: list[tuple[CVRPInstance, Solution, float]] = []

    skipped = 0
    for vrp_file in sorted(folder.glob("*.vrp")):
        if limit is not None and len(instances) >= limit:
            break
        try:
            instance = load_cvrplib_instance(vrp_file)
            sol_file = vrp_file.with_suffix(".sol")
            if sol_file.exists():
                solution, cost = load_cvrplib_solution(sol_file)
                instances.append((instance, solution, cost))
            else:
                logging.warning(f"Solution file {sol_file} not found for instance {vrp_file}")
                skipped += 1
        except Exception as e:
            logging.error(f"Error loading {vrp_file}: {e}")
            skipped += 1

    if skipped > 0:
        logging.warning(f"Loaded {len(instances)} instances from {folder}, skipped {skipped} due to errors.")
    else:
        logging.info(f"Successfully loaded {len(instances)} instances from {folder} with solutions.")

    return instances
