from __future__ import annotations

import logging
from pathlib import Path

from .core import CVRPInstance, Route, Solution


def _parse_coord_line(line: str) -> tuple[int, float, float]:
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid coordinate line: {line!r}")
    idx = int(parts[0])
    x = float(parts[1])
    y = float(parts[2])
    return idx, x, y


def _parse_demand_line(line: str) -> tuple[int, int]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid demand line: {line!r}")
    idx = int(parts[0])
    demand = int(parts[1])
    return idx, demand


def load_cvrplib_instance(file_path: str | Path) -> CVRPInstance:
    """Load a basic CVRPLib .vrp file (NODE_COORD_SECTION + DEMAND_SECTION)."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    name = path.stem
    capacity: int | None = None

    node_coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot_nodes: list[int] = []

    mode: str | None = None

    for raw in text:
        line = raw.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("NAME"):
            if ":" in line:
                name = line.split(":", 1)[1].strip() or name
            continue
        if upper.startswith("CAPACITY"):
            capacity = int(line.split(":", 1)[1].strip())
            continue

        if upper == "NODE_COORD_SECTION":
            mode = "coords"
            continue
        if upper == "DEMAND_SECTION":
            mode = "demands"
            continue
        if upper == "DEPOT_SECTION":
            mode = "depot"
            continue
        if upper == "EOF":
            break

        if mode == "coords":
            idx, x, y = _parse_coord_line(line)
            node_coords[idx] = (x, y)
        elif mode == "demands":
            idx, d = _parse_demand_line(line)
            demands[idx] = d
        elif mode == "depot":
            node = int(line)
            if node == -1:
                mode = None
            else:
                depot_nodes.append(node)

    if capacity is None:
        raise ValueError(f"CAPACITY not found in {path}")
    if not node_coords or not demands:
        raise ValueError(f"Missing NODE_COORD_SECTION or DEMAND_SECTION in {path}")

    depot_idx = depot_nodes[0] if depot_nodes else 1

    # Reindex to internal format where depot is node 0.
    customer_ids = [idx for idx in sorted(node_coords.keys()) if idx != depot_idx]
    coords = [node_coords[depot_idx]]
    demand_list = [0]
    for idx in customer_ids:
        coords.append(node_coords[idx])
        demand_list.append(int(demands.get(idx, 0)))

    return CVRPInstance(
        name=name,
        capacity=capacity,
        demands=demand_list,
        coords=coords,
    )


def load_cvrplib_solution(file_path: str | Path) -> Solution:
    """Load a CVRPLib solution file, returning routes and total distance.
    
    A CVRPLib solution file typically contains lines like:
        Route #1: 9 6 3 4 19 31 12
        Route #2: 28 14 34 23 2 35 8 15
        Route #3: 16 11 24 27 25 5 20
        Route #4: 10 7 26
        Route #5: 1 22 32 13 17 30 29 33 18 21
        Cost 799
    """

    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    routes: list[Route] = []
    total_distance: int | None = None

    for line in text:
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("ROUTE"):
            parts = line.split(":")
            if len(parts) < 2:
                continue
            route_str = parts[1].strip()
            if route_str:
                route_nodes = [int(x) for x in route_str.split()]
                routes.append(route_nodes)
        elif line.upper().startswith("COST"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                total_distance = int(parts[1])

    if total_distance is None:
        raise ValueError(f"COST not found in solution file {path}")

    return routes, total_distance


def load_cvrplib_folder(folder_path: str | Path, limit: int | None = None) -> list[tuple[CVRPInstance, Solution]]:
    """Load all .vrp instances and their corresponding .sol solutions from a folder."""
    folder = Path(folder_path)
    instances: list[tuple[CVRPInstance, Solution]] = []

    skipped = 0
    for vrp_file in sorted(folder.glob("*.vrp")):
        if limit is not None and len(instances) >= limit:
            break
        try:
            instance = load_cvrplib_instance(vrp_file)
            sol_file = vrp_file.with_suffix(".sol")
            if sol_file.exists():
                solution = load_cvrplib_solution(sol_file)
                instances.append((instance, solution))
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