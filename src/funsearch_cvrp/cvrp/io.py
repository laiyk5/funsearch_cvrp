from __future__ import annotations

from pathlib import Path

from .core import CVRPInstance


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


def load_cvrplib_folder(folder: str | Path, limit: int | None = None) -> list[CVRPInstance]:
    """Load all .vrp files in a folder, optionally with a limit."""
    folder_path = Path(folder)
    files = sorted(folder_path.glob("*.vrp"))
    if limit is not None:
        files = files[:limit]

    if not files:
        raise FileNotFoundError(f"No .vrp files found in {folder_path}")

    return [load_cvrplib_instance(p) for p in files]
