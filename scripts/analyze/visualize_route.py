#!/usr/bin/env python3
"""
Visualize CVRP routes - plot depot, customers, and solution routes.

Usage:
    # Visualize a CVRPLib instance with your solution
    python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp --routes "[[1,2,3],[4,5,6]]"

    # Visualize with auto-generated nearest neighbor solution
    python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp --auto-solve

    # Visualize synthetic instance
    python scripts/analyze/visualize_route.py --synthetic --size 50 --seed 2026 --auto-solve

    # Compare multiple solutions
    python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp \
        --compare "nearest_neighbor,clarke_wright,weighted_greedy"

    # Visualize FunSearch-generated heuristic (from results file)
    python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp \
        --funsearch --commit 7d9a1d6 --iteration 2

    # Visualize FunSearch with specific iteration ID
    python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp \
        --funsearch --iteration-id 6

    # Visualize custom heuristic code
    python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp \
        --heuristic-file outputs/7d9a1d6/20250412_153033/generated/iterative_search_results/heuristic_iter02.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


from funsearch_cvrp.cvrp.core import (
    CVRPInstance,
    evaluate_heuristic,
    nearest_neighbor_heuristic,
    weighted_greedy_heuristic,
)
from funsearch_cvrp.cvrp.baselines import clarke_wright_savings_heuristic, with_two_opt
from funsearch_cvrp.cvrp.io import load_cvrplib_instance
from funsearch_cvrp.cvrp.utils import generate_synthetic_benchmarks
from funsearch_cvrp.funsearch.interface import LLMInterface


# Color palette for routes
ROUTE_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e91e63", "#ff5722", "#795548", "#607d8b",
    "#ff9800", "#4caf50", "#2196f3", "#9c27b0", "#f44336",
]


def parse_routes(routes_str: str) -> list[list[int]]:
    """Parse routes from string like '[[1,2,3],[4,5,6]]'."""
    return json.loads(routes_str)


def load_heuristic_from_code(code: str) -> Callable[[CVRPInstance], list[list[int]]]:
    """Load a heuristic function from code string.

    Args:
        code: Python code string containing custom_heuristic function

    Returns:
        Callable heuristic function
    """
    # Create safe namespace with allowed imports
    namespace = {
        "math": math,
        "random": random,
        "CVRPInstance": CVRPInstance,
    }

    # Execute code in namespace
    exec(code, namespace)

    # Get the heuristic function
    if "custom_heuristic" not in namespace:
        raise ValueError("Code must define a 'custom_heuristic(instance)' function")

    return namespace["custom_heuristic"]


def load_heuristic_from_file(file_path: Path) -> Callable[[CVRPInstance], list[list[int]]]:
    """Load a heuristic function from a Python file."""
    code = file_path.read_text()
    return load_heuristic_from_code(code)


def find_funsearch_results(
    commit: str | None = None,
    timestamp: str | None = None
) -> Path | None:
    """Find FunSearch results JSON file."""
    base_dir = Path("outputs")

    if commit:
        commit_dir = base_dir / commit
    else:
        latest_link = base_dir / "latest"
        if latest_link.exists():
            commit_dir = latest_link.resolve().parent
        else:
            return None

    if timestamp:
        results_dir = commit_dir / timestamp
    else:
        timestamps = sorted([d for d in commit_dir.iterdir() if d.is_dir()])
        if not timestamps:
            return None
        results_dir = timestamps[-1]

    results_file = results_dir / "iterative_search_results.json"
    return results_file if results_file.exists() else None


def load_heuristic_from_funsearch(
    iteration: int | None = None,
    iteration_id: int | None = None,
    results_file: Path | None = None,
) -> tuple[Callable[[CVRPInstance], list[list[int]]], dict]:
    """Load a heuristic from FunSearch results.

    Args:
        iteration: Iteration number (0-indexed)
        iteration_id: Specific ID within iteration
        results_file: Path to results JSON file

    Returns:
        Tuple of (heuristic_function, result_info)
    """
    if results_file is None:
        results_file = find_funsearch_results()
        if results_file is None:
            raise FileNotFoundError("No FunSearch results found. Run an experiment first.")

    with open(results_file) as f:
        results = json.load(f)

    # Find the result
    target = None
    if iteration_id is not None:
        # Find by specific ID
        for r in results:
            if r.get("id") == iteration_id:
                target = r
                break
        if target is None:
            raise ValueError(f"Iteration ID {iteration_id} not found in results")
    elif iteration is not None:
        # Find by iteration number
        for r in results:
            if r.get("iteration") == iteration:
                target = r
                break
        if target is None:
            raise ValueError(f"Iteration {iteration} not found in results")
    else:
        # Default to best (lowest score)
        target = min(results, key=lambda x: x.get("stability", {}).get("avg_score", float("inf")))

    code = target.get("heuristic_code")
    if not code:
        raise ValueError(f"No heuristic code found for iteration")

    heuristic = load_heuristic_from_code(code)
    return heuristic, target


def plot_cvrp_solution(
    instance: CVRPInstance,
    routes: list[list[int]],
    title: str = "CVRP Solution",
    output_path: Path | None = None,
    show_plot: bool = True,
    ax: plt.Axes | None = None,
):
    """Plot a CVRP solution.

    Args:
        instance: CVRP problem instance
        routes: List of routes (each route is a list of customer indices)
        title: Plot title
        output_path: Path to save plot (if None, just display)
        show_plot: Whether to show the plot
        ax: Existing axes to plot on (for multi-panel plots)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure

    # Extract coordinates
    depot_coord = instance.coords[0]
    customer_coords = instance.coords[1:]
    demands = instance.demands[1:]

    # Plot depot
    ax.scatter(
        depot_coord[0], depot_coord[1],
        c="red", s=300, marker="s", zorder=5,
        edgecolors="black", linewidths=2, label="Depot"
    )
    ax.annotate("DEPOT", (depot_coord[0], depot_coord[1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=10, fontweight="bold", color="red")

    # Plot customers
    xs = [c[0] for c in customer_coords]
    ys = [c[1] for c in customer_coords]

    # Size by demand
    sizes = [100 + d * 30 for d in demands]

    scatter = ax.scatter(xs, ys, c="lightblue", s=sizes, zorder=4,
                        edgecolors="navy", linewidths=1, alpha=0.7)

    # Annotate customer IDs
    for i, (x, y, d) in enumerate(zip(xs, ys, demands), start=1):
        ax.annotate(f"{i}", (x, y), ha="center", va="center",
                   fontsize=8, fontweight="bold", color="navy")

    # Calculate route metrics
    total_distance = 0.0
    route_distances = []
    route_loads = []

    # Plot routes
    for i, route in enumerate(routes):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]

        # Build full route including depot
        full_route = [0] + route + [0]

        # Calculate route distance
        route_dist = 0.0
        route_load = sum(instance.demands[c] for c in route)

        # Draw route edges
        for j in range(len(full_route) - 1):
            from_node = full_route[j]
            to_node = full_route[j + 1]
            from_coord = instance.coords[from_node]
            to_coord = instance.coords[to_node]

            # Distance
            dist = np.hypot(from_coord[0] - to_coord[0], from_coord[1] - to_coord[1])
            route_dist += dist

            # Draw arrow
            ax.annotate(
                "",
                xy=to_coord, xytext=from_coord,
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=2,
                    alpha=0.7,
                    connectionstyle="arc3,rad=0.05",
                ),
                zorder=3,
            )

        route_distances.append(route_dist)
        route_loads.append(route_load)
        total_distance += route_dist

    # Set labels and title
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    # Build info text
    info_lines = [
        f"Instance: {instance.name}",
        f"Customers: {instance.n_customers}",
        f"Vehicle Capacity: {instance.capacity}",
        f"Routes: {len(routes)}",
        f"Total Distance: {total_distance:.2f}",
    ]

    # Add per-route info
    for i, (route, dist, load) in enumerate(zip(routes, route_distances, route_loads), 1):
        color_idx = (i - 1) % len(ROUTE_COLORS)
        color_name = ["Red", "Blue", "Green", "Orange", "Purple", "Teal", "Pink"][color_idx % 7]
        info_lines.append(f"Route {i} ({color_name}): {len(route)} customers, dist={dist:.1f}, load={load}/{instance.capacity}")

    info_text = "\n".join(info_lines)
    ax.set_title(f"{title}\n{info_text}", fontsize=11, loc="left", pad=10)

    # Equal aspect ratio
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show_plot and ax is None:
        plt.show()
        plt.close()

    return fig, ax


def compare_solvers(instance: CVRPInstance, solvers: dict[str, callable], output_dir: Path | None = None):
    """Compare multiple solvers on the same instance."""
    n_solvers = len(solvers)

    if n_solvers <= 2:
        n_cols = n_solvers
        n_rows = 1
    elif n_solvers <= 4:
        n_cols = 2
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_solvers + 2) // 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    if n_solvers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    for idx, (name, solver) in enumerate(solvers.items()):
        routes = solver(instance)
        metrics = evaluate_heuristic([instance], solver)
        score = metrics["avg_distance"] + 20.0 * metrics["avg_num_routes"]

        title = f"{name}\nDistance: {metrics['avg_distance']:.2f}, Routes: {metrics['avg_num_routes']:.1f}, Score: {score:.2f}"
        plot_cvrp_solution(instance, routes, title=title, show_plot=False, ax=axes[idx])

    # Hide empty subplots
    for idx in range(n_solvers, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"CVRP Solution Comparison - {instance.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"comparison_{instance.name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize CVRP routes")

    # Instance selection
    parser.add_argument("--instance", type=str, help="Path to .vrp file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic instance")
    parser.add_argument("--size", type=int, default=50, help="Synthetic instance size")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for synthetic")

    # Solution selection
    parser.add_argument("--routes", type=str, help="Routes as JSON string, e.g., '[[1,2,3],[4,5,6]]'")
    parser.add_argument("--auto-solve", action="store_true", help="Use nearest neighbor solver")
    parser.add_argument("--solver", type=str, default="nearest_neighbor",
                       choices=["nearest_neighbor", "clarke_wright", "weighted_greedy", "clarke_wright_2opt"],
                       help="Solver to use with --auto-solve")
    parser.add_argument("--weights", type=str, default="1.0,0.0,1.0",
                       help="Weights for weighted_greedy solver (w1,w2,w3)")

    # FunSearch heuristics
    parser.add_argument("--funsearch", action="store_true", help="Use FunSearch-generated heuristic")
    parser.add_argument("--commit", type=str, help="Git commit hash for FunSearch results")
    parser.add_argument("--timestamp", type=str, help="Timestamp folder for FunSearch results")
    parser.add_argument("--iteration", type=int, help="Iteration number to visualize (0-indexed)")
    parser.add_argument("--iteration-id", type=int, help="Specific iteration ID to visualize")
    parser.add_argument("--heuristic-file", type=str, help="Path to Python file with custom_heuristic function")
    parser.add_argument("--heuristic-code", type=str, help="Python code string with custom_heuristic function")

    # Comparison mode
    parser.add_argument("--compare", type=str, help="Compare multiple solvers: comma-separated list")

    # Output
    parser.add_argument("--output", type=str, help="Output file path (default: show plot)")
    parser.add_argument("--output-dir", type=str, help="Output directory (for comparison mode)")
    parser.add_argument("--no-show", action="store_true", help="Don't show plot, just save")

    args = parser.parse_args()

    # Load instance
    if args.instance:
        instance = load_cvrplib_instance(args.instance)
    elif args.synthetic:
        instances = generate_synthetic_benchmarks(seed=args.seed, sizes=[args.size])
        instance = instances[0]
    else:
        print("Error: Specify --instance or --synthetic")
        sys.exit(1)

    print(f"Loaded instance: {instance.name}")
    print(f"  Customers: {instance.n_customers}")
    print(f"  Capacity: {instance.capacity}")
    print(f"  Total demand: {sum(instance.demands)}")

    # Comparison mode
    if args.compare:
        solver_names = [s.strip() for s in args.compare.split(",")]
        solvers = {}

        for name in solver_names:
            if name == "nearest_neighbor":
                solvers["Nearest Neighbor"] = nearest_neighbor_heuristic
            elif name == "clarke_wright":
                solvers["Clarke-Wright"] = clarke_wright_savings_heuristic
            elif name == "clarke_wright_2opt":
                solvers["Clarke-Wright + 2-opt"] = with_two_opt(clarke_wright_savings_heuristic)
            elif name == "weighted_greedy":
                weights = tuple(float(w) for w in args.weights.split(","))
                solvers[f"Weighted Greedy {weights}"] = lambda inst, w=weights: weighted_greedy_heuristic(inst, w)

        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        compare_solvers(instance, solvers, output_dir)
        return

    # Single solution mode
    solver_name = "Unknown"
    result_info = None

    if args.routes:
        routes = parse_routes(args.routes)
        solver_name = "Custom Routes"
    elif args.funsearch:
        # Load from FunSearch results
        results_file = find_funsearch_results(args.commit, args.timestamp)
        if results_file is None:
            print("Error: No FunSearch results found. Run an experiment first.")
            sys.exit(1)

        print(f"Loading FunSearch results from: {results_file}")
        heuristic, result_info = load_heuristic_from_funsearch(
            iteration=args.iteration,
            iteration_id=args.iteration_id,
            results_file=results_file,
        )

        iter_num = result_info.get("iteration", "?")
        iter_id = result_info.get("id", "?")
        score = result_info.get("stability", {}).get("avg_score", "?")

        solver_name = f"FunSearch (Iter {iter_num}, ID {iter_id}, Score {score:.2f})"
        print(f"Using heuristic: {solver_name}")

        routes = heuristic(instance)

    elif args.heuristic_file:
        # Load from file
        file_path = Path(args.heuristic_file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        print(f"Loading heuristic from: {file_path}")
        heuristic = load_heuristic_from_file(file_path)
        solver_name = f"Custom ({file_path.name})"
        routes = heuristic(instance)

    elif args.heuristic_code:
        # Load from code string
        heuristic = load_heuristic_from_code(args.heuristic_code)
        solver_name = "Custom Code"
        routes = heuristic(instance)

    elif args.auto_solve:
        if args.solver == "nearest_neighbor":
            routes = nearest_neighbor_heuristic(instance)
        elif args.solver == "clarke_wright":
            routes = clarke_wright_savings_heuristic(instance)
        elif args.solver == "clarke_wright_2opt":
            routes = with_two_opt(clarke_wright_savings_heuristic)(instance)
        elif args.solver == "weighted_greedy":
            weights = tuple(float(w) for w in args.weights.split(","))
            routes = weighted_greedy_heuristic(instance, weights)
        solver_name = args.solver
    else:
        print("Error: Specify --routes, --auto-solve, --funsearch, --heuristic-file, --heuristic-code, or --compare")
        sys.exit(1)

    # Calculate metrics
    total_customers = sum(len(r) for r in routes)
    print(f"\nSolution:")
    print(f"  Routes: {len(routes)}")
    print(f"  Customers served: {total_customers}/{instance.n_customers}")

    # Calculate actual distance
    total_dist = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for c in route:
            total_dist += math.hypot(
                instance.coords[prev][0] - instance.coords[c][0],
                instance.coords[prev][1] - instance.coords[c][1]
            )
            prev = c
        total_dist += math.hypot(
            instance.coords[prev][0] - instance.coords[0][0],
            instance.coords[prev][1] - instance.coords[0][1]
        )
    print(f"  Total distance: {total_dist:.2f}")

    # Plot
    output_path = Path(args.output) if args.output else None
    show_plot = not args.no_show

    plot_cvrp_solution(
        instance,
        routes,
        title=f"CVRP Solution - {solver_name}",
        output_path=output_path,
        show_plot=show_plot,
    )

    if output_path:
        print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
