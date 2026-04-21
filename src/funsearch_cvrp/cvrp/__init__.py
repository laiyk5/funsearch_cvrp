"""CVRP core package."""

from .utils import generate_synthetic_benchmarks

from .core import (
    CVRPInstance,
    euclid,
    route_distance,
    solution_distance,
    is_valid_solution,
    evaluate_solver,
)
from .baselines import (
    clarke_wright_savings_heuristic,
    two_opt_route,
    two_opt_improvement,
    with_two_opt,
    nearest_neighbor_heuristic,
    weighted_greedy_heuristic,
)
from .io import load_cvrplib_folder, load_cvrplib_instance

__all__ = [
    "CVRPInstance",
    "euclid",
    "route_distance",
    "solution_distance",
    "is_valid_solution",
    "nearest_neighbor_heuristic",
    "weighted_greedy_heuristic",
    "evaluate_solver",
    "generate_synthetic_benchmarks",
    "clarke_wright_savings_heuristic",
    "two_opt_route",
    "two_opt_improvement",
    "with_two_opt",
    "load_cvrplib_instance",
    "load_cvrplib_folder",
]
