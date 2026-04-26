"""Benchmark baseline and evolved solvers on CVRPLib datasets.

Usage:
    python scripts/benchmark/benchmark_solvers.py
    python scripts/benchmark/benchmark_solvers.py --program outputs/20260425_180501/run_funsearch/best_program.py
    python scripts/benchmark/benchmark_solvers.py --datasets A,X --no-2opt
    python scripts/benchmark/benchmark_solvers.py --output my_results.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from funsearch_cvrp.cvrp.baselines import (
    clarke_wright_savings_heuristic,
    nearest_neighbor_heuristic,
    weighted_greedy_heuristic,
    with_two_opt,
)
from funsearch_cvrp.cvrp.core import (
    CVRPInstance,
    gap_score,
    is_valid_solution,
    make_greedy_solver,
    make_savings_solver,
    solution_distance,
)
from funsearch_cvrp.cvrp.io import load_cvrplib_folder, load_cvrplib_instance


def load_evolved_function(path: Path):
    """Load the evolved function from a Python file and detect spec type."""
    spec = importlib.util.spec_from_file_location("evolved_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "savings"):
        return ("savings", module.savings)
    elif hasattr(module, "priority"):
        return ("priority", module.priority)
    else:
        raise ValueError(f"No 'savings' or 'priority' function found in {path}")


def run_solver(name: str, solver_fn, instance: CVRPInstance, optimal_map: dict) -> dict:
    """Run a single solver on a single instance."""
    t0 = time.perf_counter()
    try:
        solution = solver_fn(instance)
    except Exception as e:
        return {
            "instance": instance.name,
            "cost": None,
            "gap": None,
            "time": time.perf_counter() - t0,
            "valid": False,
            "error": str(e),
        }
    elapsed = time.perf_counter() - t0

    valid, reason = is_valid_solution(instance, solution)
    if not valid:
        return {
            "instance": instance.name,
            "cost": None,
            "gap": None,
            "time": elapsed,
            "valid": False,
            "error": reason,
        }

    cost = solution_distance(instance, solution)
    opt = optimal_map.get(instance.name)
    gap = gap_score(cost, opt) * 100 if opt else None

    return {
        "instance": instance.name,
        "cost": cost,
        "gap": gap,
        "time": elapsed,
        "valid": True,
        "error": None,
    }


def print_progress(dataset: str, solver: str, idx: int, total: int, instance_name: str):
    """Print a progress line to console."""
    print(f"  [{dataset}] {solver:<30} ({idx+1}/{total}) {instance_name}", flush=True)


def build_solvers(evolved_fn, use_2opt: bool) -> dict[str, Callable]:
    """Build the solver dictionary."""
    spec_type, fn = evolved_fn
    if spec_type == "savings":
        solvers = {
            "Clarke-Wright": clarke_wright_savings_heuristic,
            "Nearest Neighbor": nearest_neighbor_heuristic,
            "Weighted Greedy (0.5,0.3,0.2)": lambda inst: weighted_greedy_heuristic(inst, (0.5, 0.3, 0.2)),
            "Evolved Savings": make_savings_solver(fn, two_opt=False),
        }
        if use_2opt:
            solvers["Clarke-Wright + 2opt"] = with_two_opt(clarke_wright_savings_heuristic)
            solvers["Nearest Neighbor + 2opt"] = with_two_opt(nearest_neighbor_heuristic)
            solvers["Evolved Savings + 2opt"] = make_savings_solver(fn, two_opt=True)
    else:  # priority
        solvers = {
            "Nearest Neighbor": nearest_neighbor_heuristic,
            "Evolved Priority": make_greedy_solver(fn),
        }
        if use_2opt:
            solvers["Nearest Neighbor + 2opt"] = with_two_opt(nearest_neighbor_heuristic)
            solvers["Evolved Priority + 2opt"] = lambda inst: with_two_opt(lambda i: make_greedy_solver(fn)(i))(inst)
    return solvers


def benchmark_dataset(
    set_name: str,
    folder: Path,
    solvers: dict[str, Callable],
    all_dataset_results: dict,
    benchmark_dir: Path,
):
    """Benchmark all solvers on one dataset."""
    if not folder.exists():
        print(f"Skipping {set_name}: folder not found")
        return None

    # Load instances
    if set_name == "XL":
        vrp_files = sorted(folder.glob("*.vrp"))
        instances = [load_cvrplib_instance(f) for f in vrp_files]
        optimal_map = {}
    else:
        loaded = load_cvrplib_folder(folder)
        instances = [inst for inst, _sol, _cost in loaded]
        optimal_map = {inst.name: cost for inst, _sol, cost in loaded if cost is not None}

    print(f"\n{'='*80}")
    print(f"Dataset: {set_name} — {len(instances)} instances, {len(optimal_map)} with known optimal")
    print(f"{'='*80}")

    all_results: dict[str, list[dict]] = {}
    for solver_name, solver_fn in solvers.items():
        print(f"\n  -> Running {solver_name}...")
        results = []
        for idx, instance in enumerate(instances):
            print_progress(set_name, solver_name, idx, len(instances), instance.name)
            result = run_solver(solver_name, solver_fn, instance, optimal_map)
            results.append(result)
        all_results[solver_name] = results

        # Save after every solver so progress is never lost
        all_dataset_results[set_name] = all_results
        _save_dataset(set_name, all_results, benchmark_dir)
        print(f"     [saved] {benchmark_dir}/benchmark_{set_name}.json")

    # Summary
    print(f"\n{'='*80}")
    print(f"Dataset: {set_name} — Summary")
    print(f"{'='*80}")
    print(f"{'Solver':<35} {'Avg Cost':>12} {'Avg Gap%':>10} {'Avg Time(s)':>12} {'Valid':>6}")
    print("-" * 80)

    summary_rows = []
    for solver_name, results in all_results.items():
        valid_results = [r for r in results if r["valid"]]
        n_valid = len(valid_results)
        n_total = len(results)
        if n_valid == 0:
            print(f"{solver_name:<35} {'N/A':>12} {'N/A':>10} {'N/A':>12} {f'0/{n_total}':>6}")
            summary_rows.append({"solver": solver_name, "avg_cost": None, "avg_gap": None, "avg_time": None, "valid": 0, "total": n_total})
            continue
        avg_cost = sum(r["cost"] for r in valid_results) / n_valid
        avg_time = sum(r["time"] for r in results) / n_total
        gaps = [r["gap"] for r in valid_results if r["gap"] is not None]
        avg_gap = sum(gaps) / len(gaps) if gaps else None
        gap_str = f"{avg_gap:.2f}" if avg_gap is not None else "N/A"
        print(f"{solver_name:<35} {avg_cost:>12.1f} {gap_str:>10} {avg_time:>12.4f} {f'{n_valid}/{n_total}':>6}")
        summary_rows.append({"solver": solver_name, "avg_cost": avg_cost, "avg_gap": avg_gap, "avg_time": avg_time, "valid": n_valid, "total": n_total})

    _save_summary(set_name, summary_rows, benchmark_dir)
    return all_results


def _save_dataset(set_name: str, all_results: dict, benchmark_dir: Path):
    """Write per-dataset detailed results to JSON."""
    serializable = {
        solver: [
            {k: v for k, v in r.items() if k != "error" or v is not None}
            for r in records
        ]
        for solver, records in all_results.items()
    }
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    path = benchmark_dir / f"benchmark_{set_name}.json"
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def _save_summary(set_name: str, summary_rows: list, benchmark_dir: Path):
    """Write per-dataset summary to JSON."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    path = benchmark_dir / f"summary_{set_name}.json"
    with open(path, "w") as f:
        json.dump({"dataset": set_name, "solvers": summary_rows}, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CVRP solvers")
    parser.add_argument(
        "--program",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "latest" / "run_funsearch" / "best_program.py",
        help="Path to the evolved program file (default: outputs/latest/run_funsearch/best_program.py)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="A,X,XL",
        help="Comma-separated dataset names to benchmark (default: A,X,XL)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "latest" / "benchmark_results.json",
        help="Output JSON path (default: outputs/latest/benchmark_results.json)",
    )
    parser.add_argument(
        "--no-2opt",
        action="store_true",
        help="Disable 2-opt post-optimization for all datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    best_program_path = args.program
    if not best_program_path.exists():
        print(f"No program file found at {best_program_path}")
        sys.exit(1)

    print(f"Using evolved program: {best_program_path}")
    evolved_fn = load_evolved_function(best_program_path)
    print(f"  Detected spec: {evolved_fn[0]}")

    # Derive benchmark output directory from program path
    # e.g. outputs/20260426_054254_ds_flash_pri_ecd/run_funsearch/best_program.py
    #   -> outputs/20260426_054254_ds_flash_pri_ecd/benchmark/
    exp_dir = best_program_path.parent.parent
    benchmark_dir = exp_dir / "benchmark"
    print(f"  Benchmark dir: {benchmark_dir}")

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    datasets = {}
    for name in dataset_names:
        folder = PROJECT_ROOT / "data" / "cvrplib" / name
        datasets[name] = folder

    use_2opt = not args.no_2opt
    solvers = build_solvers(evolved_fn, use_2opt=use_2opt)

    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"2-opt enabled: {use_2opt}")

    all_dataset_results = {}
    for set_name, folder in datasets.items():
        # XL defaults to no 2opt for speed unless explicitly requested
        dataset_solvers = solvers if use_2opt else build_solvers(evolved_fn, use_2opt=False)
        results = benchmark_dataset(set_name, folder, dataset_solvers, all_dataset_results, benchmark_dir)
        if results:
            all_dataset_results[set_name] = results

    print(f"\n\nAll results saved to {benchmark_dir}")


if __name__ == "__main__":
    main()
