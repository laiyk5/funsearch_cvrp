"""Run baseline CVRP heuristics and save results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.funsearch_cvrp.cvrp import (
    CVRPInstance,
    clarke_wright_savings_heuristic,
    evaluate_solver,
    generate_synthetic_benchmarks,
    nearest_neighbor_heuristic,
    with_two_opt,
)
from src.funsearch_cvrp.cvrp.io import load_cvrplib_folder
from src.funsearch_cvrp.utils.output_manager import get_output_dir, save_run_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


BASELINE_SOLVERS: dict[str, callable] = {
    "nearest_neighbor": nearest_neighbor_heuristic,
    "clarke_wright": clarke_wright_savings_heuristic,
    "nearest_neighbor_2opt": with_two_opt(nearest_neighbor_heuristic),
    "clarke_wright_2opt": with_two_opt(clarke_wright_savings_heuristic),
}


def run_baselines_on_instances(
    instances: list[CVRPInstance],
    solvers: dict[str, callable] | None = None,
) -> dict:
    """Run all baseline solvers on a list of instances.

    Returns a dict with per-solver results and a summary table.
    """
    solvers = solvers or BASELINE_SOLVERS
    results: dict = {}

    for name, solver in solvers.items():
        logging.info(f"Running solver: {name}")
        result = evaluate_solver(instances, solver)
        results[name] = result

        valid_str = "VALID" if result["is_valid_solver"] else "INVALID"
        logging.info(
            f"  {name}: avg_distance={result['avg_distance']:.2f}, "
            f"avg_routes={result['avg_num_routes']:.2f}, validity={valid_str}"
        )

    return results


def save_results(output_dir: Path, results: dict, instances: list[CVRPInstance]) -> None:
    """Save baseline results to the output directory."""
    results_file = output_dir / "baseline_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved to {results_file}")

    # Also write a human-readable summary
    summary_file = output_dir / "baseline_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Baseline Results Summary\n")
        f.write("=" * 60 + "\n\n")

        for name, result in results.items():
            f.write(f"Solver: {name}\n")
            f.write(f"  Avg Distance: {result['avg_distance']:.2f}\n")
            f.write(f"  Avg Routes:   {result['avg_num_routes']:.2f}\n")
            f.write(f"  Valid:        {result['is_valid_solver']}\n")
            if result["invalid_cases"]:
                f.write(f"  Invalid Cases: {len(result['invalid_cases'])}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("Per-Instance Details\n")
        f.write("=" * 60 + "\n\n")

        for i, inst in enumerate(instances):
            f.write(f"Instance: {inst.name} (customers={inst.n_customers})\n")
            for name, result in results.items():
                detail = result["details"][i]
                f.write(
                    f"  {name:30s} distance={detail['distance']:.1f}  routes={detail['num_routes']}\n"
                )
            f.write("\n")

    logging.info(f"Summary saved to {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline CVRP heuristics")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "cvrplib"],
        default="synthetic",
        help="Dataset to use (default: synthetic)",
    )
    parser.add_argument(
        "--cvrplib-dir",
        default="data/cvrplib/A",
        help="CVRPLib directory (used with --dataset cvrplib)",
    )
    parser.add_argument(
        "--limit-instances",
        type=int,
        default=None,
        help="Limit number of CVRPLib instances to evaluate",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[20, 50, 100],
        help="Synthetic instance sizes (default: 20 50 100)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=list(BASELINE_SOLVERS.keys()),
        default=None,
        help="Specific solvers to run (default: all)",
    )

    args = parser.parse_args()

    # Load instances
    if args.dataset == "synthetic":
        logging.info(f"Generating synthetic instances with sizes {args.sizes}")
        instances = generate_synthetic_benchmarks(sizes=args.sizes)
    else:
        logging.info(f"Loading CVRPLib instances from {args.cvrplib_dir}")
        instances_sols = load_cvrplib_folder(args.cvrplib_dir, limit=args.limit_instances)
        instances = [inst for inst, _ in instances_sols]
        logging.info(f"Loaded {len(instances)} instances")

    # Select solvers
    solvers = BASELINE_SOLVERS
    if args.solvers:
        solvers = {k: v for k, v in BASELINE_SOLVERS.items() if k in args.solvers}

    # Run baselines
    results = run_baselines_on_instances(instances, solvers)

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_output_dir()

    save_run_info(output_dir, extra_info={
        "dataset": args.dataset,
        "num_instances": len(instances),
        "instance_names": [inst.name for inst in instances],
        "solvers": list(solvers.keys()),
    })
    save_results(output_dir, results, instances)

    logging.info(f"All done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
