from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from funsearch_cvrp.cvrp.baselines import clarke_wright_savings_heuristic, with_two_opt
from funsearch_cvrp.cvrp.core import CVRPInstance, evaluate_heuristic, generate_synthetic_benchmarks, nearest_neighbor_heuristic, weighted_greedy_heuristic
from funsearch_cvrp.cvrp.io import load_cvrplib_folder
from funsearch_cvrp.cvrp.search import SearchConfig, run_sample_efficient_search


def _method_result(name: str, metrics: dict) -> dict:
    return {
        "name": name,
        "avg_distance": round(metrics["avg_distance"], 3),
        "avg_num_routes": round(metrics["avg_num_routes"], 3),
        "score": round(metrics["avg_distance"] + 20.0 * metrics["avg_num_routes"], 3),
        "details": metrics["details"],
    }


def _build_instances(args: argparse.Namespace) -> list[CVRPInstance]:
    if args.dataset == "synthetic":
        return generate_synthetic_benchmarks(seed=args.seed)
    return load_cvrplib_folder(args.cvrplib_dir, limit=args.limit_instances)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full CVRP project benchmark suite")
    parser.add_argument("--dataset", choices=["synthetic", "cvrplib"], default="synthetic")
    parser.add_argument("--cvrplib-dir", default="", help="Folder containing .vrp files when dataset=cvrplib")
    parser.add_argument("--limit-instances", type=int, default=None, help="Optional limit for loaded CVRPLib files")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = _build_instances(args)

    methods: list[tuple[str, Callable[[CVRPInstance], list[list[int]]]]] = [
        ("Nearest Neighbor", nearest_neighbor_heuristic),
        ("Nearest Neighbor + 2-opt", with_two_opt(nearest_neighbor_heuristic)),
        ("Clarke-Wright Savings", clarke_wright_savings_heuristic),
        ("Clarke-Wright Savings + 2-opt", with_two_opt(clarke_wright_savings_heuristic)),
    ]

    random_weights = (1.0, 0.0, 1.0)
    methods.append(
        (
            "Weighted Greedy (fixed)",
            lambda inst: weighted_greedy_heuristic(inst, random_weights),
        )
    )

    method_results: list[dict] = []
    for name, solver in methods:
        metrics = evaluate_heuristic(instances, solver)
        method_results.append(_method_result(name, metrics))

    search_cfg = SearchConfig(
        seed=args.seed,
        init_population=12,
        iterations=60,
        top_k=6,
        mutation_std=0.28,
        early_stage_instances=min(2, len(instances)),
    )
    search_result = run_sample_efficient_search(instances, search_cfg)
    best_weights = tuple(search_result["best_weights"])

    funsearch_solver = lambda inst: weighted_greedy_heuristic(inst, best_weights)
    funsearch_metrics = evaluate_heuristic(instances, funsearch_solver)
    method_results.append(_method_result("Sample-Efficient FunSearch", funsearch_metrics))

    funsearch_2opt_metrics = evaluate_heuristic(instances, with_two_opt(funsearch_solver))
    method_results.append(_method_result("Sample-Efficient FunSearch + 2-opt", funsearch_2opt_metrics))

    ranked = sorted(method_results, key=lambda x: x["score"])
    best = ranked[0]

    summary = {
        "project": "Full CVRP FunSearch Project",
        "dataset_type": args.dataset,
        "dataset_names": [inst.name for inst in instances],
        "instance_count": len(instances),
        "search_config": {
            "seed": search_cfg.seed,
            "init_population": search_cfg.init_population,
            "iterations": search_cfg.iterations,
            "top_k": search_cfg.top_k,
            "mutation_std": search_cfg.mutation_std,
            "early_stage_instances": search_cfg.early_stage_instances,
        },
        "funsearch": {
            "best_weights": [round(v, 4) for v in best_weights],
            "best_score_from_search": search_result["best_score"],
            "unique_candidates": search_result["unique_candidates"],
        },
        "methods": method_results,
        "ranking": [
            {
                "rank": i + 1,
                "name": item["name"],
                "avg_distance": item["avg_distance"],
                "avg_num_routes": item["avg_num_routes"],
                "score": item["score"],
            }
            for i, item in enumerate(ranked)
        ],
        "best_method": best,
    }

    (out_dir / "full_project_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "full_search_history.json").write_text(json.dumps(search_result["history"], indent=2), encoding="utf-8")

    print("Saved outputs/full_project_results.json")
    print("Saved outputs/full_search_history.json")
    print("Best method:", best["name"], "score=", best["score"])


if __name__ == "__main__":
    main()
