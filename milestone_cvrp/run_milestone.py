from __future__ import annotations

import json
from pathlib import Path

from cvrp_core import evaluate_heuristic, generate_synthetic_benchmarks, nearest_neighbor_heuristic, weighted_greedy_heuristic
from sample_efficient_search import SearchConfig, run_sample_efficient_search


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = generate_synthetic_benchmarks(seed=2026)

    baseline_metrics = evaluate_heuristic(instances, nearest_neighbor_heuristic)

    search_result = run_sample_efficient_search(
        instances,
        SearchConfig(
            seed=2026,
            init_population=10,
            iterations=45,
            top_k=5,
            mutation_std=0.28,
            early_stage_instances=2,
        ),
    )

    best_weights = tuple(search_result["best_weights"])
    searched_metrics = evaluate_heuristic(instances, lambda inst: weighted_greedy_heuristic(inst, best_weights))

    summary = {
        "dataset": [inst.name for inst in instances],
        "baseline": {
            "name": "Nearest Neighbor",
            "avg_distance": round(baseline_metrics["avg_distance"], 3),
            "avg_num_routes": round(baseline_metrics["avg_num_routes"], 3),
        },
        "sample_efficient_funsearch": {
            "best_weights": [round(w, 4) for w in best_weights],
            "avg_distance": round(searched_metrics["avg_distance"], 3),
            "avg_num_routes": round(searched_metrics["avg_num_routes"], 3),
            "unique_candidates": search_result["unique_candidates"],
            "best_score": search_result["best_score"],
        },
        "improvement": {
            "distance_delta": round(baseline_metrics["avg_distance"] - searched_metrics["avg_distance"], 3),
            "distance_improvement_percent": round(
                100.0 * (baseline_metrics["avg_distance"] - searched_metrics["avg_distance"]) / baseline_metrics["avg_distance"],
                2,
            ),
            "routes_delta": round(baseline_metrics["avg_num_routes"] - searched_metrics["avg_num_routes"], 3),
        },
    }

    (out_dir / "milestone_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "search_history.json").write_text(json.dumps(search_result["history"], indent=2), encoding="utf-8")

    print("Saved outputs/milestone_results.json")
    print("Saved outputs/search_history.json")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
