from __future__ import annotations

import unittest

import sys
from pathlib import Path

from funsearch_cvrp.cvrp.baselines import clarke_wright_savings_heuristic, with_two_opt
from funsearch_cvrp.cvrp.core import evaluate_heuristic, generate_synthetic_benchmarks, nearest_neighbor_heuristic
from funsearch_cvrp.search import SearchConfig, run_sample_efficient_search


class TestCVRPProject(unittest.TestCase):
    def setUp(self) -> None:
        self.instances = generate_synthetic_benchmarks(seed=2026)

    def test_baseline_runs(self) -> None:
        metrics = evaluate_heuristic(self.instances, nearest_neighbor_heuristic)
        self.assertGreater(metrics["avg_distance"], 0.0)
        self.assertGreater(metrics["avg_num_routes"], 0.0)

    def test_savings_runs(self) -> None:
        metrics = evaluate_heuristic(self.instances, clarke_wright_savings_heuristic)
        self.assertGreater(metrics["avg_distance"], 0.0)

    def test_two_opt_composition_runs(self) -> None:
        metrics = evaluate_heuristic(self.instances, with_two_opt(nearest_neighbor_heuristic))
        self.assertGreater(metrics["avg_distance"], 0.0)

    def test_search_returns_candidate(self) -> None:
        result = run_sample_efficient_search(
            self.instances,
            SearchConfig(seed=2026, init_population=6, iterations=12, top_k=3, mutation_std=0.25, early_stage_instances=2),
        )
        self.assertEqual(len(result["best_weights"]), 3)
        self.assertGreater(result["unique_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
