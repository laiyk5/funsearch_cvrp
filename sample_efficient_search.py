from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from cvrp_core import CVRPInstance, evaluate_heuristic, weighted_greedy_heuristic


@dataclass
class SearchConfig:
    seed: int = 2026
    init_population: int = 8
    iterations: int = 35
    top_k: int = 4
    mutation_std: float = 0.35
    early_stage_instances: int = 2


@dataclass
class Candidate:
    weights: tuple[float, float, float]
    score: float
    avg_routes: float
    signature: tuple[float, float, float]


def _signature(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    # Rounded signature is a fast proxy for duplicate functional behavior.
    return tuple(round(w, 2) for w in weights)


def _score(metrics: dict) -> float:
    # Lower distance and fewer routes are better.
    return metrics["avg_distance"] + 20.0 * metrics["avg_num_routes"]


def _solver_from_weights(weights: tuple[float, float, float]) -> Callable[[CVRPInstance], list[list[int]]]:
    return lambda inst: weighted_greedy_heuristic(inst, weights)


def run_sample_efficient_search(instances: list[CVRPInstance], config: SearchConfig) -> dict:
    rng = random.Random(config.seed)
    if len(instances) < config.early_stage_instances:
        raise ValueError("Not enough instances for early-stage evaluation")

    early_instances = instances[: config.early_stage_instances]
    evaluated_signatures: set[tuple[float, float, float]] = set()
    history: list[dict] = []
    population: list[Candidate] = []

    def evaluate_candidate(weights: tuple[float, float, float]) -> Candidate | None:
        sig = _signature(weights)
        if sig in evaluated_signatures:
            return None
        evaluated_signatures.add(sig)

        # Stage 1: quick check on a subset (sample-efficient).
        early_metrics = evaluate_heuristic(early_instances, _solver_from_weights(weights))
        early_s = _score(early_metrics)

        # Stage 2: full evaluation only if promising.
        threshold = 999999.0 if not population else sorted(c.score for c in population)[min(len(population) - 1, config.top_k - 1)]
        if early_s > threshold * 1.07 and population:
            # Too weak in early stage; skip full expensive evaluation.
            history.append({
                "weights": list(weights),
                "signature": list(sig),
                "status": "early_pruned",
                "early_score": round(early_s, 3),
            })
            return None

        full_metrics = evaluate_heuristic(instances, _solver_from_weights(weights))
        final_score = _score(full_metrics)
        cand = Candidate(weights=weights, score=final_score, avg_routes=full_metrics["avg_num_routes"], signature=sig)
        history.append(
            {
                "weights": list(weights),
                "signature": list(sig),
                "status": "full_eval",
                "score": round(final_score, 3),
                "avg_distance": round(full_metrics["avg_distance"], 3),
                "avg_num_routes": round(full_metrics["avg_num_routes"], 3),
            }
        )
        return cand

    # Initial population
    for _ in range(config.init_population):
        weights = (rng.uniform(0.4, 1.8), rng.uniform(-0.5, 1.3), rng.uniform(0.2, 1.6))
        cand = evaluate_candidate(weights)
        if cand is not None:
            population.append(cand)

    population.sort(key=lambda c: c.score)

    # Main loop
    for _ in range(config.iterations):
        if not population:
            break
        parent = rng.choice(population[: min(config.top_k, len(population))])
        new_weights = tuple(parent.weights[i] + rng.gauss(0.0, config.mutation_std) for i in range(3))
        cand = evaluate_candidate(new_weights)
        if cand is not None:
            population.append(cand)
            population.sort(key=lambda c: c.score)
            population = population[: max(config.top_k * 3, 8)]

    if not population:
        raise RuntimeError("Search failed to produce any valid candidate")

    best = population[0]
    return {
        "best_weights": list(best.weights),
        "best_signature": list(best.signature),
        "best_score": round(best.score, 3),
        "best_avg_routes": round(best.avg_routes, 3),
        "unique_candidates": len(evaluated_signatures),
        "history": history,
    }
