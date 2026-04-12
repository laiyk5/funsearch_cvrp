from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, List, Optional

from cvrp_core import CVRPInstance, evaluate_heuristic, weighted_greedy_heuristic
from llm_interface import LLMInterface
from function_equivalence import FunctionEquivalenceDetector


@dataclass
class SearchConfig:
    seed: int = 2026
    init_population: int = 8
    iterations: int = 35
    top_k: int = 4
    mutation_std: float = 0.35
    early_stage_instances: int = 2
    use_llm: bool = True
    llm_generation_ratio: float = 0.5  # 50%的概率使用LLM生成


@dataclass
class Candidate:
    weights: Optional[tuple[float, float, float]]
    heuristic_code: Optional[str]
    solver: Callable[[CVRPInstance], List[List[int]]]
    score: float
    avg_routes: float
    signature: str


def _score(metrics: dict) -> float:
    # Lower distance and fewer routes are better.
    return metrics["avg_distance"] + 20.0 * metrics["avg_num_routes"]


def _solver_from_weights(weights: tuple[float, float, float]) -> Callable[[CVRPInstance], List[List[int]]]:
    return lambda inst: weighted_greedy_heuristic(inst, weights)


def run_sample_efficient_search(instances: list[CVRPInstance], config: SearchConfig) -> dict:
    rng = random.Random(config.seed)
    if len(instances) < config.early_stage_instances:
        raise ValueError("Not enough instances for early-stage evaluation")

    early_instances = instances[: config.early_stage_instances]
    evaluated_signatures: set[str] = set()
    history: list[dict] = []
    population: list[Candidate] = []

    # 初始化LLM接口和功能等价检测器
    llm = LLMInterface()
    equivalence_detector = FunctionEquivalenceDetector(early_instances)

    def evaluate_candidate(solver: Callable[[CVRPInstance], List[List[int]]], 
                         weights: Optional[tuple[float, float, float]] = None,
                         heuristic_code: Optional[str] = None) -> Candidate | None:
        # 生成行为签名
        sig = equivalence_detector.get_behavior_signature(solver)
        if sig is None:
            return None
        if sig in evaluated_signatures:
            return None
        evaluated_signatures.add(sig)

        # Stage 1: quick check on a subset (sample-efficient).
        early_metrics = evaluate_heuristic(early_instances, solver)
        early_s = _score(early_metrics)

        # Stage 2: full evaluation only if promising.
        threshold = 999999.0 if not population else sorted(c.score for c in population)[min(len(population) - 1, config.top_k - 1)]
        if early_s > threshold * 1.07 and population:
            # Too weak in early stage; skip full expensive evaluation.
            history.append({
                "weights": list(weights) if weights else None,
                "signature": sig,
                "status": "early_pruned",
                "early_score": round(early_s, 3),
                "type": "heuristic" if heuristic_code else "weighted"
            })
            return None

        full_metrics = evaluate_heuristic(instances, solver)
        final_score = _score(full_metrics)
        cand = Candidate(
            weights=weights,
            heuristic_code=heuristic_code,
            solver=solver,
            score=final_score,
            avg_routes=full_metrics["avg_num_routes"],
            signature=sig
        )
        history.append(
            {
                "weights": list(weights) if weights else None,
                "signature": sig,
                "status": "full_eval",
                "score": round(final_score, 3),
                "avg_distance": round(full_metrics["avg_distance"], 3),
                "avg_num_routes": round(full_metrics["avg_num_routes"], 3),
                "type": "heuristic" if heuristic_code else "weighted"
            }
        )
        return cand

    # Initial population
    for _ in range(config.init_population):
        # 一半概率使用权重优化，一半概率使用LLM生成
        if config.use_llm and rng.random() < 0.5:
            # 使用LLM生成启发式算法
            heuristic_code = llm.generate_heuristic()
            if llm.validate_heuristic(heuristic_code):
                solver = llm.load_heuristic(heuristic_code)
                if solver:
                    cand = evaluate_candidate(solver, heuristic_code=heuristic_code)
                    if cand is not None:
                        population.append(cand)
        else:
            # 使用权重优化
            weights = (rng.uniform(0.4, 1.8), rng.uniform(-0.5, 1.3), rng.uniform(0.2, 1.6))
            solver = _solver_from_weights(weights)
            cand = evaluate_candidate(solver, weights=weights)
            if cand is not None:
                population.append(cand)

    population.sort(key=lambda c: c.score)

    # Main loop
    for _ in range(config.iterations):
        if not population:
            break
        
        # 决定使用哪种方式生成新候选
        if config.use_llm and rng.random() < config.llm_generation_ratio:
            # 使用LLM生成新的启发式算法
            heuristic_code = llm.generate_heuristic()
            if llm.validate_heuristic(heuristic_code):
                solver = llm.load_heuristic(heuristic_code)
                if solver:
                    cand = evaluate_candidate(solver, heuristic_code=heuristic_code)
                    if cand is not None:
                        population.append(cand)
                        population.sort(key=lambda c: c.score)
                        population = population[: max(config.top_k * 3, 8)]
        else:
            # 使用进化算法生成新权重
            parent = rng.choice(population[: min(config.top_k, len(population))])
            if parent.weights:
                new_weights = tuple(parent.weights[i] + rng.gauss(0.0, config.mutation_std) for i in range(3))
                solver = _solver_from_weights(new_weights)
                cand = evaluate_candidate(solver, weights=new_weights)
                if cand is not None:
                    population.append(cand)
                    population.sort(key=lambda c: c.score)
                    population = population[: max(config.top_k * 3, 8)]

    if not population:
        raise RuntimeError("Search failed to produce any valid candidate")

    best = population[0]
    return {
        "best_weights": list(best.weights) if best.weights else None,
        "best_heuristic_code": best.heuristic_code,
        "best_signature": best.signature,
        "best_score": round(best.score, 3),
        "best_avg_routes": round(best.avg_routes, 3),
        "unique_candidates": len(evaluated_signatures),
        "history": history,
    }
