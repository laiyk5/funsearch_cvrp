"""Run FunSearch with a module-based CVRP specification."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
import inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from funsearch_cvrp.funsearch import code_manipulation, config as config_lib, evaluator, funsearch, programs_database, sampler
from funsearch_cvrp.config import config as project_config
from funsearch_cvrp.cvrp.core import CVRPInstance, make_greedy_solver, make_savings_solver, gap_score, is_valid_solution, solution_distance
from funsearch_cvrp.cvrp.io import load_cvrplib_folder
from datetime import datetime as _dt
from funsearch_cvrp.utils.output_manager import (
    get_output_dir,
    get_git_commit_hash,
    is_git_dirty,
)

_logger = logging.getLogger('funsearch')


# ---------------------------------------------------------------------------
# Cross-platform sandbox
# ---------------------------------------------------------------------------

class CrossPlatformSandbox(evaluator.Sandbox):
    """Sandbox that works on both Unix and Windows."""

    def run(
        self,
        evolved_fn_code: str,
        function_name: str,
        evaluate_fn: Callable,
        test_input: Any,
        _timeout_seconds: int,
    ) -> tuple[Any, bool]:
        try:
            ast.parse(evolved_fn_code)
        except SyntaxError as e:
            _logger.debug("Sandbox syntax error in\n%s\n--> %s", evolved_fn_code, e)
            return None, False

        namespace = {}
        err_stream = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(err_stream):
                try:
                    exec(evolved_fn_code, namespace)
                except Exception as e:
                    _logger.debug("Sandbox exec error: %s", e)
                    return None, False

                if function_name not in namespace:
                    _logger.debug("Sandbox: function '%s' not found", function_name)
                    return None, False

                func = namespace[function_name]
                try:
                    result = evaluate_fn(test_input, func)
                except Exception as e:
                    _logger.debug("Sandbox evaluation error: %s", e)
                    return None, False

        stderr_output = err_stream.getvalue()
        if stderr_output:
            _logger.debug("Sandbox stderr: %s", stderr_output)

        return result, True


# ---------------------------------------------------------------------------
# Iteration-limited sampler
# ---------------------------------------------------------------------------

class LimitedSampler(sampler.Sampler):
    """Sampler that stops after a fixed number of iterations."""

    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        evaluators: list[evaluator.Evaluator],
        llm: sampler.LLM,
        max_iterations: int,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = 0,
        start_iteration: int = 0,
        database_log_path: Path | None = None,
        log_roll_every: int = 0,
        log_dir: Path | None = None,
        test_inputs: list | None = None,
        test_sandbox=None,
        function_to_evolve: str = "",
        evaluate_fn=None,
        test_eval_path: Path | None = None,
        test_eval_every: int = 0,
    ) -> None:
        super().__init__(database, evaluators, llm)
        self._max_iterations = max_iterations
        self._iteration = start_iteration
        self._checkpoint_path = checkpoint_path
        self._checkpoint_every = checkpoint_every
        self._database_log_path = database_log_path
        self._log_roll_every = log_roll_every
        self._log_dir = log_dir
        self._last_rolled_block = start_iteration // log_roll_every if log_roll_every else 0
        self._test_inputs = test_inputs or []
        self._test_sandbox = test_sandbox
        self._function_to_evolve = function_to_evolve
        self._evaluate_fn = evaluate_fn
        self._test_eval_path = test_eval_path
        self._test_eval_every = test_eval_every

    @property
    def iteration(self) -> int:
        return self._iteration

    def _log_database(self) -> None:
        """Record current database state to database.jsonl."""
        if self._database_log_path is None:
            return
        record = {
            "iteration": self._iteration,
            "best_score_per_island": [
                float(s) if s != -float("inf") else None
                for s in self._database._best_score_per_island
            ],
            "overall_best": max(
                (s for s in self._database._best_score_per_island if s != -float("inf")),
                default=None,
            ),
            "timestamp": time.time(),
        }
        with open(self._database_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _maybe_roll_log(self) -> None:
        """Archive evolution.log and eval_history every N iterations."""
        if not self._log_roll_every or not self._log_dir:
            return
        block = self._iteration // self._log_roll_every
        if block <= self._last_rolled_block:
            return
        self._last_rolled_block = block

        prev_block = block - 1
        start_it = prev_block * self._log_roll_every
        end_it = start_it + self._log_roll_every - 1

        # Roll evolution.log
        logs_dir = self._log_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        current_log = logs_dir / "evolution.log"
        archived_log = logs_dir / f"evolution_iter_{start_it:06d}_{end_it:06d}.log"

        funsearch_logger = logging.getLogger('funsearch')
        for h in list(funsearch_logger.handlers):
            if isinstance(h, logging.FileHandler):
                funsearch_logger.removeHandler(h)
                h.flush()
                h.close()
        if current_log.exists():
            current_log.rename(archived_log)
        new_h = logging.FileHandler(current_log)
        new_h.setLevel(logging.DEBUG)
        new_h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        funsearch_logger.addHandler(new_h)

        # Roll eval_history.jsonl
        eval_dir = self._log_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        current_eval = eval_dir / "eval.jsonl"
        archived_eval = eval_dir / f"eval_iter_{start_it:06d}_{end_it:06d}.jsonl"

        if current_eval.exists():
            current_eval.rename(archived_eval)
        for e in self._evaluators:
            e._eval_history_path = current_eval

        # Roll database_log.jsonl
        db_dir = self._log_dir / "database"
        db_dir.mkdir(parents=True, exist_ok=True)
        current_db = db_dir / "database.jsonl"
        archived_db = db_dir / f"database_iter_{start_it:06d}_{end_it:06d}.jsonl"

        if current_db.exists():
            current_db.rename(archived_db)
        self._database_log_path = current_db

        # Roll sampler_log.jsonl
        sampler_dir = self._log_dir / "sampler"
        sampler_dir.mkdir(parents=True, exist_ok=True)
        current_sampler = sampler_dir / "sampler.jsonl"
        archived_sampler = sampler_dir / f"sampler_iter_{start_it:06d}_{end_it:06d}.jsonl"

        if current_sampler.exists():
            current_sampler.rename(archived_sampler)
        self._llm._sampler_log_path = current_sampler

        _logger.info("Rolled evolution.log -> %s", archived_log.name)

    def _evaluate_best_on_test(self) -> None:
        """Evaluate the current best program on held-out test inputs."""
        if not self._test_inputs or not self._test_sandbox or not self._test_eval_path:
            return

        # Find the best program overall across islands
        best = None
        best_score = -float("inf")
        for island_id in range(len(self._database._islands)):
            p = self._database._best_program_per_island[island_id]
            s = self._database._best_score_per_island[island_id]
            if p is not None and s > best_score:
                best = p
                best_score = s

        if best is None:
            return

        scores_per_test = {}
        for idx, test_input in enumerate(self._test_inputs):
            result, runs_ok = self._test_sandbox.run(
                str(best).strip(),
                self._function_to_evolve,
                self._evaluate_fn,
                test_input,
                30,
            )
            if runs_ok and result is not None:
                scores_per_test[idx] = result

        if scores_per_test:
            test_score = scores_per_test[list(scores_per_test.keys())[-1]]
            record = {
                "iteration": self._iteration,
                "train_best": best_score,
                "test_score": test_score,
                "scores_per_test": scores_per_test,
                "timestamp": time.time(),
            }
            with open(self._test_eval_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            _logger.info(
                "TEST EVAL iter=%d train=%.4f test=%.4f",
                self._iteration, best_score, test_score,
            )

    def sample(self) -> None:
        while self._iteration < self._max_iterations:
            generation_time = time.time()
            prompt = self._database.get_prompt()
            samples = self._llm.draw_samples(prompt.code)
            for sample in samples:
                if self._iteration >= self._max_iterations:
                    return
                chosen_evaluator = self._evaluators[self._iteration % len(self._evaluators)]
                scores = chosen_evaluator.analyse(
                    sample, prompt.island_id, prompt.version_generated,
                    generation_time=generation_time,
                    iteration=self._iteration,
                )
                self._iteration += 1
                self._log_database()
                self._maybe_roll_log()

                if self._test_eval_every > 0 and self._iteration % self._test_eval_every == 0:
                    self._evaluate_best_on_test()

                if self._checkpoint_every > 0 and self._checkpoint_path:
                    if self._iteration % self._checkpoint_every == 0:
                        self._database.save(
                            self._checkpoint_path,
                            metadata={"completed_iterations": self._iteration},
                        )
                        _logger.info(
                            "Checkpoint saved at iteration %d/%d",
                            self._iteration,
                            self._max_iterations,
                        )


# ---------------------------------------------------------------------------
# Mock LLM for testing
# ---------------------------------------------------------------------------

class MockLLM(sampler.LLM):
    """Mock LLM for testing FunSearch pipeline without API calls."""

    def __init__(self, samples_per_prompt: int, *, sampler_log_path: Path | None = None) -> None:
        super().__init__(samples_per_prompt, sampler_log_path=sampler_log_path)
        self._count = 0

    def _draw_sample(self, prompt: str) -> str:
        import time as _time
        generation_time = _time.time()
        self._count += 1
        variants = [
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  d_dep = math.hypot(instance.coords[candidate][0] - instance.coords[0][0], instance.coords[candidate][1] - instance.coords[0][1])\n  return -d_cur + 0.5 * d_dep',
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  if remaining_capacity < instance.demands[candidate]:\n    return -1e6\n  return -(d_cur / (remaining_capacity + 1))',
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  d_dep = math.hypot(instance.coords[0][0] - instance.coords[candidate][0], instance.coords[0][1] - instance.coords[candidate][1])\n  return -d_cur - 0.1 * d_dep * instance.demands[candidate]',
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  return -d_cur / (instance.demands[candidate] + 1)',
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  d_dep = math.hypot(instance.coords[0][0] - instance.coords[candidate][0], instance.coords[0][1] - instance.coords[candidate][1])\n  ratio = remaining_capacity / instance.capacity if instance.capacity else 0\n  return -d_cur + 0.3 * d_dep + 2.0 * ratio',
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  d_dep = math.hypot(instance.coords[0][0] - instance.coords[candidate][0], instance.coords[0][1] - instance.coords[candidate][1])\n  if remaining_capacity >= instance.demands[candidate]:\n    return -d_cur\n  return -d_cur - 10.0 * (instance.demands[candidate] - remaining_capacity)',
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  d_dep = math.hypot(instance.coords[0][0] - instance.coords[candidate][0], instance.coords[0][1] - instance.coords[candidate][1])\n  return -d_cur * (remaining_capacity / instance.capacity) + d_dep * 0.2',
            'import math\n  d_cur = math.hypot(instance.coords[current_node][0] - instance.coords[candidate][0], instance.coords[current_node][1] - instance.coords[candidate][1])\n  d_dep = math.hypot(instance.coords[0][0] - instance.coords[candidate][0], instance.coords[0][1] - instance.coords[candidate][1])\n  if len(unserved) > 0:\n    return -d_cur + d_dep / len(unserved)\n  return -d_cur',
        ]
        code = variants[self._count % len(variants)]
        _logger.debug('MOCK LLM PROMPT:\n%s', prompt)
        _logger.debug('MOCK LLM RESPONSE:\n%s', code)
        self._write_sampler_log({
            "generation_time": generation_time,
            "model": "mock",
            "temperature": 0,
            "max_tokens": 0,
            "prompt": prompt,
            "raw_response": code,
            "extracted_code": code,
        })
        return code


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_best_programs(database: programs_database.ProgramsDatabase) -> list[dict]:
    results = []
    for island_id in range(len(database._islands)):
        best_program = database._best_program_per_island[island_id]
        best_score = database._best_score_per_island[island_id]
        best_scores_per_test = database._best_scores_per_test_per_island[island_id]

        if best_program is not None:
            results.append({
                "island_id": island_id,
                "best_score": best_score,
                "scores_per_test": (
                    {str(k): v for k, v in best_scores_per_test.items()}
                    if best_scores_per_test else {}
                ),
                "program": str(best_program),
            })
    return results


def save_funsearch_results(
    output_dir: Path,
    database: programs_database.ProgramsDatabase,
    config: config_lib.Config,
    inputs: list[Any],
    duration_seconds: float,
    llm_calls: int,
) -> None:
    best_programs = extract_best_programs(database)

    results = {
        "config": {
            "num_islands": config.programs_database.num_islands,
            "functions_per_prompt": config.programs_database.functions_per_prompt,
            "reset_period": config.programs_database.reset_period,
            "samples_per_prompt": config.samples_per_prompt,
            "num_evaluators": config.num_evaluators,
            "num_samplers": config.num_samplers,
            "llm_model": config.llm.model if config.llm else None,
            "llm_temperature": config.llm.temperature if config.llm else None,
        },
        "inputs": [str(i) for i in inputs],
        "duration_seconds": duration_seconds,
        "llm_calls": llm_calls,
        "best_programs": best_programs,
        "overall_best": max(
            (p["best_score"] for p in best_programs),
            default=None,
        ),
    }

    results_file = output_dir / "funsearch_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _logger.info(f"Results saved to {results_file}")

    if best_programs:
        best = max(best_programs, key=lambda p: p["best_score"])
        best_code_file = output_dir / "best_program.py"
        with open(best_code_file, "w", encoding="utf-8") as f:
            f.write(best["program"])
        _logger.info(f"Best program saved to {best_code_file}")


# ---------------------------------------------------------------------------
# CVRP evaluation harness (gap-based)
# ---------------------------------------------------------------------------

def evaluate_cvrp(instance_and_optimal: tuple[CVRPInstance, float], priority_fn) -> float:
    """Evaluate one CVRP instance using the evolved priority function.

    Returns:
        ``-gap`` where gap = (distance - optimal) / optimal.
        Higher is better (FunSearch maximizes).
        Invalid solutions return a large negative penalty.
    """
    instance, optimal = instance_and_optimal
    solver = make_greedy_solver(priority_fn)
    routes = solver(instance)

    valid, _ = is_valid_solution(instance, routes)
    if not valid:
        return -1e9

    dist = solution_distance(instance, routes)
    gap = (dist - optimal) / optimal
    return -gap


def _create_robustness_instances() -> list[tuple[CVRPInstance, float]]:
    """Create synthetic edge-case instances that stress-test evolved code.

    Covers:
      - customer at the same coordinates as the depot (d_i_depot = 0)
      - two customers at identical coordinates (d_i_j = 0)
      - customer with demand equal to capacity
    """
    instances = []

    # Instance 1: one customer at the depot location
    inst = CVRPInstance(
        name="robust_depot_overlap",
        capacity=10,
        demands=[0, 5, 3],
        coords=[(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)],
    )
    # optimal: depot->2->depot = 20, depot->1->depot = 0 (but 1 is at depot)
    # Actually optimal routes: [1] (distance 0) and [2] (distance 20)
    instances.append((inst, 20.0))

    # Instance 2: two customers at identical coordinates
    inst = CVRPInstance(
        name="robust_identical_coords",
        capacity=10,
        demands=[0, 4, 4],
        coords=[(0.0, 0.0), (5.0, 5.0), (5.0, 5.0)],
    )
    # optimal: [1,2] -> depot->1->2->depot = 0 (identical coords, distance 0 between them)
    # Wait, distance from depot to (5,5) is ~7.07. Route [1,2] = 7.07 + 0 + 7.07 = 14.14
    instances.append((inst, 14.142135623730951))

    # Instance 3: one customer with demand = capacity
    inst = CVRPInstance(
        name="robust_full_capacity",
        capacity=10,
        demands=[0, 10, 3],
        coords=[(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)],
    )
    # optimal: [1] -> 10, [2] -> 20 = 30
    instances.append((inst, 30.0))

    return instances


def evaluate_cvrp_savings(instance_and_optimal: tuple[CVRPInstance, float], savings_fn) -> float:
    """Evaluate one CVRP instance using the evolved savings function.

    Uses a Clarke-Wright-style savings construction followed by 2-opt.

    Returns:
        ``-gap`` where gap = (distance - optimal) / optimal.
        Higher is better (FunSearch maximizes).
        Invalid solutions return a large negative penalty.
    """
    instance, optimal = instance_and_optimal
    solver = make_savings_solver(savings_fn, two_opt=True)
    routes = solver(instance)

    valid, _ = is_valid_solution(instance, routes)
    if not valid:
        return -1e9

    dist = solution_distance(instance, routes)
    gap = (dist - optimal) / optimal
    return -gap


PROMPT_PRIORITY = """You are improving a priority function for a Capacitated Vehicle Routing Problem (CVRP) solver.

The solver uses a greedy constructive heuristic. At each step it builds a route by choosing
the next customer with the highest priority score returned by your function.

The function signature is:

def priority(current_node, candidate, instance, remaining_capacity, route, route_demand, unserved) -> float:
    ...

Where:
- current_node: int — the node the vehicle is at now (0 = depot)
- candidate: int — the customer under consideration
- instance: CVRPInstance with attributes:
    - coords: list[(x, y)] — 2-D coordinates, index 0 is the depot
    - demands: list[int] — demand per node, demands[0] = 0
    - capacity: int — vehicle capacity
    - n_customers: int — number of customers
- remaining_capacity: int — unused capacity on the current vehicle
- route: list[int] — customers already assigned to the current route
- route_demand: int — sum of demands already on the current route
- unserved: set[int] — customers not yet in any route

Higher return value = higher priority for `candidate` to be chosen next.

Tips:
- Prefer closer customers (lower Euclidean distance from current_node).
- Consider remaining capacity vs. candidate demand.
- Consider the distance from candidate back to the depot.
- Balance short-term gains with long-term route efficiency.
"""

PROMPT_SAVINGS = """You are improving a savings function for a Capacitated Vehicle Routing Problem (CVRP) solver.

The solver uses a Clarke-Wright-style savings algorithm:
1. Start with individual routes (depot -> customer -> depot) for each customer.
2. Compute the "savings" of merging two routes: if route A ends at customer i
   and route B starts at customer j, merging them eliminates two depot visits
   (i->depot and depot->j) and replaces them with a direct i->j link.
3. Repeatedly merge the pair of routes with the highest savings until no more
   merges are feasible (capacity constraints).
4. Apply 2-opt local search to each route for refinement.

Your function computes the savings value for a potential merge.

The function signature is:

def savings(i, j, instance) -> float:
    ...

Where:
- i: int — last customer of the first route
- j: int — first customer of the second route
- instance: CVRPInstance with attributes:
    - coords: list[(x, y)] — 2-D coordinates, index 0 is the depot
    - demands: list[int] — demand per node, demands[0] = 0
    - capacity: int — vehicle capacity
    - n_customers: int — number of customers

Higher return value = higher priority for merging the routes ending at i and starting at j.

Tips:
- The classic Clarke-Wright savings is: distance(i, depot) + distance(depot, j) - distance(i, j)
- Consider demand balance: merging routes with complementary demands is often good.
- Consider spatial clustering: merging nearby routes is usually beneficial.
- Consider the angle between i->depot and depot->j — more collinear routes may merge better.
- Avoid creating routes that are too long or have poor shape.
- IMPORTANT: Always guard against division by zero. Some customers may be at the same coordinates as the depot or at the exact same location as another customer. Check `d_i_depot < 1e-9` or `d_i_j < 1e-9` before dividing and return a safe fallback (e.g., `float('inf')`) when distances are zero.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> config_lib.Config:
    """Build FunSearch config from project config.ini and CLI overrides."""
    db_config = config_lib.ProgramsDatabaseConfig(
        num_islands=args.num_islands,
        functions_per_prompt=args.functions_per_prompt,
        reset_period=args.reset_period,
        score_bucket_precision=2,  # bucket gap scores to 2 decimal places
    )

    if args.mock_llm:
        llm_config = None
    else:
        model = project_config.get("LLM", "openai_model", fallback="gpt-4")
        api_key = project_config.get("LLM", "openai_api_key", fallback="")
        base_url = project_config.get("LLM", "openai_base_url", fallback="")
        temperature = project_config.getfloat("LLM", "llm_temperature", fallback=0.7)
        max_tokens = project_config.getint("SEARCH", "llm_max_tokens", fallback=2000)

        if args.model:
            model = args.model
        if args.temperature is not None:
            temperature = args.temperature
        if args.max_tokens is not None:
            max_tokens = args.max_tokens

        llm_config = config_lib.LLMConfig(
            model=model,
            api_key=api_key if api_key else None,
            base_url=base_url if base_url else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return config_lib.Config(
        programs_database=db_config,
        llm=llm_config,
        num_evaluators=args.num_evaluators,
        num_samplers=1,
        samples_per_prompt=args.samples_per_prompt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FunSearch on CVRP")
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of LLM samples to generate (default: 20)",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use a mock LLM for testing (no API calls)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override LLM model name from config",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override LLM temperature from config",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override LLM max tokens from config",
    )
    parser.add_argument(
        "--num-islands",
        type=int,
        default=10,
        help="Number of islands (default: 10)",
    )
    parser.add_argument(
        "--functions-per-prompt",
        type=int,
        default=3,
        help="Functions per prompt (default: 3)",
    )
    parser.add_argument(
        "--reset-period",
        type=int,
        default=600,
        help="Island reset period in seconds (default: 600)",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=5,
        help="Samples per LLM prompt (default: 5)",
    )
    parser.add_argument(
        "--num-evaluators",
        type=int,
        default=1,
        help="Number of evaluators (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint .pkl file to resume from",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save checkpoint every N iterations (0 = only at end)",
    )
    parser.add_argument(
        "--dataset-folder",
        default=None,
        help="CVRPLib dataset folder (default: from config.ini CVRP.dataset_folder)",
    )
    parser.add_argument(
        "--limit-instances",
        type=int,
        default=None,
        help="Limit number of CVRP instances to evaluate",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help="Fraction of instances to hold out for test evaluation (default: 0.0 = no holdout)",
    )
    parser.add_argument(
        "--test-eval-every",
        type=int,
        default=10,
        help="Evaluate best program on test set every N iterations (default: 10)",
    )
    parser.add_argument(
        "--spec",
        default="priority",
        choices=["priority", "savings"],
        help="Specification to evolve: priority (greedy) or savings (Clarke-Wright style) (default: priority)",
    )

    args = parser.parse_args()

    # Import the specification module dynamically
    if args.spec == "savings":
        from specifications import cvrp_spec_savings as spec_module
        evaluate_fn = evaluate_cvrp_savings
        prompt_text = PROMPT_SAVINGS
        function_to_evolve = "savings"
    else:
        from specifications import cvrp_spec as spec_module
        evaluate_fn = evaluate_cvrp
        prompt_text = PROMPT_PRIORITY
        function_to_evolve = "priority"

    # Load instances with known optima
    folder = args.dataset_folder or project_config.get("CVRP", "dataset_folder", fallback="data/cvrplib/A")
    instances_sols = load_cvrplib_folder(folder, limit=args.limit_instances)
    inputs = [(inst, cost) for inst, _, cost in instances_sols]
    _logger.info(f"Loaded {len(inputs)} CVRP instances with known optima from {folder}")

    # Add synthetic edge-case instances to stress-test robustness
    robustness = _create_robustness_instances()
    inputs = robustness + inputs
    _logger.info(f"Prepended {len(robustness)} robustness instances (total {len(inputs)})")

    # Split into train / test for generalization evaluation
    train_inputs = inputs
    test_inputs = []
    if args.test_split > 0 and len(inputs) > 1:
        import random
        random.seed(42)
        shuffled = list(inputs)
        random.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * args.test_split))
        test_inputs = shuffled[:n_test]
        train_inputs = shuffled[n_test:]
        _logger.info(
            "Train/test split: %d train, %d test (%.0f%% held out)",
            len(train_inputs), len(test_inputs), args.test_split * 100,
        )

    # Build config
    config = build_config(args)

    # Determine output directory.  When resuming without an explicit output-dir,
    # create a fresh directory and copy the checkpoint so the original run stays
    # intact.  The new run loads the copied checkpoint and writes all new files
    # (logs, eval, best-program, etc.) into the new directory.
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    _resumed_from = None
    if args.resume:
        _resumed_from = str(Path(args.resume).resolve())

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.resume:
        resume_source = Path(args.resume).resolve()
        output_dir = get_output_dir("run_funsearch", args={
            "iterations": args.iterations,
            "mock_llm": args.mock_llm,
            "model": config.llm.model if config.llm else "mock",
        })
        checkpoint_dest = output_dir / "checkpoint.pkl"
        import shutil
        shutil.copy2(resume_source, checkpoint_dest)
        _logger.info("Copied checkpoint from %s to %s", resume_source, checkpoint_dest)
        args.resume = str(checkpoint_dest)
    else:
        output_dir = get_output_dir("run_funsearch", args={
            "iterations": args.iterations,
            "mock_llm": args.mock_llm,
            "model": config.llm.model if config.llm else "mock",
        })

    # Write / append run entry to meta.json so analysis tools discover every run.
    _meta_file = output_dir / "meta.json"
    _run_entry = {
        "run_time": _dt.now().isoformat(),
        "git_commit": get_git_commit_hash(short=False),
        "git_dirty": is_git_dirty(),
        "args": {
            "iterations": args.iterations,
            "mock_llm": args.mock_llm,
            "model": config.llm.model if config.llm else "mock",
        },
    }
    if _resumed_from:
        _run_entry["resumed_from"] = _resumed_from
    if _meta_file.exists():
        _meta = json.loads(_meta_file.read_text())
        _meta.setdefault("runs", []).append(_run_entry)
        _meta_file.write_text(json.dumps(_meta, indent=2, ensure_ascii=False))
    else:
        _meta_file.write_text(json.dumps({
            "runs": [_run_entry]}, indent=2, ensure_ascii=False))

    # Update latest symlink if we are inside an "outputs" tree.
    run_dir = output_dir.parent
    if run_dir.parent.name == "outputs":
        latest = run_dir.parent / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)

    checkpoint_path = output_dir / "checkpoint.pkl"
    _db_dir = output_dir / "database"
    _db_dir.mkdir(parents=True, exist_ok=True)
    database_log_path = _db_dir / "database.jsonl"
    best_program_path = output_dir / "best_program.py"
    _eval_dir = output_dir / "eval"
    _eval_dir.mkdir(parents=True, exist_ok=True)
    eval_history_path = _eval_dir / "eval.jsonl"
    _sampler_dir = output_dir / "sampler"
    _sampler_dir.mkdir(parents=True, exist_ok=True)
    sampler_log_path = _sampler_dir / "sampler.jsonl"
    _test_dir = output_dir / "test"
    _test_dir.mkdir(parents=True, exist_ok=True)
    test_eval_path = _test_dir / "test_eval.jsonl"

    # Set up named logger with file + console handlers
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False
    _logs_dir = output_dir / 'logs'
    _logs_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_logs_dir / 'evolution.log')
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    _sh = logging.StreamHandler()
    _sh.setLevel(logging.WARNING)
    _sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    _logger.addHandler(_fh)
    _logger.addHandler(_sh)

    # Build template from the evolve function
    source = inspect.getsource(getattr(spec_module, function_to_evolve))
    template = code_manipulation.text_to_program(source)

    # Create or resume database
    completed_iterations = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        _logger.info(f"Resuming from checkpoint: {resume_path}")
        database, metadata = programs_database.ProgramsDatabase.load(
            resume_path,
            config.programs_database,
            template,
            function_to_evolve,
        )
        # Re-enable best-program snapshotting for the resumed session.
        database._best_program_path = best_program_path
        completed_iterations = metadata.get("completed_iterations", 0)
        _logger.info(f"Resuming from iteration {completed_iterations}/{args.iterations}")
    else:
        database = programs_database.ProgramsDatabase(
            config.programs_database, template, function_to_evolve,
            best_program_path=best_program_path,
        )

    # Create sandbox
    sandbox = CrossPlatformSandbox()

    # Create evaluators (train only)
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            evaluate_fn,
            train_inputs,
            sandbox=sandbox,
            eval_history_path=eval_history_path,
        ))

    # Send initial implementation
    if not args.resume:
        initial = template.get_function(function_to_evolve).body
        evaluators[0].analyse(initial, island_id=None, version_generated=None)

    # Create LLM
    if args.mock_llm:
        _logger.info("Using mock LLM (no API calls)")
        llm = MockLLM(samples_per_prompt=config.samples_per_prompt, sampler_log_path=sampler_log_path)
    else:
        if not config.llm or not config.llm.api_key:
            _logger.error(
                "No API key configured. Set openai_api_key in config.ini "
                "or use --mock-llm for testing."
            )
            sys.exit(1)
        llm = sampler.OpenAILLM(
            samples_per_prompt=config.samples_per_prompt,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            prompt=prompt_text,
            sampler_log_path=sampler_log_path,
        )
        _logger.info(f"Using OpenAI LLM: {config.llm.model}")

    # Run with iteration limit
    limited_sampler = LimitedSampler(
        database,
        evaluators,
        llm,
        max_iterations=args.iterations,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        start_iteration=completed_iterations,
        database_log_path=database_log_path,
        log_roll_every=1000,
        log_dir=output_dir,
        test_inputs=test_inputs,
        test_sandbox=sandbox,
        function_to_evolve=function_to_evolve,
        evaluate_fn=evaluate_fn,
        test_eval_path=test_eval_path,
        test_eval_every=args.test_eval_every,
    )

    _logger.info("Starting FunSearch...")
    start_time = time.time()
    limited_sampler.sample()
    duration = time.time() - start_time
    _logger.info(f"FunSearch completed in {duration:.1f}s")

    # Final checkpoint
    database.save(
        checkpoint_path,
        metadata={"completed_iterations": limited_sampler.iteration},
    )

    # Final test evaluation
    limited_sampler._evaluate_best_on_test()

    # Save results
    save_funsearch_results(
        output_dir,
        database,
        config,
        train_inputs,
        duration,
        llm_calls=limited_sampler._iteration,
    )

    _logger.info(f"All done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
