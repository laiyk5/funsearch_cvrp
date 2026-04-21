"""Run FunSearch on a specification file."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.funsearch import code_manipulation, config as config_lib, evaluator, funsearch, programs_database, sampler
from src.funsearch_cvrp.config import config as project_config
from src.funsearch_cvrp.cvrp.core import CVRPInstance
from src.funsearch_cvrp.cvrp.io import load_cvrplib_folder
from src.funsearch_cvrp.utils.output_manager import get_output_dir, save_run_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Windows-compatible sandbox
# ---------------------------------------------------------------------------

class CrossPlatformSandbox(evaluator.Sandbox):
    """Sandbox that works on both Unix and Windows."""

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input: Any,
        _timeout_seconds: int,
    ) -> tuple[Any, bool]:
        try:
            ast.parse(program)
        except SyntaxError:
            return None, False

        namespace = {}
        err_stream = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(err_stream):
                try:
                    exec(program, namespace)
                except Exception as e:
                    logging.debug("Sandbox exec error: %s", e)
                    return None, False

                if function_to_run not in namespace:
                    logging.debug("Sandbox: function '%s' not found", function_to_run)
                    return None, False

                func = namespace[function_to_run]
                try:
                    result = func(test_input)
                except Exception as e:
                    logging.debug("Sandbox function error: %s", e)
                    return None, False

        stderr_output = err_stream.getvalue()
        if stderr_output:
            logging.debug("Sandbox stderr: %s", stderr_output)

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
        history_path: Path | None = None,
    ) -> None:
        super().__init__(database, evaluators, llm)
        self._max_iterations = max_iterations
        self._iteration = start_iteration
        self._checkpoint_path = checkpoint_path
        self._checkpoint_every = checkpoint_every
        self._history_path = history_path

    @property
    def iteration(self) -> int:
        return self._iteration

    def _log_history(self) -> None:
        """Record current database state to history file."""
        if self._history_path is None:
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
        with open(self._history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def sample(self) -> None:
        while self._iteration < self._max_iterations:
            prompt = self._database.get_prompt()
            samples = self._llm.draw_samples(prompt.code)
            for sample in samples:
                if self._iteration >= self._max_iterations:
                    return
                chosen_evaluator = self._evaluators[self._iteration % len(self._evaluators)]
                chosen_evaluator.analyse(
                    sample, prompt.island_id, prompt.version_generated
                )
                self._iteration += 1
                self._log_history()

                if self._checkpoint_every > 0 and self._checkpoint_path:
                    if self._iteration % self._checkpoint_every == 0:
                        self._database.save(
                            self._checkpoint_path,
                            metadata={"completed_iterations": self._iteration},
                        )
                        logging.info(
                            "Checkpoint saved at iteration %d/%d",
                            self._iteration,
                            self._max_iterations,
                        )


# ---------------------------------------------------------------------------
# Mock LLM for testing
# ---------------------------------------------------------------------------

class MockLLM(sampler.LLM):
    """Mock LLM for testing FunSearch pipeline without API calls."""

    def __init__(self, samples_per_prompt: int) -> None:
        super().__init__(samples_per_prompt)
        self._count = 0

    def _draw_sample(self, _prompt: str) -> str:
        self._count += 1
        variants = [
            "    return 1.0",
            "    return -float(el[0])",
            "    return float(n - w)",
            "    return float(el.count(1))",
            "    return float(el.count(2))",
            "    return random.random()",
            "    return float(n) / (w + 1)",
            "    return -sum(el)",
        ]
        return variants[self._count % len(variants)]


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
    logging.info(f"Results saved to {results_file}")

    if best_programs:
        best = max(best_programs, key=lambda p: p["best_score"])
        best_code_file = output_dir / "best_program.py"
        with open(best_code_file, "w", encoding="utf-8") as f:
            f.write(best["program"])
        logging.info(f"Best program saved to {best_code_file}")


# ---------------------------------------------------------------------------
# CVRP instance helpers
# ---------------------------------------------------------------------------

def instance_to_dict(inst: CVRPInstance) -> dict:
    """Convert a CVRPInstance to the dict format expected by the spec."""
    return {
        "name": inst.name,
        "capacity": inst.capacity,
        "demands": inst.demands,
        "coords": inst.coords,
        "n_customers": inst.n_customers,
    }


def load_cvrp_inputs(dataset_folder: str | None = None, limit: int | None = None) -> list[dict]:
    """Load CVRP instances from the configured dataset folder."""
    folder = dataset_folder or project_config.get("CVRP", "dataset_folder", fallback="data/cvrplib/A")
    instances_sols = load_cvrplib_folder(folder, limit=limit)
    return [instance_to_dict(inst) for inst, _, _ in instances_sols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> config_lib.Config:
    """Build FunSearch config from project config.ini and CLI overrides."""
    db_config = config_lib.ProgramsDatabaseConfig(
        num_islands=args.num_islands,
        functions_per_prompt=args.functions_per_prompt,
        reset_period=args.reset_period,
    )

    if args.mock_llm:
        llm_config = None
    else:
        # Read LLM settings from project config
        model = project_config.get("LLM", "openai_model", fallback="gpt-4")
        api_key = project_config.get("LLM", "openai_api_key", fallback="")
        base_url = project_config.get("LLM", "openai_base_url", fallback="")
        temperature = project_config.getfloat("LLM", "llm_temperature", fallback=0.7)
        max_tokens = project_config.getint("SEARCH", "llm_max_tokens", fallback=2000)

        # CLI overrides
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
    parser = argparse.ArgumentParser(description="Run FunSearch on a specification file")
    parser.add_argument(
        "--spec",
        default="specifications/specification_nonsymmetric_admissible_set.txt",
        help="Path to the FunSearch specification file",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Test inputs (e.g., '8,3' '9,3'). Defaults to standard admissible set test cases.",
    )
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

    args = parser.parse_args()

    # Load specification
    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    specification = spec_path.read_text(encoding="utf-8")

    # Determine whether this is a CVRP spec (heuristic) or admissible-set spec
    is_cvrp_spec = "cvrp" in spec_path.name.lower()

    # Parse inputs
    if is_cvrp_spec:
        inputs = load_cvrp_inputs(args.dataset_folder, limit=args.limit_instances or 3)
        logging.info(f"Loaded {len(inputs)} CVRP instances from dataset")
    elif args.inputs:
        inputs = []
        for inp in args.inputs:
            parts = [int(x.strip()) for x in inp.split(",")]
            inputs.append(tuple(parts))
    else:
        inputs = [(8, 3), (9, 3), (10, 3), (11, 3)]

    logging.info(f"Specification: {spec_path}")
    logging.info(f"Test inputs: {inputs}")
    logging.info(f"Iterations: {args.iterations}")

    # Extract function names
    function_to_evolve, function_to_run = funsearch._extract_function_names(specification)
    logging.info(f"Evolving: {function_to_evolve}, Running: {function_to_run}")

    # Build config
    config = build_config(args)

    # Determine output directory early (needed for checkpoint / history paths)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_output_dir()

    checkpoint_path = output_dir / "checkpoint.pkl"
    history_path = output_dir / "history.jsonl"

    # Parse template
    template = code_manipulation.text_to_program(specification)

    # Create or resume database
    completed_iterations = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        logging.info(f"Resuming from checkpoint: {resume_path}")
        database, metadata = programs_database.ProgramsDatabase.load(
            resume_path,
            config.programs_database,
            template,
            function_to_evolve,
        )
        completed_iterations = metadata.get("completed_iterations", 0)
        logging.info(f"Resuming from iteration {completed_iterations}/{args.iterations}")
    else:
        database = programs_database.ProgramsDatabase(
            config.programs_database, template, function_to_evolve
        )

    # Create sandbox
    sandbox = CrossPlatformSandbox()

    # Create evaluators
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            sandbox=sandbox,
        ))

    # Send initial implementation only on fresh start
    if not args.resume:
        initial = template.get_function(function_to_evolve).body
        evaluators[0].analyse(initial, island_id=None, version_generated=None)

    # Create LLM
    if args.mock_llm:
        logging.info("Using mock LLM (no API calls)")
        llm = MockLLM(samples_per_prompt=config.samples_per_prompt)
    else:
        if not config.llm or not config.llm.api_key:
            logging.error(
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
        )
        logging.info(f"Using OpenAI LLM: {config.llm.model}")

    # Run with iteration limit
    limited_sampler = LimitedSampler(
        database,
        evaluators,
        llm,
        max_iterations=args.iterations,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        start_iteration=completed_iterations,
        history_path=history_path,
    )

    logging.info("Starting FunSearch...")
    start_time = time.time()
    limited_sampler.sample()
    duration = time.time() - start_time
    logging.info(f"FunSearch completed in {duration:.1f}s")

    # Final checkpoint
    database.save(
        checkpoint_path,
        metadata={"completed_iterations": limited_sampler.iteration},
    )

    # Save results

    save_run_info(output_dir, extra_info={
        "specification": str(spec_path),
        "inputs": [str(i) for i in inputs],
        "iterations": args.iterations,
        "mock_llm": args.mock_llm,
        "model": config.llm.model if config.llm else "mock",
    })
    save_funsearch_results(
        output_dir,
        database,
        config,
        inputs,
        duration,
        llm_calls=limited_sampler._iteration,
    )

    logging.info(f"All done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
