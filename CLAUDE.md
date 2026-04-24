# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sample-Efficient FunSearch implementation for solving the Capacitated Vehicle Routing Problem (CVRP). Uses LLMs to automatically generate and optimize heuristic algorithms, combining evolutionary search with LLM-generated code.

## Common Commands

### Setup

```bash
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate
```

### Configuration

Copy `config.ini.example` to `config.ini` and set your API key:

```bash
cp config.ini.example config.ini
# Edit config.ini: set openai_api_key and openai_model under [LLM]
```

### Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
pytest tests/
python -m unittest tests.test_project.TestCVRPProject.test_baseline_runs
```

### Linting/Formatting

```bash
ruff check .
ruff format .
```

### Running FunSearch

```bash
# Quick test with mock LLM (no API calls)
python scripts/run/run_funsearch.py --iterations 20 --mock-llm

# Run with real LLM (requires config.ini)
python scripts/run/run_funsearch.py --iterations 100 --dataset-folder data/cvrplib/A

# Resume from checkpoint (continues in the checkpoint's directory)
python scripts/run/run_funsearch.py --iterations 200 --resume outputs/latest/run_funsearch/checkpoint.pkl

# Limit test instances for faster evaluation
python scripts/run/run_funsearch.py --iterations 100 --limit-instances 10
```

### Visualization

```bash
python scripts/analyze/visualize_evolution.py outputs/latest/run_funsearch
```

## Architecture Overview

The project has two packages under `src/`:

### `src/funsearch_cvrp/` — CVRP domain logic

- **`cvrp/core.py`**: `CVRPInstance` dataclass, `make_greedy_solver()`, `gap_score()`, `is_valid_solution()`, `solution_distance()`. The greedy solver builds routes by repeatedly calling a user-provided `priority()` function.
- **`cvrp/baselines.py`**: `clarke_wright_savings_heuristic()`, `two_opt_route()`, `with_two_opt()` decorator.
- **`cvrp/io.py`**: CVRPLib `.vrp` file loader (`load_cvrplib_folder`).
- **`utils/output_manager.py`**: Creates `outputs/YYYYmmdd_HHMMSS/<script_name>/` directories and symlinks.
- **`config.py`**: Global `configparser.ConfigParser` instance. Loads defaults then overrides from `config.ini` if present.

### `src/funsearch/` — DeepMind FunSearch reference implementation

Standalone port (Apache 2.0):

- **`programs_database.py`**: Island-based population with clustering by behavior signature (`_get_signature`). `_reduce_score` uses the **last** test instance score (the hardest/most discriminating), not max. NEW BEST milestones trigger `best_program.py` export.
- **`sampler.py`**: `LLM` base class, `OpenAILLM` (OpenAI-compatible API with complex response post-processing), `Sampler` loop. Each LLM call is logged to `sampler.jsonl` with prompt, raw response, and extracted code.
- **`evaluator.py`**: `Evaluator` and `Sandbox` abstractions. `analyse()` logs accepted/rejected decisions to `evolution.log` and writes per-evaluation records to `eval.jsonl` (body, scores_per_test, timing, milestone flag).
- **`code_manipulation.py`**: AST-based program manipulation for LLM prompts. `text_to_function()` and `text_to_program()` parse generated code.

### `specifications/cvrp_spec.py` — Evolve target

Contains the `priority()` function that FunSearch mutates. Uses **2-space indentation**. Signature:

```python
def priority(current_node, candidate, instance, remaining_capacity, route, route_demand, unserved) -> float
```

### `scripts/run/run_funsearch.py` — Main entry point

Contains: `LimitedSampler` (iteration-limited sampling with checkpoint/history/log-rolling), `CrossPlatformSandbox` (AST-validated exec, cross-platform), `MockLLM`, `evaluate_cvrp()` (returns `-gap`), and `main()`.

Key CLI flags: `--mock-llm`, `--iterations`, `--num-islands`, `--samples-per-prompt`, `--dataset-folder`, `--limit-instances`, `--resume`, `--checkpoint-every`, `--output-dir`.

## Scoring Conventions

- FunSearch **maximizes** score. `evaluate_cvrp()` returns `-gap` where `gap = (distance - optimal) / optimal`. Invalid solutions get `-1e9`.
- `_reduce_score()` in `programs_database.py` picks the **last** test instance, not max. Tests are ordered with the hardest last. The evaluator's display score matches this.
- `ProgramsDatabaseConfig.score_bucket_precision=2` groups programs with gap rounded to 2 decimal places.

## LLM Response Post-Processing

`OpenAILLM._draw_sample()` applies a multi-stage extraction:

1. Strip markdown code fences (`` ``` ``)
2. Collect `import`/`from` lines placed before `def` by the LLM
3. Find first `def` line, keep from there
4. Parse via `text_to_function()`, trimming trailing markdown lines until it succeeds
5. Re-attach stripped imports **indented** into the body
6. `textwrap.dedent` then `textwrap.indent('  ')` to normalize to 2-space (matching the template)

## Output Organization

```
outputs/YYYYmmdd_HHMMSS/
├── run_funsearch/
│   ├── logs/
│   │   ├── evolution.log                    ← current block (text, human-readable)
│   │   └── evolution_iter_000000_000999.log ← archived every 1000 iterations
│   ├── eval/
│   │   ├── eval.jsonl                       ← current block (per-evaluation records)
│   │   └── eval_iter_000000_000999.jsonl
│   ├── database/
│   │   ├── database.jsonl                   ← current block (best scores per island)
│   │   └── database_iter_000000_000999.jsonl
│   ├── sampler/
│   │   ├── sampler.jsonl                    ← current block (LLM input/output)
│   │   └── sampler_iter_000000_000999.jsonl
│   ├── checkpoint.pkl
│   └── best_program.py                      ← exported on every NEW BEST milestone
├── visualize_evolution/                     ← analysis output (sibling to experiment)
│   ├── evolution.png
│   └── evolution_test_*.png
└── meta.json
```

All JSONL files and `evolution.log` roll every 1000 iterations. Active files use `{prefix}.jsonl`; archived use `{prefix}_iter_XXXXXX_XXXXXX.jsonl`. Analysis script output goes to `<parent_of_experiment>/<script_name>/`.

## Logging

Named logger `funsearch` (via `logging.getLogger('funsearch')`). File handler at DEBUG level writes to `logs/evolution.log`; StreamHandler at WARNING level for console. `programs_database.py` uses `import logging as _stdlib_logging` to avoid collision with `absl.logging`.

## Key Design Notes

- When resuming without `--output-dir`, the checkpoint's parent directory is reused (all files continue in-place).
- The `CrossPlatformSandbox` validates code with `ast.parse()` before `exec()`, but does not restrict the namespace.
- `--mock-llm` uses a hardcoded rotation of simple return expressions for testing the pipeline without API calls.
- `_reduce_score` uses the last test because it is the hardest/most discriminating. This is intentional from the DeepMind reference.
- Per-test signature tracking (via `eval.jsonl`) enables per-test-case evolution visualization.
