# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sample-Efficient FunSearch implementation for solving the Capacitated Vehicle Routing Problem (CVRP). Uses LLMs to automatically generate and optimize heuristic algorithms, combining evolutionary search with LLM-generated code.

## Common Commands

### Setup
```bash
uv venv
uv pip install -e .
source .venv/bin/activate
```

### Configuration
Copy `config.ini.example` to `config.ini` and set your API key:
```bash
cp config.ini.example config.ini
# Edit config.ini: set dashscope_api_key or openai_api_key under [LLM]
```

### Testing
```bash
# Run all unit tests
python -m unittest discover -s tests -p "test_*.py"

# Run a single test
python -m unittest tests.test_project.TestCVRPProject.test_baseline_runs
```

### Linting/Formatting
```bash
ruff check .
ruff format .
```

### Analyzing Results
```bash
python scripts/analyze/list_results.py
python scripts/analyze/extract_generated_codes.py
python scripts/analyze/visualize_results.py
python scripts/analyze/visualize_route.py --instance data/cvrplib/A/A-n32-k5.vrp --auto-solve
```

## Architecture Overview

The project has two independent packages under `src/`:

### `src/funsearch_cvrp/` — Custom CVRP search pipeline

- **`cvrp/core.py`**: `CVRPInstance` dataclass, `evaluate_heuristic()`, `generate_synthetic_benchmarks()`. Score function: `avg_distance + 20.0 * avg_num_routes` (lower is better).
- **`cvrp/baselines.py`**: `clarke_wright_savings_heuristic()`, `two_opt_route()`, `with_two_opt()` decorator.
- **`cvrp/io.py`**: CVRPLib `.vrp` file loader.
- **`search/__init__.py`**: Exports `SearchConfig`, `run_sample_efficient_search` (weight-based), `run_iterative_search` (LLM-based), `LLMInterface`, `FunctionEquivalenceDetector`.
- **`search/generator.py`**: `run_iterative_search()` — main LLM-driven search loop with early pruning. Evaluates on a small subset first; skips full evaluation if early score exceeds threshold.
- **`search/interface.py`**: `LLMInterface` — generates heuristics via LLM, validates and loads them via `exec()` in a sandboxed namespace (`math`, `random`, `CVRPInstance` only). Falls back to 5 hardcoded template heuristics on API failure.
- **`search/equivalence.py`**: `FunctionEquivalenceDetector` — SHA256 behavior signature to skip redundant evaluations.
- **`config.py`**: Reads `config.ini` via `configparser`. Sections: `[LLM]` (model, endpoint, API keys, temperature) and `[SEARCH]` (iterations, pruning thresholds, etc.).
- **`utils/models.py`**: `ENDPOINTS` dict mapping provider names to base URLs and model IDs. `get_client()` creates an `openai.OpenAI` client based on `config.ini`.

### `src/funsearch/` — DeepMind FunSearch reference implementation

Standalone port of the official FunSearch pipeline (Apache 2.0). Entry point is `funsearch.main()`. Uses decorator-based specification: functions marked `@funsearch.evolve` are mutated by the LLM; `@funsearch.run` is the evaluation function. Components: `programs_database.py` (island-based population), `sampler.py` (LLM sampling), `evaluator.py` (sandboxed execution), `code_manipulation.py` (AST manipulation).

### Configuration System

`config.ini` (gitignored) overrides defaults from `config.ini.example`. The `config` object in `src/funsearch_cvrp/config.py` is a `configparser.ConfigParser` instance — access values via `config["LLM"]["MODEL"]` etc. API endpoint is auto-selected from `utils/models.py:ENDPOINTS` based on the model name, or can be overridden with `openai_base_url`.

### Output Organization

Results are organized by git commit hash:
```
outputs/
├── latest -> symlink to latest run
├── {commit_hash}/
│   ├── {timestamp}/
│   │   ├── iterative_search_results.json
│   │   └── run_info.json
│   └── generated_{timestamp}/
│       └── heuristic_iter00_score1552.24.py
```
