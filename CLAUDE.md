# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Sample-Efficient FunSearch** implementation for solving the **Capacitated Vehicle Routing Problem (CVRP)**. The project uses LLMs (OpenAI GPT or Alibaba Qwen) to automatically generate and optimize heuristic algorithms, combining evolutionary search with LLM-generated code.

## Common Commands

### Setup
```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or activate existing environment
source .venv/bin/activate
```

### Configuration
Copy `.env.example` to `.env` and configure API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Testing
```bash
# Run all unit tests
python -m unittest discover -s tests -p "test_*.py"

# Test LLM connection
python scripts/check/test_llm.py

# Quick test (1 iteration, 5 heuristics)
python scripts/run/test_simple.py
```

### Running Experiments
```bash
# Milestone experiment
python scripts/run/run_milestone.py

# Full project with synthetic data
python scripts/run/run_full_project.py --dataset synthetic

# Full project with CVRPLib data
python scripts/run/run_full_project.py --dataset cvrplib --cvrplib-dir "data/A" --limit-instances 10

# Iterative LLM-based generation
python scripts/run/generate_heuristics.py
```

### Analyzing Results
```bash
# List all results (organized by git commit)
python scripts/analyze/list_results.py

# Extract generated Python code from JSON results
python scripts/analyze/extract_generated_codes.py

# Visualize results (generates charts in outputs/{commit}/{timestamp}/charts/)
python scripts/analyze/visualize_results.py

# Generate dashboard only
python scripts/analyze/visualize_results.py --dashboard-only

# Visualize specific result
python scripts/analyze/visualize_results.py --commit 7d9a1d6 --timestamp 20250412_153033

# Visualize CVRP routes on a map
python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp --auto-solve

# Compare multiple solvers visually
python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp \
    --compare "nearest_neighbor,clarke_wright,clarke_wright_2opt"

# Visualize synthetic instance with specific solver
python scripts/analyze/visualize_route.py --synthetic --size 50 --auto-solve --solver clarke_wright

# Visualize FunSearch-generated heuristic (by iteration number)
python scripts/analyze/visualize_route.py --synthetic --size 50 --funsearch --iteration 2

# Visualize FunSearch heuristic by ID
python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp --funsearch --iteration-id 6

# Visualize from a saved heuristic file
python scripts/analyze/visualize_route.py --instance data/A/A-n32-k5.vrp \
    --heuristic-file outputs/{commit}/{timestamp}/generated/heuristic_iter02.py
```

### Linting/Formatting
```bash
# Ruff is configured in pyproject.toml
ruff check .
ruff format .
```

## Architecture Overview

### Core Components

**CVRP Data Model** (`src/cvrp/core.py`)
- `CVRPInstance`: Dataclass with `name`, `capacity`, `demands`, `coords`
- `evaluate_heuristic()`: Evaluates a solver on multiple instances
- `weighted_greedy_heuristic()`: Main solver parameterized by 3 weights `(w1, w2, w3)`
- `generate_synthetic_benchmarks()`: Creates test instances with deterministic seed

**Sample-Efficient Search** (`src/cvrp/search.py`)
- `SearchConfig`: Configuration dataclass for search parameters
- `run_sample_efficient_search()`: Main algorithm combining weight optimization with LLM generation
- **Early Pruning**: Evaluates on small subset first; skips full evaluation if early score > 1.07x threshold
- **Score Function**: `avg_distance + 20.0 * avg_num_routes` (lower is better)

**LLM Interface** (`src/llm/interface.py`)
- `LLMInterface`: Generates heuristics via LLM calls
- `generate_heuristic()`: Prompts LLM to generate new CVRP algorithm
- `validate_heuristic()`: Syntax validation of generated code
- `load_heuristic()`: Uses `exec()` to load generated code as executable function

**Function Equivalence** (`src/llm/equivalence.py`)
- `FunctionEquivalenceDetector`: Prevents redundant evaluations
- `get_behavior_signature()`: SHA256 hash of behavior on test instances
- `are_equivalent()`: Compares two heuristics for functional equivalence

**Baselines** (`src/cvrp/baselines.py`)
- `clarke_wright_savings_heuristic()`: Classic savings algorithm
- `two_opt_route()`: 2-opt local improvement
- `with_two_opt()`: Decorator to compose any solver with 2-opt

### Configuration System

Environment variables loaded from `.env` file (see `src/utils/config.py`):

**API Keys** (model name determines which API to use):
- `OPENAI_API_KEY`: For GPT models
- `DASHSCOPE_API_KEY`: For Qwen/DeepSeek models via Alibaba DashScope

**Key Config Variables**:
- `OPENAI_MODEL`: Model name (gpt-4, qwen-max, etc.)
- `N_ITERATIONS`: FunSearch iterations (default: 10)
- `MIN_HEURISTICS_PER_ITER` / `MAX_HEURISTICS_PER_ITER`: 50-100
- `EARLY_PRUNING_THRESHOLD`: 1.5 (score multiplier for pruning)
- `DATASET_SEED`: 2026 (for reproducibility)

### Output Organization

Results are organized by **git commit hash** to avoid overwriting:

```
outputs/
├── latest -> symlink to latest run
├── {commit_hash}/
│   ├── {timestamp}/
│   │   ├── iterative_search_results.json
│   │   └── run_info.json
│   └── generated_{timestamp}/  # Extracted Python files
│       └── heuristic_iter00_score1552.24.py
```

### Search Algorithm Flow

1. **Initialization**: Generate random weights or use LLM to generate heuristics (50/50 split)
2. **Early Evaluation**: Test on 2 small instances first
3. **Pruning**: Skip full evaluation if early score > threshold * 1.07
4. **Full Evaluation**: Evaluate promising candidates on all instances
5. **Population Update**: Keep top candidates, mutate weights or generate new heuristics
6. **Repeat** for configured iterations

### Code Execution Safety

Generated code from LLM is executed via `exec()` in a controlled namespace with limited imports (`math`, `random`, `CVRPInstance` only). The `validate_heuristic()` method checks for forbidden keywords before execution.

### Testing Strategy

- Tests in `tests/test_project.py` use synthetic benchmarks with fixed seed (2026)
- Smoke tests verify pipeline runs without errors
- Tests check that search returns valid candidates with expected structure
