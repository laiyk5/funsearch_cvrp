# CVRP FunSearch Project - AI Agent Guide

## Project Overview

This is a **Sample-Efficient FunSearch** implementation for solving the **Capacitated Vehicle Routing Problem (CVRP)**. The project uses Large Language Models (LLMs) to automatically generate and optimize heuristic algorithms for CVRP, combining evolutionary search with LLM-generated code.

### Key Features
- **LLM-driven heuristic generation**: Uses Alibaba Tongyi Qianwen models to generate novel CVRP heuristic algorithms
- **Sample-efficient search**: Early pruning strategy to reduce computation overhead
- **Functional deduplication**: Behavior signature-based equivalence detection to avoid redundant evaluations
- **Multi-scale evaluation**: Tests on small (≤35 customers), medium (36-55), and large (≥56) instances
- **CVRPLib support**: Can load and evaluate on standard CVRPLib benchmark instances

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13+ |
| Core Dependencies | Python standard library only |
| Optional Dependencies | `python-docx>=1.1.0` (for Word report export) |
| LLM API | OpenAI-compatible API (Alibaba Tongyi Qianwen) |
| Testing | unittest |

## Project Structure

```
.
├── config.py                      # LLM and search configuration
├── models.py                      # LLM client and model definitions
├── cvrp_core.py                   # CVRP data model and baseline solvers (root version)
├── sample_efficient_search.py     # Sample-efficient search implementation (root version)
├── run_milestone.py               # Original milestone runner (root version)
├── test_llm.py                    # LLM connection test
├── test_simple.py                 # Quick test script
├── test_with_a_data.py            # Test with CVRPLib A dataset
├── milestone_cvrp/                # Main project directory
│   ├── README.md                  # Project documentation (Chinese)
│   ├── requirements.txt           # Minimal dependencies
│   ├── project_description.md     # Detailed project description (Chinese)
│   ├── cvrp_core.py               # CVRP core: data model, distance utilities, baselines
│   ├── baselines.py               # Clarke-Wright Savings, 2-opt improvement
│   ├── cvrplib_io.py              # CVRPLib .vrp file loader
│   ├── sample_efficient_search.py # Sample-efficient search with LLM integration
│   ├── llm_interface.py           # LLM interface for heuristic generation
│   ├── function_equivalence.py    # Functional equivalence detector
│   ├── generate_heuristics.py     # Iterative LLM-based heuristic generation
│   ├── run_milestone.py           # Milestone experiment runner
│   ├── run_full_project.py        # Full benchmark runner
│   ├── generate_report.py         # Markdown report generator
│   ├── generate_report_detailed.py# Detailed report generator
│   ├── generate_final_report.py   # Final project report generator
│   ├── export_report_docx.py      # Word export utility
│   ├── export_report_detailed.py  # Detailed Word export
│   ├── tests/
│   │   └── test_project.py        # Unit tests
│   └── outputs/                   # Generated results
├── A/A/                           # CVRPLib test instances (.vrp and .sol files)
├── outputs/                       # Root-level output directory
└── models/                        # Empty directory (placeholder)
```

## Core Components

### 1. CVRP Data Model (`cvrp_core.py`)
- `CVRPInstance`: Dataclass representing a CVRP problem instance
- `euclid()`: Euclidean distance calculation
- `route_distance()`: Calculate total distance of a route
- `solution_distance()`: Calculate total distance of a solution
- `nearest_neighbor_heuristic()`: Baseline greedy solver
- `weighted_greedy_heuristic()`: Weight-parameterized greedy solver
- `evaluate_heuristic()`: Evaluate a solver on multiple instances
- `generate_synthetic_benchmarks()`: Generate synthetic test instances

### 2. LLM Interface (`llm_interface.py`)
- `LLMInterface`: Main class for LLM interaction
  - `generate_heuristic()`: Generate a new heuristic algorithm using LLM
  - `validate_heuristic()`: Validate generated code
  - `load_heuristic()`: Load generated code as executable function
  - Fallback templates: Nearest neighbor, savings, sweep, cluster-first, regret heuristics

### 3. Sample-Efficient Search (`sample_efficient_search.py`)
- `SearchConfig`: Configuration dataclass for search parameters
- `run_sample_efficient_search()`: Main search algorithm
  - Combines weight optimization with LLM-generated heuristics
  - Early pruning based on small-scale evaluation
  - Maintains population of candidates with mutation

### 4. Function Equivalence Detection (`function_equivalence.py`)
- `FunctionEquivalenceDetector`: Detects functionally equivalent heuristics
  - `get_behavior_signature()`: Generate SHA256 hash of behavior on test instances
  - `are_equivalent()`: Compare two heuristics for equivalence
  - Prevents redundant evaluations of semantically identical algorithms

### 5. Baseline Algorithms (`baselines.py`)
- `clarke_wright_savings_heuristic()`: Classic savings algorithm
- `two_opt_route()`: 2-opt local improvement for single route
- `two_opt_improvement()`: Apply 2-opt to all routes
- `with_two_opt()`: Decorator to compose any solver with 2-opt

### 6. CVRPLib I/O (`cvrplib_io.py`)
- `load_cvrplib_instance()`: Load a .vrp file
- `load_cvrplib_folder()`: Load all .vrp files from a directory

## Configuration

### config.py
```python
# OpenAI API configuration (uses Alibaba Tongyi)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "DASHSCOPE_API_KEY")
OPENAI_MODEL = "qwen3-max-2025-09-23"

# LLM generation settings
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000

# Search configuration
N_ITERATIONS = 10
MIN_HEURISTICS_PER_ITER = 50
MAX_HEURISTICS_PER_ITER = 100

# Dataset configuration
DATASET_SIZES = [20, 50, 100]
DATASET_SEED = 2026

# Evaluation thresholds
EARLY_PRUNING_THRESHOLD = 1.5
LARGE_SCALE_THRESHOLD = 4.0
```

### models.py
Defines available LLM models and the `get_normal_client()` function for creating API clients.

**Required Environment Variable:**
```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

## Build and Run Commands

### Setup Environment
```bash
# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Or (Linux/Mac)
source .venv/bin/activate

# Install optional dependency for Word export
pip install python-docx>=1.1.0
```

### Run Tests
```bash
# Run all unit tests
cd milestone_cvrp
python -m unittest discover -s tests -p "test_*.py"

# Test LLM connection
python test_llm.py

# Quick test (1 iteration, 5 heuristics)
python test_simple.py
```

### Run Milestone Experiment
```bash
cd milestone_cvrp
python run_milestone.py
python generate_report.py
```

### Run Full Project with Synthetic Data
```bash
cd milestone_cvrp
python run_full_project.py --dataset synthetic
python generate_final_report.py
```

### Run with CVRPLib Data
```bash
cd milestone_cvrp
python run_full_project.py --dataset cvrplib --cvrplib-dir "A/A" --limit-instances 10
```

### Run Iterative Heuristic Generation
```bash
cd milestone_cvrp
python generate_heuristics.py

# Or from root (uses A/A dataset)
python test_with_a_data.py
```

## Code Style Guidelines

1. **Type Hints**: Use Python 3.9+ type hints (`list[int]`, `tuple[float, float]`)
2. **Docstrings**: Use Google-style docstrings for all public functions
3. **Imports**: Use `from __future__ import annotations` for forward references
4. **Naming**: 
   - Functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
5. **Comments**: Primary documentation language is Chinese (项目注释主要使用中文)

## Testing Strategy

- **Unit Tests**: Located in `milestone_cvrp/tests/test_project.py`
- **Smoke Tests**: Verify core pipeline runs without errors
- **Integration Tests**: Test LLM integration and full search pipeline
- **Dataset**: Uses synthetic benchmarks (deterministic with seed=2026) and CVRPLib A-instances

### Test Data Categories
- Small scale: ≤35 customers
- Medium scale: 36-55 customers  
- Large scale: ≥56 customers

## Output Files

### JSON Results
- `outputs/milestone_results.json`: Milestone experiment results
- `outputs/search_history.json`: Search history with pruning stats
- `outputs/full_project_results.json`: Full benchmark results
- `outputs/full_search_history.json`: Full search history
- `outputs/iterative_search_results.json`: LLM-generated heuristic results
- `outputs/test_a_data_results.json`: A-dataset test results

### Reports
- `milestone_report.md`: Basic milestone report
- `milestone_report_detailed.md`: Extended milestone report
- `final_project_report.md`: Final project report
- `milestone_report.docx`: Word format report

## Key Algorithms

### Score Function
```python
def _score(metrics: dict) -> float:
    # Lower is better; penalizes both distance and number of routes
    return metrics["avg_distance"] + 20.0 * metrics["avg_num_routes"]
```

### Weighted Greedy Heuristic
```python
# Score combines three factors:
# w1 * (-distance from current) + w2 * (demand ratio) + w3 * (-distance to depot)
score = w1 * (-d_cur) + w2 * demand_ratio + w3 * (-d_dep)
```

### Early Pruning Strategy
1. Evaluate on small-scale instances first
2. Skip full evaluation if early score > threshold * 1.07
3. Only promising candidates proceed to full evaluation

## Security Considerations

1. **API Keys**: Never commit API keys to version control. Use environment variables:
   - `DASHSCOPE_API_KEY` for Alibaba Tongyi
   - `OPENAI_API_KEY` (fallback)

2. **Code Execution**: The project uses `exec()` to load LLM-generated code. Generated code is executed in a controlled namespace with limited imports (only `math`, `random`, `CVRPInstance`).

3. **File Paths**: Use `pathlib.Path` for cross-platform path handling.

## Common Tasks

### Add a New Baseline Algorithm
1. Implement in `milestone_cvrp/baselines.py`
2. Add corresponding test in `milestone_cvrp/tests/test_project.py`
3. Register in `run_full_project.py` for comparison

### Modify Search Configuration
Edit `config.py` to adjust:
- Number of iterations
- Heuristics per iteration
- Early pruning threshold

### Add New CVRPLib Instances
Place `.vrp` and corresponding `.sol` files in `A/A/` directory.

## Troubleshooting

### LLM API Errors
- Check `DASHSCOPE_API_KEY` environment variable is set
- Verify network connectivity to Alibaba Cloud
- Check API rate limits (the code includes 2-second delays between calls)

### Import Errors
- Ensure running from correct directory (paths are relative)
- Check that virtual environment is activated

### Test Failures
- Verify `A/A/` directory contains CVRPLib instances
- Check file permissions for output directory
