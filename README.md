# FunSearch for CVRP

Sample-Efficient FunSearch implementation for solving the Capacitated Vehicle Routing Problem (CVRP) using LLM-generated heuristics.

## Project Structure

```
.
├── pyproject.toml            # Project config & dependencies
├── src/                      # Source code
│   ├── cvrp/                 # Core CVRP algorithms
│   │   ├── core.py           # Data model, distance utilities, baselines
│   │   ├── baselines.py      # Clarke-Wright Savings, 2-opt improvement
│   │   ├── io.py             # CVRPLib .vrp file loader
│   │   └── search.py         # Sample-efficient search algorithm
│   ├── llm/                  # LLM integration
│   │   ├── interface.py      # LLM interface for heuristic generation
│   │   └── equivalence.py    # Functional equivalence detection
│   └── utils/                # Utilities
│       ├── config.py         # Configuration (API keys, parameters)
│       └── models.py         # LLM client definitions
├── scripts/                  # Runnable scripts
│   ├── run_milestone.py      # Milestone experiment
│   ├── run_full_project.py   # Full benchmark
│   ├── generate_heuristics.py# Iterative LLM-based generation
│   ├── test_simple.py        # Quick test (1 iter, 5 heuristics)
│   ├── test_llm.py           # Test LLM connection
│   └── test_with_a_data.py   # Test with CVRPLib A dataset
├── tests/                    # Unit tests
│   └── test_project.py       # Core pipeline tests
├── reports/                  # Report generators
│   ├── generate_report.py
│   ├── generate_report_detailed.py
│   ├── generate_final_report.py
│   ├── export_report_docx.py
│   └── export_report_detailed.py
├── data/                     # CVRPLib test instances
│   └── A/                    # A-instances (.vrp and .sol files)
├── docs/                     # Documentation
│   ├── README.md             # Original README
│   ├── project_description.md
│   ├── CS5491 AI Project (2).docx
│   ├── code.docx
│   ├── proposal/             # Proposal documents
│   │   ├── CVRP_FunSearch_Proposal.docx
│   │   └── CVRP_FunSearch_Proposal_ICLR24.tex
│   └── report/               # LaTeX conference paper files
│       ├── iclr2024_conference.tex
│       ├── iclr2024_conference.pdf
│       └── ...
├── outputs/                  # Generated results (JSON, logs)
└── README.md                 # This file
```

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Install Dependencies

```bash
# Clone/navigate to project
cd AI-Project

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or activate the environment
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

### Configuration

#### Option 1: Environment Variables

```bash
# Set API key (required for LLM features)
export DASHSCOPE_API_KEY="your-api-key-here"
# or: export OPENAI_API_KEY="your-api-key"
```

#### Option 2: Configuration File (Recommended)

```bash
# Copy the example configuration file
cp .env.example .env

# Edit .env with your actual values
nano .env  # or use your preferred editor
```

#### Available Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (for GPT models) |
| `DASHSCOPE_API_KEY` | - | Alibaba DashScope key (for Qwen models) |
| `OPENAI_MODEL` | `gpt-4` | LLM model to use |
| `LLM_TEMPERATURE` | `0.7` | Generation temperature |
| `LLM_MAX_TOKENS` | `2000` | Max tokens per request |
| `N_ITERATIONS` | `10` | FunSearch iterations |
| `MIN_HEURISTICS_PER_ITER` | `50` | Min heuristics per iteration |
| `MAX_HEURISTICS_PER_ITER` | `100` | Max heuristics per iteration |
| `DATASET_SEED` | `2026` | Random seed for reproducibility |
| `EARLY_PRUNING_THRESHOLD` | `1.5` | Early pruning threshold |

**API Selection**: The code automatically selects the appropriate API endpoint based on the model name:
- **OpenAI API**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.
- **Alibaba DashScope API**: `qwen-max`, `qwen-turbo`, `deepseek-v3`, etc.

Both services use the same `openai` Python package.

## Quick Start

### Run Milestone Experiment
```bash
python scripts/run_milestone.py
python reports/generate_report.py
```

### Run Full Project
```bash
# With synthetic data
python scripts/run_full_project.py --dataset synthetic

# With CVRPLib data
python scripts/run_full_project.py --dataset cvrplib --cvrplib-dir "data/A" --limit-instances 10
```

### Run Tests
```bash
# Unit tests
python -m unittest discover -s tests -p "test_*.py"

# Quick test
python scripts/test_simple.py

# Test LLM connection
python scripts/test_llm.py

# Test with A dataset
python scripts/test_with_a_data.py
```

### Run Iterative Heuristic Generation
```bash
python scripts/generate_heuristics.py
```

## Key Components

| Component | Description |
|-----------|-------------|
| `src/cvrp/core.py` | CVRP data model, distance utilities, baseline heuristics |
| `src/cvrp/search.py` | Sample-efficient search with early pruning |
| `src/llm/interface.py` | LLM integration for generating heuristics |
| `src/llm/equivalence.py` | Functional equivalence detection |
| `src/cvrp/baselines.py` | Clarke-Wright Savings, 2-opt improvement |

## Configuration

Edit `src/utils/config.py` to adjust:
- LLM model and API settings
- Search parameters (iterations, population size)
- Early pruning thresholds

## Output Files

Results are saved to `outputs/`:
- `milestone_results.json` / `search_history.json`
- `full_project_results.json` / `full_search_history.json`
- `iterative_search_results.json`

Reports are generated in the root directory:
- `milestone_report.md`
- `final_project_report.md`

## Team Members

- QIN YUANCHENG (59061061)
- QIN ZIHENG (59866035)
- LAI YIKAI (59563061)
