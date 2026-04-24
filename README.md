# FunSearch for CVRP

Sample-Efficient FunSearch for the Capacitated Vehicle Routing Problem — LLM-driven evolutionary search for heuristic algorithms.

## Workflow

### 1. Configure

```bash
cp config.ini.example config.ini
```

Edit `config.ini` — set `openai_model`, `openai_api_key`, `openai_base_url` under `[LLM]`. Any OpenAI-compatible API works (OpenAI, Ollama, DeepSeek, DashScope, Hunyuan, etc.).

### 2. Check

```bash
# Verify LLM connectivity and model availability
PYTHONPATH=src python -m funsearch_cvrp.utils.check_llm

# Verify dataset is present and loadable
PYTHONPATH=src python -m funsearch_cvrp.utils.check_dataset --dataset-folder data/cvrplib/A
```

### 3. Run

```bash
# Quick dry-run with mock LLM (no API calls, validates pipeline end-to-end)
python scripts/run/run_funsearch.py --iterations 20 --mock-llm

# Real run
python scripts/run/run_funsearch.py \
    --iterations 200 \
    --dataset-folder data/cvrplib/A \
    --limit-instances 10

# With custom checkpoint interval
python scripts/run/run_funsearch.py \
    --iterations 500 \
    --dataset-folder data/cvrplib/A \
    --checkpoint-every 50
```

### 4. Resume

```bash
# Continue from a checkpoint (all files stay in the same directory)
python scripts/run/run_funsearch.py \
    --iterations 500 \
    --resume outputs/latest/run_funsearch/checkpoint.pkl
```

### 5. Analyze

```bash
# One-page dashboard
python scripts/analyze/analyze.py summary outputs/latest/run_funsearch

# Score evolution trajectories (overall + per-test)
python scripts/analyze/analyze.py evolution outputs/latest/run_funsearch

# Extract and compare best programs
python scripts/analyze/analyze.py programs outputs/latest/run_funsearch

# LLM behavior analysis (response lengths, latency, import patterns)
python scripts/analyze/analyze.py llm outputs/latest/run_funsearch

# List all experiments
PYTHONPATH=src python -m funsearch_cvrp.utils.output_manager list
```

All analysis outputs are written to `<experiment>/analysis/<subcommand>/`.

## Testing

```bash
pytest tests/ -v --cov=funsearch_cvrp --cov-report=term-missing
```

## Output Layout

```text
outputs/YYYYmmdd_HHMMSS/
├── run_funsearch/
│   ├── logs/               evolution.log (+ iter_*.log archives)
│   ├── eval/               eval.jsonl — per-evaluation records
│   ├── database/           database.jsonl — best scores per iteration
│   ├── sampler/            sampler.jsonl — LLM prompts & responses
│   ├── checkpoint.pkl
│   └── best_program.py     exported on every improvement
├── analysis/               analysis outputs
│   ├── summary/            dashboard.png, summary.md
│   ├── evolution/          overall.png, test_*.png
│   ├── programs/           code/, island_scores.png
│   └── llm/                response_lengths.png, latency.png
└── meta.json
├── latest → symlink to latest run
```

JSONL and log files roll every 1000 iterations (`{prefix}_iter_XXXXXX_XXXXXX.jsonl`).

## Key Notes

- FunSearch **maximizes** score. `evaluate_cvrp()` returns `-gap` where gap = (distance − optimal) / optimal.
- `_reduce_score` uses the **last** test instance (hardest/most discriminating).
- Resume continues in-place — logs, history, and checkpoints all append to the same directory.
- The LLM response extraction pipeline handles markdown fences, leading prose, trailing explanations, import capture, and indentation normalization.
