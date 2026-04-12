from __future__ import annotations

import json
from pathlib import Path


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    out_dir = Path("outputs")
    results_path = out_dir / "full_project_results.json"
    history_path = out_dir / "full_search_history.json"

    if not results_path.exists() or not history_path.exists():
        raise FileNotFoundError("Run run_full_project.py first to generate full outputs")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))

    ranking = results["ranking"]
    best = ranking[0]

    pruned_count = sum(1 for h in history if h.get("status") == "early_pruned")
    full_eval_count = sum(1 for h in history if h.get("status") == "full_eval")
    total = pruned_count + full_eval_count
    pruned_pct = 100.0 * pruned_count / total if total > 0 else 0.0

    table_lines = [
        "| Rank | Method | Avg Distance | Avg #Routes | Score |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in ranking:
        table_lines.append(
            f"| {row['rank']} | {row['name']} | {_fmt(row['avg_distance'])} | {_fmt(row['avg_num_routes'])} | {_fmt(row['score'])} |"
        )

    report = f"""# Final Project Report: Sample-Efficient FunSearch for CVRP

## Team
- QIN YUANCHENG (59061061)
- QIN ZIHENG (59866035)
- LAI YIKAI (59563061)

## Project Scope
This full project extends the milestone prototype into a complete benchmark pipeline with:
- Multiple deterministic baselines
- Local-search enhancement (2-opt)
- Sample-efficient FunSearch search
- Unified experiment runner and final ranking output
- Reproducible JSON outputs and this auto-generated report

## Dataset
- Dataset type: {results['dataset_type']}
- Instance count: {results['instance_count']}
- Instances: {', '.join(results['dataset_names'])}

## Method Ranking
{chr(10).join(table_lines)}

## Best Method
- Name: {best['name']}
- Avg distance: {_fmt(best['avg_distance'])}
- Avg routes: {_fmt(best['avg_num_routes'])}
- Score: {_fmt(best['score'])}

## FunSearch Details
- Best weights: {results['funsearch']['best_weights']}
- Best search score: {results['funsearch']['best_score_from_search']}
- Unique candidates: {results['funsearch']['unique_candidates']}
- Full evaluations: {full_eval_count}
- Early-pruned: {pruned_count} ({pruned_pct:.1f}%)

## Deliverables Produced
- outputs/full_project_results.json
- outputs/full_search_history.json
- final_project_report.md

## Reproducibility
Run the full pipeline:

```powershell
python run_full_project.py --dataset synthetic
python generate_final_report.py
```

All synthetic experiments are deterministic with the configured random seed.
"""

    Path("final_project_report.md").write_text(report, encoding="utf-8")
    print("Saved final_project_report.md")


if __name__ == "__main__":
    main()
