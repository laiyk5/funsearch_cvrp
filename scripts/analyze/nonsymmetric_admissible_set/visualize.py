"""Visualize FunSearch progress for the nonsymmetric admissible set problem.

Reads history.jsonl and funsearch_results.json from a run directory and produces
plots showing how the search progressed over iterations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_history(history_path: Path) -> list[dict]:
    """Load iteration history from JSONL file."""
    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_results(results_path: Path) -> dict:
    """Load final FunSearch results from JSON file."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_overall_best(records: list[dict], output_path: Path | None = None) -> None:
    """Plot overall best score vs iteration."""
    iterations = [r["iteration"] for r in records]
    best_scores = [r["overall_best"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, best_scores, linewidth=2, color="steelblue")
    ax.fill_between(iterations, best_scores, alpha=0.2, color="steelblue")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Overall Best Score", fontsize=12)
    ax.set_title("FunSearch: Best Score Over Iterations", fontsize=14)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_island_scores(records: list[dict], num_islands: int, output_path: Path | None = None) -> None:
    """Plot best score per island over iterations."""
    iterations = [r["iteration"] for r in records]
    island_scores = [
        [r["best_score_per_island"][i] for r in records]
        for i in range(num_islands)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_islands))

    for i, scores in enumerate(island_scores):
        ax.plot(iterations, scores, label=f"Island {i}", color=colors[i], alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Score", fontsize=12)
    ax.set_title("FunSearch: Best Score per Island Over Iterations", fontsize=14)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_score_distribution(records: list[dict], num_islands: int, output_path: Path | None = None) -> None:
    """Plot final score distribution across islands as a bar chart."""
    if not records:
        return

    final = records[-1]
    scores = [
        final["best_score_per_island"][i] if final["best_score_per_island"][i] is not None else 0
        for i in range(num_islands)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(num_islands), scores, color=plt.cm.viridis(np.linspace(0, 1, num_islands)))
    ax.set_xlabel("Island", fontsize=12)
    ax.set_ylabel("Best Score", fontsize=12)
    ax.set_title("FunSearch: Final Best Score per Island", fontsize=14)
    ax.set_xticks(range(num_islands))
    ax.set_xticklabels([f"{i}" for i in range(num_islands)])
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_improvement_rate(records: list[dict], output_path: Path | None = None) -> None:
    """Plot how often the overall best score improved."""
    improvements = []
    prev_best = None
    for r in records:
        best = r["overall_best"]
        if best is not None and (prev_best is None or best > prev_best):
            improvements.append(1)
            prev_best = best
        else:
            improvements.append(0)

    # Cumulative improvements
    cum_improvements = np.cumsum(improvements)
    iterations = [r["iteration"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(iterations, improvements, color="coral", alpha=0.7, width=0.8)
    ax_twin = ax.twinx()
    ax_twin.plot(iterations, cum_improvements, color="darkgreen", linewidth=2, marker="o", markersize=3)
    ax_twin.set_ylabel("Cumulative Improvements", color="darkgreen", fontsize=12)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Improvement (1 = yes)", fontsize=12)
    ax.set_title("FunSearch: Score Improvements Over Iterations", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize FunSearch progress for nonsymmetric admissible set"
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to a FunSearch run directory (containing history.jsonl and funsearch_results.json). "
             "Defaults to outputs/latest/",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots (default: same as run-dir)",
    )

    args = parser.parse_args()

    # Determine run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path("outputs/latest").resolve()
        if not run_dir.exists():
            # Try to follow symlink
            latest_link = Path("outputs/latest")
            if latest_link.is_symlink():
                run_dir = latest_link.resolve()

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    history_path = run_dir / "history.jsonl"
    results_path = run_dir / "funsearch_results.json"

    if not history_path.exists():
        print(f"History file not found: {history_path}")
        print("Make sure the FunSearch run was executed with the latest run_funsearch.py")
        sys.exit(1)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    # Load data
    records = load_history(history_path)
    results = load_results(results_path)

    print(f"Loaded {len(records)} history records from {history_path}")
    print(f"Overall best score: {results.get('overall_best', 'N/A')}")
    print(f"Config: {results.get('config', {})}")

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    num_islands = results.get("config", {}).get("num_islands", 10)

    # Generate plots
    plot_overall_best(records, output_path=output_dir / "best_score_over_time.png")
    plot_island_scores(records, num_islands, output_path=output_dir / "island_scores_over_time.png")
    plot_score_distribution(records, num_islands, output_path=output_dir / "final_island_distribution.png")
    plot_improvement_rate(records, output_path=output_dir / "improvement_rate.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
