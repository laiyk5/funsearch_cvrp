"""Plot test evaluation results from test_eval.jsonl.

Usage:
    python scripts/analyze/plot_test_eval.py [experiment_dir]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_test_eval(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def plot_test_eval(records: list[dict], output_dir: Path) -> None:
    if not records:
        print("No test eval records found.")
        return

    iterations = [r["iteration"] for r in records]
    train_best = [r["train_best"] for r in records]
    test_score = [r["test_score"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Train best vs Test score
    ax = axes[0]
    ax.plot(iterations, train_best, "o-", label="Train best", linewidth=1.5, markersize=4)
    ax.plot(iterations, test_score, "s-", label="Test score", linewidth=1.5, markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Train Best vs Test Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Generalization gap
    ax = axes[1]
    gap = [t - train_best[i] for i, t in enumerate(test_score)]
    colors = ["green" if g <= 0 else "red" for g in gap]
    ax.bar(iterations, gap, color=colors, alpha=0.7, width=max(1, min(5, len(iterations) * 0.1)))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test - Train (negative = test is better)")
    ax.set_title("Generalization Gap")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / "test_eval.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Plot 3: Per-test-instance scores (if multiple test instances)
    if records and "scores_per_test" in records[0]:
        test_keys = sorted(records[0]["scores_per_test"].keys(), key=int)
        if len(test_keys) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            for key in test_keys:
                scores = [r["scores_per_test"][key] for r in records]
                ax.plot(iterations, scores, "o-", label=f"Test instance {key}", linewidth=1, markersize=3)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Score")
            ax.set_title("Per-Test-Instance Scores")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            out_path2 = output_dir / "test_eval_per_instance.png"
            fig.savefig(out_path2, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Plot test evaluation results")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to run_funsearch directory (default: outputs/latest/run_funsearch)",
    )
    args = parser.parse_args()

    if args.experiment_dir is None:
        args.experiment_dir = Path("outputs/latest/run_funsearch")

    test_eval_path = args.experiment_dir / "test" / "test_eval.jsonl"
    if not test_eval_path.exists():
        print(f"No test_eval.jsonl found at {test_eval_path}")
        sys.exit(1)

    records = load_test_eval(test_eval_path)
    output_dir = args.experiment_dir.parent / "analysis" / "test"
    plot_test_eval(records, output_dir)


if __name__ == "__main__":
    main()
