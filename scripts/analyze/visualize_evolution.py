"""Visualize FunSearch evolution: overall best score and per-test scores over time.

Usage:
    python scripts/analyze/visualize_evolution.py outputs/20260425_040429/run_funsearch
    python scripts/analyze/visualize_evolution.py outputs/20260425_040429/run_funsearch -o custom.png

Output files are written to <parent_of_experiment>/visualize_evolution/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_overall_best(history: list[dict], save_path: Path) -> None:
    iterations = [r["iteration"] for r in history]
    overall_best = [r["overall_best"] for r in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iterations, overall_best, "b-", linewidth=1.2, alpha=0.7)
    ax.scatter(iterations, overall_best, s=12, c="b", zorder=5)

    best_so_far = -float("inf")
    for i, score in zip(iterations, overall_best):
        if score is not None and score > best_so_far:
            best_so_far = score
            ax.annotate(
                f"{score:.3f}",
                (i, score),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=7,
                ha="center",
                color="darkblue",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title("Best Overall Score over Iterations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overall  -> {save_path.name}")


def plot_per_test_scores(eval_history: list[dict], out_dir: Path) -> None:
    accepted = [r for r in eval_history if r.get("accepted") and r.get("iteration") is not None]
    if not accepted:
        print("  (no accepted evaluations — skipping per-test plots)")
        return

    all_keys: set[str] = set()
    for r in accepted:
        all_keys.update(r.get("scores_per_test", {}).keys())
    test_keys = sorted(all_keys, key=lambda k: int(k))

    accepted.sort(key=lambda r: r["iteration"])

    # Build best-so-far per test at every unique iteration
    current_best: dict[str, float] = {k: -float("inf") for k in test_keys}
    # Snapshots: list of (iteration, {test_key: best_so_far})
    snapshots: list[tuple[int, dict[str, float]]] = []

    for r in accepted:
        it = r["iteration"]
        # Record snapshot at each new iteration
        if not snapshots or it != snapshots[-1][0]:
            snapshots.append((it, dict(current_best)))
        for k in test_keys:
            score = r.get("scores_per_test", {}).get(k)
            if score is not None and score > current_best[k]:
                current_best[k] = score
    if snapshots:
        snapshots.append((snapshots[-1][0] + 1, dict(current_best)))

    iters = [s[0] for s in snapshots]

    for test_key in test_keys:
        scores = [s[1][test_key] for s in snapshots]
        scores = [s if s != -float("inf") else None for s in scores]

        fig, ax = plt.subplots(figsize=(10, 4))
        valid_x, valid_y = [], []
        for x, y in zip(iters, scores):
            if y is not None:
                valid_x.append(x)
                valid_y.append(y)
        if valid_x:
            ax.step(valid_x, valid_y, where="post", color="steelblue",
                    linewidth=1.2, alpha=0.85)
            ax.scatter(valid_x, valid_y, s=10, c="steelblue", zorder=5)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title(f"Best Score on Test {test_key} over Iterations")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = f"evolution_test_{test_key}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  per-test -> {len(test_keys)} files (evolution_test_*.png)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FunSearch evolution")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to a run_funsearch output directory",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Save figure to file instead of showing (e.g. evolution.png)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    history_path = output_dir / "database" / "database.jsonl"
    eval_history_path = output_dir / "eval" / "eval.jsonl"

    if not history_path.exists():
        print(f"Error: database.jsonl not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    history = load_jsonl(history_path)
    eval_history = load_jsonl(eval_history_path) if eval_history_path.exists() else []

    # Output goes to: <parent_of_experiment>/visualize_evolution/
    run_dir = output_dir.parent  # e.g. outputs/20260425_040429
    out_dir = run_dir / "visualize_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overall_best(history, out_dir / "evolution.png")

    if eval_history:
        plot_per_test_scores(eval_history, out_dir)
    else:
        print("  (eval_history.jsonl not available — skip per-test plots)")

    print(f"Done -> {out_dir}/")


if __name__ == "__main__":
    main()
