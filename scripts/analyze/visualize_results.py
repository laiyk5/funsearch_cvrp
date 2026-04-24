#!/usr/bin/env python3
"""
Visualize FunSearch results: island comparison and per-test score profiles.

Usage:
    python scripts/analyze/visualize_results.py
    python scripts/analyze/visualize_results.py outputs/latest/run_funsearch
    python scripts/analyze/visualize_results.py outputs/latest/run_funsearch -o custom.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_island_comparison(data: dict, save_path: Path) -> None:
    best = data.get("best_programs", [])
    if not best:
        print("  (no best_programs — skipping)")
        return

    islands = [p["island_id"] for p in best]
    scores = [p["best_score"] for p in best]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(islands, scores, color="steelblue", alpha=0.85)
    ax.axhline(y=data.get("overall_best", 0), color="red", linestyle="--",
               linewidth=1, label=f'overall best = {data["overall_best"]:.4f}')
    ax.set_xlabel("Island")
    ax.set_ylabel("Score")
    ax.set_title("Best Score per Island")
    ax.set_xticks(islands)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  island comparison -> {save_path.name}")


def plot_test_profiles(data: dict, save_path: Path) -> None:
    best = data.get("best_programs", [])
    if not best:
        return

    # Collect per-test scores from all islands
    all_keys: set[str] = set()
    for p in best:
        all_keys.update(p.get("scores_per_test", {}).keys())
    if not all_keys:
        return
    test_keys = sorted(all_keys, key=lambda k: int(k))
    n_tests = len(test_keys)

    # Build a matrix: rows = islands, cols = tests
    matrix = np.full((len(best), n_tests), np.nan)
    for i, p in enumerate(best):
        for j, k in enumerate(test_keys):
            matrix[i, j] = p.get("scores_per_test", {}).get(k, np.nan)

    fig, ax = plt.subplots(figsize=(max(10, n_tests * 0.3), 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(n_tests))
    ax.set_xticklabels(test_keys, rotation=90, fontsize=6)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels([f"island {p['island_id']}" for p in best], fontsize=7)
    ax.set_xlabel("Test instance")
    ax.set_ylabel("Island")
    ax.set_title("Per-Test Scores Heatmap (best per island)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  test profiles    -> {save_path.name}")


def plot_duration_info(data: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    config = data.get("config", {})
    lines = [
        f"Duration: {data.get('duration_seconds', '?')}s",
        f"LLM calls: {data.get('llm_calls', '?')}",
        f"Model: {config.get('llm_model', '?')}",
        f"Temperature: {config.get('llm_temperature', '?')}",
        f"Islands: {config.get('num_islands', '?')}",
        f"Samples/prompt: {config.get('samples_per_prompt', '?')}",
        f"Overall best: {data.get('overall_best', '?'):.4f}" if data.get('overall_best') is not None else "Overall best: N/A",
    ]
    ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=11, family="monospace")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary          -> {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize FunSearch final results")
    parser.add_argument("output_dir", type=str, nargs="?", default=None,
                        help="Path to a run_funsearch output directory (default: outputs/latest/run_funsearch)")
    parser.add_argument("-o", "--output", default=None,
                        help="Save a specific file (otherwise all in {experiment}/visualize_results/)")
    args = parser.parse_args()

    # Resolve experiment directory
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        latest = Path("outputs/latest")
        if not latest.exists():
            print("Error: no outputs/latest found. Provide an explicit path.", file=sys.stderr)
            sys.exit(1)
        exp_dir = latest.resolve() / "run_funsearch"

    results_file = exp_dir / "funsearch_results.json"
    if not results_file.exists():
        print(f"Error: {results_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {results_file}")
    data = load_results(results_file)

    # Output: sibling to experiment dir
    run_dir = exp_dir.parent
    out_dir = run_dir / "visualize_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        plot_island_comparison(data, Path(args.output))
    else:
        plot_island_comparison(data, out_dir / "island_comparison.png")
        plot_test_profiles(data, out_dir / "test_profiles.png")
        plot_duration_info(data, out_dir / "summary.png")

    print(f"Done -> {out_dir}/")


if __name__ == "__main__":
    main()
