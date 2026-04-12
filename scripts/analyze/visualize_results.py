#!/usr/bin/env python3
"""
Visualize CVRP FunSearch results.

Usage:
    python scripts/analyze/visualize_results.py
    python scripts/analyze/visualize_results.py --commit 7d9a1d6 --timestamp 20250412_153033
    python scripts/analyze/visualize_results.py --output-dir my_charts/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> list[dict]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def plot_score_trends(results: list[dict], output_path: Path | None = None):
    """Plot score trends across iterations for all scales."""
    iterations = [r["iteration"] for r in results]
    small_scores = [r["small_scale"]["score"] for r in results]
    medium_scores = [r["medium_scale"]["score"] for r in results]
    large_scores = [r["large_scale"]["score"] for r in results]
    stability_scores = [r["stability"]["avg_score"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iterations, small_scores, marker="o", label="Small Scale", linewidth=2)
    ax.plot(iterations, medium_scores, marker="s", label="Medium Scale", linewidth=2)
    ax.plot(iterations, large_scores, marker="^", label="Large Scale", linewidth=2)
    ax.plot(iterations, stability_scores, marker="d", label="Stability (Avg)", linewidth=2, linestyle="--")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Score (lower is better)", fontsize=12)
    ax.set_title("Score Trends Across Iterations", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Mark best iteration
    best_idx = np.argmin(stability_scores)
    best_iter = iterations[best_idx]
    best_score = stability_scores[best_idx]
    ax.axvline(x=best_iter, color="red", linestyle=":", alpha=0.7, label=f"Best: Iter {best_iter}")
    ax.scatter([best_iter], [best_score], color="red", s=100, zorder=5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_gap_analysis(results: list[dict], output_path: Path | None = None):
    """Plot optimality gap trends."""
    iterations = [r["iteration"] for r in results]
    small_gaps = [r["small_scale"]["gap"] for r in results]
    medium_gaps = [r["medium_scale"]["gap"] for r in results]
    large_gaps = [r["large_scale"]["gap"] for r in results]
    avg_gaps = [r["stability"]["avg_gap"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iterations, small_gaps, marker="o", label="Small Scale", linewidth=2)
    ax.plot(iterations, medium_gaps, marker="s", label="Medium Scale", linewidth=2)
    ax.plot(iterations, large_gaps, marker="^", label="Large Scale", linewidth=2)
    ax.plot(iterations, avg_gaps, marker="d", label="Average Gap", linewidth=2, linestyle="--")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title("Optimality Gap Trends", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_distance_vs_routes(results: list[dict], output_path: Path | None = None):
    """Scatter plot of distance vs number of routes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scales = ["small_scale", "medium_scale", "large_scale"]
    titles = ["Small Scale", "Medium Scale", "Large Scale"]

    for ax, scale, title in zip(axes, scales, titles, strict=False):
        distances = [r[scale]["avg_distance"] for r in results]
        routes = [r[scale]["avg_num_routes"] for r in results]
        iterations = [r["iteration"] for r in results]

        scatter = ax.scatter(distances, routes, c=iterations, cmap="viridis", s=100, alpha=0.7)
        ax.set_xlabel("Avg Distance", fontsize=11)
        ax.set_ylabel("Avg # Routes", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Iteration", fontsize=10)

    plt.suptitle("Distance vs Routes Trade-off", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_score_distribution(results: list[dict], output_path: Path | None = None):
    """Box plot of score distribution across scales."""
    small_scores = [r["small_scale"]["score"] for r in results]
    medium_scores = [r["medium_scale"]["score"] for r in results]
    large_scores = [r["large_scale"]["score"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        [small_scores, medium_scores, large_scores],
        tick_labels=["Small Scale", "Medium Scale", "Large Scale"],
        patch_artist=True,
    )

    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Score Distribution Across Scales", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def create_dashboard(results: list[dict], output_path: Path | None = None):
    """Create a comprehensive dashboard with all plots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    iterations = [r["iteration"] for r in results]

    # 1. Score trends (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(iterations, [r["small_scale"]["score"] for r in results], marker="o", label="Small", linewidth=2)
    ax1.plot(iterations, [r["medium_scale"]["score"] for r in results], marker="s", label="Medium", linewidth=2)
    ax1.plot(iterations, [r["large_scale"]["score"] for r in results], marker="^", label="Large", linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Score")
    ax1.set_title("Score Trends", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gap trends (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(iterations, [r["small_scale"]["gap"] for r in results], marker="o", label="Small", linewidth=2)
    ax2.plot(iterations, [r["medium_scale"]["gap"] for r in results], marker="s", label="Medium", linewidth=2)
    ax2.plot(iterations, [r["large_scale"]["gap"] for r in results], marker="^", label="Large", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Gap (%)")
    ax2.set_title("Optimality Gap", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Score distribution (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    scores_data = [
        [r["small_scale"]["score"] for r in results],
        [r["medium_scale"]["score"] for r in results],
        [r["large_scale"]["score"] for r in results],
    ]
    bp = ax3.boxplot(scores_data, tick_labels=["Small", "Medium", "Large"], patch_artist=True)
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel("Score")
    ax3.set_title("Score Distribution", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Distance vs Routes (bottom, spans 2 columns)
    ax4 = fig.add_subplot(gs[2, :])
    x = np.arange(len(results))
    width = 0.35
    dists = [r["stability"]["avg_score"] for r in results]
    bars = ax4.bar(x, dists, width, label="Stability Score", color="#9b59b6", alpha=0.7)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Stability Score")
    ax4.set_title("Stability Score by Iteration", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(i) for i in iterations])
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.suptitle("CVRP FunSearch Results Dashboard", fontsize=16, fontweight="bold", y=0.995)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize CVRP FunSearch results")
    parser.add_argument("--commit", default=None, help="Git commit hash (default: latest)")
    parser.add_argument("--timestamp", default=None, help="Timestamp folder (default: latest)")
    parser.add_argument("--output-dir", default=None, help="Custom directory to save charts (default: outputs/{commit}/{timestamp}/charts/)")
    parser.add_argument("--dashboard-only", action="store_true", help="Only generate dashboard")
    parser.add_argument("--show", action="store_true", help="Show plots instead of saving")
    args = parser.parse_args()

    # Determine results path
    base_dir = Path("outputs")

    if args.commit:
        commit_dir = base_dir / args.commit
    else:
        latest_link = base_dir / "latest"
        if latest_link.exists():
            commit_dir = latest_link.resolve().parent
        else:
            print("Error: No results found. Run an experiment first.")
            sys.exit(1)

    if args.timestamp:
        results_dir = commit_dir / args.timestamp
    else:
        # Use latest timestamp
        timestamps = sorted([d for d in commit_dir.iterdir() if d.is_dir()])
        if not timestamps:
            print(f"Error: No timestamp folders in {commit_dir}")
            sys.exit(1)
        results_dir = timestamps[-1]

    results_file = results_dir / "iterative_search_results.json"
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    print(f"Loading results from: {results_file}")
    results = load_results(results_file)
    print(f"Loaded {len(results)} iterations")

    if args.show:
        output_dir = None
    else:
        # Use custom output dir if specified, otherwise use outputs/{commit}/{timestamp}/charts/
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = results_dir / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving charts to: {output_dir}")

    # Generate charts
    if args.dashboard_only:
        create_dashboard(results, output_dir / "dashboard.png" if output_dir else None)
    else:
        create_dashboard(results, output_dir / "dashboard.png" if output_dir else None)
        plot_score_trends(results, output_dir / "score_trends.png" if output_dir else None)
        plot_gap_analysis(results, output_dir / "gap_analysis.png" if output_dir else None)
        plot_distance_vs_routes(results, output_dir / "distance_vs_routes.png" if output_dir else None)
        plot_score_distribution(results, output_dir / "score_distribution.png" if output_dir else None)

    if output_dir:
        print(f"\nAll charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
