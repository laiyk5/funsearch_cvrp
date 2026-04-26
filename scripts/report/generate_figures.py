#!/usr/bin/env python3
"""Generate publication-quality figures for the FunSearch CVRP report."""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use a clean, publication-ready style
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUTPUT_DIR = Path(__file__).parent.parent.parent / "docs" / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(filepath: Path) -> list[dict]:
    if not filepath.exists():
        return []
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_rolling_jsonl(exp_dir: Path, subdir: str, stem: str) -> list[dict]:
    d = exp_dir / "run_funsearch" / subdir
    if not d.exists():
        return []
    records = []
    for f in sorted(d.glob(f"{stem}_iter_*.jsonl")):
        records.extend(load_jsonl(f))
    active = d / f"{stem}.jsonl"
    records.extend(load_jsonl(active))
    return records


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "qwen_priority": {
        "dirs": [
            Path("outputs/20260425_062150_ecd_1"),
            Path("outputs/20260425_072309_ecd_2"),
            Path("outputs/20260425_154054_ecd_3"),
        ],
        "label": "Priority (Qwen 7B)",
        "color": "#E67E22",  # dark orange
        "linestyle": "-",
    },
    "qwen_cw": {
        "dirs": [Path("outputs/20260425_180407_cw_qwen2_5_coding_7b")],
        "label": "Savings (Qwen 7B)",
        "color": "#16A085",  # teal
        "linestyle": "-",
    },
    "ds_priority": {
        "dirs": [Path("outputs/20260426_054254_ds_flash_pri_ecd")],
        "label": "Priority (DeepSeek)",
        "color": "#8E44AD",  # purple
        "linestyle": "-",
    },
    "ds_cw": {
        "dirs": [Path("outputs/20260426_135214_ds_flash_cw")],
        "label": "Savings (DeepSeek)",
        "color": "#C0392B",  # red
        "linestyle": "-",
    },
}


def merge_resume_databases(dirs: list[Path]) -> list[dict]:
    """Merge database records from resumed runs, offsetting iterations."""
    all_records = []
    iteration_offset = 0
    for d in dirs:
        records = load_rolling_jsonl(d, "database", "database")
        if not records:
            continue
        # Offset iterations so they continue from previous run
        min_it = min(r["iteration"] for r in records)
        for r in records:
            r = dict(r)
            r["iteration"] = r["iteration"] - min_it + iteration_offset
            all_records.append(r)
        if records:
            iteration_offset = max(r["iteration"] for r in records) + 1
    return all_records


def merge_resume_evals(dirs: list[Path]) -> list[dict]:
    """Merge eval records from resumed runs."""
    all_records = []
    iteration_offset = 0
    for d in dirs:
        records = load_rolling_jsonl(d, "eval", "eval")
        if not records:
            continue
        # Try to find iteration field; fallback to index
        has_it = "iteration" in records[0]
        if has_it:
            min_it = min(r["iteration"] for r in records)
            for r in records:
                r = dict(r)
                r["iteration"] = r["iteration"] - min_it + iteration_offset
                all_records.append(r)
            iteration_offset = max(r["iteration"] for r in records) + 1
        else:
            for i, r in enumerate(records):
                r = dict(r)
                r["iteration"] = i + iteration_offset
                all_records.append(r)
            iteration_offset += len(records)
    return all_records


def get_merged_data(key: str):
    exp = EXPERIMENTS[key]
    db = merge_resume_databases(exp["dirs"])
    ev = merge_resume_evals(exp["dirs"])
    return db, ev


# ---------------------------------------------------------------------------
# Figure 1: Combined evolution trajectories
# ---------------------------------------------------------------------------
def fig_evolution_all():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for key, meta in EXPERIMENTS.items():
        db, _ = get_merged_data(key)
        if not db:
            continue
        iters = [r["iteration"] for r in db]
        scores = [r["overall_best"] for r in db]
        ax.plot(
            iters, scores,
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=1.5,
            label=meta["label"],
        )

        # Mark milestones
        ev = get_merged_data(key)[1]
        milestones = [r for r in ev if r.get("is_milestone")]
        if milestones:
            db_map = {r["iteration"]: r["overall_best"] for r in db}
            m_iters = [r["iteration"] for r in milestones]
            m_scores = [db_map.get(it, None) for it in m_iters]
            m_pairs = [(it, sc) for it, sc in zip(m_iters, m_scores) if sc is not None]
            if m_pairs:
                ax.scatter(
                    [p[0] for p in m_pairs],
                    [p[1] for p in m_pairs],
                    color=meta["color"],
                    s=15,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.3,
                )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Score (negative gap)")
    ax.set_title("Evolution Trajectories Across All Experiments")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_evolution_all.png")
    plt.close(fig)
    print("Saved fig_evolution_all.png")


# ---------------------------------------------------------------------------
# Figure 2: Per-island trajectories (Qwen priority)
# ---------------------------------------------------------------------------
def fig_per_island():
    db, _ = get_merged_data("qwen_priority")
    if not db:
        print("Skipping fig_per_island: no data")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    n_islands = len(db[0].get("best_score_per_island", []))
    iters = [r["iteration"] for r in db]

    cmap = plt.cm.get_cmap("tab10", n_islands)
    for i in range(n_islands):
        scores = [
            r["best_score_per_island"][i]
            for r in db
            if i < len(r["best_score_per_island"])
        ]
        if any(s > -1 for s in scores):
            ax.plot(
                iters[: len(scores)],
                scores,
                alpha=0.7,
                linewidth=1.0,
                color=cmap(i),
                label=f"Island {i}",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Score per Island")
    ax.set_title("Per-Island Trajectories — Priority (Qwen 7B)")
    ax.legend(loc="lower right", ncol=2, fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_per_island.png")
    plt.close(fig)
    print("Saved fig_per_island.png")


# ---------------------------------------------------------------------------
# Figure 3: Cross-dataset generalization bar chart
# ---------------------------------------------------------------------------
def fig_cross_dataset():
    # Data from benchmarks
    experiments = [
        ("Priority\n(Qwen)", {"A": 11.65, "X": 15.87, "XL": None}),
        ("Savings\n(Qwen)", {"A": 1.68, "X": 19.84, "XL": None}),
        ("Priority\n(DeepSeek)", {"A": 29.79, "X": 17.81, "XL": None}),
        ("Savings\n(DeepSeek)", {"A": None, "X": 22.09, "XL": None}),
    ]

    # Baselines
    baselines = {
        "A": {"NN": 40.58, "CW": 5.06, "CW+2opt": 4.78},
        "X": {"NN": 25.47, "CW": 6.01},
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))

    for idx, (dataset, ax) in enumerate(zip(["A", "X"], axes)):
        labels = []
        values = []
        colors = []

        # Add baselines
        if dataset in baselines:
            for name, val in baselines[dataset].items():
                labels.append(name)
                values.append(val)
                colors.append("#95A5A6")  # gray

        # Add evolved
        exp_colors = ["#E67E22", "#16A085", "#8E44AD", "#C0392B"]
        for i, (name, gaps) in enumerate(experiments):
            val = gaps.get(dataset)
            if val is not None:
                labels.append(name.replace("\n", " "))
                values.append(val)
                colors.append(exp_colors[i])

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor="white", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center",
                ha="left",
                fontsize=7,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Gap above optimal (%)")
        ax.set_title(f"CVRPLib {dataset}-set" if dataset != "A" else "CVRPLib A-set (train)")
        ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xlim(0, max(values) * 1.25)
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_cross_dataset.png")
    plt.close(fig)
    print("Saved fig_cross_dataset.png")


# ---------------------------------------------------------------------------
# Figure 4: Train/Test gap evolution (DeepSeek priority)
# ---------------------------------------------------------------------------
def fig_train_test_gap():
    db, _ = get_merged_data("ds_priority")
    test_records = []
    for d in EXPERIMENTS["ds_priority"]["dirs"]:
        test_records.extend(load_rolling_jsonl(d, "test", "test_eval"))

    if not db or not test_records:
        print("Skipping fig_train_test_gap: no data")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    db_map = {r["iteration"]: r["overall_best"] for r in db}
    test_iters = [r["iteration"] for r in test_records]
    train_gaps = [-db_map.get(it, np.nan) * 100 for it in test_iters]
    test_gaps = [-r.get("test_score", np.nan) * 100 for r in test_records]

    ax.plot(test_iters, train_gaps, "o-", color="#8E44AD", markersize=2,
            linewidth=1.2, label="Train (penalized)")
    ax.plot(test_iters, test_gaps, "s-", color="#E74C3C", markersize=2,
            linewidth=1.2, label="Test")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gap above optimal (%)")
    ax.set_title("Train vs. Test Gap — DeepSeek Priority")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_train_test_gap.png")
    plt.close(fig)
    print("Saved fig_train_test_gap.png")


# ---------------------------------------------------------------------------
# Figure 5: Validation gap & penalty (DeepSeek CW)
# ---------------------------------------------------------------------------
def fig_val_penalty():
    val_records = []
    for d in EXPERIMENTS["ds_cw"]["dirs"]:
        val_records.extend(load_rolling_jsonl(d, "val", "val_eval"))

    if not val_records:
        print("Skipping fig_val_penalty: no data")
        return

    fig, ax1 = plt.subplots(figsize=(5.5, 3.0))

    iters = [r["iteration"] for r in val_records]
    train_clean = [-r.get("train_clean", np.nan) * 100 for r in val_records]
    val_scores = [-r.get("val_score", np.nan) * 100 for r in val_records]
    gaps = [r.get("gap", np.nan) * 100 for r in val_records]

    color_train = "#8E44AD"
    color_val = "#E74C3C"
    color_gap = "#2C3E50"

    ax1.plot(iters, train_clean, "o-", color=color_train, markersize=2,
             linewidth=1.2, label="Train (clean)")
    ax1.plot(iters, val_scores, "s-", color=color_val, markersize=2,
             linewidth=1.2, label="Validation")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Gap (%)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_ylim(0, max(max(train_clean), max(val_scores)) * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(iters, gaps, "^-", color=color_gap, markersize=2,
             linewidth=1.0, alpha=0.7, label="Generalization gap")
    ax2.set_ylabel("Gen. Gap (pp)", color=color_gap)
    ax2.tick_params(axis="y", labelcolor=color_gap)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               framealpha=0.9, fontsize=7)

    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_val_penalty.png")
    plt.close(fig)
    print("Saved fig_val_penalty.png")


# ---------------------------------------------------------------------------
# Figure 6: Milestone timing distribution
# ---------------------------------------------------------------------------
def fig_milestones():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    all_milestones = {}
    for key, meta in EXPERIMENTS.items():
        _, ev = get_merged_data(key)
        milestones = [r["iteration"] for r in ev if r.get("is_milestone")]
        if milestones:
            all_milestones[meta["label"]] = milestones

    if not all_milestones:
        print("Skipping fig_milestones: no data")
        return

    colors = [meta["color"] for meta in EXPERIMENTS.values()]
    for i, (label, m_iters) in enumerate(all_milestones.items()):
        ax.hist(
            m_iters,
            bins=20,
            alpha=0.6,
            color=colors[i],
            label=f"{label} (n={len(m_iters)})",
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Milestones")
    ax.set_title("Milestone Timing Distribution")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_milestones.png")
    plt.close(fig)
    print("Saved fig_milestones.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating report figures...")
    fig_evolution_all()
    fig_per_island()
    fig_cross_dataset()
    fig_train_test_gap()
    fig_val_penalty()
    fig_milestones()
    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
