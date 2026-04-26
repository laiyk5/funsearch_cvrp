#!/usr/bin/env python3
"""FunSearch analysis tool — single entry point for all research analyses.

Usage:
    python scripts/analyze/analyze.py list                     # list experiments
    python scripts/analyze/analyze.py evolution [dir]          # score trajectories
    python scripts/analyze/analyze.py programs [dir]           # extract & compare programs
    python scripts/analyze/analyze.py llm [dir]                # LLM behavior analysis
    python scripts/analyze/analyze.py summary [dir]            # one-page dashboard
    python scripts/analyze/analyze.py validation [dir]         # train/val/test plots

All outputs are written to ``<experiment>/analysis/<subcommand>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# -- use non-interactive backend when saving to file -------------------------
matplotlib.use("Agg")

# -- project root ------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analyze.lib.data import (
    resolve_experiment,
    resolve_history,
    load_final_results,
    load_database_log,
    load_eval_log,
    load_sampler_log,
    load_meta,
    load_val_log,
    load_val_per_island_log,
    load_test_log,
)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _out_dir(exp_dir: Path, subcommand: str) -> Path:
    """Return ``<exp_dir>/analysis/<subcommand>/``, creating if needed."""
    d = exp_dir.parent / "analysis" / subcommand
    d.mkdir(parents=True, exist_ok=True)
    return d




# ===========================================================================
# Subcommand: islands
# ===========================================================================


def cmd_islands(args: argparse.Namespace) -> None:
    exp_dir = resolve_experiment(args.experiment)
    out_dir = _out_dir(exp_dir, "islands")
    print(f"Experiment: {exp_dir}")
    if args.full_history:
        hist = resolve_history(exp_dir)
        print(f"  Full history: {len(hist)} run(s)")

    db = load_database_log(exp_dir, full_history=args.full_history)
    if not db:
        print("  No database log found.")
        return

    its = [r["iteration"] for r in db]
    n_islands = len(db[0].get("best_score_per_island", []))
    if n_islands == 0:
        print("  No per-island data.")
        return

    # --- Overlay plot: all islands on one axis ---
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10" if n_islands <= 10 else "tab20")
    for island_id in range(n_islands):
        scores = [r["best_score_per_island"][island_id] for r in db]
        ax.plot(its, scores, linewidth=1.0, alpha=0.7,
                color=cmap(island_id % cmap.N), label=f"I{island_id}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Score")
    ax.set_title("Per-Island Best Score Evolution")
    ax.legend(loc="upper left", fontsize=7, ncol=min(n_islands, 5))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fpath = out_dir / "islands_overlay.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overlay  -> {fpath.name}")

    # --- Grid plot: one subplot per island ---
    cols = min(5, n_islands)
    rows = (n_islands + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5),
                             sharex=True, sharey=True)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = [axes] if rows == 1 else [[a] for a in axes]
    else:
        axes = axes.tolist()

    overall_best = [r["overall_best"] for r in db]
    for island_id in range(n_islands):
        r = island_id // cols
        c = island_id % cols
        ax = axes[r][c]
        scores = [r["best_score_per_island"][island_id] for r in db]
        ax.plot(its, scores, linewidth=1.0, color="steelblue")
        ax.plot(its, overall_best, linewidth=0.8, color="red",
                linestyle="--", alpha=0.5, label="overall best")
        ax.set_title(f"Island {island_id}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(n_islands, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.suptitle("Best Score per Island", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fpath = out_dir / "islands_grid.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  grid     -> {fpath.name}")

    print(f"Done -> {out_dir}/")


# ===========================================================================
# Subcommand: evolution
# ===========================================================================


def _plot_overall(records: list[dict], save_path: Path) -> Path:
    its = [r["iteration"] for r in records]
    best = [r["overall_best"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(its, best, "b-", linewidth=1.2, alpha=0.7)
    ax.scatter(its, best, s=12, c="b", zorder=5)

    best_sofar = -float("inf")
    for i, s in zip(its, best):
        if s is not None and s > best_sofar:
            best_sofar = s
            ax.annotate(f"{s:.3f}", (i, s), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha="center", color="darkblue")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title("Best Overall Score over Iterations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_per_test(eval_records: list[dict], out_dir: Path) -> list[Path]:
    accepted = [r for r in eval_records if r.get("accepted") and r.get("iteration") is not None]
    if not accepted:
        print("  (no accepted evaluations)")
        return []

    all_keys: set[str] = set()
    for r in accepted:
        all_keys.update(r.get("scores_per_test", {}).keys())
    test_keys = sorted(all_keys, key=lambda k: int(k))
    accepted.sort(key=lambda r: r["iteration"])

    current_best = {k: -float("inf") for k in test_keys}
    snapshots: list[tuple[int, dict[str, float]]] = []
    for r in accepted:
        it = r["iteration"]
        if not snapshots or it != snapshots[-1][0]:
            snapshots.append((it, dict(current_best)))
        for k in test_keys:
            s = r.get("scores_per_test", {}).get(k)
            if s is not None and s > current_best[k]:
                current_best[k] = s
    if snapshots:
        snapshots.append((snapshots[-1][0] + 1, dict(current_best)))

    iters = [s[0] for s in snapshots]
    saved = []
    for tk in test_keys:
        scores = [s[1][tk] for s in snapshots]
        scores = [s if s != -float("inf") else None for s in scores]
        vx, vy = [], []
        for x, y in zip(iters, scores):
            if y is not None:
                vx.append(x)
                vy.append(y)
        if not vx:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(vx, vy, where="post", color="steelblue", linewidth=1.2)
        ax.scatter(vx, vy, s=10, c="steelblue", zorder=5)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Score")
        ax.set_title(f"Best Score on Test {tk}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fpath = out_dir / f"test_{tk}.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath)
    return saved


def cmd_evolution(args: argparse.Namespace) -> None:
    exp_dir = resolve_experiment(args.experiment)
    out_dir = _out_dir(exp_dir, "evolution")
    print(f"Experiment: {exp_dir}")
    if args.full_history:
        hist = resolve_history(exp_dir)
        print(f"  Full history: {len(hist)} run(s)")

    db = load_database_log(exp_dir, full_history=args.full_history)
    ev = load_eval_log(exp_dir, full_history=args.full_history)

    if not db:
        print("  No database log found.")
        return

    p1 = _plot_overall(db, out_dir / "overall.png")
    print(f"  overall  -> {p1.name}")

    paths = _plot_per_test(ev, out_dir)
    print(f"  per-test -> {len(paths)} files")

    print(f"Done -> {out_dir}/")


# ===========================================================================
# Subcommand: programs
# ===========================================================================


def cmd_programs(args: argparse.Namespace) -> None:
    exp_dir = resolve_experiment(args.experiment)
    out_dir = _out_dir(exp_dir, "programs")
    print(f"Experiment: {exp_dir}")

    data = load_final_results(exp_dir)
    best = data.get("best_programs", [])
    if not best:
        print("  No best_programs in results.")
        return

    # -- Extract .py files ---------------------------------------------------
    code_dir = out_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for p in best:
        island = p["island_id"]
        score = p["best_score"]
        fpath = code_dir / f"island{island:02d}_score{score:.4f}.py"
        with open(fpath, "w") as f:
            f.write(f'"""island={island}  score={score:.4f}"""\n')
            f.write(p.get("program", ""))
    print(f"  extracted -> {len(best)} files in code/")

    # -- Score bar chart -----------------------------------------------------
    islands = [p["island_id"] for p in best]
    scores = [p["best_score"] for p in best]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(islands, scores, color="steelblue", alpha=0.85)
    ax.axhline(y=data.get("overall_best", 0), color="red", linestyle="--",
               label=f"overall best = {data['overall_best']:.4f}")
    ax.set_xlabel("Island"); ax.set_ylabel("Score")
    ax.set_title("Best Score per Island")
    ax.set_xticks(islands); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fpath = out_dir / "island_scores.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart    -> {fpath.name}")

    # -- Program length vs score ---------------------------------------------
    lengths = [len(p.get("program", "")) for p in best]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(lengths, scores, c=islands, cmap="viridis", s=60)
    for i, l, s in zip(islands, lengths, scores):
        ax.annotate(str(i), (l, s), textcoords="offset points",
                    xytext=(3, 3), fontsize=6)
    ax.set_xlabel("Program length (chars)"); ax.set_ylabel("Score")
    ax.set_title("Score vs Program Length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fpath = out_dir / "score_vs_length.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart    -> {fpath.name}")

    print(f"Done -> {out_dir}/")


# ===========================================================================
# Subcommand: llm
# ===========================================================================


def cmd_llm(args: argparse.Namespace) -> None:
    exp_dir = resolve_experiment(args.experiment)
    out_dir = _out_dir(exp_dir, "llm")
    print(f"Experiment: {exp_dir}")
    if args.full_history:
        hist = resolve_history(exp_dir)
        print(f"  Full history: {len(hist)} run(s)")

    records = load_sampler_log(exp_dir, full_history=args.full_history)
    if not records:
        print("  No sampler log found.")
        return

    # -- Response length distribution ---------------------------------------
    raw_lens = [len(r.get("raw_response", "")) for r in records]
    extracted_lens = [len(r.get("extracted_code", "")) for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(raw_lens, bins=30, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Characters"); ax1.set_ylabel("Count")
    ax1.set_title(f"Raw Response Length (n={len(records)})")
    ax1.grid(True, alpha=0.3)

    ax2.hist(extracted_lens, bins=30, color="darkorange", alpha=0.8)
    ax2.set_xlabel("Characters"); ax2.set_ylabel("Count")
    ax2.set_title(f"Extracted Code Length (n={len(records)})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fpath = out_dir / "response_lengths.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  lengths  -> {fpath.name}")

    # -- Latency histogram ---------------------------------------------------
    times = [
        r.get("gen_to_eval_s")  # we store this in eval records, not sampler
        for r in load_eval_log(exp_dir)
        if r.get("gen_to_eval_s")
    ]
    if times:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(times, bins=30, color="steelblue", alpha=0.8)
        ax.set_xlabel("Seconds (LLM call → evaluation done)")
        ax.set_ylabel("Count")
        ax.set_title(f"End-to-End Latency (n={len(times)}, median={np.median(times):.1f}s)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fpath = out_dir / "latency.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  latency  -> {fpath.name}")

    # -- Common import / pattern usage ----------------------------------------
    import_counter: Counter = Counter()
    for r in records:
        code = r.get("extracted_code", "")
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                import_counter[line] += 1
    if import_counter:
        print(f"  top imports: {import_counter.most_common(5)}")

    print(f"Done -> {out_dir}/")


# ===========================================================================
# Subcommand: tokens
# ===========================================================================


def _estimate_tokens(text: str, encoder=None) -> int:
    """Estimate token count. Use tiktoken if available, else chars/4."""
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, len(text) // 4)


def _load_tiktoken_encoder(model: str):
    """Try to load a tiktoken encoder for the given model."""
    try:
        import tiktoken
        # Map local model names to tiktoken encoding names
        if "gpt-4" in model.lower():
            return tiktoken.encoding_for_model("gpt-4")
        if "gpt-3.5" in model.lower() or "gpt-35" in model.lower():
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        if "qwen" in model.lower():
            # Qwen uses cl100k_base-like encoding
            return tiktoken.get_encoding("cl100k_base")
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        return None


def cmd_tokens(args: argparse.Namespace) -> None:
    exp_dir = resolve_experiment(args.experiment)
    out_dir = _out_dir(exp_dir, "tokens")
    print(f"Experiment: {exp_dir}")
    if args.full_history:
        hist = resolve_history(exp_dir)
        print(f"  Full history: {len(hist)} run(s)")

    records = load_sampler_log(exp_dir, full_history=args.full_history)
    if not records:
        print("  No sampler log found.")
        return

    # Try to load tiktoken for the dominant model
    dominant_model = Counter(r.get("model", "unknown") for r in records).most_common(1)[0][0]
    encoder = _load_tiktoken_encoder(dominant_model)
    method = "tiktoken" if encoder else "chars/4 heuristic"
    print(f"  Tokenizer: {method} (model={dominant_model})")

    total_input = 0
    total_output = 0
    total_extracted = 0
    per_model: dict[str, dict] = {}

    for r in records:
        model = r.get("model", "unknown")
        if model not in per_model:
            per_model[model] = {"calls": 0, "input": 0, "output": 0, "extracted": 0}

        inp = _estimate_tokens(r.get("prompt", ""), encoder)
        out = _estimate_tokens(r.get("raw_response", ""), encoder)
        ext = _estimate_tokens(r.get("extracted_code", ""), encoder)

        total_input += inp
        total_output += out
        total_extracted += ext
        per_model[model]["calls"] += 1
        per_model[model]["input"] += inp
        per_model[model]["output"] += out
        per_model[model]["extracted"] += ext

    total = total_input + total_output

    # Rough cost map (USD per 1M tokens) — update as needed
    cost_map = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "qwen": {"input": 0.0, "output": 0.0},  # local = free
    }
    # Determine cost rate
    rate = {"input": 0.0, "output": 0.0}
    for key, rates in cost_map.items():
        if key in dominant_model.lower():
            rate = rates
            break

    est_cost = (total_input / 1e6) * rate["input"] + (total_output / 1e6) * rate["output"]

    lines = [
        f"# Token Usage Report",
        f"",
        f"**Method:** {method}",
        f"**Total LLM calls:** {len(records)}",
        f"",
        f"## Overall",
        f"| Metric | Tokens |",
        f"|---|---|",
        f"| Input (prompts) | {total_input:,} |",
        f"| Output (raw responses) | {total_output:,} |",
        f"| Extracted code | {total_extracted:,} |",
        f"| **Total** | **{total:,}** |",
        f"",
        f"## Per Model",
        f"| Model | Calls | Input | Output | Extracted |",
        f"|---|---|---|---|---|",
    ]
    for model, stats in sorted(per_model.items()):
        lines.append(
            f"| {model} | {stats['calls']} | {stats['input']:,} | {stats['output']:,} | {stats['extracted']:,} |"
        )

    if rate["input"] > 0 or rate["output"] > 0:
        lines.extend([
            f"",
            f"## Estimated Cost",
            f"| Rate | USD |",
            f"|---|---|",
            f"| Input / 1M tokens | ${rate['input']:.2f} |",
            f"| Output / 1M tokens | ${rate['output']:.2f} |",
            f"| **Estimated total** | **${est_cost:.4f}** |",
        ])
    else:
        lines.extend([
            f"",
            f"## Estimated Cost",
            f"Model appears to be local (no API cost).",
        ])

    # Per-call histogram data (binned by 100-token buckets)
    output_buckets: Counter = Counter()
    for r in records:
        out = _estimate_tokens(r.get("raw_response", ""), encoder)
        output_buckets[out // 100 * 100] += 1

    lines.extend([
        f"",
        f"## Output Token Distribution",
        f"| Bucket (tokens) | Count |",
        f"|---|---|",
    ])
    for bucket in sorted(output_buckets):
        lines.append(f"| {bucket}-{bucket+99} | {output_buckets[bucket]} |")

    md_path = out_dir / "tokens.md"
    md_path.write_text("\n".join(lines))
    print(f"  report -> {md_path.name}")
    print(f"  Total tokens: {total:,}  (input {total_input:,} + output {total_output:,})")
    if est_cost > 0:
        print(f"  Est. cost: ${est_cost:.4f}")
    print(f"Done -> {out_dir}/")


# ===========================================================================
# Subcommand: summary
# ===========================================================================


def cmd_summary(args: argparse.Namespace) -> None:
    exp_dir = resolve_experiment(args.experiment)
    out_dir = _out_dir(exp_dir, "summary")
    print(f"Experiment: {exp_dir}")
    if args.full_history:
        hist = resolve_history(exp_dir)
        print(f"  Full history: {len(hist)} run(s)")

    data = load_final_results(exp_dir, full_history=args.full_history)
    db = load_database_log(exp_dir, full_history=args.full_history)
    ev = load_eval_log(exp_dir, full_history=args.full_history)
    sampler = load_sampler_log(exp_dir, full_history=args.full_history)
    meta = load_meta(exp_dir)

    # Prefer live data for in-progress runs, fall back to final snapshot
    config = data.get("config", {})
    overall = data.get("overall_best")
    duration = data.get("duration_seconds", 0)
    best = data.get("best_programs", [])

    if db:
        overall = db[-1].get("overall_best", overall)
        duration = db[-1].get("timestamp", 0) - db[0].get("timestamp", 0)
        num_islands = len(db[-1].get("best_score_per_island", []))
        if num_islands:
            config.setdefault("num_islands", num_islands)
        # Build best-programs snapshot from live db
        scores = db[-1].get("best_score_per_island", [])
        best = [{"island_id": i, "best_score": s, "program": ""}
                for i, s in enumerate(scores)]

    # Get model info from meta.json args if not in config
    runs = meta.get("runs", [])
    if runs and not config.get("llm_model"):
        args = runs[-1].get("args", {})
        config.setdefault("llm_model", args.get("model", "?"))

    n_evals = len([r for r in ev if r.get("accepted")])
    n_milestones = len([r for r in ev if r.get("is_milestone")])
    n_rejected = len([r for r in ev if not r.get("accepted")])
    llm_calls = len(sampler) if sampler else data.get("llm_calls", "?")
    iterations = db[-1]["iteration"] if db else llm_calls

    # Build a text summary
    lines = [
        "# FunSearch Run Summary",
        "",
        f"**Started:** {datetime.fromtimestamp(int(db[0]['timestamp'])).isoformat() if db else '?'}",
        f"**Duration:** {duration:.0f}s  ({duration/60:.1f} min)" if duration else "",
        "",
        "## Configuration",
        f"- Model: `{config.get('llm_model', '?')}`  temp={config.get('llm_temperature', '?')}",
        f"- Islands: {config.get('num_islands', '?')}  samples/prompt: {config.get('samples_per_prompt', '?')}",
        f"- Iterations: {iterations}",
        "",
        "## Results",
        f"- **Overall best score:** {overall:.4f}" if overall is not None else "- Overall best: N/A",
        f"- Accepted evaluations: {n_evals}",
        f"- Milestones (new best): {n_milestones}",
        f"- Rejected: {n_rejected}",
        f"- LLM calls: {llm_calls}",
        "",
        "## Best Programs",
    ]
    for p in sorted(best, key=lambda p: p["best_score"], reverse=True)[:5]:
        lines.append(f"- Island {p['island_id']}: score={p['best_score']:.4f} ({len(p.get('program',''))} chars)")

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines))
    print(f"  summary -> {md_path.name}")

    # Dashboard figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: score evolution
    ax1 = fig.add_subplot(gs[0, 0])
    if db:
        its = [r["iteration"] for r in db]
        ax1.plot(its, [r["overall_best"] for r in db], "b-", linewidth=1.2)
        ax1.set_xlabel("Iteration"); ax1.set_ylabel("Score")
        ax1.set_title("Overall Best Score")
        ax1.grid(True, alpha=0.3)

    # Top-right: island comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if best:
        ax2.bar([p["island_id"] for p in best], [p["best_score"] for p in best],
                color="steelblue", alpha=0.85)
        ax2.set_xlabel("Island"); ax2.set_ylabel("Score")
        ax2.set_title("Best per Island")
        ax2.grid(True, alpha=0.3, axis="y")

    # Bottom-left: accept/reject pie
    ax3 = fig.add_subplot(gs[1, 0])
    labels = ["Accepted", "Rejected"]
    sizes = [n_evals, n_rejected]
    if sum(sizes) > 0:
        ax3.pie(sizes, labels=labels, autopct="%1.0f%%", colors=["#2ecc71", "#e74c3c"])
    else:
        ax3.text(0.5, 0.5, "No evaluation data yet", ha="center", va="center",
                 transform=ax3.transAxes)
    ax3.set_title("Evaluation Outcomes")

    # Bottom-right: timing info
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    info = [
        f"Model: {config.get('llm_model', '?')}",
        f"Islands: {config.get('num_islands', '?')}",
        f"Iterations: {iterations}",
        f"Duration: {duration:.0f}s",
        f"Overall best: {overall:.4f}" if overall is not None else "",
        f"Milestones: {n_milestones}",
    ]
    ax4.text(0.1, 0.5, "\n".join(info), transform=ax4.transAxes,
             fontsize=11, family="monospace", va="center")

    fig.suptitle("FunSearch Run Dashboard", fontsize=14, fontweight="bold")
    fpath = out_dir / "dashboard.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  dashboard -> {fpath.name}")

    print(f"Done -> {out_dir}/")


# ===========================================================================
# Subcommand: validation
# ===========================================================================


def cmd_validation(args: argparse.Namespace) -> None:
    exp_dir = resolve_experiment(args.experiment)
    out_dir = _out_dir(exp_dir, "validation")
    print(f"Experiment: {exp_dir}")

    val_records = load_val_log(exp_dir, full_history=args.full_history)
    test_records = load_test_log(exp_dir, full_history=args.full_history)
    val_per_island = load_val_per_island_log(exp_dir, full_history=args.full_history)

    if not val_records and not test_records:
        print("  No validation or test data found.")
        print("  Run with --val-split and --test-split to generate data.")
        return

    # --- Plot 1: Train / Val / Test scores over time ---
    fig, ax = plt.subplots(figsize=(12, 5))

    if val_records:
        its = [r["iteration"] for r in val_records]
        train_scores = [r.get("train_clean", r.get("train_best")) for r in val_records]
        val_scores = [r["val_score"] for r in val_records]
        ax.plot(its, train_scores, "b-", linewidth=1.5, label="Train (clean)", alpha=0.8)
        ax.plot(its, val_scores, "g-", linewidth=1.5, label="Validation", alpha=0.8)

        # Gap shading
        gaps = [t - v for t, v in zip(train_scores, val_scores)]
        ax.fill_between(its, val_scores, train_scores, alpha=0.15, color="red",
                        label="Generalization gap")

    if test_records:
        its_test = [r["iteration"] for r in test_records]
        test_scores = [r["test_score"] for r in test_records]
        ax.plot(its_test, test_scores, "r-", linewidth=1.5, label="Test", alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (negative gap)")
    ax.set_title("Train / Validation / Test Scores Over Time")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fpath = out_dir / "train_val_test.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  train/val/test -> {fpath.name}")

    # --- Plot 2: Generalization gap and penalty ---
    if val_records:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        its = [r["iteration"] for r in val_records]
        gaps = [r.get("gap", 0) for r in val_records]
        penalties = [r.get("gen_penalty", 0) for r in val_records]

        ax1.plot(its, gaps, "r-", linewidth=1.5)
        ax1.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax1.set_ylabel("Gap (train - val)")
        ax1.set_title("Generalization Gap Over Time")
        ax1.grid(True, alpha=0.3)

        ax2.plot(its, penalties, "purple", linewidth=1.5)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Penalty")
        ax2.set_title("Generalization Penalty Over Time")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fpath = out_dir / "gap_and_penalty.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  gap/penalty    -> {fpath.name}")

    # --- Plot 3: Per-island validation scores ---
    if val_per_island:
        fig, ax = plt.subplots(figsize=(14, 6))

        island_ids = sorted({r["island_id"] for r in val_per_island})
        cmap = plt.get_cmap("tab10" if len(island_ids) <= 10 else "tab20")

        for idx, island_id in enumerate(island_ids):
            sub = [r for r in val_per_island if r["island_id"] == island_id]
            sub.sort(key=lambda r: r["iteration"])
            its = [r["iteration"] for r in sub]
            val_scores = [r["val_score"] for r in sub]
            train_scores = [r.get("train_clean", r.get("train_score")) for r in sub]
            color = cmap(idx % cmap.N)

            ax.plot(its, val_scores, "-", linewidth=1.0, alpha=0.8,
                    color=color, label=f"I{island_id} val")
            ax.plot(its, train_scores, "--", linewidth=0.8, alpha=0.5,
                    color=color)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title("Per-Island Train (dashed) vs Validation (solid)")
        ax.legend(loc="lower right", ncol=5, fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fpath = out_dir / "per_island_val.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  per-island     -> {fpath.name}")

    print(f"Done -> {out_dir}/")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="FunSearch analysis toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    # evolution
    p_evo = sub.add_parser("evolution", help="Score trajectory plots")
    p_evo.add_argument("experiment", nargs="?", default=None,
                       help="Experiment dir or run_funsearch/ path")
    p_evo.add_argument("--full-history", action="store_true",
                       help="Follow resumed_from chain in meta.json")

    # islands
    p_isl = sub.add_parser("islands", help="Per-island score evolution")
    p_isl.add_argument("experiment", nargs="?", default=None,
                       help="Experiment dir or run_funsearch/ path")
    p_isl.add_argument("--full-history", action="store_true",
                       help="Follow resumed_from chain in meta.json")

    # programs
    p_prog = sub.add_parser("programs", help="Extract and compare programs")
    p_prog.add_argument("experiment", nargs="?", default=None,
                        help="Experiment dir or run_funsearch/ path")

    # llm
    p_llm = sub.add_parser("llm", help="LLM behavior analysis")
    p_llm.add_argument("experiment", nargs="?", default=None,
                       help="Experiment dir or run_funsearch/ path")
    p_llm.add_argument("--full-history", action="store_true",
                       help="Follow resumed_from chain in meta.json")

    # tokens
    p_tok = sub.add_parser("tokens", help="Token usage analysis")
    p_tok.add_argument("experiment", nargs="?", default=None,
                       help="Experiment dir or run_funsearch/ path")
    p_tok.add_argument("--full-history", action="store_true",
                       help="Follow resumed_from chain in meta.json")

    # summary
    p_sum = sub.add_parser("summary", help="One-page dashboard")
    p_sum.add_argument("experiment", nargs="?", default=None,
                       help="Experiment dir or run_funsearch/ path")
    p_sum.add_argument("--full-history", action="store_true",
                       help="Follow resumed_from chain in meta.json")

    # validation
    p_val = sub.add_parser("validation", help="Validation and test score plots")
    p_val.add_argument("experiment", nargs="?", default=None,
                       help="Experiment dir or run_funsearch/ path")
    p_val.add_argument("--full-history", action="store_true",
                       help="Follow resumed_from chain in meta.json")

    args = parser.parse_args()

    dispatch = {
        "evolution": cmd_evolution,
        "islands": cmd_islands,
        "programs": cmd_programs,
        "llm": cmd_llm,
        "tokens": cmd_tokens,
        "summary": cmd_summary,
        "validation": cmd_validation,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
