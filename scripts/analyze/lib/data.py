"""Data loading helpers for FunSearch experiment outputs.

An experiment directory looks like::

    outputs/20260425_040429/run_funsearch/

The helpers here resolve that directory and load the rolling files (database,
eval, sampler) transparently, merging archived blocks with the current one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def resolve_experiment(path: str | Path | None = None) -> Path:
    """Return the ``run_funsearch/`` directory for an experiment.

    Resolution order:
    1. Explicit path ending in ``run_funsearch`` — use as-is.
    2. Explicit path to a timestamp dir — append ``run_funsearch``.
    3. ``None`` — follow ``outputs/latest/run_funsearch``.
    """
    if path is None:
        latest = Path("outputs/latest")
        if not latest.exists():
            sys.exit("No outputs/latest symlink. Provide an explicit path.")
        return latest.resolve() / "run_funsearch"

    p = Path(path)
    if p.name == "run_funsearch":
        return p
    # Assume p is a timestamp dir (or symlink to one)
    exp = p / "run_funsearch"
    if exp.exists():
        return exp
    sys.exit(f"Cannot find run_funsearch/ under {path}")


def load_meta(exp_dir: Path | None = None) -> dict:
    """Load ``meta.json`` for the experiment."""
    p = resolve_experiment(exp_dir) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def resolve_history(exp_dir: Path | None = None) -> list[Path]:
    """Return the chain of ``run_funsearch/`` directories from oldest to newest.

    Walks the ``resumed_from`` field in ``meta.json`` recursively.
    """
    current = resolve_experiment(exp_dir)
    chain: list[Path] = [current]
    seen: set[Path] = {current.resolve()}

    while True:
        meta = load_meta(current)
        resumed = None
        for run in meta.get("runs", []):
            if "resumed_from" in run:
                src = Path(run["resumed_from"]).resolve()
                # The checkpoint sits inside run_funsearch/
                if src.name == "checkpoint.pkl":
                    src = src.parent
                if src not in seen:
                    resumed = src
                    seen.add(src)
                break
        if resumed is None or not resumed.exists():
            break
        chain.append(resumed)
        current = resumed

    # oldest first
    return list(reversed(chain))


def _read_jsonl(filepath: Path) -> list[dict]:
    if not filepath.exists():
        return []
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_rolling(subdir: str, stem: str, exp_dir: Path | None = None) -> list[dict]:
    """Load a rolling JSONL file by merging archived blocks with the current
    active file.

    *subdir* — directory name inside the experiment (``database``, ``eval``,
    ``sampler``).
    *stem* — the active file name without extension (``database``, ``eval``,
    ``sampler``).
    *exp_dir* — the ``run_funsearch/`` directory.
    """
    d = (resolve_experiment(exp_dir) if exp_dir else resolve_experiment()) / subdir
    if not d.exists():
        return []
    records = []
    for f in sorted(d.glob(f"{stem}_iter_*.jsonl")):
        records.extend(_read_jsonl(f))
    active = d / f"{stem}.jsonl"
    records.extend(_read_jsonl(active))
    return records


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_final_results(exp_dir: Path | None = None, *, full_history: bool = False) -> dict:
    """Load ``funsearch_results.json`` (the final snapshot)."""
    if full_history:
        dirs = resolve_history(exp_dir)
        for d in reversed(dirs):
            p = d / "funsearch_results.json"
            if p.exists():
                return json.loads(p.read_text())
        return {}
    p = resolve_experiment(exp_dir) / "funsearch_results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_database_log(exp_dir: Path | None = None, *, full_history: bool = False) -> list[dict]:
    """Load all database log records (best scores per island per iteration)."""
    if full_history:
        records: list[dict] = []
        for d in resolve_history(exp_dir):
            records.extend(_load_rolling("database", "database", d))
        return records
    return _load_rolling("database", "database", exp_dir)


def load_eval_log(exp_dir: Path | None = None, *, full_history: bool = False) -> list[dict]:
    """Load all evaluation records (per-program scores, timing, milestone)."""
    if full_history:
        records: list[dict] = []
        for d in resolve_history(exp_dir):
            records.extend(_load_rolling("eval", "eval", d))
        return records
    return _load_rolling("eval", "eval", exp_dir)


def load_sampler_log(exp_dir: Path | None = None, *, full_history: bool = False) -> list[dict]:
    """Load all sampler records (LLM prompts and responses)."""
    if full_history:
        records: list[dict] = []
        for d in resolve_history(exp_dir):
            records.extend(_load_rolling("sampler", "sampler", d))
        return records
    return _load_rolling("sampler", "sampler", exp_dir)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

