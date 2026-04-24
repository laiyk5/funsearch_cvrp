"""Output directory manager.

Output structure:
    outputs/
        <YYYYmmdd_HHMMSS>/
            <script_name>/
                meta.json    # written automatically by get_output_dir()
                ...
        latest -> <YYYYmmdd_HHMMSS>/   # symlink to most recent run
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_commit_hash(short: bool = True) -> str:
    if short:
        return _git("rev-parse", "--short", "HEAD")
    return _git("rev-parse", "HEAD")


def get_git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def is_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_output_dir(script_name: str, base_dir: str = "outputs", args: dict | None = None) -> Path:
    """Create (or reuse) an output directory for a script and append a run entry to meta.json.

    Directory layout: <base_dir>/<YYYYmmdd_HHMMSS>/<script_name>/

    If the directory already exists (reuse case), the new run is appended to the
    existing runs list in meta.json rather than overwriting it.

    Args:
        script_name: Unique name for the script (e.g. "run_funsearch").
        base_dir: Root outputs folder.
        args: Script arguments/parameters to record in this run's entry.

    Returns:
        Path to the script's output directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    output_dir = run_dir / script_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _append_meta(output_dir, args)
    _update_latest_symlink(Path(base_dir), run_dir)

    return output_dir


def _append_meta(output_dir: Path, args: dict | None = None) -> None:
    """Append a run entry to meta.json, creating it if needed."""
    meta_file = output_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"runs": []}

    entry: dict = {
        "run_time": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(short=False),
        "git_dirty": is_git_dirty(),
    }
    if args:
        entry["args"] = args

    meta.setdefault("runs", []).append(entry)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def update_meta(output_dir: Path, extra: dict) -> None:
    """Merge extra fields into the latest run entry in meta.json."""
    meta_file = output_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"runs": []}

    if meta.get("runs"):
        meta["runs"][-1].update(extra)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _update_latest_symlink(base: Path, target: Path) -> None:
    latest = base / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target.resolve(), target_is_directory=True)


# ---------------------------------------------------------------------------
# Result listing
# ---------------------------------------------------------------------------

def list_results(base_dir: str = "outputs") -> list[dict]:
    """Return a list of run metadata dicts, newest first."""
    base = Path(base_dir)
    if not base.exists():
        return []

    runs = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir() or d.name == "latest":
            continue
        meta_file = d / "meta.json"
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        runs.append({"dir": d.name, **meta})
    return runs


def print_results_summary(base_dir: str = "outputs") -> None:
    runs = list_results(base_dir)
    if not runs:
        print(f"No results found in {base_dir}/")
        return

    print(f"\n{'='*60}")
    print(f"Results in {base_dir}/")
    print(f"{'='*60}\n")
    for r in runs:
        dirty = " (dirty)" if r.get("git_dirty") else ""
        commit = r.get("git_commit", "unknown")[:8]
        print(f"  {r['dir']}  commit={commit}{dirty}")
    print()
