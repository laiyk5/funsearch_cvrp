"""Lightweight periodic monitor for an active FunSearch run.

Runs cheap analyses (summary, overall evolution, LLM stats) on the latest
experiment every N seconds and writes them to outputs/latest/analysis/.

Usage:
    python scripts/monitor/monitor_run.py
    python scripts/monitor/monitor_run.py --interval 120  # every 2 min
    python scripts/monitor/monitor_run.py --once          # single shot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SCRIPT = PROJECT_ROOT / "scripts" / "analyze" / "analyze.py"


TEST_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "analyze" / "plot_test_eval.py"


def run_analysis(exp_dir: Path, cheap: bool = True) -> None:
    """Run cheap analysis subcommands on an experiment directory."""
    subcommands = ["summary", "evolution", "llm", "tokens"]
    if not cheap:
        subcommands.extend(["programs", "islands"])

    for sub in subcommands:
        cmd = [sys.executable, str(ANALYSIS_SCRIPT), sub, str(exp_dir)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
        except Exception:
            pass

    # Generate test-eval plots if data exists.
    if (exp_dir / "test" / "test_eval.jsonl").exists():
        try:
            subprocess.run(
                [sys.executable, str(TEST_EVAL_SCRIPT), str(exp_dir)],
                capture_output=True,
                timeout=60,
            )
        except Exception:
            pass


def find_latest_experiment() -> Path | None:
    """Resolve the outputs/latest symlink to the actual experiment."""
    latest = PROJECT_ROOT / "outputs" / "latest"
    if not latest.exists():
        return None
    run_dir = latest / "run_funsearch"
    if run_dir.exists():
        return run_dir
    return None


def is_run_active() -> bool:
    """Check if a FunSearch process is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_funsearch.py"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Monitor FunSearch run")
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Seconds between analysis updates (default: 120)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run analysis once and exit (no loop)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include expensive analyses (programs, islands)",
    )
    args = parser.parse_args()

    print(f"Monitor started — interval={args.interval}s")
    print(f"Project root: {PROJECT_ROOT}")

    while True:
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            print("No active experiment found.")
        else:
            active = is_run_active()
            status = "RUNNING" if active else "idle"
            print(f"[{time.strftime('%H:%M:%S')}] Analyzing {exp_dir.name} ({status}) ...")
            run_analysis(exp_dir, cheap=not args.full)
            print(f"[{time.strftime('%H:%M:%S')}] Done.")

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
