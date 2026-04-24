#!/usr/bin/env python3
"""
Extract generated programs from funsearch_results.json to standalone .py files.

Usage:
    # Extract latest results
    python scripts/analyze/extract_generated_codes.py

    # Extract a specific results file
    python scripts/analyze/extract_generated_codes.py --input outputs/latest/run_funsearch/funsearch_results.json

    # Save to a custom directory
    python scripts/analyze/extract_generated_codes.py --input results.json --output-dir ./extracted/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.funsearch_cvrp.utils.output_manager import print_results_summary


def get_default_output_dir(input_file: Path) -> Path:
    """Derive output dir sibling to the experiment directory."""
    try:
        resolved = input_file.resolve()
        # input is like outputs/YYYYmmdd_HHMMSS/run_funsearch/funsearch_results.json
        experiment_dir = resolved.parent.parent  # outputs/YYYYmmdd_HHMMSS
        if experiment_dir.name and experiment_dir.parent.name == "outputs":
            return experiment_dir / "extracted_codes"
    except Exception:
        pass
    return input_file.parent / "extracted_codes"


def extract_codes(input_file: Path, output_dir: Path | None = None):
    """Extract best programs from a funsearch_results.json file."""
    if not input_file.exists():
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        return 0

    if output_dir is None:
        output_dir = get_default_output_dir(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file) as f:
        data = json.load(f)

    best_programs = data.get("best_programs", [])
    if not best_programs:
        print("No best_programs found in", input_file)
        return 0

    print(f"Extracting {len(best_programs)} programs from {input_file} ...")

    for entry in best_programs:
        island = entry.get("island_id", "?")
        score = entry.get("best_score", "N/A")
        program = entry.get("program", "")
        if not program:
            continue

        fname = f"island{island:02d}_score{score:.4f}.py"
        fpath = output_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f'"""\n')
            f.write(f"FunSearch evolved priority function\n")
            f.write(f"Source: {input_file}\n")
            f.write(f"Island: {island}\n")
            f.write(f"Score: {score}\n")
            f.write(f"Extracted: {datetime.now().isoformat()}\n")
            f.write(f'"""\n\n')
            f.write(program)

        print(f"  -> {fname}")

    print(f"\nDone — {len(best_programs)} programs saved to {output_dir}/")
    return len(best_programs)


def find_results_files(base_dir: str = "outputs") -> list[Path]:
    """Find all funsearch_results.json files in outputs/."""
    base = Path(base_dir)
    if not base.exists():
        return []
    files = list(base.rglob("funsearch_results.json"))
    return sorted(files, reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Extract generated programs from FunSearch results")
    parser.add_argument("--input", "-i", type=Path, help="Path to funsearch_results.json")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory")
    parser.add_argument("--list", "-l", action="store_true", help="List available results")
    args = parser.parse_args()

    if args.list:
        print_results_summary("outputs")
        return

    if args.input:
        extract_codes(args.input, args.output_dir)
        return

    # Default: extract from latest run
    latest_link = Path("outputs/latest")
    if latest_link.exists():
        results_file = latest_link.resolve() / "run_funsearch" / "funsearch_results.json"
        if results_file.exists():
            extract_codes(results_file, args.output_dir)
            return

    # Fall back: find newest
    files = find_results_files()
    if files:
        extract_codes(files[0], args.output_dir)
    else:
        print("Error: no funsearch_results.json found. Run an experiment first.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
