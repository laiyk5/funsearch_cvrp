"""Benchmark only the evolved solver on XL dataset."""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import importlib.util
from funsearch_cvrp.cvrp.core import make_savings_solver, is_valid_solution, solution_distance
from funsearch_cvrp.cvrp.io import load_cvrplib_instance

# Load evolved function
prog_path = PROJECT_ROOT / "outputs/20260426_135214_ds_flash_cw/run_funsearch/best_program.py"
spec = importlib.util.spec_from_file_location("evolved", prog_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.savings

solver = make_savings_solver(fn, two_opt=False)

folder = PROJECT_ROOT / "data/cvrplib/XL"
vrp_files = sorted(folder.glob("*.vrp"))
print(f"XL dataset: {len(vrp_files)} instances")

results = []
for idx, f in enumerate(vrp_files):
    inst = load_cvrplib_instance(f)
    t0 = time.perf_counter()
    try:
        sol = solver(inst)
    except Exception as e:
        print(f"  ERROR {inst.name}: {e}")
        results.append({"instance": inst.name, "cost": None, "gap": None, "time": time.perf_counter()-t0, "valid": False, "error": str(e)})
        continue
    elapsed = time.perf_counter() - t0
    valid, reason = is_valid_solution(inst, sol)
    if not valid:
        print(f"  INVALID {inst.name}: {reason}")
        results.append({"instance": inst.name, "cost": None, "gap": None, "time": elapsed, "valid": False, "error": reason})
        continue
    cost = solution_distance(inst, sol)
    print(f"  ({idx+1}/{len(vrp_files)}) {inst.name:20s} cost={cost:.1f} time={elapsed:.2f}s")
    results.append({"instance": inst.name, "cost": cost, "gap": None, "time": elapsed, "valid": True, "error": None})

# Save
out_dir = PROJECT_ROOT / "outputs/20260426_135214_ds_flash_cw/benchmark"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "benchmark_XL_evolved_only.json"
with open(out_path, "w") as f:
    json.dump({"Evolved Savings": results}, f, indent=2)

# Summary
valid_results = [r for r in results if r["valid"]]
avg_cost = sum(r["cost"] for r in valid_results) / len(valid_results)
avg_time = sum(r["time"] for r in results) / len(results)
print(f"\nDone. Valid: {len(valid_results)}/{len(results)}  Avg cost: {avg_cost:.1f}  Avg time: {avg_time:.2f}s")
print(f"Saved to {out_path}")
