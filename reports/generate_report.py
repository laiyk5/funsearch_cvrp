from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out_dir = Path("outputs")
    results_path = out_dir / "milestone_results.json"
    if not results_path.exists():
        raise FileNotFoundError("Run run_milestone.py first to generate outputs/milestone_results.json")

    results = json.loads(results_path.read_text(encoding="utf-8"))

    baseline = results["baseline"]
    method = results["sample_efficient_funsearch"]
    improvement = results["improvement"]
    dataset_list = ", ".join(results["dataset"])

    report = f"""# Milestone Report Draft: Sample-Efficient FunSearch for CVRP

## Group Members
- QIN YUANCHENG (59061061)
- QIN ZIHENG (59866035)
- LAI YIKAI (59563061)

## 1. Problem Description and Motivation
We study the Capacitated Vehicle Routing Problem (CVRP), where a fleet of vehicles with fixed capacity must serve customer demands with minimum total travel distance. CVRP is a standard benchmark in combinatorial optimization and is practically relevant to logistics and delivery planning.

Our selected topic is **sample-efficient FunSearch for CVRP**. Instead of manually designing routing heuristics, we frame heuristic design as a search process over executable heuristic programs. The key milestone goal is to reduce evaluation cost by avoiding repeated or low-value candidate evaluations.

## 2. Method/Approach Design
### 2.1 Baseline
- Baseline heuristic: Nearest Neighbor routing with capacity feasibility.

### 2.2 Sample-Efficient FunSearch Prototype
- Heuristic family: weighted greedy selection with score terms:
  - distance from current node,
  - demand ratio,
  - distance to depot.
- Search strategy:
  - initialize a population of candidate weight vectors,
  - mutate top candidates,
  - fast duplicate filtering using rounded signatures,
  - two-stage evaluation (early subset pruning + full evaluation).
- Objective: minimize `avg_distance + 20 * avg_num_routes`.

### 2.3 Benchmark Setup
- Current benchmark for milestone: synthetic CVRP instances `{dataset_list}` generated with fixed seed (reproducible).
- Next stage: replace/extend with CVRPLib Uchoa sets for final project evaluation.

## 3. Preliminary Results
| Method | Avg Distance | Avg #Routes |
|---|---:|---:|
| {baseline['name']} | {baseline['avg_distance']} | {baseline['avg_num_routes']} |
| Sample-Efficient FunSearch (best found) | {method['avg_distance']} | {method['avg_num_routes']} |

- Distance improvement: **{improvement['distance_delta']}** ({improvement['distance_improvement_percent']}%).
- Route count change: **{improvement['routes_delta']}**.
- Unique candidate signatures evaluated: **{method['unique_candidates']}**.
- Best weight vector found: **{method['best_weights']}**.

## 4. Code Artifacts Submitted
- `cvrp_core.py`: CVRP instance generation, heuristics, and evaluator.
- `sample_efficient_search.py`: sample-efficient search with duplicate checking and early pruning.
- `run_milestone.py`: experiment runner and result export.
- `outputs/milestone_results.json`, `outputs/search_history.json`: preliminary result logs.

## 5. Next Steps
- Integrate real CVRPLib/Uchoa instances.
- Add stronger baselines (e.g., OR-Tools, LKH wrapper where available).
- Improve functional-duplicate detection beyond rounded signatures.
- Run larger-scale ablations for sample-efficiency vs solution quality trade-off.
"""

    Path("milestone_report.md").write_text(report, encoding="utf-8")
    print("Saved milestone_report.md")


if __name__ == "__main__":
    main()
