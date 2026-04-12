# Final Project Report: Sample-Efficient FunSearch for CVRP

## Team
- QIN YUANCHENG (59061061)
- QIN ZIHENG (59866035)
- LAI YIKAI (59563061)

## Project Scope
This full project extends the milestone prototype into a complete benchmark pipeline with:
- Multiple deterministic baselines
- Local-search enhancement (2-opt)
- Sample-efficient FunSearch search
- Unified experiment runner and final ranking output
- Reproducible JSON outputs and this auto-generated report

## Dataset
- Dataset type: synthetic
- Instance count: 5
- Instances: SYN-01-N20, SYN-02-N24, SYN-03-N28, SYN-04-N32, SYN-05-N36

## Method Ranking
| Rank | Method | Avg Distance | Avg #Routes | Score |
|---:|---|---:|---:|---:|
| 1 | Clarke-Wright Savings + 2-opt | 755.306 | 6.800 | 891.306 |
| 2 | Clarke-Wright Savings | 755.839 | 6.800 | 891.839 |
| 3 | Sample-Efficient FunSearch + 2-opt | 923.539 | 6.600 | 1055.539 |
| 4 | Sample-Efficient FunSearch | 930.839 | 6.600 | 1062.839 |
| 5 | Nearest Neighbor + 2-opt | 962.622 | 6.600 | 1094.622 |
| 6 | Nearest Neighbor | 985.844 | 6.600 | 1117.844 |
| 7 | Weighted Greedy (fixed) | 1077.837 | 6.600 | 1209.837 |

## Best Method
- Name: Clarke-Wright Savings + 2-opt
- Avg distance: 755.306
- Avg routes: 6.800
- Score: 891.306

## FunSearch Details
- Best weights: [0.197, 0.3153, -0.1856]
- Best search score: 1062.839
- Unique candidates: 72
- Full evaluations: 69
- Early-pruned: 3 (4.2%)

## Deliverables Produced
- outputs/full_project_results.json
- outputs/full_search_history.json
- final_project_report.md

## Reproducibility
Run the full pipeline:

```powershell
python run_full_project.py --dataset synthetic
python generate_final_report.py
```

All synthetic experiments are deterministic with the configured random seed.
