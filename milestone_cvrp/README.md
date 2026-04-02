# CVRP Full Project (Sample-Efficient FunSearch)

This folder now contains a full, reproducible project pipeline for the topic:

Sample-Efficient FunSearch for CVRP.

## Project Structure

- `cvrp_core.py`: CVRP data model, distance utilities, synthetic benchmark generation, baseline greedy solvers.
- `sample_efficient_search.py`: sample-efficient search with early pruning and functional deduplication.
- `baselines.py`: additional baselines (Clarke-Wright Savings) and 2-opt local improvement wrappers.
- `cvrplib_io.py`: CVRPLib `.vrp` loader.
- `run_milestone.py`: original milestone experiment runner.
- `run_full_project.py`: full benchmark runner with method ranking and JSON outputs.
- `generate_report.py`: milestone markdown report.
- `generate_report_detailed.py`: extended milestone markdown report.
- `generate_final_report.py`: final project markdown report from full outputs.
- `export_report_docx.py`, `export_report_detailed.py`: Word report exporters.
- `tests/test_project.py`: smoke tests for core pipeline.
- `outputs/`: generated experiment results.

## Environment

Use the workspace virtual environment:

```powershell
& "c:/Users/17714/Desktop/AI project/.venv/Scripts/Activate.ps1"
```

## Quick Start (Milestone)

```powershell
python run_milestone.py
python generate_report.py
python generate_report_detailed.py
```

## Quick Start (Full Project)

Run full synthetic benchmark and final report:

```powershell
python run_full_project.py --dataset synthetic
python generate_final_report.py
```

Run tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## CVRPLib Support

If you have CVRPLib `.vrp` files in a folder, run:

```powershell
python run_full_project.py --dataset cvrplib --cvrplib-dir "path/to/vrp_files" --limit-instances 10
```

## Output Files

- Milestone:
	- `outputs/milestone_results.json`
	- `outputs/search_history.json`
- Full project:
	- `outputs/full_project_results.json`
	- `outputs/full_search_history.json`
- Reports:
	- `milestone_report.md`
	- `milestone_report_detailed.md`
	- `final_project_report.md`

## Notes

- Synthetic benchmarks are deterministic with fixed seed.
- Core algorithmic code uses only Python standard library.
- Word export scripts require `python-docx`.
