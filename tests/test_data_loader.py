"""Tests for analysis data loader: resolve_experiment, rolling file reads."""

import json
import os
from pathlib import Path

import pytest

# Add scripts/ to path for import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze.lib.data import (
    resolve_experiment,
    load_final_results,
    load_database_log,
    load_eval_log,
    load_sampler_log,
)


def _create_experiment(base: Path) -> Path:
    """Create a minimal experiment directory structure and return exp_dir."""
    exp_dir = base / "run_funsearch"
    exp_dir.mkdir(parents=True)
    return exp_dir


class TestResolveExperiment:
    def test_explicit_run_funsearch_dir(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        result = resolve_experiment(exp_dir)
        assert result == exp_dir

    def test_timestamp_dir_appends_run_funsearch(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        result = resolve_experiment(tmp_path)
        assert result == exp_dir

    def test_none_follows_latest_symlink(self, tmp_path: Path, monkeypatch):
        # Create a fake latest symlink
        run_dir = tmp_path / "20260425_test" / "run_funsearch"
        run_dir.mkdir(parents=True)
        latest = tmp_path / "latest"
        latest.symlink_to(tmp_path / "20260425_test", target_is_directory=True)

        # Override the outputs/latest path used by resolve_experiment
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr("scripts.analyze.lib.data.Path.exists", lambda self: True)

    def test_exits_on_missing(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda _: False)
        with pytest.raises(SystemExit):
            resolve_experiment(Path("/nonexistent/path"))


class TestLoadFinalResults:
    def test_loads_best_programs(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        data = {
            "best_programs": [
                {"island_id": 0, "best_score": -0.3, "program": "def p(): pass"},
                {"island_id": 1, "best_score": -0.5, "program": "def p(): return 1"},
            ],
            "overall_best": -0.3,
            "config": {},
        }
        (exp_dir / "funsearch_results.json").write_text(json.dumps(data))
        result = load_final_results(exp_dir)
        assert len(result["best_programs"]) == 2
        assert result["overall_best"] == -0.3


class TestLoadRollingFiles:
    def test_loads_active_file(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        eval_dir = exp_dir / "eval"
        eval_dir.mkdir()
        (eval_dir / "eval.jsonl").write_text(
            '{"iteration": 1, "accepted": true}\n'
            '{"iteration": 2, "accepted": true}\n'
        )
        records = load_eval_log(exp_dir)
        assert len(records) == 2
        assert records[0]["iteration"] == 1

    def test_merges_archived_blocks(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        eval_dir = exp_dir / "eval"
        eval_dir.mkdir()
        (eval_dir / "eval_iter_000000_000999.jsonl").write_text(
            '{"iteration": 1, "accepted": true}\n')
        (eval_dir / "eval_iter_001000_001999.jsonl").write_text(
            '{"iteration": 1000, "accepted": true}\n')
        (eval_dir / "eval.jsonl").write_text(
            '{"iteration": 2000, "accepted": false}\n')

        records = load_eval_log(exp_dir)
        assert len(records) == 3
        iterations = {r["iteration"] for r in records}
        assert iterations == {1, 1000, 2000}

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        (exp_dir / "eval").mkdir()
        assert load_eval_log(exp_dir) == []

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        assert load_eval_log(exp_dir) == []

    def test_loads_database_log(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        db_dir = exp_dir / "database"
        db_dir.mkdir()
        (db_dir / "database.jsonl").write_text(
            '{"iteration": 1, "overall_best": -0.3, "best_score_per_island": [1.0]}\n')
        records = load_database_log(exp_dir)
        assert len(records) == 1

    def test_loads_sampler_log(self, tmp_path: Path):
        exp_dir = _create_experiment(tmp_path)
        sampler_dir = exp_dir / "sampler"
        sampler_dir.mkdir()
        (sampler_dir / "sampler.jsonl").write_text(
            '{"model": "gpt-4", "prompt": "...", "raw_response": "..."}\n')
        records = load_sampler_log(exp_dir)
        assert len(records) == 1
