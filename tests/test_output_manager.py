"""Tests for output_manager: meta.json, list_experiments_table, symlinks."""

import json
from pathlib import Path
from unittest import mock

import pytest

from funsearch_cvrp.utils.output_manager import (
    get_output_dir,
    list_experiments_table,
    get_git_commit_hash,
    is_git_dirty,
    update_meta,
)


class TestGitHelpers:
    def test_get_git_commit_hash_short(self):
        h = get_git_commit_hash(short=True)
        assert len(h) <= 8
        assert h != "unknown"

    def test_get_git_commit_hash_full(self):
        h = get_git_commit_hash(short=False)
        assert len(h) == 40 or h == "unknown"
        # Verify no "--long" prefix bug
        assert not h.startswith("--long")

    def test_is_git_dirty_returns_bool(self):
        result = is_git_dirty()
        assert isinstance(result, bool)


class TestGetOutputDir:
    def test_creates_directory_and_meta(self, tmp_path: Path):
        d = get_output_dir("test_script", base_dir=str(tmp_path))
        assert d.exists()
        assert (d / "meta.json").exists()

    def test_meta_contains_run_entry(self, tmp_path: Path):
        d = get_output_dir("test_script", base_dir=str(tmp_path),
                           args={"foo": "bar"})
        meta = json.loads((d / "meta.json").read_text())
        assert len(meta["runs"]) == 1
        run = meta["runs"][0]
        assert "run_time" in run
        assert "git_commit" in run
        assert run["args"]["foo"] == "bar"

    def test_second_call_appends_run(self, tmp_path: Path):
        d = get_output_dir("test_script", base_dir=str(tmp_path))
        # Second call on same dir should append
        d2 = get_output_dir("test_script", base_dir=str(tmp_path))
        assert d == d2
        meta = json.loads((d / "meta.json").read_text())
        assert len(meta["runs"]) == 2

    def test_creates_latest_symlink(self, tmp_path: Path):
        get_output_dir("test_script", base_dir=str(tmp_path))
        latest = tmp_path / "latest"
        assert latest.is_symlink()

    def test_update_meta_merges_fields(self, tmp_path: Path):
        d = get_output_dir("test_script", base_dir=str(tmp_path))
        update_meta(d, {"duration_s": 42.0})
        meta = json.loads((d / "meta.json").read_text())
        assert meta["runs"][-1]["duration_s"] == 42.0


class TestListExperimentsTable:
    def test_empty_when_no_outputs(self, tmp_path: Path):
        result = list_experiments_table(base_dir=str(tmp_path))
        assert result == []

    def test_lists_experiments(self, tmp_path: Path):
        # Create a fake experiment structure
        exp_dir = tmp_path / "20260425_120000" / "run_funsearch"
        exp_dir.mkdir(parents=True)
        (exp_dir / "meta.json").write_text(json.dumps({
            "runs": [{
                "run_time": "2026-04-25T12:00:00",
                "git_commit": "abc123def4567890",
                "git_dirty": False,
                "args": {"model": "gpt-4", "iterations": 10},
            }]
        }))
        result = list_experiments_table(base_dir=str(tmp_path))
        assert len(result) >= 1
        latest = result[0]
        assert latest["dir"] == "20260425_120000"
        assert latest["commit"] == "abc123de"
        assert latest["model"] == "gpt-4"
        assert latest["n_runs"] == 1
        assert latest["dirty"] is False

    def test_handles_legacy_long_prefix(self, tmp_path: Path):
        """Old meta.json had '--long\\n<hash>' in git_commit."""
        exp_dir = tmp_path / "20260422_000000" / "run_funsearch"
        exp_dir.mkdir(parents=True)
        (exp_dir / "meta.json").write_text(json.dumps({
            "runs": [{
                "run_time": "2026-04-22",
                "git_commit": "--long\nabcdef1234567890",
                "git_dirty": True,
                "args": {},
            }]
        }))
        result = list_experiments_table(base_dir=str(tmp_path))
        assert result[0]["commit"] == "abcdef12"

    def test_handles_unknown_commit(self, tmp_path: Path):
        exp_dir = tmp_path / "20260422_000000" / "run_funsearch"
        exp_dir.mkdir(parents=True)
        (exp_dir / "meta.json").write_text(json.dumps({
            "runs": [{"run_time": "2026-04-22", "args": {}}]
        }))
        result = list_experiments_table(base_dir=str(tmp_path))
        assert result[0]["commit"] == "unknown"
