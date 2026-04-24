"""Tests for evaluator: evaluation history writing, milestone detection."""

import json
from pathlib import Path

import pytest

from funsearch_cvrp.funsearch import evaluator
from funsearch_cvrp.funsearch import programs_database
from funsearch_cvrp.funsearch import code_manipulation
from funsearch_cvrp.funsearch import config as cfg_lib


class TestReduceScore:
    """Tests for _reduce_score — last test instance convention."""

    def test_reduce_picks_last_key(self):
        scores = {0: -1.0, 1: -2.0, 2: -0.5}
        result = programs_database._reduce_score(scores)
        assert result == -0.5

    def test_reduce_single_entry(self):
        scores = {0: -0.3}
        result = programs_database._reduce_score(scores)
        assert result == -0.3


class TestGetSignature:
    def test_signature_sorted(self):
        scores = {2: -0.5, 0: -1.0, 1: -0.8}
        sig = programs_database._get_signature(scores)
        assert sig == (-1.0, -0.8, -0.5)

    def test_signature_with_precision(self):
        scores = {0: -0.1234, 1: -0.5678}
        sig = programs_database._get_signature(scores, precision=2)
        assert sig == (-0.12, -0.57)


class TestEvalHistory:
    """Test evaluation history writing to JSONL."""

    def _make_db(self, num_islands=2):
        config = cfg_lib.ProgramsDatabaseConfig(num_islands=num_islands)
        template = code_manipulation.text_to_program(
            "def priority(a, b):\n  return 0.0")
        return programs_database.ProgramsDatabase(
            config, template, "priority")

    def test_writes_eval_record(self, tmp_path: Path):
        log_path = tmp_path / "eval.jsonl"
        db = self._make_db()
        ev = evaluator.Evaluator(
            db,
            code_manipulation.text_to_program(
                "def priority(a, b):\n  return 0.0"),
            "priority",
            lambda inp, fn: 42.0,
            inputs=[("test",)],
            sandbox=_FakeSandbox(return_value=42.0),
            eval_history_path=log_path,
        )
        ev.analyse("  return 1.0", island_id=0, version_generated=1,
                   generation_time=100.0, iteration=5)

        records = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        assert len(records) == 1
        r = records[0]
        assert r["iteration"] == 5
        assert r["island_id"] == 0
        assert r["accepted"] is True
        assert r["is_milestone"] is True  # first program on island 0
        assert "body" in r
        assert "scores_per_test" in r
        assert r["generation_time"] == 100.0
        assert "eval_duration_s" in r

    def test_milestone_false_when_not_better(self, tmp_path: Path):
        log_path = tmp_path / "eval.jsonl"
        db = self._make_db()
        template = code_manipulation.text_to_program(
            "def priority(a, b):\n  return 0.0")
        ev = evaluator.Evaluator(
            db, template, "priority",
            lambda inp, fn: 42.0,
            inputs=[("test",)],
            sandbox=_FakeSandbox(return_value=42.0),
            eval_history_path=log_path,
        )
        # Register a strong program on island 0
        db.register_program(
            template.get_function("priority"), island_id=None,
            scores_per_test={0: 100.0})

        # Now evaluate a weaker one
        ev = evaluator.Evaluator(
            db, template, "priority",
            lambda inp, fn: 1.0,  # weak score
            inputs=[("test",)],
            sandbox=_FakeSandbox(return_value=1.0),
            eval_history_path=log_path,
        )
        ev.analyse("  return 0.0", island_id=0, version_generated=1,
                   generation_time=200.0, iteration=6)

        records = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        r = records[0]
        assert r["is_milestone"] is False

    def test_writes_rejected_record(self, tmp_path: Path):
        log_path = tmp_path / "eval.jsonl"
        db = self._make_db()
        ev = evaluator.Evaluator(
            db,
            code_manipulation.text_to_program(
                "def priority(a, b):\n  return 0.0"),
            "priority",
            lambda inp, fn: 42.0,
            inputs=[("test",)],
            sandbox=_FakeSandbox(return_value=42.0, should_fail=True),
            eval_history_path=log_path,
        )
        ev.analyse("  return 1.0", island_id=0, version_generated=1,
                   iteration=0)

        records = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        assert len(records) == 1
        r = records[0]
        assert r["accepted"] is False
        assert r["reject_reason"] == "sandbox_failed"

    def test_no_file_written_when_path_is_none(self, tmp_path: Path):
        db = self._make_db()
        ev = evaluator.Evaluator(
            db,
            code_manipulation.text_to_program(
                "def priority(a, b):\n  return 0.0"),
            "priority",
            lambda inp, fn: 42.0,
            inputs=[("test",)],
            sandbox=_FakeSandbox(return_value=42.0),
        )
        # Should not raise
        ev.analyse("  return 1.0", island_id=0, version_generated=1,
                   iteration=0)


# ---------------------------------------------------------------------------
# Test sandbox
# ---------------------------------------------------------------------------


class _FakeSandbox(evaluator.Sandbox):
    def __init__(self, return_value=0.0, should_fail=False):
        self.return_value = return_value
        self.should_fail = should_fail

    def run(self, evolved_fn_code, function_name, evaluate_fn, test_input,
            timeout_seconds):
        if self.should_fail:
            return None, False
        return self.return_value, True
