"""Tests for programs_database: NEW BEST export, score_bucket, softmax."""

import json
from pathlib import Path

import numpy as np
import pytest

from funsearch_cvrp.funsearch import programs_database
from funsearch_cvrp.funsearch import code_manipulation
from funsearch_cvrp.funsearch import config as cfg_lib


class TestBestProgramExport:
    """Tests that best_program.py is saved on every NEW BEST milestone."""

    def _make_db(self, num_islands=2, best_program_path=None):
        config = cfg_lib.ProgramsDatabaseConfig(num_islands=num_islands)
        template = code_manipulation.text_to_program(
            "def priority(a, b):\n  return 0.0")
        return programs_database.ProgramsDatabase(
            config, template, "priority",
            best_program_path=best_program_path)

    def test_exports_on_new_best(self, tmp_path: Path):
        bp_path = tmp_path / "best.py"
        db = self._make_db(best_program_path=bp_path)
        program = code_manipulation.text_to_function(
            "def priority(a, b):\n  return a + b\n")

        db.register_program(program, island_id=0,
                            scores_per_test={0: 42.0})
        assert bp_path.exists()
        content = bp_path.read_text()
        assert "return a + b" in content

    def test_does_not_export_when_not_better(self, tmp_path: Path):
        bp_path = tmp_path / "best.py"
        db = self._make_db(best_program_path=bp_path)

        # Register a strong program first
        strong = code_manipulation.text_to_function(
            "def priority(a, b):\n  return 100.0\n")
        db.register_program(strong, island_id=0,
                            scores_per_test={0: 100.0})
        # Clear the file so we can detect a re-write
        bp_path.write_text("")

        # Register a weaker one
        weak = code_manipulation.text_to_function(
            "def priority(a, b):\n  return 1.0\n")
        db.register_program(weak, island_id=0,
                            scores_per_test={0: 1.0})
        assert bp_path.read_text() == ""  # unchanged

    def test_exports_best_overall_even_on_other_island_milestone(self, tmp_path: Path):
        bp_path = tmp_path / "best.py"
        db = self._make_db(best_program_path=bp_path)

        # Island 0 has strong program
        strong = code_manipulation.text_to_function(
            "def priority(a, b):\n  return 100.0\n")
        db.register_program(strong, island_id=0, scores_per_test={0: 100.0})
        original = bp_path.read_text()

        # Island 1 gets a weaker program — still a NEW BEST for island 1
        moderate = code_manipulation.text_to_function(
            "def priority(a, b):\n  return 50.0\n")
        db.register_program(moderate, island_id=1,
                            scores_per_test={0: 50.0})
        # File is re-exported but still contains the overall best (strong, 100)
        assert bp_path.exists()
        assert "return 100.0" in bp_path.read_text()


class TestScoreBucket:
    def test_programs_cluster_by_signature(self):
        config = cfg_lib.ProgramsDatabaseConfig(
            num_islands=1, score_bucket_precision=2)
        template = code_manipulation.text_to_program(
            "def priority(a, b):\n  return 0.0")
        db = programs_database.ProgramsDatabase(
            config, template, "priority")

        p1 = code_manipulation.text_to_function(
            "def priority(a, b):\n  return a\n")
        p2 = code_manipulation.text_to_function(
            "def priority(a, b):\n  return b\n")

        # Same signature (rounded to 2 places)
        db.register_program(p1, island_id=0, scores_per_test={0: -0.123})
        db.register_program(p2, island_id=0, scores_per_test={0: -0.124})

        # Both should land in the same cluster (signature = (-0.12,))
        island = db._islands[0]
        assert len(island._clusters) == 1


class TestSoftmax:
    def test_softmax_valid(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = programs_database._softmax(logits, temperature=1.0)
        assert len(probs) == 3
        assert np.isclose(sum(probs), 1.0)
        assert np.all(probs >= 0)

    def test_softmax_temperature(self):
        logits = np.array([0.0, 0.0, 0.0])
        probs = programs_database._softmax(logits, temperature=1.0)
        assert np.allclose(probs, [1/3, 1/3, 1/3])

    def test_softmax_rejects_non_finite(self):
        with pytest.raises(ValueError):
            programs_database._softmax(
                np.array([1.0, float("nan"), 3.0]), temperature=1.0)
