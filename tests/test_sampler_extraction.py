"""Tests for LLM response extraction: code_manipulation, _trim_function_body, dedent."""

import textwrap

import pytest

from funsearch_cvrp.funsearch import code_manipulation
from funsearch_cvrp.funsearch.evaluator import _trim_function_body


class TestTrimFunctionBody:
    """Tests for _trim_function_body — the core extraction helper."""

    def test_preserves_simple_body(self):
        body = _trim_function_body("  import math\n  return 1.0")
        assert "import math" in body
        assert "return 1.0" in body

    def test_strips_empty_input(self):
        assert _trim_function_body("") == ""
        assert _trim_function_body("  ") == ""

    def test_trims_trailing_non_code(self):
        """If trailing markdown makes parsing fail, lines are trimmed."""
        result = _trim_function_body("  d = 1\n  return d\n\n### explanation")
        assert "###" not in result
        assert "d = 1" in result

    def test_handles_nested_code(self):
        body = _trim_function_body(
            "  if a > 0:\n    return a\n  return b")
        assert "if a > 0" in body
        assert "return a" in body
        assert "return b" in body


class TestTextToFunction:
    """Tests for code_manipulation.text_to_function."""

    def test_parses_simple_function(self):
        fn = code_manipulation.text_to_function(
            "def foo(x, y):\n  return x + y")
        assert fn.name == "foo"
        assert fn.args == "x, y"
        assert "return x + y" in fn.body

    def test_parses_function_with_imports(self):
        fn = code_manipulation.text_to_function(
            "import math\n\ndef calc(a, b):\n  return math.hypot(a, b)")
        assert fn.name == "calc"
        assert "math.hypot" in fn.body

    def test_parses_function_with_docstring(self):
        fn = code_manipulation.text_to_function(
            'def bar(x):\n  """doc."""\n  return x')
        assert fn.docstring == "doc."
        assert "return x" in fn.body

    def test_rejects_multiple_functions(self):
        with pytest.raises(ValueError, match="Only one function expected"):
            code_manipulation.text_to_function(
                "def a():\n  pass\n\ndef b():\n  pass")


class TestRenameFunctionCalls:
    """Tests for code_manipulation.rename_function_calls."""

    def test_renames_direct_call(self):
        result = code_manipulation.rename_function_calls(
            "  return priority(a, b)", "priority", "priority_v0")
        assert "priority_v0" in result
        assert "priority(" not in result

    def test_does_not_rename_unrelated(self):
        result = code_manipulation.rename_function_calls(
            "  return other_fn(a, b)", "priority", "priority_v0")
        assert result.strip() == "return other_fn(a, b)"


class TestIndentationNormalization:
    """Tests for textwrap.dedent + indent pipeline."""

    def test_dedent_reduce_4space(self):
        body = "    import math\n    d = 1\n    return d"
        dedented = textwrap.dedent(body)
        assert dedented == "import math\nd = 1\nreturn d"

    def test_reindent_to_2space(self):
        body = "    import math\n    d = 1"
        normalized = textwrap.indent(textwrap.dedent(body), "  ")
        lines = normalized.splitlines()
        for line in lines:
            if line.strip():
                assert line.startswith("  "), f"bad: {line!r}"

    def test_preserves_nested_blocks(self):
        body = "    if a > 0:\n        return a\n    return b"
        normalized = textwrap.indent(textwrap.dedent(body), "  ")
        lines = normalized.splitlines()
        # check nested indentation is preserved relative to base
        assert lines[1].startswith("    ")  # nested return a has 4-space


class TestGetFunctionsCalled:
    def test_finds_calls(self):
        called = code_manipulation.get_functions_called(
            "return foo(x) + bar(y)")
        assert "foo" in called
        assert "bar" in called

    def test_empty_for_no_calls(self):
        called = code_manipulation.get_functions_called("return x + 1")
        assert called == set()
