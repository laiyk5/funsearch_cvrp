"""Tests for the funsearch module — current public API."""

from absl.testing import absltest
from absl.testing import parameterized

from funsearch_cvrp.funsearch import code_manipulation
from funsearch_cvrp.funsearch import config
from funsearch_cvrp.funsearch import funsearch


class FunsearchTest(parameterized.TestCase):

    def test_module_exports_main(self):
        self.assertTrue(callable(funsearch.main))

    def test_code_manipulation_text_to_program(self):
        src = '''def priority(a, b):
      """Test."""
      return a + b
    '''
        program = code_manipulation.text_to_program(src)
        self.assertLen(program.functions, 1)
        self.assertEqual(program.functions[0].name, "priority")
        self.assertEqual(program.functions[0].args, "a, b")
        self.assertIn("return a + b", program.functions[0].body)

    def test_code_manipulation_text_to_function(self):
        src = "def foo(x, y):\n  return x * y"
        fn = code_manipulation.text_to_function(src)
        self.assertEqual(fn.name, "foo")
        self.assertEqual(fn.args, "x, y")

    def test_config_defaults(self):
        db_config = config.ProgramsDatabaseConfig()
        self.assertEqual(db_config.num_islands, 10)
        self.assertEqual(db_config.functions_per_prompt, 3)


if __name__ == "__main__":
    absltest.main()
