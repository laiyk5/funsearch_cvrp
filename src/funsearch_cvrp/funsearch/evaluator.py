# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Callable, Sequence
import copy
import json
import logging as _logging
import time as _time
from pathlib import Path
from typing import Any

from . import code_manipulation
from . import programs_database

_logger = _logging.getLogger('funsearch')


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and its definition string."""
  body = _trim_function_body(generated_code)
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  # Return the Function object and a standalone function definition that can
  # be exec'd in isolation (the harness is supplied externally).
  return evolved_function, str(evolved_function).strip()


def _calls_ancestor(code: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(code):
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Sandbox:
  """Sandbox for executing generated code."""

  def run(
      self,
      evolved_fn_code: str,
      function_name: str,
      evaluate_fn: Callable,
      test_input: Any,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Execute `evolved_fn_code`, extract `function_name`, then call
    ``evaluate_fn(test_input, function)``.  Returns the result and whether
    execution succeeded."""
    raise NotImplementedError(
        'Must provide a sandbox for executing untrusted code.')


class SimpleSandbox(Sandbox):
  """Simple sandbox executing code in isolated namespace with timeout."""

  def run(
      self,
      evolved_fn_code: str,
      function_name: str,
      evaluate_fn: Callable,
      test_input: Any,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Exec evolved function code, then call evaluate_fn(test_input, func)."""
    import contextlib
    import io
    import signal

    namespace = {}

    try:
      ast.parse(evolved_fn_code)
    except SyntaxError:
      return None, False

    def timeout_handler(signum, frame):
      raise TimeoutError("Execution timeout")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
      exec(evolved_fn_code, namespace)

      if function_name not in namespace:
        return None, False

      func = namespace[function_name]

      with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
          result = evaluate_fn(test_input, func)

      signal.alarm(0)
      return result, True

    except Exception:
      return None, False
    finally:
      signal.signal(signal.SIGALRM, old_handler)
      signal.alarm(0)


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      evaluate_fn: Callable,
      inputs: Sequence[Any],
      timeout_seconds: int = 30,
      sandbox: Sandbox | None = None,
      *,
      eval_history_path: str | Path | None = None,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._evaluate_fn = evaluate_fn
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = sandbox if sandbox is not None else SimpleSandbox()
    self._eval_history_path: Path | None = (
        Path(eval_history_path) if eval_history_path else None)

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
      *,
      generation_time: float | None = None,
      iteration: int | None = None,
  ) -> None:
    """Compiles the sample into a function and evaluates it on test inputs."""
    eval_start = _time.time()
    new_function, evolved_fn_code = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)

    if _calls_ancestor(evolved_fn_code, self._function_to_evolve):
      self._maybe_write_eval_history(
          iteration=iteration, island_id=island_id,
          generation_time=generation_time, eval_time=eval_start,
          body=new_function.body, scores_per_test={},
          accepted=False, reject_reason='calls_ancestor')
      _logger.debug('[island=%s] REJECTED (calls ancestor)\n%s', island_id, new_function.body)
      return

    scores_per_test = {}
    any_failed = False
    for idx, current_input in enumerate(self._inputs):
      test_output, runs_ok = self._sandbox.run(
          evolved_fn_code,
          self._function_to_evolve,
          self._evaluate_fn,
          current_input,
          self._timeout_seconds,
      )
      if runs_ok and test_output is not None:
        if not isinstance(test_output, (int, float)):
          raise ValueError('evaluate_fn did not return an int/float score.')
        scores_per_test[idx] = test_output
      else:
        any_failed = True

    # Reject programs that crash on ANY instance — robustness is mandatory.
    if any_failed:
      _logger.debug('[island=%s] REJECTED (crash on instance %d)\n%s', island_id, idx, new_function.body)
      self._maybe_write_eval_history(
          iteration=iteration, island_id=island_id,
          generation_time=generation_time, eval_time=eval_start,
          body=new_function.body, scores_per_test={},
          accepted=False, reject_reason='instance_crash')
      return None

    if scores_per_test:
      # Use the last test instance as the reduced score, matching
      # programs_database._reduce_score — the last test is the
      # hardest / most discriminating instance.
      score = scores_per_test[list(scores_per_test.keys())[-1]]
      signature = tuple(round(scores_per_test[k], 2) for k in sorted(scores_per_test.keys()))
      _logger.debug('[island=%s] ACCEPTED score=%.4f sig=%s\n%s', island_id, score, signature, new_function.body)

      # Check whether this will become a new best.
      is_milestone = False
      if island_id is not None:
        prev_best = self._database._best_score_per_island[island_id]
        is_milestone = score > prev_best

      self._database.register_program(new_function, island_id, scores_per_test)
      self._maybe_write_eval_history(
          iteration=iteration, island_id=island_id,
          generation_time=generation_time, eval_time=eval_start,
          body=new_function.body, scores_per_test=scores_per_test,
          accepted=True, is_milestone=is_milestone)
      return scores_per_test

    _logger.debug('[island=%s] REJECTED (sandbox failed)\n%s', island_id, new_function.body)
    self._maybe_write_eval_history(
        iteration=iteration, island_id=island_id,
        generation_time=generation_time, eval_time=eval_start,
        body=new_function.body, scores_per_test={},
        accepted=False, reject_reason='sandbox_failed')
    return None

  def _maybe_write_eval_history(
      self,
      *,
      iteration: int | None,
      island_id: int | None,
      generation_time: float | None,
      eval_time: float,
      body: str,
      scores_per_test: dict,
      accepted: bool,
      is_milestone: bool = False,
      reject_reason: str = '',
  ) -> None:
    """Write an evaluation record to the history file."""
    if self._eval_history_path is None:
      return
    eval_end = _time.time()
    record: dict[str, Any] = {
        'eval_time': eval_end,
        'eval_duration_s': round(eval_end - eval_time, 3),
    }
    if iteration is not None:
      record['iteration'] = iteration
    if island_id is not None:
      record['island_id'] = island_id
    if generation_time is not None:
      record['generation_time'] = generation_time
      record['gen_to_eval_s'] = round(eval_end - generation_time, 3)
    record['accepted'] = accepted
    if not accepted and reject_reason:
      record['reject_reason'] = reject_reason
    record['is_milestone'] = is_milestone
    record['scores_per_test'] = {str(k): v for k, v in scores_per_test.items()}
    record['body'] = body
    with open(self._eval_history_path, 'a', encoding='utf-8') as f:
      f.write(json.dumps(record, ensure_ascii=False) + '\n')
