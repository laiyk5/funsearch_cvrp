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

"""A single-threaded implementation of the FunSearch pipeline."""
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from . import code_manipulation
from . import config as config_lib
from . import evaluator
from . import programs_database
from . import sampler


def main(
    evolve_func: Callable,
    evaluate_fn: Callable,
    inputs: Sequence[Any],
    config: config_lib.Config,
    prompt: str | None = None,
    llm: sampler.LLM | None = None,
    sandbox: evaluator.Sandbox | None = None,
):
  """Launches a FunSearch experiment.

  Args:
    evolve_func: The seed function to evolve. Its source code is extracted via
      ``inspect.getsource()`` and used as the template for the programs
      database.
    evaluate_fn: Callable with signature ``evaluate_fn(test_input,
      evolved_func) -> float``.  This is your real evaluation harness (e.g.
      ``evaluate_cvrp``).  It receives one test input and the evolved function
      object, and must return a scalar score (higher = better).
    inputs: Sequence of test inputs. Each input is passed individually to
      ``evaluate_fn``.
    config: FunSearch configuration.
    prompt: Optional context string prepended to every LLM prompt. Use this
      to provide background on types, helper functions, and the problem domain.
    llm: Optional LLM instance. If None, an OpenAILLM is created from
      ``config.llm``.
    sandbox: Optional sandbox for executing evolved code. If None, a
      SimpleSandbox is used.
  """
  function_to_evolve = evolve_func.__name__

  # Extract source of the evolve function and build the template Program.
  source = inspect.getsource(evolve_func)
  template = code_manipulation.text_to_program(source)

  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve)

  # Create sandbox if not provided
  if sandbox is None:
    sandbox = evaluator.SimpleSandbox()

  evaluators = []
  for _ in range(config.num_evaluators):
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        evaluate_fn,
        inputs,
        sandbox=sandbox,
    ))

  # Send the initial implementation to be analysed by one evaluator.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)

  # Create LLM if not provided
  if llm is None:
    llm = sampler.OpenAILLM(
        samples_per_prompt=config.samples_per_prompt,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        prompt=prompt,
    )

  samplers = [sampler.Sampler(database, evaluators, llm)
              for _ in range(config.num_samplers)]

  for s in samplers:
    s.sample()
