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

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np

from . import evaluator
from . import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    raise NotImplementedError('Must provide a language model.')

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      llm: LLM,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = llm

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    while True:
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code)
      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)



class OpenAILLM(LLM):
  """OpenAI LLM implementation for FunSearch."""

  def __init__(
      self,
      samples_per_prompt: int,
      model: str = "gpt-4",
      temperature: float = 0.7,
      max_tokens: int = 1000,
      api_key: str | None = None,
  ) -> None:
    super().__init__(samples_per_prompt)
    self._model = model
    self._temperature = temperature
    self._max_tokens = max_tokens
    if api_key:
      import openai
      openai.api_key = api_key

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    import openai
    response = openai.chat.completions.create(
        model=self._model,
        messages=[
            {"role": "system", "content": "You are an expert Python programmer. Complete the given function with an improved implementation."},
            {"role": "user", "content": prompt},
        ],
        temperature=self._temperature,
        max_tokens=self._max_tokens,
    )
    return response.choices[0].message.content
