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
import json
import logging as _logging
import textwrap
from pathlib import Path

import numpy as np

from . import evaluator
from . import programs_database

_logger = _logging.getLogger('funsearch')


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, *, sampler_log_path: Path | None = None) -> None:
    self._samples_per_prompt = samples_per_prompt
    self._sampler_log_path = sampler_log_path

  def _write_sampler_log(self, record: dict) -> None:
    if self._sampler_log_path is None:
      return
    with open(self._sampler_log_path, "a", encoding="utf-8") as f:
      f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
      base_url: str | None = None,
      prompt: str | None = None,
      *,
      sampler_log_path: Path | None = None,
  ) -> None:
    super().__init__(samples_per_prompt, sampler_log_path=sampler_log_path)
    self._model = model
    self._temperature = temperature
    self._max_tokens = max_tokens
    self._api_key = api_key
    self._base_url = base_url
    self._prompt = prompt

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    import httpx
    import time as _time
    from openai import OpenAI

    generation_time = _time.time()

    client_kwargs = {}
    if self._api_key:
      client_kwargs["api_key"] = self._api_key
    if self._base_url:
      client_kwargs["base_url"] = self._base_url

    client_kwargs["http_client"] = httpx.Client(trust_env=False)
    client = OpenAI(**client_kwargs)

    system_msg = "You are an expert Python programmer. Complete the given function with an improved implementation."
    if self._prompt:
      system_msg += "\n\n" + self._prompt

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    _logger.debug('LLM REQUEST model=%s temp=%.2f max_tokens=%d', self._model, self._temperature, self._max_tokens)
    _logger.debug('LLM PROMPT:\n%s', prompt)

    response = client.chat.completions.create(
        model=self._model,
        messages=messages,
        temperature=self._temperature,
        max_tokens=self._max_tokens,
    )
    raw_response = response.choices[0].message.content
    _logger.debug('LLM RAW RESPONSE:\n%s', raw_response)

    # Strip markdown code fences if present
    raw = raw_response
    if "```" in raw:
        lines = raw.splitlines()
        inner = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(inner)
    # The LLM may have prefixed explanatory text; find the first `def ` line
    # and keep everything from there.  Also collect any `import` / `from`
    # statements that appear before the function definition — the LLM often
    # places them above the `def`.
    lines = raw.splitlines()
    imports = []
    def_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
        elif stripped.startswith("def "):
            def_idx = i
            break
    if def_idx is not None:
        raw = "\n".join(imports + lines[def_idx:])
    # If LLM returned a full function definition, extract just the body.
    # If trailing markdown/text causes parsing to fail, trim from the end
    # until it succeeds.
    from . import code_manipulation
    candidate = raw.strip()
    stripped_imports = []
    # The candidate may start with imports followed by a `def`; try parsing
    # it as a full function, falling back to extracting only the body.
    while candidate:
        try:
            fn = code_manipulation.text_to_function(candidate)
            raw = fn.body
            # Re-attach any imports that were stripped before the `def`.
            # The extracted body is indented, so imports must be indented too.
            if stripped_imports:
                raw = "\n".join("    " + imp for imp in stripped_imports) + "\n" + raw
            break
        except Exception:
            # If candidate starts with imports + a function definition,
            # try stripping the imports so we can parse just the def.
            candidate_lines = candidate.splitlines()
            if candidate_lines and candidate_lines[0].lstrip().startswith(("import ", "from ")):
                stripped_imports.append(candidate_lines[0])
                candidate = "\n".join(candidate_lines[1:])
            else:
                # Trim last line and try again
                if len(candidate_lines) <= 1:
                    break
                candidate = "\n".join(candidate_lines[:-1])
    # Normalize indentation: the LLM may use 4-space indent while the template
    # expects 2-space.  Dedent then re-indent to a consistent 2 spaces.
    extracted = textwrap.indent(textwrap.dedent(raw), '  ')
    _logger.debug('LLM EXTRACTED CODE:\n%s', extracted)

    self._write_sampler_log({
        "generation_time": generation_time,
        "model": self._model,
        "temperature": self._temperature,
        "max_tokens": self._max_tokens,
        "prompt": prompt,
        "raw_response": raw_response,
        "extracted_code": extracted,
    })

    return extracted
