FunSearch Approach
==================

FunSearch uses **LLM + evolutionary search** to automatically discover heuristic
algorithms for CVRP, without hand-crafting rules.

Limitations of Classical Methods
---------------------------------

- Require hand-crafted heuristic rules
- Poor generalisation across problem scales
- Difficult to escape local optima

How FunSearch Works
-------------------

FunSearch treats algorithm discovery as a program synthesis problem. Instead of
tuning parameters, it evolves the *code itself*:

1. **LLM generation**: sample candidate algorithm code from a prompt
2. **Evaluation**: score candidates on benchmark instances
3. **Evolutionary selection**: register high-scoring programs into the database
4. **Iteration**: repeat until a high-quality algorithm emerges

The key insight is that the LLM acts as a mutation operator — it reads existing
programs and writes improved versions, guided by the evolutionary pressure of the
programs database.

Specification Format
--------------------

A FunSearch run is driven by a *specification file* — a Python module with two
decorated functions:

.. code-block:: python

   @funsearch.evolve
   def priority(item, bins):
       """Return a priority score for placing `item` into each bin."""
       return -bins  # initial implementation

   @funsearch.run
   def evaluate(instances):
       """Score a batch of problem instances. Higher is better."""
       ...

- ``@funsearch.evolve`` marks the function the LLM will rewrite each iteration.
  The initial body serves as the seed program.
- ``@funsearch.run`` marks the evaluation harness. It receives a list of test
  instances and must return a scalar score (higher = better).

The evaluator calls the ``evolve`` function internally; FunSearch only replaces
that one function while keeping the rest of the harness fixed.

Programs Database
-----------------

The programs database is the core data structure that drives the evolutionary
search. It maintains a diverse population of programs and produces prompts for
the LLM.

Islands
^^^^^^^

The database is partitioned into **islands** (default: 10). Each island evolves
independently, which preserves diversity and prevents premature convergence to a
single local optimum.

Every new program generated from a prompt on island *i* is registered back into
island *i*. Programs added at the very start (the seed) are broadcast to all
islands simultaneously.

Clusters
^^^^^^^^

Within each island, programs are grouped into **clusters** by their *score
signature* — a tuple of per-test scores across all evaluation inputs:

.. code-block:: python

   signature = (score_on_test_0, score_on_test_1, ..., score_on_test_n)

Two programs land in the same cluster if and only if they produce identical
scores on every test input, meaning they are *behaviourally equivalent* even if
their code differs. Each cluster stores its representative score (the score on
the last test input) and all programs that share that signature.

When sampling a program from a cluster, shorter programs are preferred — they
are assigned higher probability via a softmax over negated lengths. This acts as
an implicit Occam's razor, favouring simpler implementations.

Cluster Sampling and Temperature Schedule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build a prompt, the island samples ``functions_per_prompt`` clusters (default:
3) using a **softmax over cluster scores**:

.. math::

   P(\text{cluster}_i) = \text{softmax}\!\left(\frac{s_i}{T}\right)

where :math:`s_i` is the cluster's score and :math:`T` is the current
temperature.

The temperature follows a **sawtooth schedule** that decreases within each
period and then resets:

.. math::

   T = T_0 \times \left(1 - \frac{n \bmod P}{P}\right)

where :math:`T_0` is the initial temperature, :math:`n` is the total number of
programs registered so far, and :math:`P` is the period length.

- **High temperature** (early in a period): near-uniform sampling → exploration
- **Low temperature** (late in a period): peaked on best clusters → exploitation

This oscillation prevents the search from locking onto a single cluster too
early while still exploiting good solutions as the period progresses.

Island Reset
^^^^^^^^^^^^

Every ``reset_period`` seconds (default: 600 s), the **bottom half** of islands
(ranked by best score) are reset:

1. The weak island is discarded and replaced with a fresh empty island.
2. A random island from the surviving top half is chosen as the *founder*.
3. The founder's best program is seeded into the new island.

This mechanism eliminates stagnant sub-populations while preserving the
diversity of the surviving islands. The new island starts fresh but with a
strong seed, giving it a head start.

Prompt Construction
-------------------

When the LLM is asked to generate a new program, it receives a prompt built from
the sampled programs. The prompt is structured as follows:

1. The full specification template (imports, harness code, type definitions).
2. The sampled programs, **sorted by score ascending** (worst first, best last),
   renamed to ``_v0``, ``_v1``, …, ``_v{N-1}``. Each function after ``_v0``
   gets a docstring ``"Improved version of _v{i-1}."``.
3. An **empty function header** ``_v{N}`` with docstring
   ``"Improved version of _v{N-1}."``, which the LLM must complete.

Example prompt structure:

.. code-block:: python

   # ... template preamble ...

   def priority_v0(item, bins):
       """Original heuristic."""
       return -bins

   def priority_v1(item, bins):
       """Improved version of `priority_v0`."""
       return -(bins - item)

   def priority_v2(item, bins):
       """Improved version of `priority_v1`."""
       # <LLM completes this>

Presenting programs in ascending score order gives the LLM a clear improvement
trajectory: it sees what was tried, how it improved, and is asked to continue
that trend.

Sampler Loop
------------

One or more **samplers** run concurrently (or sequentially in the single-process
implementation). Each sampler:

1. Calls ``database.get_prompt()`` to obtain a prompt and its ``island_id``.
2. Sends the prompt to the LLM and draws one or more code completions.
3. Parses each completion to extract the generated function body.
4. Sends each parsed function to the **evaluator**.

The sampler loop runs indefinitely until the configured number of iterations or
a time budget is exhausted.

Evaluator and Sandbox
---------------------

The evaluator receives a generated function and:

1. **Compiles** it — checks for syntax errors.
2. **Runs** it inside a ``SimpleSandbox``: an isolated ``exec`` namespace
   containing only ``math``, ``random``, and the problem-specific types.
   A ``SIGALRM`` timeout kills runs that exceed the per-call time limit.
3. **Scores** the function by calling the ``@funsearch.run`` harness on all
   test inputs.
4. **Registers** the scored function in the programs database under the
   originating ``island_id``.

Functions that raise exceptions, time out, or produce invalid solutions receive
a score of ``-inf`` and are discarded.

Configuration Reference
-----------------------

Key parameters in ``ProgramsDatabaseConfig``:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``num_islands``
     - 10
     - Number of independent sub-populations
   * - ``reset_period``
     - 600 s
     - Seconds between island resets
   * - ``functions_per_prompt``
     - 3
     - Programs sampled per prompt
   * - ``cluster_sampling_temperature_init``
     - 0.1
     - Initial softmax temperature :math:`T_0`
   * - ``cluster_sampling_temperature_period``
     - 30 000
     - Programs-registered period :math:`P` for temperature schedule

Evaluation Dimensions
---------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Description
     - Weight
   * - Average distance
     - Mean total route distance across all test instances
     - 1.0
   * - Route count
     - Number of vehicles used (fewer is better)
     - 20.0
   * - Stability
     - Standard deviation across problem scales
     - Tiebreaker
   * - Gap
     - Percentage gap to known optimal solution
     - Reference metric

Gap formula:

.. math::

   \text{Gap} = \frac{\text{solver distance} - \text{optimal distance}}{\text{optimal distance}} \times 100\%

Example: solver distance = 850, optimal = 784 → Gap = **8.4%**

References
----------

1. **Romera-Paredes et al. (2023)** — *"Mathematical discoveries from program search with large language models"* — FunSearch (Nature)
