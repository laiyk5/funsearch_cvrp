Design Notes: Applying FunSearch to CVRP
=========================================

This page documents design decisions and trade-offs specific to applying
FunSearch to CVRP, based on analysis of the implementation.

Cluster Fragmentation from Many Test Cases
------------------------------------------

The programs database groups programs into clusters by their *score signature*
— a tuple of per-instance scores across all evaluation inputs. Two programs
share a cluster only if they produce identical scores on every test input.

With dozens of CVRP instances, this condition is almost never satisfied: nearly
every program ends up in its own cluster. This defeats the purpose of
clustering, because:

- The softmax sampling over clusters degenerates to near-uniform sampling over
  all past programs
- The exploration/exploitation dynamic driven by the temperature schedule
  collapses
- Evolutionary pressure weakens — there are no "good clusters" to exploit

**Mitigations:**

- **Bucket scores** — round each per-instance score before forming the
  signature, so programs with *similar* (not identical) performance co-cluster
- **Subsample test cases** — use a small fixed subset (5–8 representative
  instances) for the signature; use the full set only for the reduced score
- **Aggregate by scale** — use a coarser signature like
  ``(score_small, score_medium, score_large)`` bucketed by instance size

The original FunSearch paper used bin-packing with ~10 test inputs, which is
already borderline. For CVRP with dozens of instances, one of the above
mitigations is likely necessary.

Preventing Brute-Force Solutions
---------------------------------

FunSearch gives the LLM full freedom to write any Python code, so it could in
principle generate exhaustive search algorithms that find good solutions but are
computationally intractable.

Two mechanisms prevent this:

**Sandbox timeout**
  The ``SimpleSandbox`` kills any function that exceeds the per-call time limit
  via ``SIGALRM``. Set this aggressively (1–2 seconds per instance). A
  brute-force search over even 20 customers times out immediately and receives
  a score of ``-inf``, so it is never selected as a seed.

**Multi-scale evaluation**
  Evaluating on small, medium, and large instances catches accidentally
  super-linear code. A solution that passes the timeout on N=20 but blows up
  on N=100 scores poorly on large instances, dragging down its aggregate score
  and reducing its selection probability.

Together these create a natural selection pressure toward fast, general
heuristics rather than slow exact methods.

Evolving Only the Core Heuristic
----------------------------------

The LLM does not need to generate a complete CVRP solver. The ``@funsearch.evolve``
function should be the minimal *decision kernel* — a pure scoring function
called once per candidate, with no loops over all customers and no route
construction logic.

The full solver infrastructure (capacity checking, route construction, distance
calculation, score aggregation) lives in the ``@funsearch.run`` harness and is
never modified by the LLM.

Example decomposition:

.. code-block:: python

   @funsearch.evolve
   def score_candidate(customer_idx, current_load, capacity, distances, demands):
       """Score how desirable it is to visit this customer next."""
       return -distances[customer_idx]  # seed: nearest neighbor

   @funsearch.run
   def evaluate(instances):
       total = 0
       for inst in instances:
           routes = greedy_construct(inst, score_candidate)  # fixed harness
           total += compute_score(routes, inst)
       return -total  # higher is better

The LLM only rewrites ``score_candidate`` — a function that receives local
context (current load, distances, demands) and returns a scalar priority. The
greedy construction loop, capacity enforcement, and scoring all remain fixed.

This design has several advantages:

- **Prevents brute-force by construction** — the evolved function is called
  per candidate customer, not over the whole problem, so it cannot implement
  exhaustive search
- **Keeps the search space tractable** — the LLM is asked to improve a small,
  well-defined function rather than rewrite an entire solver
- **Separates concerns** — correctness guarantees (feasibility, capacity
  constraints) are enforced by the fixed harness, not the evolved code

The key design question is: *what is the minimal decision function that, if
improved, would improve solution quality?* That function goes in
``@funsearch.evolve``.
