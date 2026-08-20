# Construct Collapse in Scalar Preference Evaluation

A synthetic contextual-bandit simulation testing whether scalar (weighted-sum)
preference aggregation can conceal epistemically important behavior — specifically,
whether a policy can achieve a high aggregate reward while systematically hiding
higher rates of confident wrongness under insufficient evidence, and whether a
stratified / CRS-style (constrained) reward regime avoids this.

Full results and interpretation are in [`RESULTS.md`](./RESULTS.md). This README
covers setup and reproduction.

## Status

8 experiments run so far (see `RESULTS.md` for detail): primary scalar-vs-CRS-vs-oracle
comparison, a smoothness-weight Pareto sweep, a policy-class robustness check (MLP vs.
linear), a constraint-ablation, a noisy-evaluator ablation, a τ-sweep, a
non-convex-frontier analysis explaining *why* the scalar regime collapses sharply
while CRS behaves as a smooth dial, and an exact Bayes-optimal policy check that
eliminates training-algorithm artifacts entirely and pins the collapse threshold
to a closed-form value (w_s* = 2/3.5). All core claims have held up under every
stress test run so far.

**Open / not yet run:** distribution-shift test (train under one evidence-proportion
mix, evaluate under another); the natural-language LLM extension (real model outputs
+ rubric judges) described as an optional follow-up in the original experiment design.

**Publication plan:** core findings are being scoped into an 8-page submission for
the TAE (Trust-AI-Eval) NeurIPS 2026 workshop (deadline Aug 29, 2026); deeper
material here (non-convexity theory, exact-optimal-policy validation) is earmarked
for a longer companion paper, not blocked by the workshop deadline since TAE
submissions are non-archival.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e .
```

## Reproducing the experiments

Each script is deterministic (seeded) and writes CSVs to `results/`. Some are slow
enough that they checkpoint incrementally and are safe to re-run (they skip
already-completed work).

```bash
uv run python scripts/run_primary.py       # Section 1: scalar vs CRS vs oracle, 10 seeds
uv run python scripts/run_sweep.py         # Section 2: linear-policy Pareto sweep over w_s, 50 seeds
uv run python scripts/run_mlp_sweep.py     # Section 5: MLP policy-class robustness check (checkpointed -- re-run to resume)
uv run python scripts/run_ablations.py     # Section 3-4: constraint ablation + noisy-evaluator ablation
uv run python scripts/run_tau_sweep.py     # Section 6: CRS epistemic-floor tau sweep, 30 seeds
uv run python scripts/run_exact_check.py   # Section 8: exact Bayes-optimal policy, no training loop, N=500k
```

Figures in `figures/` were generated from these results with ad hoc plotting code
(not currently checked in as scripts — regenerate-on-demand if that's wanted).

## Repo layout

```
src/construct_collapse/
  sim.py       # core: data generation, reward tables, linear policy, REINFORCE + CRS Lagrangian training, eval
  mlp_sim.py   # MLP policy (manual backprop + Adam) for the policy-class robustness check
  analytic.py  # exact Bayes-optimal policy (closed-form posterior, no training loop)
scripts/       # one driver script per experiment section, writes to results/
results/       # csv outputs, raw + aggregated
figures/       # png plots referenced from RESULTS.md
RESULTS.md     # full write-up: methodology, results, and honest caveats per section
```

## A design note on the reward table

`sim.py`'s `epistemic_reward()` has 6 of its 12 (evidence × action) cells marked
`DOC` (taken directly from the source experiment design doc) and 6 marked `FILL`
(extrapolated to complete the grid). The `FILL` cells are a judgment call, not
something derived from the original spec — worth a sanity check against intended
semantics before citing exact numbers from this repo in the paper.

## Caveats

This is a synthetic simulation demonstrating a *possible and reproducible* failure
mode under one transparent, adjustable model of evaluator preferences — it is not
evidence about what any specific real RLHF pipeline does. Numbers here (e.g. the
2.8x selective-risk gap, the w_s≈0.52-0.55 collapse threshold) are properties of
this reward table and feature set, not universal constants. See `RESULTS.md` for
the full scope discussion.
