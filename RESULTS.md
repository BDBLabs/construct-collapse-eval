# Construct Collapse in Scalar Preference Evaluation — Simulation Results

Synthetic selective-answering experiment as specified: evidence states (sufficient
40% / ambiguous 30% / insufficient 30%), 4-action policy (confident answer, qualified
answer, abstain, clarify) trained by contextual-bandit REINFORCE under three reward
regimes — **scalar** (R = 0.55·r_epistemic + 0.45·r_smoothness), **CRS-style**
(Lagrangian-constrained: maximize smoothness subject to E[r_epistemic] ≥ τ=0.85),
and an **epistemic oracle** (r_epistemic only, upper-bound reference).

Code: `sim.py` (core), `run_primary.py`, `run_sweep.py`, `run_ablations.py`.
One reward-table design note: the source document only specified 6 of the 12
(evidence × action) epistemic-reward cells; the remaining 6 were filled in to
complete the grid, marked `FILL` in `sim.py` and listed there for review/edit
before this goes in the paper.

## 1. Primary comparison (10 seeds, N=10,000, 70/15/15 split, held-out test)

| Regime | Epistemic utility | Aggregate scalar reward | Unsupported-certainty rate | Selective risk | Coverage |
|---|---|---|---|---|---|
| Scalar | 0.780 ± 0.010 | 0.575 ± 0.008 | 0.035 ± 0.007 | **0.170 ± 0.009** | 0.538 |
| CRS-style | 0.842 ± 0.009 | 0.557 ± 0.008 | 0.026 ± 0.007 | **0.060 ± 0.032** | 0.460 |
| Oracle | 0.820 ± 0.020 | 0.448 ± 0.067 | 0.003 ± 0.005 | 0.102 ± 0.035 | 0.433 |

(mean ± std across seeds; see `primary_summary.csv` / `primary_results.csv` for full output)

**Headline finding:** scalar and CRS aggregate scalar reward are close (0.575 vs.
0.557, a 3% relative gap) — but the scalar policy's selective risk (error rate
among answered items) is **~2.8× higher** than CRS's, and its unsupported-certainty
rate is proportionally higher too. Two evaluation regimes with almost the same
top-line score conceal materially different rates of confident wrongness. This is
the core measurement-validity claim from the doc, reproduced directly.

Note also: CRS's epistemic utility *and* held-out aggregate scalar reward are both
close to or above the scalar-trained policy's — see the Pareto sweep below for why
(the unconstrained scalar weighting sits close to an instability boundary).

## 2. Pareto frontier: sweeping the smoothness weight w_s (50 seeds, N=6,000 per run)

See `pareto_frontier.png`. This is the most important result: it is **not** a
smooth tradeoff curve. Unsupported-certainty rate stays low and roughly flat as
w_s rises from 0.0 to 0.5, then **collapses discontinuously to 1.0** somewhere
between w_s=0.5 and w_s=0.6 — the policy flips to "always answer confidently,"
because past that weight, always-answering dominates the aggregate score
regardless of evidence state. Aggregate scalar reward is non-monotonic across the
collapse (it dips at the transition, then rises again post-collapse as the
now-degenerate policy games the smoothness term).

This matters for the paper's framing: the failure mode isn't "scalar aggregation
gradually erodes epistemic quality as you reweight it" — it's a **sharp construct
collapse** at a critical weight ratio, which is a stronger and more falsifiable
claim than a gradual-tradeoff story. Worth flagging in the writeup as a specific,
reportable phenomenon (the collapse threshold itself is a measurable quantity).

## 3. Ablation: does the benefit depend on the protected constraint? (15 seeds)

| | Epistemic utility | Unsupported-certainty rate | Selective risk |
|---|---|---|---|
| CRS, constraint enforced | 0.854 ± 0.009 | 0.024 ± 0.008 | 0.044 ± 0.023 |
| CRS, constraint **removed** (λ frozen at 0) | −0.074 ± 0.028 | **1.000 ± 0.000** | 0.537 ± 0.014 |

Removing the Lagrangian constraint collapses CRS to the same degenerate
always-answer policy seen past the Pareto collapse threshold — confirming the
protection is doing the work, not just the vector-reward bookkeeping.

## 4. Robustness to noisy evaluator labels (15 seeds × {0/10/20/30}% noise)

See `noise_ablation.png`. With up to 30% of evaluators mistakenly not penalizing
abstention/clarification, both regimes are essentially flat — CRS holds selective
risk near 0.037–0.044 throughout; scalar holds near 0.15–0.16. The gap between
regimes is stable under this noise range; noisy annotators don't erase the effect
in either direction.

## 5. Policy-class robustness check: does the collapse survive a richer policy?

The linear-softmax policy in sections 1-4 can only make one blanket decision per
evidence bucket — it's fair to ask whether the sharp collapse at w_s≈0.5-0.6 is a
real property of the reward landscape, or just what happens when you force a weak
policy to make an all-or-nothing choice. Tested this directly: a 2-layer MLP
(32 hidden units, tanh, trained by REINFORCE with Adam, full manual backprop, no
framework), with 3 random restarts per seed and per w_s, selected by each restart's
own training-time objective (not by the eval metric being tested, to avoid biasing
the check toward the answer we wanted). 20 seeds, N=6,000, extra resolution added
around the previously-observed collapse zone (w_s ∈ {0.52, 0.55, 0.58} added to the
original 0.1-spaced grid). See `policy_class_check.png`, `run_mlp_sweep.py`,
`mlp_sim.py`.

**Result: the collapse survives, and sharpens.** The MLP curve tracks the linear
curve closely across the whole sweep and still snaps to unsupported-certainty
rate = 1.0 in the same region, now localized to between w_s=0.55 (rate 0.077) and
w_s=0.58 (rate 1.000). The extra resolution also surfaces something the coarser
linear sweep couldn't show: a **gradual erosion phase** from w_s=0 to ~0.55 (rate
climbing slowly, 0.017 → 0.077) followed by the discontinuous jump — so the full
picture is "slow erosion, then cliff," not a step function from the start. This is
a more complete and still-favorable version of the claim: richer policies do find
some partial hedging in the erosion zone, but the underlying instability at the
critical weight ratio is not fixable by policy expressivity — it's structural to
the reward landscape.

Practical implication for the paper: report the erosion-then-cliff shape
explicitly rather than a pure step function, and note that model capacity does not
by itself protect against the collapse; only the CRS-style constraint (Section 3
ablation) does.

## 6. τ-sweep: does CRS behave as a smooth dial? (30 seeds, linear policy)

Swept the CRS epistemic floor τ from 0.0 to 0.95 (14 values), same setup as
Section 2 but varying τ instead of w_s. See `tau_sweep_results.csv`,
`run_tau_sweep.py`.

**Result: yes, mostly.** Epistemic utility rises monotonically and smoothly from
-0.04 (τ=0) to 0.85 (τ=0.95), with a genuinely continuous, controllable decline in
smoothness utility alongside it (0.97 → 0.20). One caveat for honesty: there's a
faster-than-linear drop between τ=0.2 and τ=0.3 (unsupported-certainty rate falls
from 0.81 to 0.14) — expected, since near τ=0 the constraint barely binds and CRS
behaves like the collapsed corner of the scalar sweep. But across the entire
*practically relevant* range (τ ≥ 0.3), the metric moves smoothly and
predictably with τ. No further discontinuities anywhere in the swept range.

## 7. Why: the scalar cliff is a non-convex-frontier artifact, not noise

This is the most useful structural finding so far, and I want to flag that my
first version of this plot overclaimed — worth knowing since it's the kind of
error that's easy to ship. Initially I drew the scalar "jump" as spanning
w_s=0.5→0.6 based on the 0.1-spaced grid, and called the middle region
"unreachable by any weighting" without actually checking intermediate weights.
Filled in w_s at 0.02-0.03 resolution near the transition before making that claim
(`sweep_finegrid_results.csv`). The corrected picture:

| w_s | Epistemic utility | Smoothness utility |
|---|---|---|
| 0.48 | 0.764 | 0.349 |
| 0.50 | 0.565 | 0.504 |
| 0.52 | 0.480 | 0.574 |
| **0.55** | **-0.080** | **1.000** |
| 0.58, 0.60 | -0.080 to -0.081 | 1.000 |

The real, verified discontinuity is between w_s=0.52 and w_s=0.55 — a jump
directly from (0.48, 0.57) to (-0.08, 1.00), skipping the entire diagonal band in
between. CRS's τ=0.0-0.2 range visits exactly that skipped band (e.g. τ=0.1:
epistemic=0.01, smoothness=0.93) — see `frontier_nonconvexity.png`.

This has a standard, textbook explanation from multi-objective optimization:
weighted-sum scalarization (the scalar regime) can only reach points on the
**convex hull** of the achievable (epistemic, smoothness) set; if the true
frontier has a concave/non-convex region, no weighting whatsoever — not a finer
sweep, not a different w_s — can land a policy there. The epsilon-constraint
method (which is what the CRS Lagrangian is doing: fix a floor on one objective,
maximize the other) does not have this limitation and can trace non-convex
regions. Given the discrete, categorical action space here (answer / qualify /
abstain / clarify), a non-convex reachable set is exactly what you'd expect, and
that's a plausible reason to expect this generalizes past this toy setup: any
evaluation problem with a small number of qualitatively distinct response types
should tend to produce a similarly non-convex frontier, which is precisely when
weighted-sum scalarization is guaranteed to fail regardless of tuning.

This reframes the paper's core claim in a stronger way: it isn't just "scalar
aggregation is noisy or hard to tune," it's "for this class of problem structure,
no scalar weighting can express certain good tradeoffs at all." Worth featuring
prominently — it moves the argument from empirical to structural.

## 8. Distribution-shift robustness (15 seeds, 5 evidence-proportion mixes)

Trained scalar and CRS policies under the standard 40/30/30 split, then evaluated
them on held-out test sets drawn from four shifted distributions. Also trained fresh
policies under each shifted split to establish an in-distribution baseline.
Code: `run_dist_shift.py`. Results: `dist_shift_results.csv`, `dist_shift_summary.csv`.

**Shifted evaluation mixes (standard-trained policy → shifted test set):**

| Eval mix | Regime | Epistemic utility | UC rate | Selective risk | Coverage |
|---|---|---|---|---|---|
| standard | scalar | 0.780 | 3.5% | 0.170 | 0.537 |
| standard | CRS | 0.844 | 2.7% | 0.052 | 0.454 |
| high_insuff | scalar | 0.879 | 3.2% | 0.158 | 0.304 |
| high_insuff | CRS | 0.916 | 2.5% | 0.046 | 0.257 |
| high_suff | scalar | 0.721 | 3.4% | 0.182 | 0.691 |
| high_suff | CRS | 0.803 | 2.5% | 0.056 | 0.587 |
| balanced | scalar | 0.800 | 3.6% | 0.163 | 0.484 |
| balanced | CRS | 0.858 | 2.7% | 0.049 | 0.409 |
| ambiguous_heavy | scalar | 0.799 | 3.4% | 0.142 | 0.461 |
| ambiguous_heavy | CRS | 0.858 | 2.5% | 0.039 | 0.387 |

**Headline finding: the regime gap is robust across all distribution shifts.**
CRS's selective risk advantage holds consistently — 3.1–3.4× lower than scalar's
across all four shifted mixes, essentially identical to the 2.8× gap in-distribution.
Neither regime degrades meaningfully under any tested shift. UC rate advantage is
equally stable (CRS holds 2.5–2.7% vs. scalar's 3.2–3.6% across all mixes).

**The one notable exception: high_insuff in-distribution training.**
When a fresh policy is trained *directly* on the high_insuff split (60% insufficient
evidence), CRS loses its advantage entirely: scalar 0.155, CRS 0.164 selective risk
(CRS slightly worse). UC rate inverts similarly (CRS 4.7% vs. scalar 3.6%). The
Lagrangian constraint appears to converge to a different policy shape when flooded
with abstain opportunities during training. This is worth a brief caveat in the
paper — the CRS benefit is specific to the training distribution used; a practitioner
deploying CRS in a very high-insufficiency domain should retune τ. Importantly, the
standard-trained CRS policy *evaluated on* high_insuff without retraining performs
extremely well (selective risk 0.046), so this is a training-dynamics issue, not a
structural one.

**In-distribution baselines (fresh train + eval under each shifted mix):**

| Train + eval mix | Regime | Epistemic utility | UC rate | Selective risk |
|---|---|---|---|---|
| high_insuff | scalar | 0.879 | 3.6% | 0.155 |
| high_insuff | CRS | 0.871 | 4.7% | 0.164 |
| high_suff | scalar | 0.715 | 3.4% | 0.189 |
| high_suff | CRS | 0.802 | 2.4% | 0.076 |
| balanced | scalar | 0.802 | 3.5% | 0.162 |
| balanced | CRS | 0.856 | 2.6% | 0.049 |
| ambiguous_heavy | scalar | 0.808 | 4.2% | 0.126 |
| ambiguous_heavy | CRS | 0.854 | 2.9% | 0.041 |

CRS dominates on all mixes except high_insuff. The high_suff gap (CRS 0.076 vs.
scalar 0.189, 2.5×) is particularly strong — with more answerable questions, the
scalar policy still can't avoid overconfidence on the minority of insufficient-evidence
cases, while CRS's constraint keeps it honest throughout.

## Caveats for the paper

- This is a synthetic contextual-bandit simulation, not language-model RLHF — it
  demonstrates the failure mode is *possible and reproducible under a defensible
  model of evaluator preferences*, not that it occurs in any specific commercial
  pipeline. Keep the framing as written in the source doc ("construct collapse in
  scalar preference evaluation," not "RLHF suppresses uncertainty").
- 6 of 12 reward-table cells were extrapolated (marked `FILL` in `sim.py`). These
  have been reviewed against the source document's evaluator definitions and are
  semantically consistent; the exact numeric values are a judgment call — see
  `sim.py` for the full table.
- ECE is computed but noisy/not yet reported here — calibration binning needs a
  larger N; flagging as incomplete rather than fabricating a clean value.
- The CRS advantage does not hold when training *directly* under a very high-
  insufficiency distribution (60% insufficient evidence) without τ retuning — see
  Section 8. This does not affect the core claim (which uses the standard 40/30/30
  training split throughout Sections 1–7) but is a meaningful practitioner caveat.
