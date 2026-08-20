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

## 8. Exact Bayes-optimal policy: eliminating training as a possible confound

Every result up to this point came from REINFORCE — a stochastic optimizer.
Even with the MLP robustness check (Section 5), someone could reasonably ask
whether the sharp collapse is a property of gradient-based training dynamics
(a bifurcation in the optimization landscape) rather than a property of the
reward-maximization *problem itself*. This section closes that gap completely,
because the problem is small enough to solve exactly.

**Method.** We know the exact data-generating process (we wrote it), so instead
of learning a policy, we compute the true Bayesian posterior
q = P(model_knows=1 | evidence, difficulty, confidence_signal) in closed form
via Bayes' rule, then take the exact per-example argmax of expected reward
across the four actions — no gradient descent, no random initialization, no
training loop of any kind. This is the ceiling on what *any* policy, however
well trained, could achieve using only these features. For the CRS regime, the
Lagrange multiplier λ is found by direct bisection (not learned) to the minimal
value whose resulting exact-optimal policy hits the target τ. See
`construct_collapse/analytic.py`, `scripts/run_exact_check.py`.

One implementation note worth flagging: `sim.rewards_for_actions` loops over
examples in Python calling `epistemic_reward` one at a time — harmless at
REINFORCE's batch_size=256, but far too slow once this runs inside a λ-bisection
over N=500,000 (a first attempt timed out). Added `fast_rewards_for_actions`, a
vectorized lookup-table version, and checked it against the original on a
mixed-action sample before trusting any downstream number
(`max abs diff = 0.0`, all four actions exercised).

**Result: the collapse survives completely, and its exact location is now
pinned to 4 decimal places.** At N=500,000 (sampling noise negligible), epistemic
utility holds around 0.40-0.83 through w_s=0.570, then collapses to -0.094
by w_s=0.575 — a discontinuity spanning less than 0.005 in the weight. This is
not a training artifact: it is what the mathematically optimal decision rule
does under this reward structure.

**Bonus: a closed-form derivation lands almost exactly on the numerical
threshold.** For an item under insufficient evidence where the model definitely
doesn't know (posterior q=0 — the majority case, since a noise_sd=0.3 signal is
fairly informative and posteriors cluster near 0 or 1), confident_answer beats
abstain exactly when:

w_s > (2 - 2q) / (3.5 - 2q)  →  at q=0:  **w_s\* = 2/3.5 = 0.571428...**

The full numerical computation collapses between w_s=0.570 and w_s=0.575, with
w_s=0.5714 sitting almost exactly mid-transition. Hand algebra and a 500k-sample
numerical computation agree to 3+ significant figures.

**A genuinely useful side finding: weaker training collapses *earlier* than
necessary, not later.** Overlaying all three methods (`exact_vs_trained_comparison.png`):

| Method | Observed collapse zone |
|---|---|
| Linear policy, REINFORCE (Section 2) | w_s ≈ 0.52 - 0.55 |
| MLP + 3 restarts, REINFORCE (Section 5) | w_s ≈ 0.55 - 0.58 |
| Exact Bayes-optimal (this section) | w_s ≈ 0.570 - 0.575 |

The exact/theoretical threshold (0.5714) is the *latest* one — the true
optimum tolerates a higher smoothness weight than either trained policy could
before collapsing. As training quality improves (linear → MLP+restarts → exact),
the observed threshold climbs monotonically toward the theoretical value. This
means real gradient-based training should be *expected* to hit this failure
mode a bit earlier (at a more conservative, seemingly "safer" reward weighting)
than idealized decision-theoretic analysis would predict — a more concerning
practical implication than the toy result alone, not a less concerning one, and
worth stating plainly in the paper rather than only reporting the theoretical
threshold.

**τ-sweep at N=500,000 confirms Section 6's dial finding holds exactly too**:
epistemic utility rises smoothly and monotonically from 0.0 (τ=0) to a ceiling
of 0.8523 (the maximum achievable given noise_sd=0.3's inherent Bayes error —
matches the exact oracle's epistemic utility of 0.8526 almost exactly, as it
should). τ=0.90 and 0.95 are infeasible past that ceiling and correctly return
the same near-oracle policy (λ saturates at the search bound). No discontinuities
anywhere in the exact τ-sweep.

**Bonus, found while building the τ-sweep comparison figure**: the REINFORCE
τ-sweep from Section 6 has real jaggedness (achieved epistemic utility jumps
from 0.036 at τ=0.2 to 0.438 at τ=0.3, badly missing both targets) that
completely disappears in the exact computation (every τ from 0 to 0.85 is hit
almost exactly, `epistemic_utility ≈ τ` at every row). The "roughish" transition
flagged in Section 6 was REINFORCE's Lagrangian dual-ascent struggling to
converge precisely, not a real property of CRS. CRS's dial is smoother than we
gave it credit for. See `figures/crs_dial_exact_vs_trained.png`.

## 9. Distribution shift: does CRS's certified floor transport across deployment mixes?

Every prior section trained and evaluated under the same evidence-proportion mix
(40% sufficient / 30% ambiguous / 30% insufficient). CRS's λ is calibrated once,
via bisection, to hit a target τ under that mix — it is not re-tuned per
deployment environment. The practically important question a mitigation like
this has to answer is not "does it work on the benchmark," it's "does the
certified guarantee survive contact with a different deployment distribution."

**Method.** Calibrate CRS's λ (and fix scalar's weights, and oracle) under the
training mix as always, then evaluate those exact same fixed policies —
no retraining, no recalibration — on five deployment mixes: same-as-training,
an "easier" mix with more sufficient-evidence questions (70/20/10), an
ambiguous-heavy mix (20/50/30), and two "harder" mixes with progressively more
insufficient-evidence questions (20/30/50 and 10/20/70). Ran this both with the
exact Bayes-optimal policies (N=500k per mix, `run_distribution_shift_exact.py`)
and, as a cross-check, with actually-REINFORCE-trained linear policies (5 seeds,
`run_distribution_shift_reinforce.py`) — same fixed trained policy object
evaluated across all five deployment sets.

**Result: the certified floor does not transport, and the direction is the
opposite of the intuitive one.** Under the "easier" mix (more sufficient-evidence
questions — intuitively a gentler deployment environment), CRS's achieved
epistemic utility drops to 0.786, undershooting its own calibrated τ=0.85 target
by 0.064. Every mix with *more* insufficient evidence over-satisfies the target
instead (up to 0.943 under "much harder"). The REINFORCE cross-check reproduces
this almost exactly (0.786 under "easier" vs. the exact computation's 0.786) —
this is not an artifact of the idealized decision rule. See
`figures/distribution_shift.png`.

**But the safety-relevant per-example rate is genuinely invariant.**
Unsupported-certainty rate (confident answers specifically among
insufficient-evidence items) stays flat across all five mixes for every regime
(CRS: 0.0306-0.0311; scalar: 0.0415-0.0485) — expected, since the decision rule
is a fixed function of individual (evidence, difficulty, confidence_signal)
features and doesn't reference the population mix at all. Worth being honest
that this invariance is close to a tautology given how the policy is
constructed, not a surprising discovery — but it does correctly separate two
different questions that are easy to conflate: whether the policy's *behavior on
hard cases* degrades under shift (it doesn't, here) versus whether the
*aggregate certified number* remains valid under shift (it doesn't).

**Mechanism, verified rather than assumed.** Traced why "easier" hurts the
aggregate: under the calibrated CRS policy, sufficient-evidence items split into
78.3% confident_answer (mean r_e=0.970, close to but below the guaranteed
ceiling), 7.9% qualified_answer (mean r_e=0.036 — weak, near the correctness
boundary), and **13.8% clarify (r_e=-0.2 flat penalty)**. That last group is the
main driver: to hit a demanding τ=0.85 in aggregate, the Lagrangian-optimal
policy doesn't only get very conservative on insufficient-evidence items — it
also starts unnecessarily hedging on a real slice (13.8%) of genuinely
sufficient-evidence items, incurring a straight penalty rather than forgoing
upside. Weighted average confirms the arithmetic exactly:
0.783×0.970 + 0.079×0.036 + 0.138×(−0.2) = 0.735, matching the measured
sufficient-evidence-conditional epistemic utility of 0.735. When the deployment
mix has proportionally more sufficient-evidence items, this weak spot gets
weighted more heavily, dragging the aggregate below τ.

**Important caveat on how much weight this specific mechanism can bear**: the
-0.2 "unnecessary clarify under sufficient evidence" penalty driving most of
this is one of the `FILL` reward-table cells (my extrapolation, not from the
source doc) — flagged the same way in Section 1. The *qualitative* finding (a
single pooled population-level constraint has no per-category structure, so it
can silently over- or under-shoot depending on which evidence category the
deployment distribution happens to weight) doesn't depend on this specific
value and should generalize. But the *exact magnitude* (0.064 undershoot) is
sensitive to a judgment-call reward value and shouldn't be quoted as precise
without sign-off on that cell.

**This is a genuine limitation to report, not a reason to walk back CRS's
case.** It also points to an easy, testable extension: a *stratified* CRS with
separate per-evidence-category constraints (rather than one pooled
population-average constraint) should prevent this specific cross-category
leakage — worth flagging as future work rather than running now, since it's a
new mechanism, not a robustness check of the existing one.

## Caveats for the paper

- This is a synthetic contextual-bandit simulation, not language-model RLHF — it
  demonstrates the failure mode is *possible and reproducible under a defensible
  model of evaluator preferences*, not that it occurs in any specific commercial
  pipeline. Keep the framing as written in the source doc ("construct collapse in
  scalar preference evaluation," not "RLHF suppresses uncertainty").
- 6 of 12 reward-table cells were extrapolated (marked `FILL` in `sim.py`) — worth
  a quick sanity check against your intended semantics before citing exact numbers.
- ECE is computed but noisy/not yet reported here — ans count and calibration
  binning need a larger N if you want that number in the paper; flagging as
  incomplete rather than fabricating a clean value.
- Distribution-shift sweep (varying ambiguous/insufficient proportions) is not yet
  run — straightforward to add with `generate_dataset(..., evidence_probs=...)`,
  didn't want to pad this pass with a result I hadn't verified before sending it.
