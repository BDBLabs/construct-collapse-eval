"""
Exact / Bayes-optimal decision rule for the selective-answering task.

Removes the training algorithm entirely as a possible confound. Instead of
learning a policy via REINFORCE (subject to local optima, restart luck,
learning-rate sensitivity, convergence noise), we compute the decision that
directly maximizes expected reward under the TRUE data-generating process:

  1. model_knows ~ Bernoulli(p_know(evidence, difficulty))          [ground truth]
  2. confidence_signal ~ Normal(model_knows, noise_sd^2)            [noisy observation]

A rational decision-maker who can observe (evidence, difficulty, confidence_signal)
but not model_knows directly should act on the exact Bayesian posterior
q = P(model_knows=1 | evidence, difficulty, confidence_signal), computed via
Bayes' rule with the known generative model above (closed form, since both the
prior and the likelihood are known exactly -- we wrote the generator).

Given q, the four actions have closed-form expected epistemic reward (smoothness
reward is action-only, so it's just a constant per action -- see sim.R_S_BY_ACTION):
  confident_answer : E[r_e] = q*(+1.0) + (1-q)*(-1.0) = 2q - 1
  qualified_answer  : E[r_e] = q*(+0.9) + (1-q)*(-0.4) = 1.3q - 0.4
  abstain / clarify : E[r_e] = constant given evidence state (reward doesn't
                       depend on correctness for these two actions)

The optimal action under scalar weights (w_e, w_s) is just the argmax of
w_e*E[r_e] + w_s*r_s across the four actions -- a closed-form, per-example
decision with no optimization at all. The CRS regime (maximize r_s subject to
population E[r_e] >= tau) is solved by bisecting on the Lagrange multiplier
lambda in the same objective (w_e=lambda, w_s=1), since increasing lambda can
only weakly increase achieved r_e -- standard Lagrangian dual monotonicity.

This is the ceiling on what ANY policy -- however well trained, however
expressive -- could achieve using only these features. If REINFORCE-trained
policies approach it, that validates the training. If the discontinuity
persists here, it is a property of the reward-maximization problem itself.
"""
import numpy as np
from .sim import (EVIDENCE, A_CONF, A_QUAL, A_ABST, A_CLAR, R_S_BY_ACTION,
                   epistemic_reward, rewards_for_actions, BASE_P_KNOW)

# Vectorized reward lookup: sim.rewards_for_actions loops in Python per-example
# (fine at REINFORCE's batch_size=256, far too slow once this runs inside a
# lambda bisection over N~500k). Build small (action x evidence x correct)
# lookup tables once and index with numpy fancy indexing instead.
_EPI_TABLE = np.zeros((4, 3, 2))
for _a in range(4):
    for _e in range(3):
        for _c in (False, True):
            _EPI_TABLE[_a, _e, int(_c)] = epistemic_reward(_e, _a, _c)
_RS_TABLE = np.array([R_S_BY_ACTION[a] for a in range(4)])


def fast_rewards_for_actions(data, actions):
    """Vectorized equivalent of sim.rewards_for_actions (evaluator_noise_p=0 case).
    Verified to match exactly in analytic_selftest.py."""
    correct = data["model_knows"].astype(int)
    evidence = data["evidence_idx"]
    r_e = _EPI_TABLE[actions, evidence, correct]
    r_s = _RS_TABLE[actions]
    return r_e, r_s


def posterior_knows(data, noise_sd=0.3):
    """Exact P(model_knows=1 | evidence, difficulty, confidence_signal) via Bayes' rule."""
    evidence = data["evidence_idx"]
    difficulty = data["difficulty"]
    c = data["confidence_signal"]
    base_p = np.array([BASE_P_KNOW[EVIDENCE[e]] for e in evidence])
    prior = np.clip(base_p - 0.3 * difficulty, 0.02, 0.98)

    # likelihoods (normalizing constant cancels in the ratio -- omitted)
    lik1 = np.exp(-0.5 * ((c - 1.0) / noise_sd) ** 2)
    lik0 = np.exp(-0.5 * ((c - 0.0) / noise_sd) ** 2)
    num = prior * lik1
    den = num + (1 - prior) * lik0
    return num / den


def _per_action_expected_values(data, noise_sd=0.3):
    """Returns dict action -> (E[r_e] array, r_s scalar).
    epistemic_reward for ABST/CLAR only depends on evidence state (3 possible
    values), so precompute a 3-entry lookup and index into it -- avoids an
    O(N) Python-level function-call loop, which matters once this runs inside
    a lambda bisection (many calls x large N)."""
    q = posterior_knows(data, noise_sd)
    evidence = data["evidence_idx"]
    abst_lookup = np.array([epistemic_reward(e, A_ABST, True) for e in range(len(EVIDENCE))])
    clar_lookup = np.array([epistemic_reward(e, A_CLAR, True) for e in range(len(EVIDENCE))])
    r_abst = abst_lookup[evidence]
    r_clar = clar_lookup[evidence]
    return {
        A_CONF: (2 * q - 1, R_S_BY_ACTION[A_CONF]),
        A_QUAL: (1.3 * q - 0.4, R_S_BY_ACTION[A_QUAL]),
        A_ABST: (r_abst, R_S_BY_ACTION[A_ABST]),
        A_CLAR: (r_clar, R_S_BY_ACTION[A_CLAR]),
    }


def optimal_actions(data, w_e, w_s, noise_sd=0.3):
    """Exact per-example argmax of w_e*E[r_e] + w_s*r_s across the 4 actions. No training."""
    ev = _per_action_expected_values(data, noise_sd)
    n = len(data["evidence_idx"])
    V = np.zeros((n, 4))
    for a in (A_CONF, A_QUAL, A_ABST, A_CLAR):
        re, rs = ev[a]
        V[:, a] = w_e * re + w_s * rs
    return V.argmax(axis=1)


def optimal_actions_crs(data, tau, noise_sd=0.3, lambda_hi=200.0, iters=60):
    """Bisect on Lagrange multiplier lambda (objective = r_s + lambda*r_e) to find the
    minimal lambda whose exact-optimal actions achieve population mean realized r_e >= tau.
    Realized r_e uses TRUE model_knows (ground truth), matching how the REINFORCE/empirical
    CRS training's constraint was evaluated -- so results are directly comparable."""
    def achieved(lam):
        actions = optimal_actions(data, w_e=lam, w_s=1.0, noise_sd=noise_sd)
        r_e, _ = fast_rewards_for_actions(data, actions)
        return r_e.mean(), actions

    r_e_hi, actions_hi = achieved(lambda_hi)
    if r_e_hi < tau:
        return actions_hi, lambda_hi  # tau infeasible even at (near-)oracle; return best achievable

    lo, hi = 0.0, lambda_hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        r_e_mid, _ = achieved(mid)
        if r_e_mid >= tau:
            hi = mid
        else:
            lo = mid
    _, final_actions = achieved(hi)
    return final_actions, hi


def evaluate_actions(data, actions, w_e=0.55, w_s=0.45):
    """Mirrors sim.evaluate()'s metric definitions exactly, but for precomputed
    (deterministic, exactly-optimal) actions rather than a trained policy."""
    r_e, r_s = fast_rewards_for_actions(data, actions)
    scalar = w_e * r_e + w_s * r_s

    insuff_mask = data["evidence_idx"] == EVIDENCE.index("insufficient")
    ucr = (actions[insuff_mask] == A_CONF).mean() if insuff_mask.any() else np.nan
    cal_abst = (actions[insuff_mask] == A_ABST).mean() if insuff_mask.any() else np.nan

    answered_mask = np.isin(actions, [A_CONF, A_QUAL])
    if answered_mask.any():
        incorrect_answered = answered_mask & (~data["model_knows"])
        selective_risk = incorrect_answered.sum() / answered_mask.sum()
        coverage = answered_mask.mean()
    else:
        selective_risk, coverage = np.nan, 0.0

    return {
        "epistemic_utility": r_e.mean(),
        "smoothness_utility": r_s.mean(),
        "aggregate_scalar_reward": scalar.mean(),
        "unsupported_certainty_rate": ucr,
        "calibrated_abstention_rate": cal_abst,
        "selective_risk": selective_risk,
        "coverage": coverage,
    }
