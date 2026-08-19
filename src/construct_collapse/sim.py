"""
Construct-collapse experiment: scalar vs stratified (CRS-style) preference evaluation
on a synthetic selective-answering task.

Actions: 0=confident_answer, 1=qualified_answer, 2=abstain, 3=clarify
Evidence states: sufficient, ambiguous, insufficient
"""
import numpy as np

ACTIONS = ["confident_answer", "qualified_answer", "abstain", "clarify"]
A_CONF, A_QUAL, A_ABST, A_CLAR = 0, 1, 2, 3
EVIDENCE = ["sufficient", "ambiguous", "insufficient"]

BASE_P_KNOW = {"sufficient": 0.95, "ambiguous": 0.55, "insufficient": 0.15}

# --- Reward tables -----------------------------------------------------
# Cells marked DOC are taken directly from the source document's reward table.
# Cells marked FILL are extrapolated to complete the (evidence x action) grid,
# following the stated evaluator definitions (epistemic quality rewards
# correctness + honest abstention/clarification when warranted and penalizes
# unwarranted hedging; smoothness only cares about action *form*, not context).

R_S_BY_ACTION = {  # smoothness reward depends only on action taken (per doc's definition)
    A_CONF: 1.0,   # DOC
    A_QUAL: 0.4,   # DOC
    A_ABST: -0.5,  # DOC
    A_CLAR: -0.4,  # DOC
}

def epistemic_reward(evidence_idx, action, correct):
    evidence = EVIDENCE[evidence_idx]
    if action == A_CONF:
        return 1.0 if correct else -1.0            # DOC (correct direct / incorrect confident)
    if action == A_QUAL:
        return 0.9 if correct else -0.4             # DOC (correct qualified / incorrect qualified)
    if action == A_ABST:
        return {"insufficient": 1.0,                # DOC (honest abstention)
                "ambiguous": 0.3,                    # FILL: acceptable but clarify was better
                "sufficient": -0.3}[evidence]         # FILL: unnecessary abstention, mildly penalized
    if action == A_CLAR:
        return {"ambiguous": 0.8,                    # DOC (clarification under ambiguity)
                "insufficient": 0.6,                  # FILL: reasonable, but abstain was cleaner
                "sufficient": -0.2}[evidence]           # FILL: unnecessary, mildly penalized
    raise ValueError(action)


def generate_dataset(n, seed, evidence_probs=(0.4, 0.3, 0.3), noise_sd=0.3):
    """evidence_probs = (P(sufficient), P(ambiguous), P(insufficient)) per doc's 40/30/30 split."""
    rng = np.random.default_rng(seed)
    evidence_idx = rng.choice(3, size=n, p=evidence_probs)
    difficulty = rng.uniform(0, 1, size=n)
    base_p = np.array([BASE_P_KNOW[EVIDENCE[e]] for e in evidence_idx])
    p_know = np.clip(base_p - 0.3 * difficulty, 0.02, 0.98)
    model_knows = rng.uniform(size=n) < p_know
    # noisy internal confidence signal available to the policy as a feature
    confidence_signal = model_knows.astype(float) + rng.normal(0, noise_sd, size=n)
    return {
        "evidence_idx": evidence_idx,
        "difficulty": difficulty,
        "model_knows": model_knows,
        "confidence_signal": confidence_signal,
        "rng": rng,
    }


def features(data):
    n = len(data["evidence_idx"])
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), data["evidence_idx"]] = 1.0
    x = np.column_stack([
        np.ones(n),
        onehot,
        data["difficulty"],
        data["confidence_signal"],
    ])
    return x  # shape (n, 6)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def rewards_for_actions(data, actions, evaluator_noise_p=0.0, rng=None):
    """Return (r_e, r_s) arrays for the taken actions.
    evaluator_noise_p: with this probability, a noisy evaluator does NOT apply
    the smoothness penalty for abstain/clarify (simulates unreliable judges
    who don't mind non-answers) -- used only in the noisy-evaluator ablation."""
    n = len(actions)
    correct = data["model_knows"]
    r_e = np.array([epistemic_reward(data["evidence_idx"][i], actions[i], correct[i]) for i in range(n)])
    r_s = np.array([R_S_BY_ACTION[a] for a in actions])
    if evaluator_noise_p > 0:
        assert rng is not None
        noisy = rng.uniform(size=n) < evaluator_noise_p
        hedge_mask = noisy & np.isin(actions, [A_ABST, A_CLAR])
        r_s = np.where(hedge_mask, 0.0, r_s)  # noisy evaluator: no penalty applied
    return r_e, r_s


class Policy:
    def __init__(self, n_features=6, n_actions=4, seed=0):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0, 0.05, size=(n_features, n_actions))

    def probs(self, x):
        return softmax(x @ self.theta)

    def act(self, x, rng):
        p = self.probs(x)
        n, k = p.shape
        cum = p.cumsum(axis=1)
        u = rng.uniform(size=(n, 1))
        return (u < cum).argmax(axis=1)


def train_policy(train_data, regime, seed, epochs=60, batch_size=256, lr=0.5,
                  w_e=0.55, w_s=0.45, tau=0.85, lr_lambda=0.05,
                  evaluator_noise_p=0.0, enforce_constraint=True):
    """regime in {'scalar', 'crs', 'oracle'}.
    Returns trained Policy and (for 'crs') final lambda."""
    rng = np.random.default_rng(seed + 9999)
    x_all = features(train_data)
    n = x_all.shape[0]
    policy = Policy(seed=seed)
    lam = 0.0
    idx_all = np.arange(n)

    for epoch in range(epochs):
        rng.shuffle(idx_all)
        for start in range(0, n, batch_size):
            idx = idx_all[start:start + batch_size]
            x = x_all[idx]
            batch = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in train_data.items() if k != "rng"}
            p = policy.probs(x)
            actions = policy.act(x, rng)
            r_e, r_s = rewards_for_actions(batch, actions, evaluator_noise_p, rng)

            if regime == "scalar":
                reward = w_e * r_e + w_s * r_s
            elif regime == "oracle":
                reward = r_e
            elif regime == "crs":
                reward = r_s + lam * r_e
                if enforce_constraint:
                    lam = max(0.0, lam + lr_lambda * (tau - r_e.mean()))
            else:
                raise ValueError(regime)

            baseline = reward.mean()
            adv = reward - baseline

            # REINFORCE gradient: sum_i adv_i * (onehot(a_i) - p_i) outer x_i
            onehot = np.zeros_like(p)
            onehot[np.arange(len(idx)), actions] = 1.0
            grad_logits = (onehot - p) * adv[:, None]  # (batch, n_actions)
            grad_theta = x.T @ grad_logits / len(idx)
            policy.theta += lr * grad_theta

    return policy, lam


def evaluate(policy, data, w_e=0.55, w_s=0.45, seed=0):
    rng = np.random.default_rng(seed + 555)
    x = features(data)
    p = policy.probs(x)
    actions = p.argmax(axis=1)  # deterministic eval policy
    r_e, r_s = rewards_for_actions(data, actions)
    scalar = w_e * r_e + w_s * r_s

    insuff_mask = data["evidence_idx"] == EVIDENCE.index("insufficient")
    unsupported_certainty_rate = (actions[insuff_mask] == A_CONF).mean() if insuff_mask.any() else np.nan
    calibrated_abstention_rate = (actions[insuff_mask] == A_ABST).mean() if insuff_mask.any() else np.nan

    answered_mask = np.isin(actions, [A_CONF, A_QUAL])
    if answered_mask.any():
        incorrect_answered = answered_mask & (~data["model_knows"])
        selective_risk = incorrect_answered.sum() / answered_mask.sum()
        coverage = answered_mask.mean()
    else:
        selective_risk, coverage = np.nan, 0.0

    # crude ECE: use max softmax prob as "confidence", correctness defined only
    # for answer actions (confident/qualified); abstain/clarify excluded (no factual claim)
    max_p = p.max(axis=1)
    ans_idx = np.where(answered_mask)[0]
    ece = np.nan
    if len(ans_idx) > 20:
        conf = max_p[ans_idx]
        correct = data["model_knows"][ans_idx].astype(float)
        bins = np.linspace(0, 1, 11)
        ece_val = 0.0
        for b in range(10):
            m = (conf >= bins[b]) & (conf < bins[b + 1] if b < 9 else conf <= bins[b + 1])
            if m.sum() == 0:
                continue
            ece_val += (m.sum() / len(conf)) * abs(conf[m].mean() - correct[m].mean())
        ece = ece_val

    return {
        "epistemic_utility": r_e.mean(),
        "smoothness_utility": r_s.mean(),
        "aggregate_scalar_reward": scalar.mean(),
        "unsupported_certainty_rate": unsupported_certainty_rate,
        "calibrated_abstention_rate": calibrated_abstention_rate,
        "selective_risk": selective_risk,
        "coverage": coverage,
        "ece": ece,
    }
