"""
Richer policy class check: does the sharp construct-collapse threshold survive
a nonlinear (MLP) policy, or is it an artifact of the linear-softmax policy
being too weak to discover a partial-hedging strategy?

Reuses generate_dataset / features / rewards_for_actions / evaluate from sim.py
unchanged -- only the policy class and its training loop are new. `evaluate()`
only requires policy.probs(x), so MLPPolicy is a drop-in replacement.
"""
import numpy as np
from .sim import generate_dataset, features, rewards_for_actions, evaluate, softmax

class MLPPolicy:
    """5 -> H (tanh) -> 4 (softmax). No bias trick via features; explicit biases."""
    def __init__(self, n_in=5, n_hidden=32, n_actions=4, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.W1 = rng.normal(0, scale1, size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, scale2, size=(n_hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        # Adam state
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def _forward(self, x5):
        h_pre = x5 @ self.W1 + self.b1
        h = np.tanh(h_pre)
        logits = h @ self.W2 + self.b2
        p = softmax(logits)
        return h_pre, h, logits, p

    def probs(self, x6):
        # x6 comes from sim.features(): [1, is_suff, is_amb, is_insuff, difficulty, conf]
        # drop the leading bias column, MLP has its own biases
        x5 = x6[:, 1:]
        _, _, _, p = self._forward(x5)
        return p

    def act(self, x6, rng):
        p = self.probs(x6)
        n = p.shape[0]
        cum = p.cumsum(axis=1)
        u = rng.uniform(size=(n, 1))
        return (u < cum).argmax(axis=1)

    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self._params()
        for k in params:
            g = grads[k]
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
            mhat = self._m[k] / (1 - beta1 ** self._t)
            vhat = self._v[k] / (1 - beta2 ** self._t)
            params[k] += lr * mhat / (np.sqrt(vhat) + eps)  # gradient ASCENT


def train_mlp_policy(train_data, regime, seed, epochs=80, batch_size=256, lr=0.01,
                      n_hidden=32, w_e=0.55, w_s=0.45, tau=0.85, lr_lambda=0.05,
                      evaluator_noise_p=0.0, n_restarts=3):
    """Trains n_restarts independent MLPs, selects the one with the best
    achieved TRAINING objective under `regime` (not the eval metric we're
    checking -- avoids biasing restart selection toward the answer we want),
    and returns that policy. This mimics standard practice of taking the
    best checkpoint by training loss/reward across seeds."""
    x_all = features(train_data)
    n = x_all.shape[0]
    best_policy, best_lam, best_score = None, 0.0, -np.inf

    for r in range(n_restarts):
        rng = np.random.default_rng(seed * 1000 + r)
        policy = MLPPolicy(n_hidden=n_hidden, seed=seed * 1000 + r)
        lam = 0.0
        idx_all = np.arange(n)

        for epoch in range(epochs):
            rng.shuffle(idx_all)
            for start in range(0, n, batch_size):
                idx = idx_all[start:start + batch_size]
                x6 = x_all[idx]
                x5 = x6[:, 1:]
                batch = {k: v[idx] for k, v in train_data.items() if k != "rng"}

                h_pre, h, logits, p = policy._forward(x5)
                actions = policy.act(x6, rng)
                r_e, r_s = rewards_for_actions(batch, actions, evaluator_noise_p, rng)

                if regime == "scalar":
                    reward = w_e * r_e + w_s * r_s
                elif regime == "oracle":
                    reward = r_e
                elif regime == "crs":
                    reward = r_s + lam * r_e
                    lam = max(0.0, lam + lr_lambda * (tau - r_e.mean()))
                else:
                    raise ValueError(regime)

                baseline = reward.mean()
                adv = reward - baseline

                onehot = np.zeros_like(p)
                onehot[np.arange(len(idx)), actions] = 1.0
                dlogits = (onehot - p) * adv[:, None]

                dW2 = h.T @ dlogits / len(idx)
                db2 = dlogits.mean(axis=0)
                dh = dlogits @ policy.W2.T
                dh_pre = dh * (1 - h ** 2)
                dW1 = x5.T @ dh_pre / len(idx)
                db1 = dh_pre.mean(axis=0)

                policy.adam_step({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, lr=lr)

        # score this restart by its OWN training objective, on the full train set
        x6_full = x_all
        x5_full = x6_full[:, 1:]
        _, _, _, p_full = policy._forward(x5_full)
        actions_full = p_full.argmax(axis=1)  # deterministic
        r_e_f, r_s_f = rewards_for_actions(train_data, actions_full)
        if regime == "scalar":
            score = (w_e * r_e_f + w_s * r_s_f).mean()
        elif regime == "oracle":
            score = r_e_f.mean()
        elif regime == "crs":
            # selection criterion for CRS: smoothness reward AMONG restarts that
            # satisfy the constraint; if none satisfy it, prefer highest r_e
            feasible = r_e_f.mean() >= tau - 0.05
            score = r_s_f.mean() if feasible else (r_e_f.mean() - 10.0)

        if score > best_score:
            best_score, best_policy, best_lam = score, policy, lam

    return best_policy, best_lam
