import os
import numpy as np
import pandas as pd
from construct_collapse.sim import generate_dataset, train_policy, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N = 8000
SEEDS = list(range(15))

# --- Ablation 1: CRS with vs without the protected epistemic constraint ---
rows = []
for seed in SEEDS:
    data = generate_dataset(N, seed=seed)
    n_train, n_val = int(0.7 * N), int(0.15 * N)
    train = {k: v[:n_train] for k, v in data.items() if k != "rng"}
    test = {k: v[n_train + n_val:] for k, v in data.items() if k != "rng"}
    for enforce in [True, False]:
        policy, lam = train_policy(train, regime="crs", seed=seed, enforce_constraint=enforce)
        m = evaluate(policy, test, seed=seed)
        m["constraint_enforced"] = enforce
        m["seed"] = seed
        rows.append(m)
df1 = pd.DataFrame(rows)
df1.to_csv(os.path.join(RESULTS_DIR, "ablation_constraint.csv"), index=False)
print("=== Constraint ablation ===")
print(df1.groupby("constraint_enforced")[
    ["epistemic_utility", "smoothness_utility", "unsupported_certainty_rate", "selective_risk"]
].agg(["mean", "std"]).round(4).to_string())

# --- Ablation 2: noisy evaluators (evaluator doesn't penalize abstain/clarify) ---
rows = []
for noise_p in [0.0, 0.1, 0.2, 0.3]:
    for seed in SEEDS:
        data = generate_dataset(N, seed=seed)
        n_train, n_val = int(0.7 * N), int(0.15 * N)
        train = {k: v[:n_train] for k, v in data.items() if k != "rng"}
        test = {k: v[n_train + n_val:] for k, v in data.items() if k != "rng"}
        for regime in ["scalar", "crs"]:
            policy, lam = train_policy(train, regime=regime, seed=seed, evaluator_noise_p=noise_p)
            m = evaluate(policy, test, seed=seed)
            m["regime"] = regime
            m["noise_p"] = noise_p
            m["seed"] = seed
            rows.append(m)
df2 = pd.DataFrame(rows)
df2.to_csv(os.path.join(RESULTS_DIR, "ablation_noise.csv"), index=False)
print("\n=== Noisy evaluator ablation ===")
print(df2.groupby(["regime", "noise_p"])[
    ["unsupported_certainty_rate", "selective_risk", "epistemic_utility"]
].agg(["mean", "std"]).round(4).to_string())
