import os
import numpy as np
import pandas as pd
from construct_collapse.sim import generate_dataset, train_policy, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N = 10000
SEEDS = list(range(10))  # 10 seeds for primary result (50 used later in the weight sweep)
REGIMES = ["scalar", "crs", "oracle"]

def split(data, n_train, n_val):
    idx = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, None),
    }
    out = {}
    for name, sl in idx.items():
        out[name] = {k: v[sl] for k, v in data.items() if k != "rng"}
    return out

rows = []
for seed in SEEDS:
    data = generate_dataset(N, seed=seed)
    n_train, n_val = int(0.7 * N), int(0.15 * N)
    parts = split(data, n_train, n_val)

    for regime in REGIMES:
        policy, lam = train_policy(parts["train"], regime=regime, seed=seed)
        metrics = evaluate(policy, parts["test"], seed=seed)
        metrics["regime"] = regime
        metrics["seed"] = seed
        metrics["final_lambda"] = lam
        rows.append(metrics)
    print(f"seed {seed} done")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "primary_results.csv"), index=False)

summary = df.groupby("regime").agg(["mean", "std"])
print(summary)
summary.to_csv(os.path.join(RESULTS_DIR, "primary_summary.csv"))
