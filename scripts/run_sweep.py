import os
import numpy as np
import pandas as pd
from construct_collapse.sim import generate_dataset, train_policy, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N = 6000  # smaller per-run N since we now run many (weight x seed) combos
SEEDS = list(range(50))
W_S_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

rows = []
for w_s in W_S_VALUES:
    for seed in SEEDS:
        data = generate_dataset(N, seed=seed)
        n_train, n_val = int(0.7 * N), int(0.15 * N)
        train = {k: v[:n_train] for k, v in data.items() if k != "rng"}
        test = {k: v[n_train + n_val:] for k, v in data.items() if k != "rng"}

        policy, _ = train_policy(train, regime="scalar", seed=seed, w_e=1 - w_s, w_s=w_s, epochs=40)
        m = evaluate(policy, test, w_e=1 - w_s, w_s=w_s, seed=seed)
        m["w_s"] = w_s
        m["seed"] = seed
        rows.append(m)
    print(f"w_s={w_s} done")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "sweep_results.csv"), index=False)

agg = df.groupby("w_s").agg(
    scalar_mean=("aggregate_scalar_reward", "mean"),
    scalar_ci=("aggregate_scalar_reward", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    ucr_mean=("unsupported_certainty_rate", "mean"),
    ucr_ci=("unsupported_certainty_rate", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    risk_mean=("selective_risk", "mean"),
    risk_ci=("selective_risk", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
).reset_index()
agg.to_csv(os.path.join(RESULTS_DIR, "sweep_summary.csv"), index=False)
print(agg.round(4).to_string())
