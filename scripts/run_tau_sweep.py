import os
import numpy as np
import pandas as pd
from construct_collapse.sim import generate_dataset, train_policy, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N = 6000
SEEDS = list(range(30))
TAU_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

rows = []
for tau in TAU_VALUES:
    for seed in SEEDS:
        data = generate_dataset(N, seed=seed)
        n_train, n_val = int(0.7 * N), int(0.15 * N)
        train = {k: v[:n_train] for k, v in data.items() if k != "rng"}
        test = {k: v[n_train + n_val:] for k, v in data.items() if k != "rng"}

        policy, lam = train_policy(train, regime="crs", seed=seed, tau=tau)
        m = evaluate(policy, test, seed=seed)
        m["tau"] = tau
        m["seed"] = seed
        m["final_lambda"] = lam
        rows.append(m)
    print(f"tau={tau} done")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "tau_sweep_results.csv"), index=False)

agg = df.groupby("tau").agg(
    epi_mean=("epistemic_utility", "mean"),
    epi_ci=("epistemic_utility", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    smooth_mean=("smoothness_utility", "mean"),
    smooth_ci=("smoothness_utility", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    ucr_mean=("unsupported_certainty_rate", "mean"),
    ucr_ci=("unsupported_certainty_rate", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    risk_mean=("selective_risk", "mean"),
    risk_ci=("selective_risk", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    lambda_mean=("final_lambda", "mean"),
).reset_index()
agg.to_csv(os.path.join(RESULTS_DIR, "tau_sweep_summary.csv"), index=False)
print(agg.round(4).to_string())
