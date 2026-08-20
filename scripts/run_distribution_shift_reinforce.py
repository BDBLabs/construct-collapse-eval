import os
import numpy as np
import pandas as pd
from construct_collapse.sim import generate_dataset, train_policy, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_TRAIN = 60_000
N_DEPLOY = 80_000
SEEDS = list(range(5))
TRAIN_MIX = (0.4, 0.3, 0.3)
TAU = 0.85

DEPLOY_MIXES = {
    "same_as_training":   (0.4, 0.3, 0.3),
    "easier":             (0.7, 0.2, 0.1),
    "ambiguous_heavy":    (0.2, 0.5, 0.3),
    "harder":             (0.2, 0.3, 0.5),
    "much_harder":        (0.1, 0.2, 0.7),
}

rows = []
for seed in SEEDS:
    data_train = generate_dataset(N_TRAIN, seed=seed, evidence_probs=TRAIN_MIX)
    n_tr = int(0.85 * N_TRAIN)  # no held-out val needed here, just train/deploy split
    train_split = {k: v[:n_tr] for k, v in data_train.items() if k != "rng"}

    scalar_policy, _ = train_policy(train_split, regime="scalar", seed=seed)
    crs_policy, lam = train_policy(train_split, regime="crs", seed=seed, tau=TAU)

    for mix_name, mix in DEPLOY_MIXES.items():
        data_deploy = generate_dataset(N_DEPLOY, seed=seed + 500, evidence_probs=mix)
        for regime, policy in [("scalar", scalar_policy), ("crs", crs_policy)]:
            m = evaluate(policy, data_deploy, seed=seed)
            m["regime"] = regime
            m["deploy_mix"] = mix_name
            m["seed"] = seed
            rows.append(m)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "distribution_shift_reinforce_results.csv"), index=False)

agg = df.groupby(["regime", "deploy_mix"])[["epistemic_utility", "unsupported_certainty_rate", "selective_risk"]].mean().reset_index()
pd.set_option("display.width", 200)
print(agg.round(4).to_string())
