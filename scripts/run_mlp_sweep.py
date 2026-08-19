import time, os
import numpy as np
import pandas as pd
from construct_collapse.sim import generate_dataset, evaluate
from construct_collapse.mlp_sim import train_mlp_policy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N = 6000
SEEDS = list(range(20))
# extra resolution around the linear-policy collapse point (0.5-0.6)
W_S_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.52, 0.55, 0.58, 0.6, 0.65, 0.7, 0.8]
OUT = os.path.join(RESULTS_DIR, "mlp_sweep_results.csv")
TIME_BUDGET_SEC = 80  # stop cleanly well before any external limit, resume on next call

combos = [(w_s, seed) for w_s in W_S_VALUES for seed in SEEDS]

done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(zip(prev["w_s"].round(6), prev["seed"]))

t0 = time.time()
n_done_this_call = 0
header_written = os.path.exists(OUT)
for w_s, seed in combos:
    if (round(w_s, 6), seed) in done:
        continue
    if time.time() - t0 > TIME_BUDGET_SEC:
        break
    data = generate_dataset(N, seed=seed)
    n_train, n_val = int(0.7 * N), int(0.15 * N)
    train = {k: v[:n_train] for k, v in data.items() if k != "rng"}
    test = {k: v[n_train + n_val:] for k, v in data.items() if k != "rng"}

    policy, _ = train_mlp_policy(train, regime="scalar", seed=seed, w_e=1 - w_s, w_s=w_s,
                                  epochs=80, n_restarts=3)
    m = evaluate(policy, test, w_e=1 - w_s, w_s=w_s, seed=seed)
    m["w_s"] = w_s
    m["seed"] = seed
    # flush immediately so an external kill mid-chunk doesn't lose progress
    row_df = pd.DataFrame([m])
    row_df.to_csv(OUT, mode="a", header=not header_written, index=False)
    header_written = True
    n_done_this_call += 1

total_done = len(done) + n_done_this_call
print(f"this call: {n_done_this_call} combos, elapsed {time.time()-t0:.1f}s. "
      f"total done: {total_done}/{len(combos)}")

if total_done >= len(combos):
    df = pd.read_csv(OUT)
    agg = df.groupby("w_s").agg(
        scalar_mean=("aggregate_scalar_reward", "mean"),
        scalar_ci=("aggregate_scalar_reward", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
        ucr_mean=("unsupported_certainty_rate", "mean"),
        ucr_ci=("unsupported_certainty_rate", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
        risk_mean=("selective_risk", "mean"),
        risk_ci=("selective_risk", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    ).reset_index()
    agg.to_csv(os.path.join(RESULTS_DIR, "mlp_sweep_summary.csv"), index=False)
    print("SWEEP COMPLETE")
    print(agg.round(4).to_string())
else:
    print("call again to continue")
