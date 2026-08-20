"""
Exact Bayes-optimal policy check (RESULTS.md Section 8).

No REINFORCE, no training loop, no restarts -- computes the true optimal
decision rule in closed form for both the w_s sweep and the tau sweep, using
N=500,000 so sampling noise is negligible. See construct_collapse/analytic.py
for the derivation and fast_rewards_for_actions for the vectorization note.
"""
import os
import numpy as np
import pandas as pd
from construct_collapse.sim import generate_dataset
from construct_collapse.analytic import optimal_actions, optimal_actions_crs, evaluate_actions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N = 500_000

data = generate_dataset(N, seed=0)

# --- w_s sweep: coarse full range + fine resolution around the collapse zone ---
w_s_grid = np.round(np.unique(np.concatenate([
    np.arange(0.0, 0.30, 0.05),
    np.arange(0.30, 0.70001, 0.005),
    np.arange(0.70, 0.80001, 0.05),
])), 4)

rows = []
for w_s in w_s_grid:
    actions = optimal_actions(data, w_e=1 - w_s, w_s=w_s)
    m = evaluate_actions(data, actions, w_e=1 - w_s, w_s=w_s)
    m["w_s"] = w_s
    rows.append(m)
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "exact_sweep_results.csv"), index=False)
print(f"w_s sweep: {len(rows)} points written")

# --- tau sweep ---
tau_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
rows = []
for tau in tau_grid:
    actions, lam = optimal_actions_crs(data, tau=tau)
    m = evaluate_actions(data, actions)
    m["tau"] = tau
    m["lambda"] = lam
    rows.append(m)
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "exact_tau_sweep_results.csv"), index=False)
print(f"tau sweep: {len(rows)} points written")

# --- closed-form cross-check ---
# For an item under insufficient evidence with posterior q=0 (the majority case:
# the confidence signal is fairly informative given noise_sd=0.3, so most
# posteriors are near 0 or 1), confident_answer beats abstain exactly when
#   w_s > (2 - 2q) / (3.5 - 2q)   =>   at q=0:  w_s* = 2/3.5
w_s_star = 2 / 3.5
print(f"\nClosed-form threshold (q=0 case): w_s* = 2/3.5 = {w_s_star:.6f}")
