"""
Distribution shift test (RESULTS.md Section 9).

Question: CRS's lambda is calibrated (via bisection) to hit a target tau under
ONE evidence-proportion mix. It is not re-tuned per deployment environment. If
the deployment mix shifts -- e.g. users increasingly ask questions the model
can't answer -- does the SAME fixed policy still deliver near-tau epistemic
quality, or does the "certified" floor silently erode?

This uses the exact Bayes-optimal approach (Section 8) specifically because it
isolates the distribution-shift effect from training noise: the policy is a
fixed, closed-form function of (evidence, difficulty, confidence_signal), so
any change in aggregate metrics under a shifted mix is due ONLY to the shift in
which examples appear, not to any change in how individual examples are
decided.
"""
import os
import pandas as pd
from construct_collapse.sim import generate_dataset
from construct_collapse.analytic import optimal_actions, optimal_actions_crs, evaluate_actions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_CALIBRATE = 500_000
N_DEPLOY = 300_000
TRAIN_MIX = (0.4, 0.3, 0.3)  # sufficient, ambiguous, insufficient -- same as every prior section
TAU = 0.85

DEPLOY_MIXES = {
    "same_as_training":   (0.4, 0.3, 0.3),
    "easier":             (0.7, 0.2, 0.1),
    "ambiguous_heavy":    (0.2, 0.5, 0.3),
    "harder":             (0.2, 0.3, 0.5),
    "much_harder":        (0.1, 0.2, 0.7),
}

# --- calibrate fixed policies under the training mix (same as every prior section) ---
data_train = generate_dataset(N_CALIBRATE, seed=0, evidence_probs=TRAIN_MIX)
scalar_w = (0.55, 0.45)  # (w_e, w_s) -- same as every prior section
crs_actions_train, lam_star = optimal_actions_crs(data_train, tau=TAU)
crs_train_metrics = evaluate_actions(data_train, crs_actions_train)
print(f"Calibrated CRS lambda under training mix: {lam_star:.4f} "
      f"(achieved epistemic_utility={crs_train_metrics['epistemic_utility']:.4f}, target tau={TAU})")

rows = []
for mix_name, mix in DEPLOY_MIXES.items():
    data_deploy = generate_dataset(N_DEPLOY, seed=1, evidence_probs=mix)

    scalar_actions = optimal_actions(data_deploy, w_e=scalar_w[0], w_s=scalar_w[1])
    crs_actions = optimal_actions(data_deploy, w_e=lam_star, w_s=1.0)  # SAME fixed lambda, not recalibrated
    oracle_actions = optimal_actions(data_deploy, w_e=1.0, w_s=0.0)

    for regime, actions in [("scalar", scalar_actions), ("crs", crs_actions), ("oracle", oracle_actions)]:
        m = evaluate_actions(data_deploy, actions)
        m["regime"] = regime
        m["deploy_mix"] = mix_name
        m["mix_sufficient"], m["mix_ambiguous"], m["mix_insufficient"] = mix
        rows.append(m)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "distribution_shift_exact_results.csv"), index=False)
pd.set_option("display.width", 200)
print(df[["deploy_mix", "regime", "epistemic_utility", "unsupported_certainty_rate", "selective_risk"]]
      .round(4).to_string())
