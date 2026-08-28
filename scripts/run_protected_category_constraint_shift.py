python
#!/usr/bin/env python3
"""
Protected-category constraint distribution-shift experiment.

Question
--------
The pooled CRS experiment calibrates one aggregate epistemic constraint:

    E[r_e] >= tau

under a calibration distribution.

That guarantee can fail under distribution shift because the aggregate
constraint permits epistemic performance to be distributed unevenly across
evidence categories.

This experiment tests the stronger alternative:

    E[r_e | evidence=k] >= tau_k   for every evidence category k.

The category floors are specified independently. They are NOT copied from
the pooled CRS solution.

For each category k, we independently solve:

    maximize E[r_s | k]

    subject to

        E[r_e | k] >= tau_k

using a category-specific Lagrange multiplier.

The resulting policy is frozen and then evaluated under several deployment
mixtures.

A successful result means:

    category floors satisfied under calibration
        AND
    category floors satisfied under every deployment mix.

This would provide evidence that category-conditional protection repairs
the specific pooled-aggregate transport failure.

A failure would be equally informative: it would show that conditioning
only on evidence category is insufficient and motivate finer-grained
conditional guarantees, uncertainty-set constraints, or DRO.

Outputs
-------
results/protected_category_constraint_shift.csv

    One row per target configuration x deployment mix.

results/protected_category_constraint_shift_targets.csv

    Calibration feasibility and target diagnostics.

results/protected_category_constraint_shift_summary.csv

    Compact deployment summary.

Important
---------
This experiment intentionally keeps the policy frozen after calibration.
Deployment distributions are NOT used to recalibrate lambda.

The aggressive target 0.74 for sufficient evidence is expected to be
infeasible in this environment and is retained specifically to expose
the feasibility boundary.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Repository import path
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================================
# Repository imports
# ============================================================================

from construct_collapse.sim import (
    EVIDENCE,
    A_CONF,
    A_QUAL,
    A_ABST,
    A_CLAR,
    R_S_BY_ACTION,
    epistemic_reward,
    generate_dataset,
)

from construct_collapse.analytic import (
    posterior_knows,
    evaluate_actions,
)


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CALIBRATE = 500_000
N_DEPLOY = 500_000

# Calibration distribution.
TRAIN_MIX = {
    "sufficient": 0.40,
    "ambiguous": 0.30,
    "insufficient": 0.30,
}

# EXACT deployment mixtures used by the original distribution-shift test.
DEPLOY_MIXES = {
    "train_40_30_30": (0.40, 0.30, 0.30),
    "easier_70_20_10": (0.70, 0.20, 0.10),
    "ambiguous_20_50_30": (0.20, 0.50, 0.30),
    "harder_20_30_50": (0.20, 0.30, 0.50),
    "much_harder_10_20_70": (0.10, 0.20, 0.70),
}


# ============================================================================
# Independently specified protected-category target grid
# ============================================================================

CATEGORY_TARGET_GRID = {
    "sufficient": [0.70, 0.72, 0.74],
    "ambiguous": [0.80, 0.85, 0.90],
    "insufficient": [0.95, 0.98, 0.99],
}

LAMBDA_HI = 200.0
BISECTION_ITERS = 60

EVIDENCE_TO_INDEX = {
    name: idx
    for idx, name in enumerate(EVIDENCE)
}


# ============================================================================
# Expected epistemic reward
# ============================================================================

def expected_action_values(data):
    """
    Compute posterior expected epistemic reward for each action.

    Returns
    -------
    dict[str, np.ndarray]
        action -> expected epistemic reward for each observation.
    """

    q = posterior_knows(data)
    evidence = data["evidence_idx"]

    abst_lookup = np.array(
        [
            epistemic_reward(
                evidence_idx=e,
                action=A_ABST,
                correct=True,
            )
            for e in range(len(EVIDENCE))
        ],
        dtype=float,
    )

    clar_lookup = np.array(
        [
            epistemic_reward(
                evidence_idx=e,
                action=A_CLAR,
                correct=True,
            )
            for e in range(len(EVIDENCE))
        ],
        dtype=float,
    )

    return {
        A_CONF: 2.0 * q - 1.0,
        A_QUAL: 1.3 * q - 0.4,
        A_ABST: abst_lookup[evidence],
        A_CLAR: clar_lookup[evidence],
    }


# ============================================================================
# Protected category policy
# ============================================================================

def protected_category_actions(data, lambdas):
    """
    Select actions using a category-specific Lagrange multiplier.

    For observation x in evidence category k:

        argmax_a [
            r_s(a, x) + lambda_k * r_e(a, x)
        ]

    Parameters
    ----------
    data:
        Dataset.

    lambdas:
        Mapping evidence category -> lambda.

    Returns
    -------
    np.ndarray
        Selected action for each observation.
    """

    rewards_e = expected_action_values(data)
    evidence_idx = data["evidence_idx"]

    actions = [
        A_CONF,
        A_QUAL,
        A_ABST,
        A_CLAR,
    ]

    scores = np.column_stack(
        [
            np.asarray(
                rewards_e[action],
                dtype=float,
            )
            for action in actions
        ]
    )

    # Scalar task reward for each action.
    #
    # R_S_BY_ACTION is an action-level task reward exposed by the simulator.
    # The category-specific epistemic multiplier is then applied to the
    # expected epistemic reward.
    r_s = np.column_stack(
        [
            np.full(
                len(data["evidence_idx"]),
                float(R_S_BY_ACTION[action]),
            )
            for action in actions
        ]
    )

    lambda_per_row = np.zeros(
        len(data["evidence_idx"]),
        dtype=float,
    )

    for category, idx in EVIDENCE_TO_INDEX.items():
        mask = evidence_idx == idx

        if not np.any(mask):
            continue

        lambda_per_row[mask] = lambdas[category]

    objective = (
        r_s
        + scores * lambda_per_row[:, None]
    )

    best = np.argmax(
        objective,
        axis=1,
    )

    return np.asarray(
        actions,
        dtype=object,
    )[best]


# ============================================================================
# Category conditional epistemic reward
# ============================================================================

def category_epistemic_metrics(data, actions):
    """
    Calculate epistemic utility conditional on evidence category.
    """

    metrics = {}

    expected_e = expected_action_values(data)

    for category, idx in EVIDENCE_TO_INDEX.items():

        mask = data["evidence_idx"] == idx

        if not np.any(mask):
            metrics[category] = np.nan
            continue

        action_values = np.zeros(
            mask.sum(),
            dtype=float,
        )

        category_actions = actions[mask]

        for action in (
            A_CONF,
            A_QUAL,
            A_ABST,
            A_CLAR,
        ):

            action_mask = (
                category_actions == action
            )

            if not np.any(action_mask):
                continue

            vals = expected_e[action][mask]

            action_values[action_mask] = (
                vals[action_mask]
            )

        metrics[category] = float(
            np.mean(action_values)
        )

    return metrics


# ============================================================================
# Calibrate one category
# ============================================================================

def calibrate_category_lambda(
    data,
    category,
    target,
):
    """
    Independently calibrate lambda for one evidence category.

    The category is observed, so all observations outside the category are
    irrelevant to this optimization.

    We maximize expected task reward subject to the conditional epistemic
    floor.

    Returns
    -------
    dict
        lambda, achieved epistemic reward, feasible flag, and diagnostics.
    """

    idx = EVIDENCE_TO_INDEX[category]

    mask = (
        data["evidence_idx"] == idx
    )

    category_data = {
        key: value[mask]
        for key, value in data.items()
    }

    # ------------------------------------------------------------------------
    # Evaluate epistemic reward at lambda = 0.
    # ------------------------------------------------------------------------

    actions_zero = protected_category_actions(
        category_data,
        {
            name: 0.0
            for name in EVIDENCE
        },
    )

    m_zero = category_epistemic_metrics(
        category_data,
        actions_zero,
    )[category]

    # ------------------------------------------------------------------------
    # Evaluate epistemic reward at a very large lambda.
    # ------------------------------------------------------------------------

    hi = LAMBDA_HI

    actions_hi = protected_category_actions(
        category_data,
        {
            name: hi
            for name in EVIDENCE
        },
    )

    m_hi = category_epistemic_metrics(
        category_data,
        actions_hi,
    )[category]

    # ------------------------------------------------------------------------
    # Feasibility.
    # ------------------------------------------------------------------------

    if m_hi < target - 1e-12:

        return {
            "lambda": np.nan,
            "achieved": float(m_hi),
            "feasible": False,
            "min_lambda": np.nan,
            "max_lambda": hi,
            "zero_lambda_epistemic": float(m_zero),
        }

    # ------------------------------------------------------------------------
    # If unconstrained optimum already satisfies the floor.
    # ------------------------------------------------------------------------

    if m_zero >= target:

        return {
            "lambda": 0.0,
            "achieved": float(m_zero),
            "feasible": True,
            "min_lambda": 0.0,
            "max_lambda": 0.0,
            "zero_lambda_epistemic": float(m_zero),
        }

    lo = 0.0

    # ------------------------------------------------------------------------
    # Bisection.
    #
    # We seek the smallest lambda whose induced policy satisfies the floor.
    # ------------------------------------------------------------------------

    for _ in range(BISECTION_ITERS):

        mid = 0.5 * (
            lo + hi
        )

        actions_mid = protected_category_actions(
            category_data,
            {
                name: mid
                for name in EVIDENCE
            },
        )

        m_mid = category_epistemic_metrics(
            category_data,
            actions_mid,
        )[category]

        if m_mid >= target:
            hi = mid
        else:
            lo = mid

    lambda_star = hi

    actions_star = protected_category_actions(
        category_data,
        {
            name: lambda_star
            for name in EVIDENCE
        },
    )

    achieved = category_epistemic_metrics(
        category_data,
        actions_star,
    )[category]

    return {
        "lambda": float(lambda_star),
        "achieved": float(achieved),
        "feasible": bool(
            achieved >= target - 1e-10
        ),
        "min_lambda": float(lo),
        "max_lambda": float(hi),
        "zero_lambda_epistemic": float(m_zero),
    }


# ============================================================================
# Evaluate a frozen policy
# ============================================================================

def evaluate_protected_policy(
    data,
    lambdas,
    target_tuple,
):
    """
    Evaluate a frozen category-protected policy.

    Returns aggregate metrics plus conditional category metrics.
    """

    actions = protected_category_actions(
        data,
        lambdas,
    )

    aggregate = evaluate_actions(
        data,
        actions,
    )

    conditional = category_epistemic_metrics(
        data,
        actions,
    )

    targets = {
        "sufficient": target_tuple[0],
        "ambiguous": target_tuple[1],
        "insufficient": target_tuple[2],
    }

    undershoots = {
        category: max(
            0.0,
            targets[category]
            - conditional[category],
        )
        for category in targets
    }

    max_undershoot = max(
        undershoots.values()
    )

    aggregate.update(
        {
            "category_sufficient_epistemic":
                conditional["sufficient"],

            "category_ambiguous_epistemic":
                conditional["ambiguous"],

            "category_insufficient_epistemic":
                conditional["insufficient"],

            "target_sufficient":
                targets["sufficient"],

            "target_ambiguous":
                targets["ambiguous"],

            "target_insufficient":
                targets["insufficient"],

            "undershoot_sufficient":
                undershoots["sufficient"],

            "undershoot_ambiguous":
                undershoots["ambiguous"],

            "undershoot_insufficient":
                undershoots["insufficient"],

            "max_category_undershoot":
                max_undershoot,

            "all_category_floors_met":
                max_undershoot <= 1e-10,
        }
    )

    return aggregate


# ============================================================================
# Main experiment
# ============================================================================

def main():

    print("=" * 78)
    print(
        "PROTECTED CATEGORY CONSTRAINT — DISTRIBUTION SHIFT"
    )
    print("=" * 78)

    print()
    print(
        f"Calibration N: {N_CALIBRATE:,}"
    )
    print(
        f"Deployment N:  {N_DEPLOY:,}"
    )
    print(
        f"Training mix:  {TRAIN_MIX}"
    )
    print()

    # ------------------------------------------------------------------------
    # Calibration data
    # ------------------------------------------------------------------------

    train_mix_tuple = (
        TRAIN_MIX["sufficient"],
        TRAIN_MIX["ambiguous"],
        TRAIN_MIX["insufficient"],
    )

    data_train = generate_dataset(
        N_CALIBRATE,
        seed=0,
        evidence_probs=train_mix_tuple,
    )

    # ------------------------------------------------------------------------
    # Target configurations
    # ------------------------------------------------------------------------

    target_tuples = list(
        product(
            CATEGORY_TARGET_GRID["sufficient"],
            CATEGORY_TARGET_GRID["ambiguous"],
            CATEGORY_TARGET_GRID["insufficient"],
        )
    )

    calibration_rows = []
    deployment_rows = []

    for tau_s, tau_a, tau_i in target_tuples:

        target_tuple = (
            tau_s,
            tau_a,
            tau_i,
        )

        target_label = (
            f"s{tau_s:.2f}_"
            f"a{tau_a:.2f}_"
            f"i{tau_i:.2f}"
        )

        print()
        print("-" * 78)
        print(
            f"TARGET CONFIGURATION: "
            f"{target_label}"
        )

        # --------------------------------------------------------------------
        # Independently calibrate each category.
        # --------------------------------------------------------------------

        calibration = {}

        for category, target in zip(
            (
                "sufficient",
                "ambiguous",
                "insufficient",
            ),
            target_tuple,
        ):

            result = calibrate_category_lambda(
                data_train,
                category,
                target,
            )

            calibration[category] = result

            lambda_value = result["lambda"]

            if np.isfinite(lambda_value):
                lambda_text = f"{lambda_value:.6f}"
            else:
                lambda_text = "nan"

            print(
                f"  {category:12s}: "
                f"target={target:.4f}  "
                f"achieved={result['achieved']:.4f}  "
                f"lambda={lambda_text}  "
                f"feasible={result['feasible']}"
            )

        feasible = all(
            calibration[c]["feasible"]
            for c in (
                "sufficient",
                "ambiguous",
                "insufficient",
            )
        )

        # --------------------------------------------------------------------
        # Preserve infeasible configurations in the calibration output.
        # --------------------------------------------------------------------

        if not feasible:

            print(
                "  >>> INFEASIBLE "
                "TARGET CONFIGURATION"
            )

            calibration_rows.append(
                {
                    "target_label": target_label,

                    "target_sufficient": tau_s,
                    "target_ambiguous": tau_a,
                    "target_insufficient": tau_i,

                    "lambda_sufficient":
                        calibration[
                            "sufficient"
                        ]["lambda"],

                    "lambda_ambiguous":
                        calibration[
                            "ambiguous"
                        ]["lambda"],

                    "lambda_insufficient":
                        calibration[
                            "insufficient"
                        ]["lambda"],

                    "achieved_sufficient":
                        calibration[
                            "sufficient"
                        ]["achieved"],

                    "achieved_ambiguous":
                        calibration[
                            "ambiguous"
                        ]["achieved"],

                    "achieved_insufficient":
                        calibration[
                            "insufficient"
                        ]["achieved"],

                    "feasible_sufficient":
                        calibration[
                            "sufficient"
                        ]["feasible"],

                    "feasible_ambiguous":
                        calibration[
                            "ambiguous"
                        ]["feasible"],

                    "feasible_insufficient":
                        calibration[
                            "insufficient"
                        ]["feasible"],

                    "feasible": False,
                }
            )

            continue

        lambdas = {
            "sufficient":
                calibration[
                    "sufficient"
                ]["lambda"],

            "ambiguous":
                calibration[
                    "ambiguous"
                ]["lambda"],

            "insufficient":
                calibration[
                    "insufficient"
                ]["lambda"],
        }

        print(
            f"  Lambdas: "
            f"s={lambdas['sufficient']:.6f} "
            f"a={lambdas['ambiguous']:.6f} "
            f"i={lambdas['insufficient']:.6f}"
        )

        # --------------------------------------------------------------------
        # Calibration diagnostics.
        # --------------------------------------------------------------------

        calibration_metrics = evaluate_protected_policy(
            data_train,
            lambdas,
            target_tuple,
        )

        calibration_rows.append(
            {
                "target_label": target_label,

                "target_sufficient": tau_s,
                "target_ambiguous": tau_a,
                "target_insufficient": tau_i,

                "lambda_sufficient":
                    lambdas["sufficient"],

                "lambda_ambiguous":
                    lambdas["ambiguous"],

                "lambda_insufficient":
                    lambdas["insufficient"],

                "achieved_sufficient":
                    calibration_metrics[
                        "category_sufficient_epistemic"
                    ],

                "achieved_ambiguous":
                    calibration_metrics[
                        "category_ambiguous_epistemic"
                    ],

                "achieved_insufficient":
                    calibration_metrics[
                        "category_insufficient_epistemic"
                    ],

                "epistemic_utility":
                    calibration_metrics[
                        "epistemic_utility"
                    ],

                "max_category_undershoot":
                    calibration_metrics[
                        "max_category_undershoot"
                    ],

                "all_category_floors_met":
                    calibration_metrics[
                        "all_category_floors_met"
                    ],

                "feasible": True,
            }
        )

        # --------------------------------------------------------------------
        # Frozen-policy deployment evaluation.
        # --------------------------------------------------------------------

        for deploy_name, deploy_mix in DEPLOY_MIXES.items():

            print()
            print(
                f"Deployment: {deploy_name}"
            )

            data_deploy = generate_dataset(
                N_DEPLOY,
                seed=1,
                evidence_probs=deploy_mix,
            )

            metrics = evaluate_protected_policy(
                data_deploy,
                lambdas,
                target_tuple,
            )

            row = {
                "target_label":
                    target_label,

                "deploy_mix":
                    deploy_name,

                "mix_sufficient":
                    deploy_mix[0],

                "mix_ambiguous":
                    deploy_mix[1],

                "mix_insufficient":
                    deploy_mix[2],

                "lambda_sufficient":
                    lambdas["sufficient"],

                "lambda_ambiguous":
                    lambdas["ambiguous"],

                "lambda_insufficient":
                    lambdas["insufficient"],

                **metrics,
            }

            deployment_rows.append(row)

            print(
                f"  epistemic="
                f"{metrics['epistemic_utility']:.4f}"
            )

            print(
                f"  sufficient="
                f"{metrics['category_sufficient_epistemic']:.4f} "
                f"/ target {tau_s:.4f}"
            )

            print(
                f"  ambiguous="
                f"{metrics['category_ambiguous_epistemic']:.4f} "
                f"/ target {tau_a:.4f}"
            )

            print(
                f"  insufficient="
                f"{metrics['category_insufficient_epistemic']:.4f} "
                f"/ target {tau_i:.4f}"
            )

            print(
                f"  max undershoot="
                f"{metrics['max_category_undershoot']:.4f}"
            )

            print(
                "  ALL FLOORS MET="
                f"{'YES' if metrics['all_category_floors_met'] else 'NO'}"
            )

    # =========================================================================
    # Save outputs
    # =========================================================================

    targets_path = (
        RESULTS_DIR
        / "protected_category_constraint_shift_targets.csv"
    )

    results_path = (
        RESULTS_DIR
        / "protected_category_constraint_shift.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "protected_category_constraint_shift_summary.csv"
    )

    targets_df = pd.DataFrame(
        calibration_rows
    )

    results_df = pd.DataFrame(
        deployment_rows
    )

    targets_df.to_csv(
        targets_path,
        index=False,
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    if len(results_df):

        summary_df = (
            results_df[
                [
                    "target_label",
                    "deploy_mix",
                    "epistemic_utility",
                    "category_sufficient_epistemic",
                    "category_ambiguous_epistemic",
                    "category_insufficient_epistemic",
                    "max_category_undershoot",
                    "all_category_floors_met",
                ]
            ]
            .sort_values(
                [
                    "target_label",
                    "deploy_mix",
                ]
            )
        )

    else:

        summary_df = pd.DataFrame(
            columns=[
                "target_label",
                "deploy_mix",
                "epistemic_utility",
                "category_sufficient_epistemic",
                "category_ambiguous_epistemic",
                "category_insufficient_epistemic",
                "max_category_undershoot",
                "all_category_floors_met",
            ]
        )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # =========================================================================
    # Final report
    # =========================================================================

    feasible_count = (
        int(
            targets_df["feasible"].sum()
        )
        if len(targets_df)
        else 0
    )

    total_count = len(
        target_tuples
    )

    print()
    print("=" * 78)
    print("EXPERIMENT COMPLETE")
    print("=" * 78)

    print()
    print(
        f"Results:  {results_path}"
    )

    print(
        f"Targets:  {targets_path}"
    )

    print(
        f"Summary:  {summary_path}"
    )

    print()
    print(
        f"Feasible target configurations: "
        f"{feasible_count}/{total_count}"
    )

    if len(results_df):

        print()
        print("-" * 78)
        print(
            "EASIER DEPLOYMENT: 70/20/10"
        )
        print("-" * 78)

        easier = results_df[
            results_df["deploy_mix"]
            == "easier_70_20_10"
        ].copy()

        print(
            easier[
                [
                    "target_label",
                    "epistemic_utility",
                    "category_sufficient_epistemic",
                    "category_ambiguous_epistemic",
                    "category_insufficient_epistemic",
                    "max_category_undershoot",
                    "all_category_floors_met",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

        print()
        print("-" * 78)
        print("ALL DEPLOYMENT RESULTS")
        print("-" * 78)

        print(
            results_df[
                [
                    "target_label",
                    "deploy_mix",
                    "epistemic_utility",
                    "max_category_undershoot",
                    "all_category_floors_met",
                ]
            ]
            .sort_values(
                [
                    "target_label",
                    "deploy_mix",
                ]
            )
            .round(4)
            .to_string(index=False)
        )

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    print(
        """
The decisive test is whether the independently specified category floors
remain satisfied after the deployment distribution changes.

A successful result is:

    category floors satisfied under calibration
        AND
    category floors satisfied under every deployment mix.

That supports category-conditional protection as a remedy for the specific
pooled-aggregate transport failure.

A failure is also scientifically useful:

    category floors satisfied under calibration
        BUT
    one or more category floors violated after deployment.

That shows evidence-category conditioning alone is insufficient and motivates
finer-grained conditional guarantees, uncertainty-set constraints, or
distributionally robust optimization.

IMPORTANT:

Infeasible target configurations are NOT counted as successful policies.
They remain in the calibration output so the feasibility boundary remains
visible.

The key paper comparison should be:

    original pooled CRS
        vs.
    feasible protected-category CRS

using the SAME deployment mixtures and SAME evaluation definitions.
"""
    )


if __name__ == "__main__":
    main()
