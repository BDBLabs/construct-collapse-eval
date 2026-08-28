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

This experiment tests a stronger alternative:

    E[r_e | evidence=k] >= tau_k  for every evidence category k.

Unlike the "matched category target" experiment, the category floors here
are specified independently rather than copied from the pooled CRS solution.

Method
------
For each target tuple:

    (tau_sufficient, tau_ambiguous, tau_insufficient)

we solve the finite-action constrained problem exactly using
category-specific Lagrange multipliers.

The category-specific Lagrangian is:

    maximize E[r_s]
    subject to

        E[r_e | sufficient]   >= tau_sufficient
        E[r_e | ambiguous]    >= tau_ambiguous
        E[r_e | insufficient] >= tau_insufficient

Because the evidence category is observed, the per-example decision rule
takes the form:

    argmax_a [ r_s(a) + lambda_k * E[r_e(a) | observation] ]

where k is the evidence category of the example.

For each category, lambda_k is calibrated independently by bisection.

The policy is then frozen and evaluated under several deployment mixtures.

This experiment deliberately distinguishes:

    aggregate protection
        from
    category-conditional protection.

A successful result would show that category floors remain satisfied under
distribution shift even when the pooled aggregate floor does not.

An unsuccessful result would be scientifically useful too: it would show that
conditioning on evidence category is still insufficient and motivate finer
conditional guarantees or distributionally robust constraints.

Outputs
-------
results/protected_category_constraint_shift.csv
    One row per target configuration x deployment mix.

results/protected_category_constraint_shift_targets.csv
    Calibration diagnostics and feasibility information.

results/protected_category_constraint_shift_summary.csv
    Compact summary of deployment performance.
"""

from __future__ import annotations

import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Repository import path
# ============================================================================

# Repository uses:
#
#     src/construct_collapse/
#
# Therefore the directory that must be placed on sys.path is <repo>/src,
# NOT merely <repo>.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================================
# Imports from the actual repository
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
    fast_rewards_for_actions,
    evaluate_actions,
)


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CALIBRATE = 500_000
N_DEPLOY = 500_000

TRAIN_MIX = {
    "sufficient": 0.4,
    "ambiguous": 0.3,
    "insufficient": 0.3,
}

# Keep these EXACTLY aligned with the original distribution-shift experiment.
DEPLOY_MIXES = {
    "train_40_30_30": (0.4, 0.3, 0.3),
    "easier_70_20_10": (0.7, 0.2, 0.1),
    "ambiguous_20_50_30": (0.2, 0.5, 0.3),
    "harder_20_30_50": (0.2, 0.3, 0.5),
    "much_harder_10_20_70": (0.1, 0.2, 0.7),
}


# ============================================================================
# Protected category target grid
# ============================================================================
#
# IMPORTANT:
#
# These are independently specified floors.
#
# We intentionally do NOT derive them from the pooled CRS solution.
#
# The sufficient-evidence maximum in this environment is approximately
# 0.737, so 0.74 is expected to be infeasible. Keeping it in the grid is
# useful because it explicitly demonstrates the feasibility boundary.
#
# Moderate:
#     sufficient   0.70
#     ambiguous    0.80
#     insufficient 0.95
#
# Strong:
#     sufficient   0.72
#     ambiguous    0.85
#     insufficient 0.98
#
# Aggressive:
#     sufficient   0.74
#     ambiguous    0.90
#     insufficient 0.99
#
# The aggressive configuration is expected to expose infeasibility rather
# than being silently treated as a successful policy.
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
# Exact expected epistemic reward
# ============================================================================

def expected_action_values(data):
    """
    Compute posterior expected epistemic reward for every action.

    Returns
    -------
    dict
        action -> np.ndarray shape (N,)
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
# Exact protected-category policy
# ============================================================================

def protected_category_actions(
    data,
    lambdas,
):
    """
    Compute the exact deterministic protected-category policy.

    Parameters
    ----------
    data:
        Dataset containing evidence, difficulty and confidence signal.

    lambdas:
        Sequence:

            (lambda_sufficient,
             lambda_ambiguous,
             lambda_insufficient)

    Returns
    -------
    np.ndarray
        Deterministic action for every example.
    """

    if len(lambdas) != len(EVIDENCE):
        raise ValueError(
            f"Expected {len(EVIDENCE)} lambdas, got {len(lambdas)}"
        )

    expected_re = expected_action_values(data)
    evidence = data["evidence_idx"]

    lambda_by_example = np.asarray(
        lambdas,
        dtype=float,
    )[evidence]

    n = len(evidence)

    values = np.empty(
        (n, len(R_S_BY_ACTION)),
        dtype=float,
    )

    for action in (
        A_CONF,
        A_QUAL,
        A_ABST,
        A_CLAR,
    ):
        values[:, action] = (
            R_S_BY_ACTION[action]
            + lambda_by_example * expected_re[action]
        )

    return values.argmax(axis=1)


# ============================================================================
# Conditional epistemic utility
# ============================================================================

def category_epistemic_utilities(data, actions):
    """
    Compute realized epistemic utility separately by evidence category.
    """

    r_e, _ = fast_rewards_for_actions(
        data,
        actions,
    )

    result = {}

    for evidence_idx, evidence_name in enumerate(EVIDENCE):

        mask = data["evidence_idx"] == evidence_idx

        if mask.any():
            result[evidence_name] = float(
                r_e[mask].mean()
            )
        else:
            result[evidence_name] = np.nan

    return result


# ============================================================================
# Category-local policy
# ============================================================================

def category_actions(
    category_data,
    lambda_value,
):
    """
    Compute the exact policy for one evidence category.

    Since every example in category_data has the same evidence state,
    lambda_value applies to every example.
    """

    expected_re = expected_action_values(
        category_data
    )

    n = len(category_data["evidence_idx"])

    values = np.empty(
        (n, len(R_S_BY_ACTION)),
        dtype=float,
    )

    for action in (
        A_CONF,
        A_QUAL,
        A_ABST,
        A_CLAR,
    ):
        values[:, action] = (
            R_S_BY_ACTION[action]
            + lambda_value * expected_re[action]
        )

    return values.argmax(axis=1)


# ============================================================================
# Category calibration
# ============================================================================

def optimize_single_category(
    data,
    evidence_idx,
    target,
    lambda_hi=LAMBDA_HI,
    iters=BISECTION_ITERS,
):
    """
    Find the smallest lambda for a category such that the realized
    conditional epistemic utility reaches target.

    Returns
    -------
    dict
        lambda
        achieved
        feasible
        max_achievable
    """

    category_mask = (
        data["evidence_idx"] == evidence_idx
    )

    if not category_mask.any():
        return {
            "lambda": np.nan,
            "achieved": np.nan,
            "feasible": False,
            "max_achievable": np.nan,
        }

    # Keep only ndarray fields and remove the RNG object.
    category_data = {
        key: value[category_mask]
        for key, value in data.items()
        if isinstance(value, np.ndarray)
    }

    def evaluate_lambda(lambda_value):
        actions = category_actions(
            category_data,
            lambda_value,
        )

        r_e, _ = fast_rewards_for_actions(
            category_data,
            actions,
        )

        return float(r_e.mean())

    # ------------------------------------------------------------------------
    # Baseline: lambda = 0
    # ------------------------------------------------------------------------

    achieved_zero = evaluate_lambda(0.0)

    # ------------------------------------------------------------------------
    # Upper bound
    # ------------------------------------------------------------------------

    achieved_hi = evaluate_lambda(
        lambda_hi
    )

    max_achievable = achieved_hi

    # Target cannot be achieved even with a very strong epistemic multiplier.
    if achieved_hi + 1e-12 < target:
        return {
            "lambda": lambda_hi,
            "achieved": achieved_hi,
            "feasible": False,
            "max_achievable": max_achievable,
        }

    # Already satisfies target with pure smoothness optimization.
    if achieved_zero >= target:
        return {
            "lambda": 0.0,
            "achieved": achieved_zero,
            "feasible": True,
            "max_achievable": max_achievable,
        }

    # ------------------------------------------------------------------------
    # Bisection
    # ------------------------------------------------------------------------

    lo = 0.0
    hi = lambda_hi

    for _ in range(iters):

        mid = 0.5 * (lo + hi)

        achieved_mid = evaluate_lambda(
            mid
        )

        if achieved_mid >= target:
            hi = mid
        else:
            lo = mid

    achieved_final = evaluate_lambda(
        hi
    )

    return {
        "lambda": hi,
        "achieved": achieved_final,
        "feasible": True,
        "max_achievable": max_achievable,
    }


# ============================================================================
# Calibrate complete protected policy
# ============================================================================

def calibrate_protected_policy(
    calibration_data,
    target_tuple,
):
    """
    Calibrate one lambda independently for each evidence category.
    """

    category_results = {}
    lambdas = []

    for evidence_idx, category in enumerate(EVIDENCE):

        target = target_tuple[evidence_idx]

        result = optimize_single_category(
            data=calibration_data,
            evidence_idx=evidence_idx,
            target=target,
        )

        category_results[category] = {
            "target": target,
            **result,
        }

        lambdas.append(
            result["lambda"]
        )

    feasible = all(
        category_results[category]["feasible"]
        for category in EVIDENCE
    )

    return {
        "lambdas": tuple(lambdas),
        "feasible": feasible,
        "categories": category_results,
    }


# ============================================================================
# Calibration record
# ============================================================================

def make_calibration_record(
    target_tuple,
    calibration,
):
    record = {
        "target_sufficient": target_tuple[0],
        "target_ambiguous": target_tuple[1],
        "target_insufficient": target_tuple[2],
        "feasible": calibration["feasible"],
        "lambda_sufficient": calibration["lambdas"][0],
        "lambda_ambiguous": calibration["lambdas"][1],
        "lambda_insufficient": calibration["lambdas"][2],
    }

    for category in EVIDENCE:

        result = calibration["categories"][category]

        record[
            f"{category}_achieved_calibration"
        ] = result["achieved"]

        record[
            f"{category}_max_calibration"
        ] = result["max_achievable"]

        record[
            f"{category}_feasible"
        ] = result["feasible"]

    return record


# ============================================================================
# Target label
# ============================================================================

def target_label(target_tuple):
    return (
        f"s{target_tuple[0]:.2f}_"
        f"a{target_tuple[1]:.2f}_"
        f"i{target_tuple[2]:.2f}"
    )


# ============================================================================
# Main experiment
# ============================================================================

def main():

    print("=" * 78)
    print("PROTECTED-CATEGORY CONSTRAINT DISTRIBUTION-SHIFT EXPERIMENT")
    print("=" * 78)
    print()

    print(
        f"Repository:          {REPO_ROOT}"
    )
    print(
        f"Calibration N:       {N_CALIBRATE:,}"
    )
    print(
        f"Evaluation N/mix:    {N_DEPLOY:,}"
    )
    print(
        f"Calibration mix:     {TRAIN_MIX}"
    )
    print()

    print("Category target grid:")

    for category, values in CATEGORY_TARGET_GRID.items():
        print(
            f"  {category:<13} {values}"
        )

    print()

    # =========================================================================
    # Calibration dataset
    # =========================================================================

    data_train = generate_dataset(
        N_CALIBRATE,
        seed=0,
        evidence_probs=(
            TRAIN_MIX["sufficient"],
            TRAIN_MIX["ambiguous"],
            TRAIN_MIX["insufficient"],
        ),
    )

    # =========================================================================
    # Target configurations
    # =========================================================================

    target_combinations = list(
        product(
            CATEGORY_TARGET_GRID["sufficient"],
            CATEGORY_TARGET_GRID["ambiguous"],
            CATEGORY_TARGET_GRID["insufficient"],
        )
    )

    print(
        f"Target configurations: "
        f"{len(target_combinations)}"
    )

    # =========================================================================
    # Calibration
    # =========================================================================

    calibrations = []

    print()
    print("=" * 78)
    print("CALIBRATION")
    print("=" * 78)

    for target_tuple in target_combinations:

        calibration = calibrate_protected_policy(
            calibration_data=data_train,
            target_tuple=target_tuple,
        )

        record = make_calibration_record(
            target_tuple,
            calibration,
        )

        calibrations.append(
            {
                "targets": target_tuple,
                "calibration": calibration,
                "record": record,
            }
        )

        status = (
            "FEASIBLE"
            if calibration["feasible"]
            else "INFEASIBLE"
        )

        print()
        print(
            f"Targets: "
            f"s={target_tuple[0]:.3f} "
            f"a={target_tuple[1]:.3f} "
            f"i={target_tuple[2]:.3f}"
        )

        for category in EVIDENCE:

            result = calibration["categories"][category]

            print(
                f"  {category:<13}"
                f" lambda={result['lambda']:.8f}"
                f" target={result['target']:.6f}"
                f" achieved={result['achieved']:.6f}"
                f" max={result['max_achievable']:.6f}"
                f" "
                f"{'OK' if result['feasible'] else 'INFEASIBLE'}"
            )

        print(
            f"  Overall: {status}"
        )

    # =========================================================================
    # Deployment evaluation
    # =========================================================================

    rows = []

    print()
    print("=" * 78)
    print("DEPLOYMENT EVALUATION")
    print("=" * 78)

    # Generate each deployment dataset once and reuse it across all target
    # configurations. This gives us common random numbers and makes
    # configuration-to-configuration comparisons cleaner.
    deployment_data = {}

    for mix_name, mix in DEPLOY_MIXES.items():

        print(
            f"Generating deployment data: "
            f"{mix_name}"
        )

        deployment_data[mix_name] = generate_dataset(
            N_DEPLOY,
            seed=1,
            evidence_probs=mix,
        )

    for calibration_record in calibrations:

        target_tuple = calibration_record["targets"]
        calibration = calibration_record["calibration"]

        label = target_label(
            target_tuple
        )

        # Infeasible policies are recorded in the target CSV but are NOT
        # presented as successful protected policies.
        if not calibration["feasible"]:
            continue

        lambdas = calibration["lambdas"]

        print()
        print("-" * 78)
        print(
            f"TARGET CONFIGURATION: {label}"
        )
        print(
            f"Lambdas: "
            f"s={lambdas[0]:.6f} "
            f"a={lambdas[1]:.6f} "
            f"i={lambdas[2]:.6f}"
        )

        for mix_name, mix in DEPLOY_MIXES.items():

            data_deploy = deployment_data[
                mix_name
            ]

            # ---------------------------------------------------------------
            # Protected policy
            # ---------------------------------------------------------------

            actions = protected_category_actions(
                data=data_deploy,
                lambdas=lambdas,
            )

            metrics = evaluate_actions(
                data_deploy,
                actions,
            )

            category_utils = (
                category_epistemic_utilities(
                    data_deploy,
                    actions,
                )
            )

            # ---------------------------------------------------------------
            # Category floor checks
            # ---------------------------------------------------------------

            undershoots = {}

            for idx, category in enumerate(EVIDENCE):

                target = target_tuple[idx]
                achieved = category_utils[category]

                undershoots[category] = max(
                    0.0,
                    target - achieved,
                )

            max_undershoot = max(
                undershoots.values()
            )

            all_floors_met = (
                max_undershoot <= 1e-12
            )

            # ---------------------------------------------------------------
            # Record
            # ---------------------------------------------------------------

            row = {
                "policy": "protected_category",
                "target_label": label,

                "target_sufficient":
                    target_tuple[0],

                "target_ambiguous":
                    target_tuple[1],

                "target_insufficient":
                    target_tuple[2],

                "lambda_sufficient":
                    lambdas[0],

                "lambda_ambiguous":
                    lambdas[1],

                "lambda_insufficient":
                    lambdas[2],

                "deploy_mix":
                    mix_name,

                "mix_sufficient":
                    mix[0],

                "mix_ambiguous":
                    mix[1],

                "mix_insufficient":
                    mix[2],

                "epistemic_utility":
                    metrics["epistemic_utility"],

                "smoothness_utility":
                    metrics["smoothness_utility"],

                "aggregate_scalar_reward":
                    metrics["aggregate_scalar_reward"],

                "unsupported_certainty_rate":
                    metrics["unsupported_certainty_rate"],

                "calibrated_abstention_rate":
                    metrics["calibrated_abstention_rate"],

                "selective_risk":
                    metrics["selective_risk"],

                "coverage":
                    metrics["coverage"],

                "category_sufficient_epistemic":
                    category_utils["sufficient"],

                "category_ambiguous_epistemic":
                    category_utils["ambiguous"],

                "category_insufficient_epistemic":
                    category_utils["insufficient"],

                "sufficient_undershoot":
                    undershoots["sufficient"],

                "ambiguous_undershoot":
                    undershoots["ambiguous"],

                "insufficient_undershoot":
                    undershoots["insufficient"],

                "max_category_undershoot":
                    max_undershoot,

                "all_category_floors_met":
                    all_floors_met,
            }

            rows.append(row)

            print()
            print(
                f"Deployment: {mix_name}"
            )

            print(
                f"  epistemic="
                f"{metrics['epistemic_utility']:.4f}"
            )

            print(
                f"  sufficient="
                f"{category_utils['sufficient']:.4f}"
                f" / target "
                f"{target_tuple[0]:.4f}"
            )

            print(
                f"  ambiguous="
                f"{category_utils['ambiguous']:.4f}"
                f" / target "
                f"{target_tuple[1]:.4f}"
            )

            print(
                f"  insufficient="
                f"{category_utils['insufficient']:.4f}"
                f" / target "
                f"{target_tuple[2]:.4f}"
            )

            print(
                f"  max undershoot="
                f"{max_undershoot:.4f}"
            )

            print(
                f"  ALL FLOORS MET="
                f"{'YES' if all_floors_met else 'NO'}"
            )

    # =========================================================================
    # Write target diagnostics
    # =========================================================================

    results_path = (
        RESULTS_DIR
        / "protected_category_constraint_shift.csv"
    )

    targets_path = (
        RESULTS_DIR
        / "protected_category_constraint_shift_targets.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "protected_category_constraint_shift_summary.csv"
    )

    results_df = pd.DataFrame(
        rows
    )

    targets_df = pd.DataFrame(
        [
            entry["record"]
            for entry in calibrations
        ]
    )

    # =========================================================================
    # Summary
    # =========================================================================

    if not results_df.empty:

        summary_rows = []

        for label, group in results_df.groupby(
            "target_label",
            sort=True,
        ):

            first = group.iloc[0]

            easier_mask = (
                group["deploy_mix"]
                == "easier_70_20_10"
            )

            if easier_mask.any():
                easier_row = group.loc[
                    easier_mask
                ].iloc[0]

                easier_epistemic = (
                    easier_row[
                        "epistemic_utility"
                    ]
                )

                easier_undershoot = (
                    easier_row[
                        "max_category_undershoot"
                    ]
                )

                easier_all_met = bool(
                    easier_row[
                        "all_category_floors_met"
                    ]
                )

            else:
                easier_epistemic = np.nan
                easier_undershoot = np.nan
                easier_all_met = False

            summary_rows.append(
                {
                    "target_label":
                        label,

                    "target_sufficient":
                        first["target_sufficient"],

                    "target_ambiguous":
                        first["target_ambiguous"],

                    "target_insufficient":
                        first["target_insufficient"],

                    "deployment_count":
                        len(group),

                    "deployments_all_floors_met":
                        int(
                            group[
                                "all_category_floors_met"
                            ].sum()
                        ),

                    "worst_category_undershoot":
                        group[
                            "max_category_undershoot"
                        ].max(),

                    "mean_epistemic_utility":
                        group[
                            "epistemic_utility"
                        ].mean(),

                    "easier_epistemic_utility":
                        easier_epistemic,

                    "easier_max_undershoot":
                        easier_undershoot,

                    "easier_all_floors_met":
                        easier_all_met,
                }
            )

        summary_df = pd.DataFrame(
            summary_rows
        )

    else:
        summary_df = pd.DataFrame()

    # =========================================================================
    # Save
    # =========================================================================

    results_df.to_csv(
        results_path,
        index=False,
    )

    targets_df.to_csv(
        targets_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # =========================================================================
    # Final output
    # =========================================================================

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

    feasible_count = int(
        targets_df["feasible"].sum()
    )

    total_count = len(
        targets_df
    )

    print()
    print(
        f"Feasible target configurations: "
        f"{feasible_count}/{total_count}"
    )

    # =========================================================================
    # Easier deployment table
    # =========================================================================

    if not results_df.empty:

        print()
        print("-" * 78)
        print("EASIER DEPLOYMENT: 70/20/10")
        print("-" * 78)

        easier = results_df[
            results_df["deploy_mix"]
            == "easier_70_20_10"
        ].copy()

        columns = [
            "target_label",
            "epistemic_utility",
            "category_sufficient_epistemic",
            "category_ambiguous_epistemic",
            "category_insufficient_epistemic",
            "max_category_undershoot",
            "all_category_floors_met",
        ]

        print(
            easier[
                columns
            ]
            .sort_values(
                "max_category_undershoot"
            )
            .round(4)
            .to_string(
                index=False
            )
        )

        # =========================================================================
        # Full deployment table
        # =========================================================================

        print()
        print("-" * 78)
        print("ALL DEPLOYMENT RESULTS")
        print("-" * 78)

        columns = [
            "target_label",
            "deploy_mix",
            "epistemic_utility",
            "max_category_undershoot",
            "all_category_floors_met",
        ]

        print(
            results_df[
                columns
            ]
            .sort_values(
                [
                    "target_label",
                    "deploy_mix",
                ]
            )
            .round(4)
            .to_string(
                index=False
            )
        )

    # =========================================================================
    # Scientific interpretation
    # =========================================================================

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    print(
        """
The decisive test is whether the independently specified category floors
remain satisfied after the deployment distribution changes.

A successful result would be:

    category floors satisfied under calibration
        AND
    category floors satisfied under every deployment mix.

That would support category-conditional protection as a remedy for the
specific pooled-aggregate transport failure.

A failure would also be informative:

    category floors satisfied under calibration
        BUT
    one or more category floors violated after deployment.

That would show that evidence-category conditioning alone is insufficient
and would motivate finer-grained conditional guarantees, uncertainty-set
constraints, or distributionally robust optimization.

IMPORTANT:

Infeasible target configurations are not counted as successful policies.
They remain in the calibration output precisely so the feasibility boundary
is visible.

The key comparison for the paper should therefore be between feasible
protected policies and the original pooled CRS policy, using the SAME
deployment mixtures and the SAME evaluation definitions.
"""
    )


if __name__ == "__main__":
    main()