#!/usr/bin/env python3
"""
Original pooled CRS vs. protected-category CRS.

Compares:

1. Original pooled CRS
   - One global epistemic floor calibrated on the 40/30/30 mixture.

2. Protected CRS
   - Independent category-conditional epistemic floors.
   - Evaluates every feasible configuration from the protected-category
     constraint-shift experiment.

Both policies are calibrated once and then frozen for evaluation under
the same deployment distributions.

Outputs:
    results/pooled_vs_protected_crs.csv
    results/pooled_vs_protected_crs_summary.csv
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

CALIBRATION_N = 500_000
EVALUATION_N = 500_000

CALIBRATION_MIX = {
    "sufficient": 0.40,
    "ambiguous": 0.30,
    "insufficient": 0.30,
}

DEPLOYMENT_MIXES = {
    "train_40_30_30": {
        "sufficient": 0.40,
        "ambiguous": 0.30,
        "insufficient": 0.30,
    },
    "easier_70_20_10": {
        "sufficient": 0.70,
        "ambiguous": 0.20,
        "insufficient": 0.10,
    },
    "ambiguous_20_50_30": {
        "sufficient": 0.20,
        "ambiguous": 0.50,
        "insufficient": 0.30,
    },
    "harder_20_30_50": {
        "sufficient": 0.20,
        "ambiguous": 0.30,
        "insufficient": 0.50,
    },
    "much_harder_10_20_70": {
        "sufficient": 0.10,
        "ambiguous": 0.20,
        "insufficient": 0.70,
    },
}

# Protected target grid used by the preceding experiment.
PROTECTED_TARGETS = {
    "sufficient": [0.70, 0.72, 0.74],
    "ambiguous": [0.80, 0.85, 0.90],
    "insufficient": [0.95, 0.98, 0.99],
}

# Original pooled CRS target.
POOLED_TARGET = 0.85

# Same lambda ceiling used by the protected experiment.
MAX_LAMBDA = 200.0

# Reproducibility.
CALIBRATION_SEED = 20260828
DEPLOYMENT_SEED = 20260829


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    category: str
    base_epistemic: float


@dataclass
class CalibrationResult:
    lambda_value: float
    achieved: float
    maximum: float
    feasible: bool


@dataclass
class Policy:
    name: str
    kind: str
    pooled_lambda: float | None
    category_lambdas: Dict[str, float] | None
    targets: Dict[str, float] | None
    calibration_feasible: bool


# ---------------------------------------------------------------------------
# Model / scoring function
# ---------------------------------------------------------------------------

def protected_score(base_epistemic: float, lambda_value: float) -> float:
    """
    Monotone constraint transformation.

    This is intentionally the same functional form used by the protected
    category experiment:

        score = base + lambda * base * (1 - base)

    followed by clipping to [0, 1].

    Larger lambda values increase epistemic utility while diminishing as
    base epistemic confidence approaches 1.
    """
    score = base_epistemic + lambda_value * base_epistemic * (
        1.0 - base_epistemic
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Synthetic evidence generator
# ---------------------------------------------------------------------------

def category_base_distribution(category: str, rng: random.Random) -> float:
    """
    Generate a base epistemic value for a category.

    The parameters reproduce the qualitative category structure used by the
    protected-category experiment:

        sufficient    -> highest baseline
        ambiguous     -> middle baseline
        insufficient  -> lowest baseline

    The exact calibration behavior is driven by the same transformation
    used by the preceding experiment.
    """

    if category == "sufficient":
        # Centered around ~0.45
        value = rng.betavariate(9.0, 11.0)

    elif category == "ambiguous":
        # Centered around ~0.34
        value = rng.betavariate(7.0, 14.0)

    elif category == "insufficient":
        # Centered around ~0.20
        value = rng.betavariate(5.0, 20.0)

    else:
        raise ValueError(f"Unknown category: {category}")

    return value


def generate_dataset(
    n: int,
    mix: Dict[str, float],
    seed: int,
) -> List[Observation]:
    """
    Generate an evaluation population with the requested category mixture.
    """

    rng = random.Random(seed)

    categories = list(mix.keys())
    weights = [mix[c] for c in categories]

    observations: List[Observation] = []

    for _ in range(n):
        category = rng.choices(categories, weights=weights, k=1)[0]
        base = category_base_distribution(category, rng)

        observations.append(
            Observation(
                category=category,
                base_epistemic=base,
            )
        )

    return observations


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def weighted_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def grouped_mean(
    observations: Iterable[Observation],
    scores: Iterable[float],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for obs, score in zip(observations, scores):
        totals[obs.category] = totals.get(obs.category, 0.0) + score
        counts[obs.category] = counts.get(obs.category, 0) + 1

    return {
        category: totals[category] / counts[category]
        for category in totals
    }


# ---------------------------------------------------------------------------
# Lambda calibration
# ---------------------------------------------------------------------------

def calibrate_lambda(
    observations: List[Observation],
    target: float,
) -> CalibrationResult:
    """
    Find the smallest lambda that reaches the requested epistemic floor.

    If the target cannot be reached before MAX_LAMBDA, the configuration
    is marked infeasible.
    """

    base_scores = [obs.base_epistemic for obs in observations]

    def achieved(lambda_value: float) -> float:
        scores = [
            protected_score(base, lambda_value)
            for base in base_scores
        ]
        return weighted_mean(scores)

    maximum = achieved(MAX_LAMBDA)

    if maximum < target:
        return CalibrationResult(
            lambda_value=MAX_LAMBDA,
            achieved=maximum,
            maximum=maximum,
            feasible=False,
        )

    lo = 0.0
    hi = MAX_LAMBDA

    for _ in range(70):
        mid = (lo + hi) / 2.0

        if achieved(mid) >= target:
            hi = mid
        else:
            lo = mid

    final_lambda = hi
    final_achieved = achieved(final_lambda)

    return CalibrationResult(
        lambda_value=final_lambda,
        achieved=final_achieved,
        maximum=maximum,
        feasible=True,
    )


# ---------------------------------------------------------------------------
# Pooled CRS calibration
# ---------------------------------------------------------------------------

def calibrate_pooled_policy(
    calibration_data: List[Observation],
) -> Policy:
    """
    Calibrate the original pooled CRS.

    Important:
        The pooled CRS sees only the aggregate population and therefore
        has ONE lambda and ONE aggregate target.
    """

    result = calibrate_lambda(
        calibration_data,
        POOLED_TARGET,
    )

    return Policy(
        name="original_pooled_crs",
        kind="pooled",
        pooled_lambda=result.lambda_value,
        category_lambdas=None,
        targets=None,
        calibration_feasible=result.feasible,
    )


# ---------------------------------------------------------------------------
# Protected CRS calibration
# ---------------------------------------------------------------------------

def calibrate_protected_policies(
    calibration_data: List[Observation],
) -> List[Policy]:
    """
    Calibrate every protected-category target configuration.

    Each category receives an independently calibrated lambda.
    """

    category_data: Dict[str, List[Observation]] = {
        category: []
        for category in PROTECTED_TARGETS
    }

    for obs in calibration_data:
        category_data[obs.category].append(obs)

    policies: List[Policy] = []

    for sufficient_target in PROTECTED_TARGETS["sufficient"]:
        for ambiguous_target in PROTECTED_TARGETS["ambiguous"]:
            for insufficient_target in PROTECTED_TARGETS["insufficient"]:

                targets = {
                    "sufficient": sufficient_target,
                    "ambiguous": ambiguous_target,
                    "insufficient": insufficient_target,
                }

                lambdas: Dict[str, float] = {}
                feasible = True

                for category, target in targets.items():
                    result = calibrate_lambda(
                        category_data[category],
                        target,
                    )

                    lambdas[category] = result.lambda_value

                    if not result.feasible:
                        feasible = False

                label = (
                    f"s{sufficient_target:.2f}_"
                    f"a{ambiguous_target:.2f}_"
                    f"i{insufficient_target:.2f}"
                )

                policies.append(
                    Policy(
                        name=f"protected_crs_{label}",
                        kind="protected",
                        pooled_lambda=None,
                        category_lambdas=lambdas,
                        targets=targets,
                        calibration_feasible=feasible,
                    )
                )

    return policies


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy: Policy,
    observations: List[Observation],
    deployment_mix: Dict[str, float],
) -> Dict[str, float | bool | str]:
    """
    Evaluate a frozen policy.

    No recalibration occurs here.

    This is critical: the policy calibrated on 40/30/30 is transported
    unchanged to the deployment distribution.
    """

    if policy.kind == "pooled":
        assert policy.pooled_lambda is not None

        scores = [
            protected_score(
                obs.base_epistemic,
                policy.pooled_lambda,
            )
            for obs in observations
        ]

    else:
        assert policy.category_lambdas is not None

        scores = [
            protected_score(
                obs.base_epistemic,
                policy.category_lambdas[obs.category],
            )
            for obs in observations
        ]

    category_scores = grouped_mean(observations, scores)
    aggregate = weighted_mean(scores)

    result: Dict[str, float | bool | str] = {
        "policy": policy.name,
        "policy_kind": policy.kind,
        "deploy_mix": "",
        "epistemic_utility": aggregate,
        "category_sufficient_epistemic": category_scores.get(
            "sufficient",
            float("nan"),
        ),
        "category_ambiguous_epistemic": category_scores.get(
            "ambiguous",
            float("nan"),
        ),
        "category_insufficient_epistemic": category_scores.get(
            "insufficient",
            float("nan"),
        ),
    }

    if policy.kind == "protected":
        assert policy.targets is not None

        undershoots = {
            category: max(
                0.0,
                policy.targets[category]
                - category_scores.get(category, 0.0),
            )
            for category in policy.targets
        }

        max_undershoot = max(undershoots.values())

        result["max_category_undershoot"] = max_undershoot
        result["all_category_floors_met"] = (
            max_undershoot <= 0.0
        )

        result["target_sufficient"] = policy.targets["sufficient"]
        result["target_ambiguous"] = policy.targets["ambiguous"]
        result["target_insufficient"] = policy.targets["insufficient"]

    else:
        pooled_undershoot = max(
            0.0,
            POOLED_TARGET - aggregate,
        )

        result["pooled_target"] = POOLED_TARGET
        result["pooled_undershoot"] = pooled_undershoot
        result["pooled_floor_met"] = pooled_undershoot <= 0.0

        # These are deliberately calculated even though the pooled policy
        # has no category-specific constraints. This is the point of the
        # comparison: reveal what the aggregate guarantee says—and does not
        # say—about categories.
        result["max_category_undershoot"] = float("nan")
        result["all_category_floors_met"] = ""

    result["deployment_sufficient_share"] = deployment_mix["sufficient"]
    result["deployment_ambiguous_share"] = deployment_mix["ambiguous"]
    result["deployment_insufficient_share"] = deployment_mix["insufficient"]

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("ORIGINAL POOLED CRS VS. PROTECTED CRS")
    print("=" * 78)
    print()
    print(f"Repository:          {ROOT}")
    print(f"Calibration N:       {CALIBRATION_N:,}")
    print(f"Evaluation N/mix:    {EVALUATION_N:,}")
    print(f"Calibration mix:     {CALIBRATION_MIX}")
    print(f"Pooled CRS target:   {POOLED_TARGET:.2f}")
    print()

    # -----------------------------------------------------------------------
    # Calibration data
    # -----------------------------------------------------------------------

    print("=" * 78)
    print("CALIBRATION")
    print("=" * 78)

    calibration_data = generate_dataset(
        n=CALIBRATION_N,
        mix=CALIBRATION_MIX,
        seed=CALIBRATION_SEED,
    )

    print()
    print("Calibrating original pooled CRS...")

    pooled_policy = calibrate_pooled_policy(
        calibration_data
    )

    if pooled_policy.pooled_lambda is None:
        raise RuntimeError("Pooled lambda was not calibrated.")

    print(
        f"  pooled lambda={pooled_policy.pooled_lambda:.8f}"
    )
    print(
        f"  target={POOLED_TARGET:.6f}"
    )

    # -----------------------------------------------------------------------
    # Protected policies
    # -----------------------------------------------------------------------

    print()
    print("Calibrating protected CRS configurations...")

    protected_policies = calibrate_protected_policies(
        calibration_data
    )

    feasible_protected = [
        policy
        for policy in protected_policies
        if policy.calibration_feasible
    ]

    print(
        f"  protected configurations: "
        f"{len(protected_policies)}"
    )
    print(
        f"  feasible configurations: "
        f"{len(feasible_protected)}"
    )

    print()
    print("Feasible protected policies:")

    for policy in feasible_protected:
        assert policy.targets is not None
        assert policy.category_lambdas is not None

        print(
            "  "
            f"{policy.name.replace('protected_crs_', '')}: "
            f"s={policy.category_lambdas['sufficient']:.6f} "
            f"a={policy.category_lambdas['ambiguous']:.6f} "
            f"i={policy.category_lambdas['insufficient']:.6f}"
        )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    print()
    print("=" * 78)
    print("DEPLOYMENT EVALUATION")
    print("=" * 78)

    all_results: List[Dict[str, object]] = []

    for deployment_index, (
        deployment_name,
        deployment_mix,
    ) in enumerate(DEPLOYMENT_MIXES.items()):

        print()
        print("-" * 78)
        print(f"DEPLOYMENT: {deployment_name}")
        print("-" * 78)

        deployment_data = generate_dataset(
            n=EVALUATION_N,
            mix=deployment_mix,
            seed=DEPLOYMENT_SEED + deployment_index,
        )

        policies = [pooled_policy] + feasible_protected

        for policy in policies:

            result = evaluate_policy(
                policy=policy,
                observations=deployment_data,
                deployment_mix=deployment_mix,
            )

            result["deploy_mix"] = deployment_name

            all_results.append(result)

            print()
            print(f"Policy: {policy.name}")
            print(
                f"  epistemic={result['epistemic_utility']:.4f}"
            )
            print(
                "  sufficient="
                f"{result['category_sufficient_epistemic']:.4f}"
            )
            print(
                "  ambiguous="
                f"{result['category_ambiguous_epistemic']:.4f}"
            )
            print(
                "  insufficient="
                f"{result['category_insufficient_epistemic']:.4f}"
            )

            if policy.kind == "pooled":
                print(
                    f"  pooled target={POOLED_TARGET:.4f}"
                )
                print(
                    f"  pooled floor met="
                    f"{result['pooled_floor_met']}"
                )
            else:
                print(
                    f"  max category undershoot="
                    f"{result['max_category_undershoot']:.6f}"
                )
                print(
                    f"  ALL CATEGORY FLOORS MET="
                    f"{result['all_category_floors_met']}"
                )

    # -----------------------------------------------------------------------
    # Write detailed results
    # -----------------------------------------------------------------------

    detailed_path = (
        RESULTS / "pooled_vs_protected_crs.csv"
    )

    fieldnames = sorted(
        {
            key
            for result in all_results
            for key in result.keys()
        }
    )

    with detailed_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    summary_rows: List[Dict[str, object]] = []

    for deployment_name in DEPLOYMENT_MIXES:
        rows = [
            row
            for row in all_results
            if row["deploy_mix"] == deployment_name
        ]

        pooled_rows = [
            row
            for row in rows
            if row["policy_kind"] == "pooled"
        ]

        protected_rows = [
            row
            for row in rows
            if row["policy_kind"] == "protected"
        ]

        if not pooled_rows:
            continue

        pooled = pooled_rows[0]

        protected_floor_successes = sum(
            bool(row["all_category_floors_met"])
            for row in protected_rows
        )

        protected_count = len(protected_rows)

        best_protected = max(
            protected_rows,
            key=lambda row: float(row["epistemic_utility"]),
            default=None,
        )

        summary_rows.append(
            {
                "deploy_mix": deployment_name,

                "pooled_epistemic_utility":
                    pooled["epistemic_utility"],

                "pooled_sufficient_epistemic":
                    pooled["category_sufficient_epistemic"],

                "pooled_ambiguous_epistemic":
                    pooled["category_ambiguous_epistemic"],

                "pooled_insufficient_epistemic":
                    pooled["category_insufficient_epistemic"],

                "pooled_floor_met":
                    pooled["pooled_floor_met"],

                "protected_feasible_policy_count":
                    protected_count,

                "protected_floor_success_count":
                    protected_floor_successes,

                "protected_floor_success_rate":
                    (
                        protected_floor_successes / protected_count
                        if protected_count
                        else float("nan")
                    ),

                "best_protected_policy":
                    (
                        best_protected["policy"]
                        if best_protected
                        else ""
                    ),

                "best_protected_epistemic_utility":
                    (
                        best_protected["epistemic_utility"]
                        if best_protected
                        else float("nan")
                    ),

                "best_protected_sufficient_epistemic":
                    (
                        best_protected[
                            "category_sufficient_epistemic"
                        ]
                        if best_protected
                        else float("nan")
                    ),

                "best_protected_ambiguous_epistemic":
                    (
                        best_protected[
                            "category_ambiguous_epistemic"
                        ]
                        if best_protected
                        else float("nan")
                    ),

                "best_protected_insufficient_epistemic":
                    (
                        best_protected[
                            "category_insufficient_epistemic"
                        ]
                        if best_protected
                        else float("nan")
                    ),
            }
        )

    summary_path = (
        RESULTS / "pooled_vs_protected_crs_summary.csv"
    )

    summary_fields = list(summary_rows[0].keys())

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    # -----------------------------------------------------------------------
    # Final interpretation
    # -----------------------------------------------------------------------

    print()
    print("=" * 78)
    print("EXPERIMENT COMPLETE")
    print("=" * 78)
    print()
    print(f"Results: {detailed_path}")
    print(f"Summary: {summary_path}")
    print()
    print(
        "The comparison deliberately keeps calibration fixed at 40/30/30."
    )
    print(
        "Policies are NOT recalibrated after the deployment distribution "
        "changes."
    )
    print()
    print(
        "The key question is whether the original pooled CRS aggregate "
        "guarantee"
    )
    print(
        "corresponds to category-level protection under transport."
    )
    print()
    print(
        "For protected CRS, success requires every independently specified "
        "category"
    )
    print(
        "floor to remain satisfied under deployment."
    )
    print()
    print(
        "For pooled CRS, only the aggregate floor is guaranteed."
    )
    print(
        "Its category-specific results are reported diagnostically rather "
        "than treated"
    )
    print(
        "as constraints."
    )
    print()


if __name__ == "__main__":
    main()