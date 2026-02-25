"""
scenario_discovery.py - Scenario discovery using gradient-boosted trees and SHAP.

Identifies which uncertain conditions drive policy failure using XGBoost
classifiers trained on re-evaluation results, with SHAP values for
interpretable feature importance decomposition.

This replaces the traditional PRIM/CART approach with modern ML-based
explainability methods that handle correlated uncertainties and
nonlinear interaction effects.

Usage:
    python scenario_discovery.py \
        --formulation ffmp \
        --reevaluation_file outputs/reevaluation/ffmp/reevaluation_results.csv

Output:
    outputs/figures/<formulation>/
        shap_summary_<policy_idx>.png
        shap_interaction_<policy_idx>.png
        feature_importance_<formulation>.png
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import get_obj_names, get_obj_directions, OUTPUTS_DIR


def train_failure_classifier(
    scenario_features: pd.DataFrame,
    failure_labels: np.ndarray,
) -> "xgb.XGBClassifier":
    """Train an XGBoost classifier to predict policy failure.

    Args:
        scenario_features: DataFrame of scenario characteristics
            (e.g., drought severity, mean inflow, seasonal distribution).
        failure_labels: Binary array (1 = failure, 0 = success).

    Returns:
        Fitted XGBClassifier.
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # Cross-validation accuracy check
    scores = cross_val_score(clf, scenario_features, failure_labels, cv=5)
    print(f"  XGBoost CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Fit on full dataset
    clf.fit(scenario_features, failure_labels)
    return clf


def compute_shap_values(clf, scenario_features: pd.DataFrame):
    """Compute SHAP values for feature importance decomposition.

    Returns:
        shap.Explanation object.
    """
    import shap

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(scenario_features)
    return shap_values


def plot_shap_summary(shap_values, scenario_features, fig_path: Path):
    """Generate SHAP summary plot (beeswarm)."""
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        scenario_features,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_interaction(shap_values, scenario_features, fig_path: Path):
    """Generate SHAP interaction plot for top feature pairs."""
    import shap

    # Get top 2 features by mean |SHAP|
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top2 = np.argsort(mean_abs)[-2:]

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        top2[1],
        shap_values.values,
        scenario_features,
        interaction_index=top2[0],
        show=False,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def define_failure(
    obj_values: pd.DataFrame,
    thresholds: dict,
) -> np.ndarray:
    """Define binary failure labels based on satisficing thresholds.

    A scenario is a "failure" if ANY objective violates its threshold.

    Args:
        obj_values: DataFrame with objective columns.
        thresholds: Dict mapping objective name to threshold value.

    Returns:
        Binary array (1 = failure, 0 = success).
    """
    directions = get_obj_directions()
    obj_names = get_obj_names()

    failure = np.zeros(len(obj_values), dtype=bool)
    for i, name in enumerate(obj_names):
        if name not in thresholds:
            continue
        threshold = thresholds[name]
        if directions[i] == 1:  # maximize: fail if below threshold
            failure |= (obj_values[name] < threshold)
        else:  # minimize: fail if above threshold
            failure |= (obj_values[name] > threshold)

    return failure.astype(int)


def extract_scenario_features(reeval_df: pd.DataFrame) -> pd.DataFrame:
    """Extract scenario characteristics for scenario discovery.

    TODO: When stochastic ensembles are implemented, this should
    extract hydrological features of each realization (mean flow,
    drought severity, seasonality, etc.) from the ensemble metadata.

    For now, returns placeholder features.
    """
    # Placeholder: will be replaced with actual scenario features
    # from the stochastic ensemble metadata
    features = pd.DataFrame({
        "realization_idx": reeval_df["realization_idx"].values,
    })
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Scenario discovery using XGBoost + SHAP."
    )
    parser.add_argument("--formulation", type=str, default="ffmp")
    parser.add_argument("--reevaluation_file", type=str, required=True)
    parser.add_argument("--solution_idx", type=int, default=None,
                        help="Analyze a specific solution (default: all)")
    args = parser.parse_args()

    reeval_file = Path(args.reevaluation_file)
    fig_dir = OUTPUTS_DIR / "figures" / args.formulation / "scenario_discovery"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading re-evaluation results from: {reeval_file}")
    df = pd.read_csv(reeval_file)
    obj_names = get_obj_names()

    # TODO: Set thresholds from baseline analysis
    thresholds = {}

    print(f"Loaded {len(df)} evaluations")
    print(f"Solutions: {df['solution_idx'].nunique()}")
    print(f"Realizations: {df['realization_idx'].nunique()}")

    # Extract scenario features
    scenario_features = extract_scenario_features(df)

    if scenario_features.shape[1] < 2:
        print("\nScenario features not yet available (requires stochastic ensemble).")
        print("Scenario discovery will be implemented after ensemble generation.")
        return

    # Analyze each solution (or a specific one)
    solutions = [args.solution_idx] if args.solution_idx is not None \
        else df["solution_idx"].unique()

    for sol_idx in solutions:
        print(f"\n--- Solution {sol_idx} ---")
        sol_df = df[df["solution_idx"] == sol_idx]

        # Define failure
        failure = define_failure(sol_df[obj_names], thresholds)
        failure_rate = failure.mean()
        print(f"  Failure rate: {failure_rate:.1%}")

        if failure_rate < 0.05 or failure_rate > 0.95:
            print("  Skipping: failure rate too extreme for meaningful discovery.")
            continue

        # Train classifier
        features = scenario_features.loc[sol_df.index]
        clf = train_failure_classifier(features, failure)

        # SHAP analysis
        shap_values = compute_shap_values(clf, features)

        plot_shap_summary(
            shap_values, features,
            fig_dir / f"shap_summary_sol{sol_idx:04d}.png",
        )
        plot_shap_interaction(
            shap_values, features,
            fig_dir / f"shap_interaction_sol{sol_idx:04d}.png",
        )
        print(f"  Plots saved to: {fig_dir}")


if __name__ == "__main__":
    main()
