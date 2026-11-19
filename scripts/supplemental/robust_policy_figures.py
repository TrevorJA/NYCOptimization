"""robust_policy_figures.py - Scenario-discovery scatters (satisficing and
regret) and operating rules of one design's most robust policy under a
satisficing criterion set.

Selects the policy of a design's re-evaluated Pareto set with the highest
E_test satisficing robustness under ``--criterion`` (ties broken by distance
to the ideal point; :func:`src.factor_mapping.select_compromise`), then
renders, with the existing builders and no plotting code of its own:

* the per-criterion pass/fail scatters of every E_test SOW over the DU
  forcing space, with the criterion set's joint panel last
  (:mod:`src.plotting.policy_profile`), on the (volume multiplier,
  utilization) and (volume multiplier, seasonal amplitude) planes, each
  panel over the logistic-regression factor map of its label
  (:mod:`src.factor_mapping`, ``model="logit"``): the fitted probability
  that a SOW passes and its P = 0.5 boundary;
* the joint panel of those two grids alone, as a two-panel figure with the
  criterion set's member conditions written out above the panels
  (:func:`src.plotting.policy_profile.fig_policy_profile_joint_failure_map`),
  and again under each classifier family of ``JOINT_COMPARISON_MODELS``
  (stem suffixed by the family), to compare factor maps of the same label;
* the matching low/high-regret scatters on the same planes: each SOW
  labelled by whether the policy is worse than the FFMP incumbent beyond the
  scored tolerance, per objective, with the criterion set's joint low-regret
  panel last (the per-SOW view of ``no_harm_freq_tau__{criterion}``), over
  the same factor map fitted to the low-regret label;
* the performance curves of the same policy across the E_test SOWs
  (:func:`src.plotting.policy_profile.fig_policy_profile_performance_cdfs`):
  per objective, the SOW fraction performing at least as well as a value,
  for the policy and the FFMP incumbent over every policy of the design's
  re-evaluated Pareto set, on the criterion set's member axes and on all
  objectives;
* the compact operating-rule figure of the same policy against the baseline
  FFMP (:func:`src.plotting.policy_rules_compact.plot_policy_rules_compact`).

Outputs, all under
``outputs/figures/{design}/{slug}/policy_inspection/robustness_{tag}/``
(beside ``parallel_coords/robustness_{tag}/``), stems prefixed
``most_robust_{criterion}_s{solution_id}``:

    ..._scenario_discovery_volume_utilization.png
    ..._scenario_discovery_volume_seasonality.png
    ..._scenario_discovery_joint.png
    ..._scenario_discovery_joint_gbt.png
    ..._regret_discovery_volume_utilization.png
    ..._regret_discovery_volume_seasonality.png
    ..._performance_cdfs.png
    ..._performance_cdfs_all.png
    ..._rules.png
    ..._selection.csv     the selected policy, its robustness and regret
                          frequency, and the incumbent's robustness

The per-SOW pass/fail and regret tables behind the scatters go to the figure-tables
tree with every other policy profile
(``outputs/comparison/{slug}/{tag}/figure_tables/policy_profile/``), as do
the factor-map fits of every panel (``..._logit_fits_...csv``: class counts,
CV skill, pseudo-R^2, factor importances, coefficients). Fit skill is
reported there only, never in the figures.

Usage (repo root, venv active; light, one core):
    python scripts/supplemental/robust_policy_figures.py
    python scripts/supplemental/robust_policy_figures.py --criterion nyc_supply
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src import results_data as rd  # noqa: E402
from src.factor_mapping import select_compromise  # noqa: E402
from src.figures.registry import FigureContext  # noqa: E402
from src.plotting import policy_profile as pp  # noqa: E402
from src.plotting import style  # noqa: E402
from src.plotting.policy_rules_compact import plot_policy_rules_compact  # noqa: E402
from src.satisficing_criteria import (FOCAL_CRITERION_ENV, active_variant,  # noqa: E402
                                      criterion_by_key)
from src.solution_selection import load_natural_front  # noqa: E402

#: Classifier family of the factor maps drawn under the SOW labels
#: (``src.factor_mapping.MODELS``).
SURFACE_MODEL = "logit"

#: Other classifier families the joint-criterion figure is also rendered with,
#: as ``..._scenario_discovery_joint_{family}.png``.
JOINT_COMPARISON_MODELS = ("gbt",)

#: Performance-CDF figures: output-stem suffix -> the objectives shown.
PERFORMANCE_CDFS = {"performance_cdfs": "focal", "performance_cdfs_all": "all"}

#: Scatter figures: output-stem suffix -> policy-profile builder.
SCATTERS = {
    "scenario_discovery_volume_utilization":
        pp.fig_policy_profile_failure_map_volume_utilization,
    "scenario_discovery_volume_seasonality":
        pp.fig_policy_profile_failure_map_volume_seasonality,
    "scenario_discovery_joint":
        pp.fig_policy_profile_joint_failure_map,
    "regret_discovery_volume_utilization":
        pp.fig_policy_profile_regret_map_volume_utilization,
    "regret_discovery_volume_seasonality":
        pp.fig_policy_profile_regret_map_volume_seasonality,
}


def select_most_robust(res, cset) -> dict:
    """The design's most robust policy under ``cset``, with its provenance.

    Cross-checks the recomputed robustness against the persisted criteria
    scorecard (the robustness axis of the parallel-axes figures), so a
    criteria variant that differs from the scored one fails here rather than
    yielding figures that disagree with the scorecard.
    """
    thr = rd.criterion_thresholds(res, cset)
    pick = select_compromise(res.raw, "best_satisficing", thresholds=thr)
    sid = pick["solution_id"]

    card = pd.read_csv(res.path / "robustness_scorecard_criteria.csv",
                       index_col="solution_id")
    scored = card[f"sat_set__{cset.key}"]
    low_regret = card.get(f"no_harm_freq_tau__{cset.key}")
    if not np.isclose(scored.loc[sid], pick["satisficing"]):
        sys.exit(f"[robust_policy] '{cset.key}' robustness of policy {sid} is "
                 f"{pick['satisficing']:.3f} under criteria variant "
                 f"'{active_variant()}' but {scored.loc[sid]:.3f} in the "
                 f"scorecard; rescore the design or set NYCOPT_CRITERIA_VARIANT.")

    inc_sat = rd.incumbent_satisfaction(res, thr)
    return {
        "design": res.design, "criterion": cset.key,
        "criteria_variant": active_variant(), "rule": pick["rule"],
        "solution_id": sid, "robustness": pick["satisficing"],
        "rank_among_policies": int((scored > scored.loc[sid]).sum()) + 1,
        "n_policies": len(scored),
        "regret_frequency": (1.0 - float(low_regret.loc[sid])
                             if low_regret is not None else np.nan),
        "incumbent_robustness": (float(np.nanmean(inc_sat.all(axis=1)))
                                 if inc_sat is not None else np.nan),
        "n_sow": len(res.raw.sow_labels),
        "realizations_per_sow": res.raw.realizations_per_sow,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--design", default="historic")
    p.add_argument("--slug", default="ffmp_obj8")
    p.add_argument("--formulation", default="ffmp")
    p.add_argument("--tag", default="etest_kn_50yr_n12500")
    p.add_argument("--criterion", default="compromise",
                   help="nyc_supply | downstream_flows | flood | compromise")
    p.add_argument("--set-suffix", default="_merged",
                   help="merged .set the re-eval scored (row == solution_id)")
    a = p.parse_args()

    cset = criterion_by_key(a.criterion)
    res = rd.load_design_results(a.tag, slug=a.slug, designs=[a.design])[a.design]
    sel = select_most_robust(res, cset)
    sid = sel["solution_id"]
    print(f"[robust_policy] {a.design}: policy {sid} is the most robust under "
          f"'{cset.label}' ({sel['robustness']:.3f} of {sel['n_sow']} E_test "
          f"SOWs; incumbent {sel['incumbent_robustness']:.3f}); high regret "
          f"against the incumbent in {sel['regret_frequency']:.3f}")

    out_dir = (config.figure_dir_for(a.design, a.slug, "policy_inspection")
               / f"robustness_{a.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"most_robust_{cset.key}_s{sid:04d}"
    pd.DataFrame([sel]).to_csv(out_dir / f"{prefix}_selection.csv", index=False)

    style.apply_style()
    style.FIGURE_FORMATS = ("png",)

    # The policy-profile builders take the profiled policy and the joint
    # (focal) criterion from the environment.
    os.environ[pp.PROFILE_POLICY_ENV] = f"{a.design}:{sid}"
    os.environ[FOCAL_CRITERION_ENV] = cset.key
    ctx = FigureContext(a.tag, a.slug)
    table_dir = ctx.tables("policy_profile", frozenset({"profile_cube"}))
    for suffix, builder in SCATTERS.items():
        stem = f"{prefix}_{suffix}"
        info = builder(ctx, out_dir / stem, table_dir,
                       surface_model=SURFACE_MODEL)
        # The regret maps must decompose the scorecard's regret frequency.
        if "joint_low_regret" in info and not np.isclose(
                1.0 - info["joint_low_regret"], sel["regret_frequency"]):
            sys.exit(f"[robust_policy] {stem}: joint regret frequency "
                     f"{1.0 - info['joint_low_regret']:.3f} differs from the "
                     f"scorecard's {sel['regret_frequency']:.3f}.")
        print(f"[figure] {stem}.png")
    for model in JOINT_COMPARISON_MODELS:
        stem = f"{prefix}_scenario_discovery_joint_{model}"
        pp.fig_policy_profile_joint_failure_map(ctx, out_dir / stem, table_dir,
                                                surface_model=model)
        print(f"[figure] {stem}.png")
    for suffix, axes in PERFORMANCE_CDFS.items():
        stem = f"{prefix}_{suffix}"
        pp.fig_policy_profile_performance_cdfs(ctx, out_dir / stem, table_dir,
                                               axes=axes)
        print(f"[figure] {stem}.png")

    set_file = (config.OUTPUTS_DIR / a.design / a.slug / "sets"
                / f"{a.slug}{a.set_suffix}.set")
    dv, *_ = load_natural_front(set_file, a.formulation)
    label = f"{style.design_label(a.design).split(' (')[0]} policy #{sid}"
    plot_policy_rules_compact(dv[sid], a.formulation, candidate_label=label,
                              output_file=out_dir / f"{prefix}_rules")
    plt.close("all")
    print(f"[figure] {prefix}_rules.png")
    print(f"figures -> {out_dir}\ntables  -> {table_dir}")


if __name__ == "__main__":
    main()
