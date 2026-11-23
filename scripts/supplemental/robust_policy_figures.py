"""robust_policy_figures.py - Scenario-discovery scatters (satisficing and
regret) and operating rules of one design's most robust policy under a
satisficing criterion set, or of each of its FFMP-dominating policies.

Selects the policy of a design's re-evaluated Pareto set with the highest
E_test satisficing robustness under ``--criterion`` (ties broken by distance
to the ideal point; :func:`src.factor_mapping.select_compromise`), or, with
``--select ffmp_dominating``, every policy that dominates the FFMP incumbent
on the historic trace re-run under ``--demand`` (the highlighted lines of
``historic_nyc_demand_{demand}_merged_parallel_coords_dominance.png``), then
renders for each, with the existing builders and no plotting code of its own:

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
``most_robust_{criterion}_s{solution_id}`` (``ffmp_dominating_...`` under
``--select ffmp_dominating``). The figures name the policy "Most robust
policy #{solution_id}" or "FFMP-dominating robust policy #{solution_id}":

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
                          frequency, and the incumbent's robustness (one
                          ``ffmp_dominating_{criterion}_selection.csv`` with
                          a row per policy under ``--select ffmp_dominating``)

The per-SOW pass/fail and regret tables behind the scatters go to the figure-tables
tree with every other policy profile
(``outputs/comparison/{slug}/{tag}/figure_tables/policy_profile/``), as do
the factor-map fits of every panel (``..._logit_fits_...csv``: class counts,
CV skill, pseudo-R^2, factor importances, coefficients). Fit skill is
reported there only, never in the figures.

Usage (repo root, venv active; light, one core):
    python scripts/supplemental/robust_policy_figures.py
    python scripts/supplemental/robust_policy_figures.py --criterion nyc_supply
    python scripts/supplemental/robust_policy_figures.py \\
        --select ffmp_dominating --figures scenario_discovery_joint_gbt
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
from src.solution_selection import dominance_mask, load_natural_front  # noqa: E402

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


#: Every figure of one policy, by output-stem suffix (``--figures``).
FIGURES = (*SCATTERS,
           *(f"scenario_discovery_joint_{m}" for m in JOINT_COMPARISON_MODELS),
           *PERFORMANCE_CDFS, "rules")

#: ``--select`` rule -> (output-stem prefix, display name of a policy). The
#: names say how the policy was selected, not which design searched it: every
#: figure here is of one design, and its name ("Historical") reads as the
#: historical operating policy rather than an optimized one.
SELECTIONS = {
    "most_robust": ("most_robust", "Most robust policy #{sid}"),
    "ffmp_dominating": ("ffmp_dominating", "FFMP-dominating robust policy #{sid}"),
}

#: Legend text of the design's Pareto set in the performance-CDF figures.
PARETO_SET_LABEL = "All {n} Pareto-approximate policies"


def policy_record(res, cset, sid: int, rule: str,
                  robustness: float | None = None) -> dict:
    """Provenance of policy ``sid`` under ``cset``, from the criteria scorecard.

    ``robustness`` is a recomputed value to cross-check against the persisted
    scorecard (the robustness axis of the parallel-axes figures), so a
    criteria variant that differs from the scored one fails here rather than
    yielding figures that disagree with the scorecard.
    """
    card = pd.read_csv(res.path / "robustness_scorecard_criteria.csv",
                       index_col="solution_id")
    scored = card[f"sat_set__{cset.key}"]
    low_regret = card.get(f"no_harm_freq_tau__{cset.key}")
    if robustness is None:
        robustness = float(scored.loc[sid])
    elif not np.isclose(scored.loc[sid], robustness):
        sys.exit(f"[robust_policy] '{cset.key}' robustness of policy {sid} is "
                 f"{robustness:.3f} under criteria variant "
                 f"'{active_variant()}' but {scored.loc[sid]:.3f} in the "
                 f"scorecard; rescore the design or set NYCOPT_CRITERIA_VARIANT.")

    inc_sat = rd.incumbent_satisfaction(res, rd.criterion_thresholds(res, cset))
    return {
        "design": res.design, "criterion": cset.key,
        "criteria_variant": active_variant(), "rule": rule,
        "solution_id": sid, "robustness": robustness,
        "rank_among_policies": int((scored > scored.loc[sid]).sum()) + 1,
        "n_policies": len(scored),
        "regret_frequency": (1.0 - float(low_regret.loc[sid])
                             if low_regret is not None else np.nan),
        "incumbent_robustness": (float(np.nanmean(inc_sat.all(axis=1)))
                                 if inc_sat is not None else np.nan),
        "n_sow": len(res.raw.sow_labels),
        "realizations_per_sow": res.raw.realizations_per_sow,
    }


def select_most_robust(res, cset) -> dict:
    """The design's most robust policy under ``cset``, with its provenance."""
    thr = rd.criterion_thresholds(res, cset)
    pick = select_compromise(res.raw, "best_satisficing", thresholds=thr)
    return policy_record(res, cset, pick["solution_id"], pick["rule"],
                         robustness=pick["satisficing"])


def ffmp_dominating_ids(res, slug: str, demand: str) -> list[int]:
    """Re-evaluated policies that dominate the FFMP incumbent on the historic
    trace re-run under NYC demand ``demand``.

    The same test as ``constant_demand_figures.py``'s dominance figure: weak
    Pareto dominance, exact comparison, on the full-precision objectives of
    ``reeval/historic_nyc_demand_{demand}/`` (``set_row`` == ``solution_id``).
    """
    obj_names = list(res.raw.obj_names)
    cd_dir = res.path.parent / f"historic_nyc_demand_{demand}"
    cd = pd.read_csv(cd_dir / "sets" / f"{slug}_merged.csv", index_col="set_row")
    pol = pd.read_csv(cd_dir / "policies.csv")
    baseline = pol.loc[pol.policy_id == "baseline", obj_names].to_numpy(float)[0]
    sol_ids = np.asarray(res.raw.solution_ids, dtype=int)
    objs = cd.loc[sol_ids, [f"{o}__{demand}" for o in obj_names]].to_numpy(float)
    dirs = np.where([res.raw.directions[n] == "maximize" for n in obj_names], 1, -1)
    ok = ~np.isnan(objs).any(axis=1)
    return [int(i) for i in sol_ids[ok][dominance_mask(objs[ok], baseline, dirs)]]


def render_policy(a, cset, sel: dict, prefix: str, label: str,
                  out_dir: Path, figures: tuple) -> Path:
    """Render ``figures`` of the policy of ``sel``; returns the table dir."""
    sid = sel["solution_id"]

    # The policy-profile builders take the profiled policy, its display name
    # and the joint (focal) criterion from the environment.
    os.environ[pp.PROFILE_POLICY_ENV] = f"{a.design}:{sid}"
    os.environ[pp.PROFILE_LABEL_ENV] = label
    os.environ[FOCAL_CRITERION_ENV] = cset.key
    ctx = FigureContext(a.tag, a.slug)
    table_dir = ctx.tables("policy_profile", frozenset({"profile_cube"}))
    for suffix, builder in SCATTERS.items():
        if suffix not in figures:
            continue
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
        suffix = f"scenario_discovery_joint_{model}"
        if suffix not in figures:
            continue
        stem = f"{prefix}_{suffix}"
        info = pp.fig_policy_profile_joint_failure_map(
            ctx, out_dir / stem, table_dir, surface_model=model)
        # The joint panel must show the scorecard's robustness.
        if not np.isclose(info["joint_pass"], sel["robustness"]):
            sys.exit(f"[robust_policy] {stem}: joint pass fraction "
                     f"{info['joint_pass']:.3f} differs from the scorecard's "
                     f"{sel['robustness']:.3f}.")
        print(f"[figure] {stem}.png")
    for suffix, axes in PERFORMANCE_CDFS.items():
        if suffix not in figures:
            continue
        stem = f"{prefix}_{suffix}"
        pp.fig_policy_profile_performance_cdfs(ctx, out_dir / stem, table_dir,
                                               axes=axes,
                                               set_label=PARETO_SET_LABEL)
        print(f"[figure] {stem}.png")

    if "rules" in figures:
        set_file = (config.OUTPUTS_DIR / a.design / a.slug / "sets"
                    / f"{a.slug}{a.set_suffix}.set")
        dv, *_ = load_natural_front(set_file, a.formulation)
        plot_policy_rules_compact(dv[sid], a.formulation, candidate_label=label,
                                  output_file=out_dir / f"{prefix}_rules")
        print(f"[figure] {prefix}_rules.png")
    plt.close("all")
    return table_dir


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
    p.add_argument("--select", choices=tuple(SELECTIONS), default="most_robust",
                   help="most_robust: the one most robust policy under "
                        "--criterion; ffmp_dominating: every policy that "
                        "dominates the FFMP incumbent under --demand")
    p.add_argument("--demand", default="constant_max",
                   help="NYC demand of the historic re-run behind "
                        "--select ffmp_dominating")
    p.add_argument("--figures", nargs="+", choices=FIGURES, default=FIGURES,
                   metavar="FIGURE",
                   help=f"figures to render (default all): {', '.join(FIGURES)}")
    a = p.parse_args()

    cset = criterion_by_key(a.criterion)
    res = rd.load_design_results(a.tag, slug=a.slug, designs=[a.design])[a.design]
    stem, label = SELECTIONS[a.select]
    if a.select == "most_robust":
        sels = [select_most_robust(res, cset)]
    else:
        ids = ffmp_dominating_ids(res, a.slug, a.demand)
        if not ids:
            sys.exit(f"[robust_policy] no policy of {a.design} dominates the "
                     f"FFMP incumbent under NYC demand '{a.demand}'.")
        sels = [policy_record(res, cset, sid, f"ffmp_dominating_{a.demand}")
                for sid in ids]

    out_dir = (config.figure_dir_for(a.design, a.slug, "policy_inspection")
               / f"robustness_{a.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.select == "most_robust":
        selection_csv = f"{stem}_{cset.key}_s{sels[0]['solution_id']:04d}_selection.csv"
    else:
        selection_csv = f"{stem}_{cset.key}_selection.csv"
    pd.DataFrame(sels).to_csv(out_dir / selection_csv, index=False)

    style.apply_style()
    style.FIGURE_FORMATS = ("png",)
    for sel in sels:
        sid = sel["solution_id"]
        print(f"[robust_policy] {a.design}: policy {sid} ({a.select}) meets "
              f"'{cset.label}' in {sel['robustness']:.3f} of {sel['n_sow']} "
              f"E_test SOWs (rank {sel['rank_among_policies']} of "
              f"{sel['n_policies']}; incumbent {sel['incumbent_robustness']:.3f}); "
              f"high regret against the incumbent in {sel['regret_frequency']:.3f}")
        table_dir = render_policy(
            a, cset, sel, f"{stem}_{cset.key}_s{sid:04d}",
            label.format(sid=sid), out_dir, tuple(a.figures))
    print(f"figures -> {out_dir}\ntables  -> {table_dir}")


if __name__ == "__main__":
    main()
