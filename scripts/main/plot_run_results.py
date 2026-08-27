"""plot_run_results.py - Render the result figures for a single optimization run.

Generates, for one (scenario, moea_slug, reeval preset), after a stakeholder
screen is applied to the re-evaluated reference set (NYC weekly delivery
reliability >= 0.5 -- see :mod:`src.pareto_filter`):

  1. Pareto parallel-coordinates over the objective set, accepted policies vs the
     screened-out ones, with the FFMP baseline overlaid.
  1b/1c. Pairwise objective tradeoff scatters (headline pairs + full
     lower-triangle matrix), colored by NYC delivery reliability, baseline starred.
  1d. Bound-normalized decision-variable ranges of criterion-satisfying subsets
     (baseline dominance, NYC-storage floors) against the full front.
  2. Hypervolume convergence vs NFE (from step-07 .metrics).
  3. DU performance distributions: each objective's DU-expected value across the
     acceptable policies vs the baseline and its satisficing threshold (the raw
     magnitudes co-reported alongside robustness, per Huang et al. 2025).
  4. DU robustness: multivariate satisficing (SOW-unit Starr domain criterion)
     across the acceptable policies, with the FFMP baseline robustness overlaid,
     plus the per-objective decomposition exposing the binding constraint.
  5. Scenario discovery: pass/fail of the most-robust policy across the sampled
     DU forcing factors (theta = m, r1, r2).
  6. Operating rules: the FFMP decision variables of a few representative policies
     (NYC-diversion-priority, Montague-flow-priority, most-robust) vs baseline.

All figures land under ``figures/{scenario}/{slug}/``. Every panel is wrapped so
one failure never blocks the others. Figures 1 (a-d) and 2 also work after step
07 alone; 3-6 need step 08 (the re-eval + robustness outputs).

Usage (from repo root, venv active):
  python3 -m scripts.main.plot_run_results \
      --slug ffmp_obj8 --scenario historic \
      --preset etest_kn_50yr_n25000_first25ch --formulation ffmp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import baseline_objectives_csv
from src.pareto_filter import filter_reference_set


def _run_dir(scenario: str, slug: str) -> Path:
    return Path("outputs") / scenario / slug


def _reeval_ref_set(run_dir: Path, slug: str) -> Path:
    """The exact reference set the re-eval scored (row index == solution_id)."""
    return run_dir / "sets" / f"{slug}_merged.set"


def _load_reeval_long(reeval_dir: Path) -> pd.DataFrame | None:
    p = reeval_dir / "reeval_raw.parquet"
    if not p.exists():
        print(f"[reeval] not found: {p}")
        return None
    return pd.read_parquet(p)


def _load_baseline_vec(formulation, obj_names, scenario):
    """Baseline objectives (natural units) aligned to the front, or None.

    Resolved through ``config.baseline_objectives_csv`` — a baseline vector is
    only comparable to a front evaluated on the SAME substrate, so ensemble
    scenarios must get the search-ensemble-scored vector (step 05
    ``--search-ensemble``), never the flat historic record. An unscored
    scenario yields None and the panels omit the overlay.
    """
    bcsv = baseline_objectives_csv(formulation, scenario)
    if not bcsv.exists():
        print(f"[baseline] not scored for scenario '{scenario}' ({bcsv}); "
              "omitting baseline overlay")
        return None
    row = pd.read_csv(bcsv).iloc[0]
    try:
        return np.array([float(row[n]) for n in obj_names])
    except KeyError:
        return None


# --------------------------------------------------------------------------- #
# Fig 1: Pareto parallel-coordinates, accepted vs screened out
# --------------------------------------------------------------------------- #
def fig_parallel_coords(ref_set, formulation, filt, out_dir, scenario):
    from src.plotting.parallel_coordinates import plot_parallel_coordinates
    plot_parallel_coordinates(ref_set, formulation,
                              out_dir / "run01_pareto_parallel_coords.png",
                              baseline_objs=_load_baseline_vec(formulation,
                                                               filt.obj_names,
                                                               scenario),
                              figsize=(13, 5.5), keep_mask=filt.mask)
    print(f"[fig1] {filt.n_accepted}/{filt.n_total} acceptable "
          f"-> 01_pareto_parallel_coords.png")


# --------------------------------------------------------------------------- #
# Fig 1b/1c: pairwise objective tradeoff scatters (headline pairs + matrix)
# --------------------------------------------------------------------------- #
def fig_tradeoff_scatter(filt, formulation, out_dir, scenario):
    from src.plotting.tradeoff_scatter import (plot_key_tradeoffs,
                                               plot_scatter_matrix)
    baseline = _load_baseline_vec(formulation, filt.obj_names, scenario)
    color_by = ("nyc_delivery_reliability_annual"
                if "nyc_delivery_reliability_annual" in filt.obj_names
                else filt.obj_names[0])
    plot_key_tradeoffs(filt.natural_obj, filt.obj_names, filt.directions,
                       color_by=color_by, baseline=baseline,
                       output_file=out_dir / "run01b_tradeoff_scatter")
    plt.close("all")
    plot_scatter_matrix(filt.natural_obj, filt.obj_names, filt.directions,
                        color_by=color_by, baseline=baseline,
                        output_file=out_dir / "run01c_tradeoff_scatter_matrix")
    plt.close("all")
    print("[fig1b/c] -> 01b_tradeoff_scatter.png, "
          "runrun01c_tradeoff_scatter_matrix.png")


# --------------------------------------------------------------------------- #
# Fig 1d: DV ranges of criterion-satisfying subsets vs the full front
# --------------------------------------------------------------------------- #
def fig_dv_ranges(filt, formulation, out_dir, scenario):
    from src.plotting.dv_ranges import default_criteria, plot_dv_ranges
    baseline = _load_baseline_vec(formulation, filt.obj_names, scenario)
    criteria = default_criteria(filt.natural_obj, filt.obj_names,
                                filt.directions, baseline=baseline)
    if not criteria:
        print("[fig1d] skipped: no computable criteria "
              "(no baseline and no NYC-storage objective)")
        return
    plot_dv_ranges(filt.dv, formulation, criteria,
                   output_file=out_dir / "run01d_dv_ranges")
    plt.close("all")
    print("[fig1d] -> 01d_dv_ranges.png")


# --------------------------------------------------------------------------- #
# Fig 2: hypervolume convergence
# --------------------------------------------------------------------------- #
def fig_hypervolume(run_dir, formulation, out_dir):
    from src.plotting.hypervolume_convergence import plot_hypervolume_convergence
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists() or not any(metrics_dir.glob("*.metrics")):
        print("[fig2] no metrics dir/files"); return
    plot_hypervolume_convergence(metrics_dir, formulation,
                                 out_dir / "run02_hypervolume_convergence.png",
                                 figsize=(8, 5))
    print("[fig2] -> 02_hypervolume_convergence.png")


# --------------------------------------------------------------------------- #
# Fig 3: DU performance distributions (raw magnitudes, acceptable policies)
# --------------------------------------------------------------------------- #
def fig_du_distributions(reeval_dir, filt, out_dir):
    """Per-objective DU-expected value across acceptable policies vs baseline
    and the satisficing threshold. The raw-magnitude co-report (Huang 2025)."""
    import json
    opt = _load_reeval_long(reeval_dir)
    base = _load_reeval_long(reeval_dir / "baseline")
    if opt is None:
        print("[fig3] no optimized reeval"); return
    meta = json.loads((reeval_dir / "reeval_raw_meta.json").read_text())
    thresholds = meta.get("thresholds", {})
    kinds = meta.get("kinds", {})

    accepted = set(int(i) for i in filt.accepted_ids)
    opt = opt[opt["solution_id"].isin(accepted)]

    # per (solution, objective): mean over SOWs = DU-expected performance
    opt_mean = (opt.groupby(["solution_id", "objective"])["value"].mean()
                .unstack("objective"))
    base_mean = (base.groupby("objective")["value"].mean()
                 if base is not None else None)

    obj_names = filt.obj_names
    dirs = filt.directions
    n = len(obj_names)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(obj_names):
        ax = axes[i]
        if name not in opt_mean.columns:
            ax.set_visible(False); continue
        vals = opt_mean[name].dropna().values
        ax.boxplot(vals, widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="steelblue", alpha=0.5),
                   medianprops=dict(color="navy"))
        ax.scatter(np.random.default_rng(0).normal(1, 0.04, len(vals)), vals,
                   s=8, alpha=0.35, color="steelblue", zorder=3)
        if base_mean is not None and name in base_mean.index:
            ax.axhline(base_mean[name], color="firebrick", lw=2,
                       label="FFMP baseline")
        if thresholds.get(name) is not None:
            ax.axhline(thresholds[name], color="darkorange", lw=1.6, ls="--",
                       label="satisficing threshold")
        ax.legend(fontsize=7, loc="best")
        arrow = "↑ better" if dirs[i] == 1 else "↓ better"
        ax.set_title(f"{name}\n({arrow})", fontsize=8)
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("DU performance of ACCEPTABLE policies across E_test "
                 f"(n={filt.n_accepted}; mean over SOWs per policy)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "run03_du_performance_distributions.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig3] -> 03_du_performance_distributions.png")


# --------------------------------------------------------------------------- #
# Fig 4: DU robustness (multivariate satisficing + baseline + decomposition)
# --------------------------------------------------------------------------- #
def fig_robustness(reeval_dir, filt, most_robust_id, out_dir):
    from src.plotting.robustness_summary import plot_du_robustness
    info = plot_du_robustness(reeval_dir, filt.accepted_ids,
                              out_dir / "run04_du_robustness.png",
                              most_robust_id=most_robust_id)
    print(f"[fig4] baseline sat_sow={info['baseline_sat_sow']:.2f}, "
          f"best accepted={info['best_accepted_sat_sow']:.2f}, "
          f"binding={info['binding_objective']} -> 04_du_robustness.png")
    return info


# --------------------------------------------------------------------------- #
# Fig 4b: incumbent-relative regret, per objective, in natural units
# --------------------------------------------------------------------------- #
def fig_regret(reeval_dir, filt, out_dir):
    """Per-objective tail regret and mean gain against the status-quo FFMP policy.

    The RQ1 companion to fig 4: the satisficing figure says how often a policy is
    acceptable by a fixed standard, this one says how much it takes away from
    current operations, and from which party.
    """
    from src.plotting.regret_summary import plot_regret_decomposition
    try:
        out = plot_regret_decomposition(reeval_dir, out_dir / "run04b_regret",
                                        accepted_ids=filt.accepted_ids)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[fig4b] skipped: {exc}")
        return None
    print(f"[fig4b] -> {Path(out).name}")
    return out


# --------------------------------------------------------------------------- #
# Fig 5: scenario discovery over DU theta factors
# --------------------------------------------------------------------------- #
def fig_scenario_discovery(reeval_dir, preset, most_robust_id, out_dir):
    from src.plotting.scenario_discovery import plot_scenario_discovery
    ensemble_dir = Path("outputs/synthetic_ensembles") / preset
    if not (ensemble_dir / "forcing_profiles.npz").exists():
        print(f"[fig5] no forcing_profiles.npz under {ensemble_dir}"); return
    info = plot_scenario_discovery(reeval_dir, ensemble_dir, most_robust_id,
                                   out_dir / "run05_scenario_discovery.png")
    print(f"[fig5] id {info['solution_id']} passes {info['n_pass']}/"
          f"{info['n_sow']} SOWs -> 05_scenario_discovery.png")


# --------------------------------------------------------------------------- #
# Fig 6: operating rules of representative policies
# --------------------------------------------------------------------------- #
def fig_operating_rules(filt, examples, out_dir, formulation="ffmp"):
    """One operating-rules panel figure per representative policy."""
    from src.plotting.policy_rules import plot_policy_rules
    for sel in examples:
        stub = out_dir / f"run06_operating_rules_{sel.rule}"
        plot_policy_rules(filt.dv[sel.index], formulation, show_baseline=True,
                          candidate_label=f"{sel.label} (id {sel.index})",
                          output_file=stub)
        plt.close("all")
    ids = ", ".join(f"{s.label}=id{s.index}" for s in examples)
    print(f"[fig6] {ids} -> 06_operating_rules_*.png")


# --------------------------------------------------------------------------- #
def _select_examples(filt, scorecard):
    """Pick 3 DISTINCT representative acceptable policies (as solution_ids).

    A thin wrapper over :func:`src.solution_selection.select_by_rules`, which
    owns the ranking + collision fall-through logic. Order: most-robust,
    NYC-priority, Montague-priority. Solutions screened out by the stakeholder
    filter score ``-inf`` so they are never chosen.

    Returns:
        ``(most_robust_solution_id, [Selection, ...])``.
    """
    from src.formulations import resolve_objective_index
    from src.solution_selection import Selection, select_by_rules

    nat = filt.natural_obj
    names = filt.obj_names
    eligible = np.asarray(filt.mask, dtype=bool)

    def _eligible_only(scores):
        return np.where(eligible, np.asarray(scores, dtype=float), -np.inf)

    sat = scorecard["sat_multivariate_sow"] if scorecard is not None else None
    robustness = np.full(nat.shape[0], -np.inf)
    if sat is not None:
        for sid in sat.index:
            if 0 <= int(sid) < robustness.size:
                robustness[int(sid)] = float(sat.loc[sid])

    rules = [
        ("Most-robust", "most_robust", _eligible_only(robustness)),
        ("NYC-diversion priority", "best_nyc_reliability",
         _eligible_only(nat[:, resolve_objective_index(
             names, "nyc_delivery_reliability_weekly")])),
        ("Montague-flow priority", "best_montague_reliability",
         _eligible_only(nat[:, resolve_objective_index(
             names, "montague_flow_reliability_weekly")])),
    ]
    examples: list[Selection] = select_by_rules(rules)
    return examples[0].index, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="ffmp_obj8_mm_moderate")
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--preset", default="etest_kn_10yr_n200")
    ap.add_argument("--formulation", default="ffmp")
    args = ap.parse_args()

    run_dir = _run_dir(args.scenario, args.slug)
    reeval_dir = run_dir / "reeval" / args.preset
    out_dir = Path("figures") / args.scenario / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_set = _reeval_ref_set(run_dir, args.slug)
    print(f"run_dir={run_dir}\nreeval_dir={reeval_dir}\nref_set={ref_set}\n"
          f"out_dir={out_dir}\n")

    # Stakeholder screen on the re-evaluated reference set (aligns to solution_id).
    filt = filter_reference_set(ref_set, args.formulation)
    print(filt.summary(), "\n")

    # Most-robust acceptable policy + representative examples (from the scorecard).
    sc_path = reeval_dir / "robustness_scorecard.csv"
    scorecard = pd.read_csv(sc_path, index_col=0) if sc_path.exists() else None
    most_robust_id, examples = _select_examples(filt, scorecard)

    tasks = [
        ("parallel_coords", lambda: fig_parallel_coords(ref_set, args.formulation, filt, out_dir, args.scenario)),
        ("tradeoff_scatter", lambda: fig_tradeoff_scatter(filt, args.formulation, out_dir, args.scenario)),
        ("dv_ranges", lambda: fig_dv_ranges(filt, args.formulation, out_dir, args.scenario)),
        ("hypervolume", lambda: fig_hypervolume(run_dir, args.formulation, out_dir)),
        ("du_distributions", lambda: fig_du_distributions(reeval_dir, filt, out_dir)),
        ("robustness", lambda: fig_robustness(reeval_dir, filt, most_robust_id, out_dir)),
        ("regret", lambda: fig_regret(reeval_dir, filt, out_dir)),
        ("scenario_discovery", lambda: fig_scenario_discovery(reeval_dir, args.preset, most_robust_id, out_dir)),
        ("operating_rules", lambda: fig_operating_rules(filt, examples, out_dir,
                                                        args.formulation)),
    ]
    ok, fail = [], []
    for name, fn in tasks:
        try:
            fn(); ok.append(name)
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            fail.append(name)
    print(f"\nDONE. ok={ok} fail={fail}")
    print(f"Figures in: {out_dir}")


if __name__ == "__main__":
    main()
