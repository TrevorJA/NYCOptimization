"""plot_run_results.py - Render the result figures for a single optimization run.

Generates, for one (scenario, moea_slug, reeval preset), after a stakeholder
screen is applied to the re-evaluated reference set (NYC weekly delivery
reliability >= 0.5 -- see :mod:`src.pareto_filter`):

  1. Pareto parallel-coordinates over the objective set, accepted policies vs the
     screened-out ones, with the FFMP baseline overlaid.
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
one failure never blocks the others. Figures 1-2 also work after step 07 alone;
3-6 need step 08 (the re-eval + robustness outputs).

Usage (from repo root, venv active):
  python3 -m scripts.main.plot_run_results \
      --slug ffmp_obj7_mm_moderate --scenario historic \
      --preset etest_kn_10yr_n200 --formulation ffmp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# --------------------------------------------------------------------------- #
# Fig 1: Pareto parallel-coordinates, accepted vs screened out
# --------------------------------------------------------------------------- #
def fig_parallel_coords(ref_set, formulation, filt, out_dir):
    from src.plotting.parallel_coordinates import plot_parallel_coordinates
    # baseline objectives (natural units) for the overlay
    baseline = None
    bcsv = Path("outputs/baseline") / f"{formulation}_baseline_objectives.csv"
    if bcsv.exists():
        row = pd.read_csv(bcsv).iloc[0]
        try:
            baseline = np.array([float(row[n]) for n in filt.obj_names])
        except KeyError:
            baseline = None
    plot_parallel_coordinates(ref_set, formulation,
                              out_dir / "01_pareto_parallel_coords.png",
                              baseline_objs=baseline, figsize=(13, 5.5),
                              keep_mask=filt.mask)
    print(f"[fig1] {filt.n_accepted}/{filt.n_total} acceptable "
          f"-> 01_pareto_parallel_coords.png")


# --------------------------------------------------------------------------- #
# Fig 2: hypervolume convergence
# --------------------------------------------------------------------------- #
def fig_hypervolume(run_dir, formulation, out_dir):
    from src.plotting.hypervolume_convergence import plot_hypervolume_convergence
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists() or not any(metrics_dir.glob("*.metrics")):
        print("[fig2] no metrics dir/files"); return
    plot_hypervolume_convergence(metrics_dir, formulation,
                                 out_dir / "02_hypervolume_convergence.png",
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

    # per (solution, objective): mean over realizations = DU-expected performance
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
                 f"(n={filt.n_accepted}; mean over realizations per policy)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "03_du_performance_distributions.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig3] -> 03_du_performance_distributions.png")


# --------------------------------------------------------------------------- #
# Fig 4: DU robustness (multivariate satisficing + baseline + decomposition)
# --------------------------------------------------------------------------- #
def fig_robustness(reeval_dir, filt, most_robust_id, out_dir):
    from src.plotting.robustness_summary import plot_du_robustness
    info = plot_du_robustness(reeval_dir, filt.accepted_ids,
                              out_dir / "04_du_robustness.png",
                              most_robust_id=most_robust_id)
    print(f"[fig4] baseline sat_sow={info['baseline_sat_sow']:.2f}, "
          f"best accepted={info['best_accepted_sat_sow']:.2f}, "
          f"binding={info['binding_objective']} -> 04_du_robustness.png")
    return info


# --------------------------------------------------------------------------- #
# Fig 5: scenario discovery over DU theta factors
# --------------------------------------------------------------------------- #
def fig_scenario_discovery(reeval_dir, preset, most_robust_id, out_dir):
    from src.plotting.scenario_discovery import plot_scenario_discovery
    ensemble_dir = Path("outputs/synthetic_ensembles") / preset
    if not (ensemble_dir / "forcing_profiles.npz").exists():
        print(f"[fig5] no forcing_profiles.npz under {ensemble_dir}"); return
    info = plot_scenario_discovery(reeval_dir, ensemble_dir, most_robust_id,
                                   out_dir / "05_scenario_discovery.png")
    print(f"[fig5] id {info['solution_id']} passes {info['n_pass']}/"
          f"{info['n_sow']} SOWs -> 05_scenario_discovery.png")


# --------------------------------------------------------------------------- #
# Fig 6: operating rules of representative policies
# --------------------------------------------------------------------------- #
def fig_operating_rules(filt, examples, out_dir):
    from src.plotting.operating_rules import plot_operating_rules
    policies = [{"label": lbl, "dv": filt.dv[sid], "color": col}
                for lbl, sid, col in examples]
    plot_operating_rules(policies, out_dir / "06_operating_rules.png")
    ids = ", ".join(f"{lbl}=id{sid}" for lbl, sid, _ in examples)
    print(f"[fig6] {ids} -> 06_operating_rules.png")


# --------------------------------------------------------------------------- #
def _select_examples(filt, scorecard):
    """Pick 3 DISTINCT representative acceptable policies (as solution_ids).

    Returns list of (label, solution_id, color). Order: most-robust, NYC-priority,
    Montague-priority. Ties/collisions fall through to the next-best candidate.
    """
    accepted = list(int(i) for i in filt.accepted_ids)
    nat = filt.natural_obj
    names = filt.obj_names
    nyc_k = names.index("nyc_delivery_reliability_weekly")
    mont_k = names.index("montague_flow_reliability_weekly")

    sat = scorecard["sat_multivariate_sow"] if scorecard is not None else None
    chosen: list[int] = []

    def _pick(score_by_id):
        for sid in sorted(accepted, key=lambda s: score_by_id(s), reverse=True):
            if sid not in chosen:
                chosen.append(sid); return sid
        return accepted[0]

    most_robust = _pick(lambda s: (float(sat.loc[s]) if sat is not None
                                   and s in sat.index else -np.inf))
    nyc_id = _pick(lambda s: nat[s, nyc_k])
    mont_id = _pick(lambda s: nat[s, mont_k])
    examples = [
        ("Most-robust", most_robust, "#2ca25f"),
        ("NYC-diversion priority", nyc_id, "#1f77b4"),
        ("Montague-flow priority", mont_id, "#ff7f0e"),
    ]
    return most_robust, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="ffmp_obj7_mm_moderate")
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
        ("parallel_coords", lambda: fig_parallel_coords(ref_set, args.formulation, filt, out_dir)),
        ("hypervolume", lambda: fig_hypervolume(run_dir, args.formulation, out_dir)),
        ("du_distributions", lambda: fig_du_distributions(reeval_dir, filt, out_dir)),
        ("robustness", lambda: fig_robustness(reeval_dir, filt, most_robust_id, out_dir)),
        ("scenario_discovery", lambda: fig_scenario_discovery(reeval_dir, args.preset, most_robust_id, out_dir)),
        ("operating_rules", lambda: fig_operating_rules(filt, examples, out_dir)),
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
