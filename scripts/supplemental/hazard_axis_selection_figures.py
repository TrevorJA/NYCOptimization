"""hazard_axis_selection_figures.py - Figures of the hazard-axis selection.

Pure post-processing of the tables persisted by
``scripts/supplemental/hazard_axis_selection_run.py`` (no image is touched):

  F1_decision_landscape  every admissible set's decision statistic by set size
                         against the random-null band, the leaders, the
                         equivalence class and the adopted set; the leading
                         sets by rank with their replicate spread.
  F2_coverage_profile    the adopted set's coverage (KS distance to uniform) on
                         every continuous metric and its tail-share profile on
                         all 21 metrics, against the random null and the other
                         leading sets.
  F3_joint_check         the marginal criterion against the joint covering
                         distance, both as z-scores against the random null,
                         for every admissible set.
  F4_adopted_marginals   the adopted set's selected members on every continuous
                         metric's scaled range: selection ECDF, pool ECDF and
                         the uniform target.

Settings in ``supplemental_config.py`` (``HAXSEL_*``); figures follow
``src/plotting/style.py`` (PNG only). Run after the run script::

    python scripts/supplemental/hazard_axis_selection_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_haxsel_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.plotting.style import apply_style, save_figure  # noqa: E402

#: Compact metric codes for tick labels.
METRIC_CODE = {
    "drought_magnitude": "Dmag", "drought_duration": "Ddur", "drought_severity": "Dsev",
    "drought_development_rate": "Ddev", "drought_termination_rate": "Dterm",
    "drought_event_count": "Dcnt", "drought_total_deficit": "Dtot",
    "lowflow_min_12month": "L12m", "lowflow_min_24month": "L24m", "lowflow_min_year": "L1yr",
    "lowflow_min_7day": "L7d", "flood_peak_discharge": "Fpk", "flood_pulse_duration": "Fdur",
    "flood_rise_rate": "Frise", "flood_pulse_count": "Fcnt", "flood_days_above": "Fdays",
    "flood_max_3day": "F3d", "flood_pulse_volume": "Fvol", "highflow_max_year": "Hyr",
    "annual_cv": "CV", "flashiness": "RB",
}
#: Fixed hue per set size (Okabe-Ito; never cycled).
SIZE_COLOR = {4: "#0072B2", 5: "#E69F00", 6: "#009E73"}
DRAW_MARKER = {0: "o", 1: "^"}
NULL_COLOR = "0.82"
POOL_COLOR = "0.55"
ADOPTED_COLOR = "black"


def _table(name: str) -> pd.DataFrame:
    path = scfg.haxsel_table_path(name)
    if not path.exists():
        raise SystemExit(f"[haxsel-fig] table missing: {path}")
    return pd.read_csv(path)


def _code(members: str) -> str:
    return "+".join(METRIC_CODE.get(m, m) for m in members.split("|"))


def _null_band(reference: pd.DataFrame, statistic: str) -> tuple[float, float]:
    """Pooled null mean and SD of a statistic over the draws."""
    g = reference.loc[reference["statistic"] == statistic]
    return float(g["mean"].mean()), float(g["sd"].mean())


def _metric_order(metrics: pd.DataFrame, continuous_only: bool) -> list[str]:
    """Metrics in cluster order (draw-0 clusters), continuous metrics first when restricted."""
    m = metrics.loc[metrics["candidate_axis"]] if continuous_only else metrics
    return m.sort_values(["cluster_id__d0", "metric"])["metric"].tolist()


def _cluster_separators(ax, metrics: pd.DataFrame, order: list[str]) -> None:
    ids = metrics.set_index("metric").loc[order, "cluster_id__d0"].to_numpy()
    for j in range(1, len(order)):
        if ids[j] != ids[j - 1]:
            ax.axvline(j - 0.5, color="0.75", lw=0.6, zorder=0)


def fig_decision_landscape(ranking: pd.DataFrame, wrapper: pd.DataFrame, reference: pd.DataFrame,
                           summary: dict) -> None:
    """F1: the decision statistic of every admissible set, and the leading sets by rank."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.6]})
    null_mean, null_sd = _null_band(reference, "cov_cluster_mean")
    rng = np.random.default_rng(0)
    leaders = set(ranking.head(scfg.HAXSEL_LEADERS)["set_id"])
    for n, g in ranking.groupby("n"):
        x = n + rng.uniform(-0.28, 0.28, size=len(g))
        lead = g["set_id"].isin(leaders)
        a.scatter(x[~lead], g.loc[~lead, "mean"], s=9, c=SIZE_COLOR[int(n)], alpha=0.35,
                  edgecolors="none", zorder=2)
        a.scatter(x[lead], g.loc[lead, "mean"], s=18, c=SIZE_COLOR[int(n)], alpha=0.95,
                  edgecolors="white", linewidths=0.4, zorder=3)
        cls = g["tie_group"].to_numpy()
        a.scatter(x[cls], g.loc[cls, "mean"], s=46, facecolors="none", edgecolors=ADOPTED_COLOR,
                  linewidths=0.9, zorder=4)
    adopted = ranking.loc[ranking["adopted"]].iloc[0]
    a.scatter([adopted["n"]], [adopted["mean"]], marker="*", s=180, c=ADOPTED_COLOR, zorder=5)
    a.axhspan(null_mean - 2 * null_sd, null_mean + 2 * null_sd, color=NULL_COLOR, zorder=0)
    a.axhline(null_mean, color="0.5", lw=0.8, ls="--", zorder=1)
    a.text(3.55, null_mean, "random 300-subsets\n(mean, ±2 SD)", ha="left", va="bottom",
           fontsize=7, color="0.35")
    a.set_xticks(sorted(ranking["n"].unique()))
    a.set_xlabel("number of selection axes")
    a.set_ylabel("cluster-averaged KS distance to uniform\n(lower = more uniform coverage)")
    a.set_title(f"All {len(ranking):,} admissible sets", fontsize=10)
    a.legend(handles=[
        Line2D([], [], marker="o", ls="", color="0.5", ms=4, label="admissible set"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=ADOPTED_COLOR, ms=7, label="equivalence class"),
        Line2D([], [], marker="*", ls="", color=ADOPTED_COLOR, ms=10, label="adopted set"),
    ], fontsize=7, loc="lower right")

    top = ranking.head(scfg.HAXSEL_RANKING_TOP).reset_index(drop=True)
    x = np.arange(len(top))
    tie = top["tie_group"].to_numpy()
    if tie.any():
        b.axvspan(np.flatnonzero(tie).min() - 0.5, np.flatnonzero(tie).max() + 0.5, color="0.93", zorder=0)
    b.axhspan(null_mean - 2 * null_sd, null_mean + 2 * null_sd, color=NULL_COLOR, zorder=0)
    for i, row in top.iterrows():
        reps = wrapper.loc[wrapper["set_id"] == row["set_id"]]
        for d, g in reps.groupby("draw"):
            b.scatter(np.full(len(g), i) + (d - 0.5) * 0.2, g["cov_cluster_mean"], s=9,
                      marker=DRAW_MARKER.get(int(d), "s"), c=SIZE_COLOR[int(row["n"])], alpha=0.5,
                      edgecolors="none", zorder=2)
        b.errorbar([i], [row["mean"]], yerr=[2 * row["se"]], fmt="_", ms=12, mew=1.6,
                   color=ADOPTED_COLOR, ecolor=ADOPTED_COLOR, capsize=2, zorder=3)
        if row["adopted"]:
            b.scatter([i], [row["mean"] - 4 * row["se"] - 0.004], marker="*", s=140, c=ADOPTED_COLOR,
                      zorder=4)
    b.set_xticks(x)
    b.set_xticklabels([_code(m) for m in top["members"]], rotation=60, ha="right", fontsize=6.5)
    b.set_xlim(-0.7, len(top) - 0.3)
    b.set_ylabel("cluster-averaged KS distance to uniform")
    s = summary["settings"]
    b.set_title(f"Leading {len(top)} sets by rank (N={s['HAXSEL_N']}, P={s['HAXSEL_POOL_P']:,}, "
                f"{len(summary['pools'])} draws × {s['HAXSEL_SEEDS']} seeds; whisker = 2 SE; "
                f"shaded = equivalence class)", fontsize=9)
    b.legend(handles=[Line2D([], [], marker="s", ls="", color=c, label=f"{n} axes")
                      for n, c in SIZE_COLOR.items()]
             + [Line2D([], [], marker=m, ls="", color="0.5", label=f"draw {d}") for d, m in DRAW_MARKER.items()],
             fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    save_figure(fig, scfg.haxsel_figure_path("F1_decision_landscape"))
    plt.close(fig)


def fig_coverage_profile(ranking: pd.DataFrame, null: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """F2: the adopted set's per-metric coverage and tail-share profiles against the null."""
    adopted = ranking.loc[ranking["adopted"]].iloc[0]
    own = set(adopted["members"].split("|"))
    others = ranking.head(scfg.HAXSEL_PROFILE_TOP)
    others = others.loc[~others["adopted"]]
    fig, (a, b) = plt.subplots(2, 1, figsize=(13.0, 7.2))
    for ax, prefix, order, ylabel, ref_line in (
        (a, "cov__", _metric_order(metrics, True),
         "KS distance to uniform\non the metric's scaled range", None),
        (b, "tail__", _metric_order(metrics, False),
         "share of selected members in the\nhazard-direction pool tail (p90)", 0.10),
    ):
        x = np.arange(len(order))
        cols = [f"{prefix}{m}" for m in order]
        nm, nsd = null[cols].mean().to_numpy(), null[cols].std(ddof=1).to_numpy()
        ax.fill_between(x, np.clip(nm - 2 * nsd, 0.0, None), nm + 2 * nsd, color=NULL_COLOR,
                        zorder=0, label="random 300-subsets (±2 SD)")
        for _, r in others.iterrows():
            ax.plot(x, r[cols].to_numpy(dtype=float), color="0.6", lw=0.8, alpha=0.6, zorder=1)
        vals = adopted[cols].to_numpy(dtype=float)
        ax.plot(x, vals, color=SIZE_COLOR[int(adopted["n"])], lw=1.8, zorder=3)
        is_own = np.array([m in own for m in order])
        ax.scatter(x[is_own], vals[is_own], s=46, c=SIZE_COLOR[int(adopted["n"])], edgecolors="black",
                   linewidths=0.8, zorder=4)
        ax.scatter(x[~is_own], vals[~is_own], s=30, facecolors="white",
                   edgecolors=SIZE_COLOR[int(adopted["n"])], linewidths=1.2, zorder=4)
        if ref_line is not None:
            ax.axhline(ref_line, color="0.4", lw=0.8, ls="--", zorder=1)
        _cluster_separators(ax, metrics, order)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_CODE.get(m, m) for m in order], rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel(ylabel)
    a.set_title(f"Adopted set {_code(adopted['members'])}: coverage of every continuous metric "
                "(filled = its own axes; gray lines = the other leading sets; "
                "vertical lines separate metric clusters)", fontsize=9)
    b.set_title("Tail-share profile on all 21 metrics (dashed = i.i.d. share 0.10)", fontsize=9)
    a.legend(handles=[
        Patch(color=NULL_COLOR, label="random 300-subsets (±2 SD)"),
        Line2D([], [], color="0.6", lw=0.8, label=f"other leading sets (top {scfg.HAXSEL_PROFILE_TOP})"),
        Line2D([], [], marker="o", color=SIZE_COLOR[int(adopted["n"])], mec="black", label="adopted set, own axis"),
        Line2D([], [], marker="o", mfc="white", color=SIZE_COLOR[int(adopted["n"])], label="adopted set, other metric"),
    ], fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    save_figure(fig, scfg.haxsel_figure_path("F2_coverage_profile"))
    plt.close(fig)


def fig_joint_check(ranking: pd.DataFrame, reference: pd.DataFrame, checks: pd.DataFrame) -> None:
    """F3: the marginal criterion against the joint covering distance, as null z-scores."""
    cm, cs = _null_band(reference, "cov_cluster_mean")
    qm, qs = _null_band(reference, "cover_q")
    z_cov = (ranking["mean"] - cm) / cs
    z_cover = (ranking["cover_q"] - qm) / qs
    leaders = ranking["set_id"].isin(set(ranking.head(scfg.HAXSEL_LEADERS)["set_id"]))
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for n, c in SIZE_COLOR.items():
        sel = (ranking["n"] == n).to_numpy()
        ax.scatter(z_cov[sel & ~leaders], z_cover[sel & ~leaders], s=9, c=c, alpha=0.35,
                   edgecolors="none", label=f"{n} axes", zorder=2)
        ax.scatter(z_cov[sel & leaders], z_cover[sel & leaders], s=20, c=c, alpha=0.95,
                   edgecolors="white", linewidths=0.4, zorder=3)
    cls = ranking["tie_group"].to_numpy()
    ax.scatter(z_cov[cls], z_cover[cls], s=50, facecolors="none", edgecolors=ADOPTED_COLOR,
               linewidths=0.9, zorder=4, label="equivalence class")
    ad = ranking["adopted"].to_numpy()
    ax.scatter(z_cov[ad], z_cover[ad], marker="*", s=200, c=ADOPTED_COLOR, zorder=5, label="adopted set")
    ax.axhline(0, color="0.5", lw=0.8, ls="--", zorder=1)
    ax.axvline(0, color="0.5", lw=0.8, ls="--", zorder=1)
    ax.text(0.02, 0.98, "random-subset reference at 0 on both axes", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color="0.35")
    rho = float(checks.loc[checks["check"] == "spearman_cov_vs_cover_q", "value"].iloc[0])
    ax.set_xlabel("marginal coverage: cluster-averaged KS distance (z-score; lower = better)")
    ax.set_ylabel(f"joint coverage: p{int(scfg.HAXSEL_COVER_Q)} pool-to-nearest-member distance\n"
                  "(z-score; lower = better)")
    ax.set_title(f"Marginal against joint coverage over all admissible sets (Spearman ρ = {rho:.2f})",
                 fontsize=9.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save_figure(fig, scfg.haxsel_figure_path("F3_joint_check"))
    plt.close(fig)


def fig_adopted_marginals(ranking: pd.DataFrame, metrics: pd.DataFrame, members: pd.DataFrame,
                          quantiles: pd.DataFrame, draw: int) -> None:
    """F4: the adopted set's selected members on every continuous metric's scaled range."""
    adopted = ranking.loc[ranking["adopted"]].iloc[0]
    own = set(adopted["members"].split("|"))
    order = _metric_order(metrics, True)
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.0, 2.75 * nrow), sharex=True, sharey=True)
    color = SIZE_COLOR[int(adopted["n"])]
    for ax, m in zip(axes.ravel(), order):
        ax.plot([0, 1], [0, 1], color="0.4", lw=0.8, ls="--", zorder=1)
        ax.plot(quantiles[m], quantiles["level"], color=POOL_COLOR, lw=1.4, zorder=2)
        v = np.sort(members[m].to_numpy(dtype=float))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=1.6, zorder=3)
        ks = float(np.max(np.maximum(np.arange(1, len(v) + 1) / len(v) - v,
                                     v - np.arange(len(v)) / len(v))))
        ax.set_title(f"{METRIC_CODE.get(m, m)}{'  (axis)' if m in own else ''}   KS {ks:.2f}",
                     fontsize=9, fontweight="bold" if m in own else "normal")
        if m in own:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color)
                sp.set_linewidth(1.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("scaled value (p1–p99 range)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative share", fontsize=8)
    fig.suptitle(f"Adopted set {_code(adopted['members'])}: selected members on every continuous metric "
                 f"(draw {draw}, seed 0; N = {len(members)})", fontsize=10)
    fig.legend(handles=[
        Line2D([], [], color=color, lw=1.6, label="selected members"),
        Line2D([], [], color=POOL_COLOR, lw=1.4, label="candidate pool"),
        Line2D([], [], color="0.4", lw=0.8, ls="--", label="uniform on the scaled range"),
    ], loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, scfg.haxsel_figure_path("F4_adopted_marginals"))
    plt.close(fig)


def main() -> None:
    """Render the four figures from the persisted tables."""
    apply_style()
    scfg.HAXSEL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(scfg.haxsel_json_path("summary").read_text(encoding="utf-8"))
    ranking, wrapper, metrics = _table("ranking"), _table("wrapper_scores"), _table("metrics")
    null, reference, checks = _table("null_scores"), _table("null_reference"), _table("joint_check")
    draw = int(scfg.HAXSEL_DRAWS[0])
    fig_decision_landscape(ranking, wrapper, reference, summary)
    fig_coverage_profile(ranking, null, metrics)
    fig_joint_check(ranking, reference, checks)
    fig_adopted_marginals(ranking, metrics, _table(f"adopted_members_d{draw}"),
                          _table(f"pool_quantiles_d{draw}"), draw)
    print(f"[haxsel-fig] wrote 4 figures -> {scfg.HAXSEL_FIGURES_DIR}")


if __name__ == "__main__":
    main()
