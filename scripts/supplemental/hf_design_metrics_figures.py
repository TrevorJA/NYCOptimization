"""hf_design_metrics_figures.py - Figures of the HF design metrics.

Pure post-processing of the tables and distribution grids persisted by
``scripts/supplemental/hf_design_metrics_run.py`` (no image is touched):

  F1_marginals_range      one panel per selection axis: scaled-axis ECDFs of
                          the candidate ensemble, the HF ensemble, its Latin
                          hypercube targets, the MC ensemble, the random
                          same-N band, and the historical windows as ticks;
                          the candidate p90 marked; range bars beneath.
  F2_coverage_diversity   ECDFs of the candidate-to-nearest-member distance
                          (minimax distance marked) and of the minimum-
                          spanning-tree edge lengths; a dot plot of the four
                          headline statistics as ratios to the random mean.
  F3_target_displacement  per-target displacement of the greedy rule, the
                          exact assignment, and the free lower bound; greedy
                          against exact per target.
  F4_measure_weights      sorted nearest-member redistribution weights (N w_i)
                          with the effective sample size, and the weights
                          against the drought-magnitude coordinate.

Settings in ``supplemental_config.py`` (``HFM_*``); figures follow
``src/plotting/style.py`` (PNG only). Run after the run script::

    python scripts/supplemental/hf_design_metrics_figures.py
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

scfg.configure_hfm_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.plotting.ensemble_composition import HAZARD_METRIC_LABELS, HISTORIC_MARK  # noqa: E402
from src.plotting.style import apply_style, design_color, save_figure  # noqa: E402

HF_COLOR = design_color("hazard_filling_stationary")
MC_COLOR = design_color("monte_carlo")
CAND_COLOR = "0.55"
BAND_COLOR = "0.80"
DRAW_ALPHA = (1.0, 0.55, 0.35)

SET_LABEL = {
    "hf": "HF ensemble",
    "lhs_targets": "Latin hypercube targets",
    "mc": "MC ensemble",
    "random": "random N-subsets",
}


def _table(name: str) -> pd.DataFrame | None:
    path = scfg.hfm_table_path(name)
    if not path.exists():
        print(f"[hfm-fig] table missing: {path}")
        return None
    return pd.read_csv(path)


def _json(name: str) -> dict | None:
    path = scfg.hfm_json_path(name)
    if not path.exists():
        print(f"[hfm-fig] json missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ecdf_xy(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def _axis_label(axis: str) -> str:
    label = HAZARD_METRIC_LABELS.get(axis, axis)
    return label.split(" (")[0] + " (scaled)"


def _set_lines(ax, dist_draw: dict, key: str, levels, draw_i: int, *, x_from_levels: bool = True):
    """Plot one draw's HF / targets / MC ECDF curves from persisted level grids."""
    sets = dist_draw.get("sets", {})
    for name, color, ls in (("hf", HF_COLOR, "-"), ("lhs_targets", HF_COLOR, "--"),
                            ("mc", MC_COLOR, "-")):
        entry = next((v for k, v in sets.items() if k.startswith(f"{name}_d")), None)
        if entry is None:
            continue
        vals = entry[key]
        ax.plot(vals, levels, ls, color=color, lw=1.6, alpha=DRAW_ALPHA[min(draw_i, 2)],
                label=SET_LABEL[name] if draw_i == 0 else None)
        if key == "cover_levels":
            ax.plot(vals[-1], 1.0, "v", color=color, ms=6, alpha=DRAW_ALPHA[min(draw_i, 2)])


###############################################################################
# Figures
###############################################################################

def fig_marginals_range(axes_tab: pd.DataFrame, points: pd.DataFrame | None, dist: dict, manifest: dict) -> None:
    """F1: per-axis scaled ECDFs with the random band, historic ticks, and range bars."""
    axes = manifest["axes"]
    levels = np.asarray(dist["levels"])
    fig, axarr = plt.subplots(2, 3, figsize=(12.5, 7.2), sharey=True)
    for k, axis in enumerate(axes):
        ax = axarr.flat[k]
        for draw_i, (draw, dd) in enumerate(sorted(dist["draws"].items())):
            band = dd["random_axis_band"][axis]
            if draw_i == 0:
                ax.fill_betweenx(levels, band["lo"], band["hi"], color=BAND_COLOR, alpha=0.6,
                                 lw=0, label=SET_LABEL["random"] + " (5th to 95th pct)")
                ax.plot(dd["candidate_axis_levels"][axis], levels, color=CAND_COLOR, lw=2.2,
                        label="candidate ensemble")
                ax.axvline(manifest["draws"][draw]["p90_scaled"][k], color=CAND_COLOR, ls=":",
                           lw=1.0, label="candidate 90th percentile")
            for name, color, ls in (("hf", HF_COLOR, "-"), ("lhs_targets", HF_COLOR, "--"),
                                    ("mc", MC_COLOR, "-")):
                entry = dd.get("sets", {}).get(f"{name}_d{draw}")
                if entry is None:
                    continue
                x, y = _ecdf_xy(entry["axis_values"][axis])
                ax.step(x, y, ls, where="post", color=color, lw=1.5,
                        alpha=DRAW_ALPHA[min(draw_i, 2)],
                        label=SET_LABEL[name] if draw_i == 0 else None)
        if points is not None:
            hist = points[points["kind"] == "historic_window"]
            if not hist.empty:
                hist = hist[hist["draw"] == hist["draw"].min()]
                ax.plot(hist[f"z_{axis}"], np.full(len(hist), -0.04), "|", color=HISTORIC_MARK,
                        ms=9, mew=1.4, label="historical 10-year windows" if k == 0 else None)
        # Range bars beneath the panel, one per set, draw 0.
        y0 = -0.10
        for name, color in (("hf", HF_COLOR), ("mc", MC_COLOR), ("lhs_targets", HF_COLOR)):
            sub = axes_tab[(axes_tab["set"] == name) & (axes_tab["axis"] == axis)
                           & (axes_tab["draw"] == axes_tab["draw"].min())]
            if sub.empty:
                continue
            ax.plot([sub["min"].iloc[0], sub["max"].iloc[0]], [y0, y0], color=color,
                    lw=3 if name != "lhs_targets" else 1.2, solid_capstyle="butt")
            y0 -= 0.05
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.24, 1.02)
        ax.set_xlabel(_axis_label(axis))
        if k % 3 == 0:
            ax.set_ylabel("cumulative fraction of members")
    handles, labels = axarr.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-axis marginals in the selector's scaled coordinates "
                 "(bars beneath: attained span of HF, MC, targets)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, scfg.hfm_figure_path("F1_marginals_range"))
    plt.close(fig)


def fig_coverage_diversity(summary: pd.DataFrame, dist: dict) -> None:
    """F2: coverage-distance and MST-edge ECDFs, and headline ratios to random."""
    levels = np.asarray(dist["levels"])
    fig, (ax_c, ax_m, ax_r) = plt.subplots(1, 3, figsize=(14, 4.6),
                                           gridspec_kw={"width_ratios": (1.15, 1.15, 1.0)})
    for draw_i, (draw, dd) in enumerate(sorted(dist["draws"].items())):
        if draw_i == 0:
            b = dd["random_cover_band"]
            ax_c.fill_betweenx(levels, b["lo"], b["hi"], color=BAND_COLOR, alpha=0.6, lw=0,
                               label=SET_LABEL["random"] + " (5th to 95th pct)")
            b = dd["random_mst_band"]
            ax_m.fill_betweenx(levels, b["lo"], b["hi"], color=BAND_COLOR, alpha=0.6, lw=0)
        _set_lines(ax_c, dd, "cover_levels", levels, draw_i)
        _set_lines(ax_m, dd, "mst_levels", levels, draw_i)
    ax_c.set_xlabel("distance from a candidate realization\nto its nearest member (scaled units)")
    ax_c.set_ylabel("cumulative fraction of candidates")
    ax_c.set_title("Coverage (triangle: minimax distance)", fontsize=10)
    ax_m.set_xlabel("minimum-spanning-tree edge length\namong members (scaled units)")
    ax_m.set_ylabel("cumulative fraction of edges")
    ax_m.set_title("Diversity", fontsize=10)

    metrics = [("minimax_distance", "minimax\ndistance"),
               ("mean_nearest_member_distance", "mean\nnearest-\nmember\ndistance"),
               ("mst_edge_mean", "MST\nmean\nedge"),
               ("mst_edge_min", "MST\nminimum\nedge\n(maximin)"),
               ("ess_over_n", "effective\nsample\nsize / N")]
    rand = summary[summary["set"] == "random"]
    xs = np.arange(len(metrics))
    for draw_i, draw in enumerate(sorted(summary["draw"].unique())):
        r = rand[rand["draw"] == draw]
        if not r.empty and draw_i == 0:
            lo = [r[m].quantile(0.05) / r[m].mean() for m, _ in metrics]
            hi = [r[m].quantile(0.95) / r[m].mean() for m, _ in metrics]
            ax_r.fill_between(xs, lo, hi, color=BAND_COLOR, alpha=0.6, lw=0, step=None)
        for name, color, marker in (("hf", HF_COLOR, "o"), ("lhs_targets", HF_COLOR, "^"),
                                    ("mc", MC_COLOR, "s")):
            s = summary[(summary["set"] == name) & (summary["draw"] == draw)]
            if s.empty:
                continue
            ys = [s[f"ratio_to_random_{m}"].iloc[0] for m, _ in metrics]
            ax_r.plot(xs + (draw_i - 1) * 0.08, ys, marker, color=color, ms=6,
                      mfc="none" if name == "lhs_targets" else color,
                      alpha=DRAW_ALPHA[min(draw_i, 2)],
                      label=SET_LABEL[name] if draw_i == 0 else None)
    ax_r.axhline(1.0, color=CAND_COLOR, lw=0.8, ls=":")
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([lab for _, lab in metrics], fontsize=7.5)
    ax_r.set_xlim(-0.5, len(metrics) - 0.5)
    ax_r.set_ylabel("ratio to the mean of random N-subsets")
    ax_r.set_title("Headline statistics relative to random", fontsize=10)
    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, scfg.hfm_figure_path("F2_coverage_diversity"))
    plt.close(fig)


def fig_target_displacement(points: pd.DataFrame, summary: pd.DataFrame) -> None:
    """F3: displacement ECDFs (greedy, exact, free) and greedy against exact per target."""
    tgt = points[points["kind"] == "target"]
    if tgt.empty:
        print("[hfm-fig] no target rows; F3 skipped")
        return
    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(11, 4.4))
    for draw_i, draw in enumerate(sorted(tgt["draw"].unique())):
        t = tgt[tgt["draw"] == draw]
        alpha = DRAW_ALPHA[min(draw_i, 2)]
        for col, color, ls, lab in (("greedy_disp", HF_COLOR, "-", "sequential rule (Eq. 7)"),
                                    ("exact_disp", "0.15", "-", "exact assignment"),
                                    ("free_disp", "0.15", ":", "nearest candidate (lower bound)")):
            x, y = _ecdf_xy(t[col])
            ax_e.step(x, y, ls, where="post", color=color, lw=1.6, alpha=alpha,
                      label=lab if draw_i == 0 else None)
        s = summary[(summary["set"] == "hf") & (summary["draw"] == draw)]
        lab = None
        if not s.empty:
            s = s.iloc[0]
            lab = (f"draw {draw}: total {s['greedy_total']:.2f} vs {s['exact_total']:.2f} "
                   f"(gap {100 * s['gap_rel']:.1f}%), Jaccard {s['jaccard_greedy_exact']:.2f}")
        ax_s.plot(t["exact_disp"], t["greedy_disp"], "o", color=HF_COLOR, ms=4, alpha=alpha,
                  label=lab)
    lim = max(tgt["greedy_disp"].max(), tgt["exact_disp"].max()) * 1.05
    ax_s.plot([0, lim], [0, lim], color="0.4", lw=0.8)
    ax_s.set_xlim(0, lim)
    ax_s.set_ylim(0, lim)
    ax_s.set_xlabel("displacement under the exact assignment (scaled units)")
    ax_s.set_ylabel("displacement under the sequential rule (scaled units)")
    ax_s.legend(frameon=False, fontsize=8, loc="upper left")
    ax_e.set_xlabel("target displacement (scaled units)")
    ax_e.set_ylabel("cumulative fraction of targets")
    ax_e.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save_figure(fig, scfg.hfm_figure_path("F3_target_displacement"))
    plt.close(fig)


def fig_measure_weights(points: pd.DataFrame, dist: dict, manifest: dict) -> None:
    """F4: sorted redistribution weights with ESS, and weights against drought magnitude."""
    fig, (ax_w, ax_m) = plt.subplots(1, 2, figsize=(11, 4.4))
    n = manifest["N"]
    rank = np.arange(1, n + 1) / n
    for draw_i, (draw, dd) in enumerate(sorted(dist["draws"].items())):
        alpha = DRAW_ALPHA[min(draw_i, 2)]
        if draw_i == 0:
            b = dd["random_weights_band"]
            ax_w.fill_between(rank, np.maximum(b["lo"], 1e-3), b["hi"], color=BAND_COLOR,
                              alpha=0.6, lw=0, label=SET_LABEL["random"] + " (5th to 95th pct)")
        for name, color, ls in (("hf", HF_COLOR, "-"), ("lhs_targets", HF_COLOR, "--"),
                                ("mc", MC_COLOR, "-")):
            entry = dd.get("sets", {}).get(f"{name}_d{draw}")
            if entry is None:
                continue
            w = np.asarray(entry["weights_sorted"])
            ax_w.plot(np.arange(1, len(w) + 1) / len(w), np.maximum(w, 1e-3), ls, color=color,
                      lw=1.6, alpha=alpha,
                      label=(f"{SET_LABEL[name]} (effective sample size / N = "
                             f"{entry['ess_over_n']:.2f})" if draw_i == 0 else None))
    ax_w.axhline(1.0, color=CAND_COLOR, lw=0.8, ls=":")
    ax_w.set_yscale("log")
    ax_w.set_xlabel("members ranked by weight (fraction of N)")
    ax_w.set_ylabel("N w_i (redistributed mass relative to equal weights;\nzero clamped to 0.001)")
    ax_w.legend(frameon=False, fontsize=8, loc="upper right")

    axis = "drought_magnitude"
    for draw_i, draw in enumerate(sorted(points["draw"].unique())):
        alpha = DRAW_ALPHA[min(draw_i, 2)]
        for kind, color, lab in (("hf_member", HF_COLOR, SET_LABEL["hf"]),
                                 ("mc_member", MC_COLOR, SET_LABEL["mc"])):
            p = points[(points["kind"] == kind) & (points["draw"] == draw)]
            if p.empty:
                continue
            ax_m.plot(p[f"z_{axis}"], np.maximum(p["n_weight"], 1e-3), "o", color=color, ms=4,
                      alpha=alpha, label=lab if draw_i == 0 else None)
    ax_m.axhline(1.0, color=CAND_COLOR, lw=0.8, ls=":")
    ax_m.set_yscale("log")
    ax_m.set_xlabel(_axis_label(axis))
    ax_m.set_ylabel("N w_i (zero clamped to 0.001)")
    ax_m.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save_figure(fig, scfg.hfm_figure_path("F4_measure_weights"))
    plt.close(fig)


def main() -> None:
    apply_style()
    scfg.HFM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _json("hfm_manifest")
    dist = _json("hfm_distributions")
    summary = _table("hfm_summary")
    axes_tab = _table("hfm_axes")
    points = _table("hfm_points")
    if manifest is None or dist is None or summary is None or axes_tab is None:
        sys.exit("[hfm-fig] the run script has not produced its tables; nothing to draw.")
    print(f"[hfm-fig] P={manifest['P']} N={manifest['N']} draws={list(dist['draws'])}")
    fig_marginals_range(axes_tab, points, dist, manifest)
    fig_coverage_diversity(summary, dist)
    if points is not None and not points.empty:
        fig_target_displacement(points, summary)
        fig_measure_weights(points, dist, manifest)
    print(f"[hfm-fig] figures -> {scfg.HFM_FIGURES_DIR}")


if __name__ == "__main__":
    main()
