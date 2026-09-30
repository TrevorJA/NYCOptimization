"""dry_envelope_figures.py - SI figure for the dry end of the E_test forcing box.

Pure post-processing of the tables written by ``dry_envelope_run.py``. One
two-panel figure: (a) the drought magnitude of paired 10-year windows at each
annual-volume multiplier level, against the stationary sample's tail quantiles
and the historical record's windows; (b) the share of windows at each level
whose drought hazard lies beyond the stationary sample's upper tail, per
drought axis. A second figure is drawn from the production tables when they
exist. Settings ``DRYENV_*`` in ``supplemental_config.py``; PNG only. Run::

    python scripts/supplemental/dry_envelope_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_dryenv_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.plotting.style import apply_style, save_figure  # noqa: E402

TABLES = scfg.DRYENV_TABLES_DIR
FIGURES = scfg.DRYENV_FIGURES_DIR

AXIS_LABELS = {
    "drought_magnitude": "drought magnitude (SSI-6 deficit-months)",
    "drought_duration": "drought duration (months)",
    "drought_severity": "drought severity (peak SSI-6 deficit)",
    "drought_total_deficit": "summed drought magnitude, all events",
    "lowflow_min_12month": "minimum 12-month mean flow (fraction of record mean)",
    "lowflow_min_24month": "minimum 24-month mean flow (fraction of record mean)",
}
SHORT = {
    "drought_magnitude": "magnitude", "drought_duration": "duration",
    "drought_severity": "severity", "drought_total_deficit": "summed magnitude",
    "lowflow_min_12month": "12-month low flow", "lowflow_min_24month": "24-month low flow",
}
COLOR_STAT = "0.35"
COLOR_HIST = "#c1272d"
COLOR_LEVEL = "#7a4fa3"
COLOR_AXES = ["#7a4fa3", "#e6a03c", "#2b8cbe", "#4daf4a", "#984ea3", "#7f7f7f"]


def _levels() -> pd.DataFrame:
    return pd.read_csv(TABLES / "dryenv_level_definitions.csv")


def figure_local() -> None:
    """Panel (a): magnitude by level; panel (b): tail-exceedance share by level."""
    levels = _levels().sort_values("em")
    windows = pd.read_csv(TABLES / "dryenv_windows.csv")
    ref = pd.read_csv(TABLES / "dryenv_reference.csv").set_index("axis")
    lvl = pd.read_csv(TABLES / "dryenv_levels.csv")
    hist = pd.read_csv(TABLES / "dryenv_historic_windows.csv")

    ax_name = "drought_magnitude"
    em = levels["em"].to_numpy()
    cur = float(levels.loc[levels["is_margin_bound"], "em"].iloc[0])
    cmip = float(levels.loc[levels["level"] == "cmip6_min", "em"].iloc[0])
    adopted = float(levels["adopted_em"].iloc[0])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))

    data = [windows.loc[windows["source"] == lv, ax_name].to_numpy() for lv in levels["level"]]
    a.boxplot(data, positions=em, widths=0.02, showfliers=True,
              flierprops={"marker": ".", "markersize": 3, "color": COLOR_LEVEL},
              medianprops={"color": COLOR_LEVEL}, boxprops={"color": COLOR_LEVEL},
              whiskerprops={"color": COLOR_LEVEL}, capprops={"color": COLOR_LEVEL},
              manage_ticks=False)
    r = ref.loc[ax_name]
    for q, ls in ((90, "--"), (99, ":")):
        a.axhline(r[f"stationary_tail{q}"], color=COLOR_STAT, ls=ls, lw=1.0)
    a.axhline(r["stationary_extreme"], color=COLOR_STAT, ls="-", lw=0.8, alpha=0.6)
    dis = hist[hist["source"] == "historic_disjoint"][ax_name]
    a.scatter(np.full(len(dis), em.max() + 0.03), dis, marker="_", s=120, color=COLOR_HIST,
              zorder=5)
    dor = float(hist.loc[hist["source"] == "historic_drought_of_record", ax_name].iloc[0])
    a.scatter([em.max() + 0.03], [dor], marker="*", s=110, color=COLOR_HIST, zorder=6)
    for x, ls, col in ((cur, "-.", "0.6"), (cmip, ":", "0.6"), (adopted, "-", COLOR_HIST)):
        a.axvline(x, color=col, ls=ls, lw=0.9)
    a.set_xlabel("annual-volume multiplier of the forcing profile")
    a.set_ylabel(AXIS_LABELS[ax_name])
    a.set_xlim(em.min() - 0.03, em.max() + 0.06)
    handles = [
        Line2D([], [], color=COLOR_LEVEL, lw=1.5, label="forced 10-yr windows (paired levels)"),
        Line2D([], [], color=COLOR_STAT, ls="--", label="stationary sample q90"),
        Line2D([], [], color=COLOR_STAT, ls=":", label="stationary sample q99"),
        Line2D([], [], color=COLOR_STAT, ls="-", alpha=0.6, label="stationary sample maximum"),
        Line2D([], [], color=COLOR_HIST, marker="_", ls="", markersize=10,
               label="historical record, disjoint windows"),
        Line2D([], [], color=COLOR_HIST, marker="*", ls="", markersize=9,
               label="1960s drought of record window"),
        Line2D([], [], color="0.6", ls="-.", label="lower bound of the widened CMIP6 box"),
        Line2D([], [], color="0.6", ls=":", label="driest CMIP6 run"),
        Line2D([], [], color=COLOR_HIST, ls="-", label="adopted lower bound"),
    ]
    a.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper right")
    a.set_title("(a) drought magnitude at fixed annual-volume levels", fontsize=10)

    for i, axn in enumerate(list(scfg.DRYENV_DROUGHT_AXES) + ["drought_total_deficit"]):
        sub = lvl[lvl["axis"] == axn].merge(levels[["level", "em"]], on="level",
                                            suffixes=("", "_lv")).sort_values("em")
        b.plot(sub["em"], sub["frac_beyond_stationary_tail99"], marker="o", ms=4,
               color=COLOR_AXES[i], label=f"{SHORT[axn]} beyond q99")
        b.plot(sub["em"], sub["frac_beyond_stationary_tail90"], marker="s", ms=3, ls="--",
               color=COLOR_AXES[i], alpha=0.6, label=f"{SHORT[axn]} beyond q90")
    b.axhline(0.01, color=COLOR_STAT, ls=":", lw=0.8)
    b.axhline(0.10, color=COLOR_STAT, ls="--", lw=0.8)
    for x, ls, col in ((cur, "-.", "0.6"), (cmip, ":", "0.6"), (adopted, "-", COLOR_HIST)):
        b.axvline(x, color=col, ls=ls, lw=0.9)
    b.set_xlabel("annual-volume multiplier of the forcing profile")
    b.set_ylabel("share of windows beyond the stationary tail")
    b.set_ylim(-0.02, 1.02)
    b.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    b.set_title("(b) share of windows beyond the stationary sample's tail", fontsize=10)

    fig.tight_layout()
    save_figure(fig, FIGURES / "F1_dry_envelope")
    plt.close(fig)


def figure_production() -> None:
    """Production leg: E_test sub-window tails by e^m bin against the pool band."""
    path = TABLES / "dryenv_production_bins.csv"
    if not path.exists():
        print("[dryenv] production tables absent; production figure skipped")
        return
    bins = pd.read_csv(path)
    ref = pd.read_csv(TABLES / "dryenv_production_reference.csv").set_index("axis")
    axes_ = list(scfg.DRYENV_DROUGHT_AXES) + ["drought_total_deficit"]
    fig, axs = plt.subplots(1, len(axes_), figsize=(3.2 * len(axes_), 4.0), sharey=False)
    for ax, axn in zip(axs, axes_):
        sub = bins[bins["axis"] == axn].copy()
        sub["x"] = (sub["em_min"] + sub["em_max"]) / 2
        sub = sub.sort_values("x")
        ax.fill_between(sub["x"], ref.loc[axn, "pool_tail90"], ref.loc[axn, "pool_tail99"],
                        color=COLOR_STAT, alpha=0.2, label="candidate ensemble q90 to q99")
        ax.plot(sub["x"], sub["tail90"], marker="s", ms=3, color=COLOR_LEVEL, ls="--",
                label="E_test sub-windows q90")
        ax.plot(sub["x"], sub["tail99"], marker="o", ms=4, color=COLOR_LEVEL,
                label="E_test sub-windows q99")
        ax.set_title(SHORT[axn], fontsize=10)
        ax.set_xlabel("annual-volume multiplier (bin centre)")
    axs[0].set_ylabel("hazard value")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, FIGURES / "F2_dry_envelope_production")
    plt.close(fig)


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_local()
    figure_production()
    print(f"[dryenv] figures in {FIGURES}")


if __name__ == "__main__":
    main()
