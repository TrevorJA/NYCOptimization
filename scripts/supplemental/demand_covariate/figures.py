"""figures.py - Figures of the NYC demand-covariate diagnostics.

Renders, from the tables ``run.py`` wrote (zero simulation), old (USGS
gauge-sum covariate) versus new (reconstruction reservoir-inflow covariate),
current FFMP policy on the historic record:

  storage_1960s_drought        combined NYC storage under both demands and under
                               the superseded constant 800 MGD reference over the
                               1960s drought with the FFMP zone boundaries, and
                               the monthly-mean demand series of the three
  storage_difference_windows   the whole-record storage difference (new - old)
                               with the screening threshold, and one panel per
                               screened window with both storage series
  demand_monthly_climatology   monthly climatology of the two demand series
  demand_difference            the two monthly-mean demand series and their
                               difference over the record, eras shaded

Manuscript style; every number lives in the CSVs, none in panel text.
Outputs -> outputs/supplemental/demand_covariate/figures. Configuration lives
in supplemental_config.py (DCV_* section) -- no CLI value flags.

Usage:
    python scripts/supplemental/demand_covariate/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_dcv_env()

from src.plotting import style  # noqa: E402
from src.simulation import _ZONE_LEVELS  # noqa: E402

#: Variant identity: the superseded covariate in muted gray, the current one in
#: Okabe-Ito blue; line style is the secondary encoding.
VARIANT_STYLE = {
    "old": {"color": "#8C8C8C", "ls": "--", "label": "USGS gauge-sum covariate (superseded)"},
    "new": {"color": "#0072B2", "ls": "-", "label": "Reservoir-inflow covariate (current)"},
}
#: The constant-demand reference (flat 800 MGD, every output before 2026-11-03),
#: drawn on the 1960s drought figure only.
REFERENCE_STYLE = {
    "constant_800": {"color": "#D55E00", "ls": "-.",
                     "label": "Constant 800 MGD demand (superseded reference)"},
}
ZONE_LABELS = {"level1b": "L1b", "level1c": "L1c", "level2": "L2",
               "level3": "L3", "level4": "L4", "level5": "L5"}
ZONE_COLOR = "#B8860B"
ERA_SHADE = {"observed_2000_2021": "#DDDDDD"}
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def _table(name: str, **kw) -> pd.DataFrame:
    path = scfg.dcv_table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/supplemental/demand_covariate/run.py first.")
    return pd.read_csv(path, **kw)


def _daily() -> pd.DataFrame:
    return _table("daily_series", index_col="date", parse_dates=True)


def _zone_curves(zones: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map the day-of-year zone boundaries onto a date index."""
    doy = np.clip(index.dayofyear.to_numpy(), 1, 366) - 1
    return pd.DataFrame({lvl: zones[lvl].to_numpy()[doy] for lvl in _ZONE_LEVELS}, index=index)


def _draw_zones(ax, curves: pd.DataFrame) -> None:
    for lvl in _ZONE_LEVELS:
        ax.plot(curves.index, curves[lvl], color=ZONE_COLOR, lw=0.8, alpha=0.7, zorder=1)
        ax.annotate(ZONE_LABELS[lvl], (curves.index[-1], curves[lvl].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=plt.rcParams["font.size"] - 2, color=ZONE_COLOR)


def _draw_storage(ax, daily: pd.DataFrame, zones: pd.DataFrame, start, end,
                  styles: dict = VARIANT_STYLE) -> None:
    sub = daily.loc[start:end]
    _draw_zones(ax, _zone_curves(zones, pd.DatetimeIndex(sub.index)))
    for v, st in styles.items():
        ax.plot(sub.index, sub[f"storage_pct_{v}"], color=st["color"], ls=st["ls"], lw=1.6,
                label=st["label"], zorder=3)
    ax.set_ylim(0, 102)
    ax.set_xlim(sub.index[0], sub.index[-1])
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)


def fig_storage_1960s() -> Path:
    daily, zones = _daily(), _table("zone_boundaries")
    start, end = scfg.DCV_DROUGHT_1960S
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.1], "hspace": 0.08})
    styles = {**VARIANT_STYLE, **REFERENCE_STYLE}
    _draw_storage(axes[0], daily, zones, start, end, styles)
    axes[0].set_ylabel("Combined NYC storage (% of capacity)")
    axes[0].legend(loc="lower left", frameon=False)
    sub = daily.loc[start:end]
    for v, st in styles.items():
        monthly = sub[f"demand_{v}"].resample("MS").mean()
        axes[1].step(monthly.index, monthly, where="post", color=st["color"], ls=st["ls"], lw=1.6)
    axes[1].set_ylabel("NYC demand,\nmonthly mean (MGD)")
    axes[1].set_ylim(0, 850)
    axes[1].grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    out = scfg.dcv_figure_path("storage_1960s_drought")
    style.save_figure(fig, out)
    plt.close(fig)
    return out


def fig_storage_difference_windows() -> Path:
    daily, zones = _daily(), _table("zone_boundaries")
    windows = _table("storage_difference_windows")
    diff = daily["storage_pct_new"] - daily["storage_pct_old"]
    n = len(windows)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(13, 3.6 + 3.1 * nrows))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.3] + [1.0] * nrows,
                          hspace=0.45, wspace=0.34, bottom=0.05, top=0.98)
    top = fig.add_subplot(gs[0, :])
    top.plot(diff.index, diff, color=VARIANT_STYLE["new"]["color"], lw=0.9)
    thr = scfg.DCV_STORAGE_DIFF_THRESHOLD_PCT
    for y in (thr, -thr):
        top.axhline(y, color="#8C8C8C", lw=0.8, ls=":")
    top.axhline(0, color="#8C8C8C", lw=0.6)
    for w in windows.itertuples():
        top.axvspan(pd.Timestamp(w.start), pd.Timestamp(w.end), color="#F0E442", alpha=0.45, lw=0)
    top.set_ylabel("Storage difference,\nnew minus old (% of capacity)")
    top.set_xlim(diff.index[0], diff.index[-1])
    top.xaxis.set_major_locator(mdates.YearLocator(10))
    top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    top.grid(axis="y", color="#E6E6E6", lw=0.6)
    handles = None
    for k, w in enumerate(windows.itertuples()):
        ax = fig.add_subplot(gs[1 + k // ncols, k % ncols])
        _draw_storage(ax, daily, zones, w.panel_start, w.panel_end)
        ax.set_title(f"{w.start[:7]} to {w.end[:7]}")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if k % ncols == 0:
            ax.set_ylabel("Combined NYC storage\n(% of capacity)")
        handles = ax.get_legend_handles_labels()
    if handles is not None:
        fig.legend(*handles, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.005))
    out = scfg.dcv_figure_path("storage_difference_windows")
    style.save_figure(fig, out)
    plt.close(fig)
    return out


def fig_demand_climatology() -> Path:
    clim = _table("demand_monthly_climatology")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = clim["month"].to_numpy()
    for v, st in VARIANT_STYLE.items():
        ax.fill_between(x, clim[f"{v}_p25"], clim[f"{v}_p75"], color=st["color"], alpha=0.15, lw=0)
        ax.plot(x, clim[f"{v}_mean"], color=st["color"], ls=st["ls"], lw=2.0, marker="o", ms=5,
                label=st["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel("NYC demand, monthly mean (MGD)")
    ax.set_ylim(0, 850)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.legend(loc="lower left", frameon=False)
    out = scfg.dcv_figure_path("demand_monthly_climatology")
    style.save_figure(fig, out)
    plt.close(fig)
    return out


def fig_demand_difference() -> Path:
    monthly = _table("demand_monthly_means", index_col="month_start", parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.08})
    for ax in axes:
        for name, start, end in scfg.DCV_ERAS:
            if name in ERA_SHADE:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=ERA_SHADE[name], alpha=0.6, lw=0)
            else:
                ax.axvline(pd.Timestamp(start), color="#BBBBBB", lw=0.6, ls=":")
    for v, st in VARIANT_STYLE.items():
        axes[0].plot(monthly.index, monthly[v], color=st["color"], ls=st["ls"], lw=0.9, label=st["label"])
    axes[0].set_ylabel("NYC demand,\nmonthly mean (MGD)")
    axes[0].set_ylim(0, 850)
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].plot(monthly.index, monthly["diff"], color=VARIANT_STYLE["new"]["color"], lw=0.9)
    axes[1].plot(monthly.index, monthly["diff"].rolling(12, center=True).mean(),
                 color="#000000", lw=1.4, label="12-month mean")
    axes[1].axhline(0, color="#8C8C8C", lw=0.6)
    axes[1].set_ylabel("Demand difference,\nnew minus old (MGD)")
    axes[1].legend(loc="lower left", frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[1].set_xlim(monthly.index[0], monthly.index[-1])
    axes[1].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    out = scfg.dcv_figure_path("demand_difference")
    style.save_figure(fig, out)
    plt.close(fig)
    return out


def main() -> None:
    style.apply_manuscript_style()
    scfg.DCV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (fig_storage_1960s, fig_storage_difference_windows,
               fig_demand_climatology, fig_demand_difference):
        print(f"[dcv_figures] {fn()}.png")


if __name__ == "__main__":
    main()
