"""drought_curtailment.py - Storage and FFMP curtailments through a drought.

A presentation-scale companion to :mod:`policy_dynamics` for a small number of
policies (typically the FFMP baseline and one searched policy) over one
drought. Panels, top to bottom:

    storage     NYC combined storage (% of capacity)
    diversion   NYC diversion as % of the 800 MGD Decree allocation
    montague    Montague flow target as % of the 1,750 cfs Decree obligation

The two curtailment panels share one y-axis: both are a share of the party's
1954 Supreme Court Decree quantity, so a cut to NYC and a cut to the
downstream target read on the same scale.

The diversion is the simulated delivery, which under a constant 800 MGD
demand equals the FFMP drought-stage limit unless storage is exhausted. The
Montague target is the simulated ``mrf_target_delMontague``: the FFMP
flow-target factors (and the searched ``mrf_target_scale_montague_*`` decision
variables) lower it in drought stages, so each policy draws its own.

Inputs come from ``src.policy_simulation.load_policy_runs`` with
:data:`REQUIRED_RESULTS_SETS`.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import config
from config import MONTAGUE_DECREE_TARGET_MGD, NYC_DECREE_DIVERSION_CAP_MGD
from src.objectives import nyc_storage_pct_daily
from src.plotting.front_overview import HIGHLIGHT_COLOR
from src.plotting.style import (INCUMBENT_COLOR, apply_style, font_rc,
                                save_figure, style_axes)
from src.policy_simulation import BASELINE_NAME

#: pywrdrb results sets the panels read; pass to ``load_policy_runs``.
REQUIRED_RESULTS_SETS = list(config.RESULTS_SETS) + ["mrf_targets"]

#: Smallest type size of the figure (points); the title steps up from it.
FONT_FLOOR = 16

#: Line style per role. The baseline is the wider line and is drawn first, so
#: where the two policies coincide (both at 100 %) it shows as a halo around
#: the policy line instead of being hidden by it.
BASELINE_STYLE = {"color": INCUMBENT_COLOR, "lw": 4.6, "alpha": 0.80, "zorder": 2}
POLICY_STYLE = {"color": HIGHLIGHT_COLOR, "lw": 2.4, "alpha": 1.0, "zorder": 3}
#: Colours of further policies, after the first.
EXTRA_POLICY_COLORS = ["#eb6834", "#1baf7a", "#9467bd"]

GRID_COLOR = "#E4E4E4"
REFERENCE_COLOR = "0.45"
NOTE_COLOR = "0.25"
#: Storage (% of capacity) below which the reservoirs count as empty: the
#: diversion then falls below the FFMP limit because there is no water to send.
EMPTY_STORAGE_PCT = 1.0


def _styles(names, baseline: str) -> dict:
    """Line keyword arguments per policy name."""
    styles, k = {}, 0
    for name in names:
        if name == baseline:
            styles[name] = dict(BASELINE_STYLE)
        else:
            styles[name] = dict(POLICY_STYLE)
            if k:
                styles[name]["color"] = EXTRA_POLICY_COLORS[(k - 1) % len(EXTRA_POLICY_COLORS)]
            k += 1
    return styles


def _window(series: pd.Series, date_range) -> pd.Series:
    return series.loc[pd.Timestamp(date_range[0]):pd.Timestamp(date_range[1])]


def storage_pct(data: dict) -> pd.Series:
    """NYC combined storage, % of capacity (daily)."""
    return nyc_storage_pct_daily(data)


def diversion_pct(data: dict) -> pd.Series:
    """NYC diversion as % of the 800 MGD Decree allocation (daily)."""
    return 100.0 * data["ibt_diversions"]["delivery_nyc"] / NYC_DECREE_DIVERSION_CAP_MGD


def montague_target_pct(data: dict) -> pd.Series:
    """Montague flow target as % of the 1,750 cfs Decree obligation (daily)."""
    return 100.0 * data["mrf_targets"]["mrf_target_delMontague"] / MONTAGUE_DECREE_TARGET_MGD


#: Panel key -> (title, y label, series function, drawstyle, smoothing days).
PANELS = OrderedDict([
    ("storage", ("NYC reservoir storage", "% of capacity",
                 storage_pct, "default", 1)),
    ("diversion", ("NYC diversion", "% of 800 MGD\nallocation",
                   diversion_pct, "default", 1)),
    ("montague", ("Montague flow target", "% of 1,750 cfs\nobligation",
                  montague_target_pct, "default", 1)),
])


def _note_empty_storage(ax, runs: dict, labels: dict, date_range) -> None:
    """Label the diversion collapse of the first policy that empties storage.

    Diversions below the FFMP stage limit are not a rule: they are the
    physical shortfall of an empty system. One note, to the right of the last
    empty-storage day, says so.
    """
    for name, data in runs.items():
        s = _window(storage_pct(data), date_range)
        empty = s.index[s.to_numpy() < EMPTY_STORAGE_PCT]
        if len(empty) == 0:
            continue
        ax.text(empty[-1] + pd.Timedelta(days=35), 27,
                f"{labels.get(name, name)}:\nreservoirs empty, diversion fails",
                ha="left", va="center", color=NOTE_COLOR, linespacing=1.2)
        return


def curtailment_table(runs: dict, date_range, smooth: bool = False) -> pd.DataFrame:
    """Plotted daily series, one column per (panel, policy), for the companion CSV."""
    cols = {}
    for key, (_, _, fn, _, days) in PANELS.items():
        for name, data in runs.items():
            s = fn(data)
            if smooth and days > 1:
                s = s.rolling(days, center=True, min_periods=1).mean()
            cols[f"{key}_pct__{name}"] = _window(s, date_range)
    return pd.DataFrame(cols)


def plot_drought_curtailment(runs: dict, labels: dict, *, date_range,
                             baseline: str = BASELINE_NAME, title: str = "",
                             subtitle: str = "", figsize=(13.0, 11.5),
                             output_file: Path | None = None) -> Figure:
    """Stack storage and the two Decree-share curtailment panels.

    Args:
        runs: Ordered mapping ``policy name -> results dict`` (from
            ``load_policy_runs`` with :data:`REQUIRED_RESULTS_SETS`).
        labels: ``policy name -> legend label``.
        date_range: ``(start, end)`` date strings of the plotted window.
        baseline: Policy drawn in the baseline style.
        title: Figure title (left-aligned).
        subtitle: Second title line in muted ink (scenario, demand).
        figsize: Figure size in inches.
        output_file: Optional path stub (no extension) for ``style.save_figure``.

    Returns:
        The matplotlib Figure.
    """
    if not runs:
        raise ValueError("runs is empty; nothing to plot")
    apply_style()
    styles = _styles(runs, baseline)
    start, end = (pd.Timestamp(d) for d in date_range)

    with plt.rc_context(font_rc(FONT_FLOOR)):
        fig, axes = plt.subplots(
            len(PANELS), 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [1.35, 1.0, 1.0], "hspace": 0.30})
        for k, (ax, (key, (ptitle, ylabel, fn, _, days))) in enumerate(
                zip(axes, PANELS.items())):
            style_axes(ax)
            for name, data in runs.items():
                s = fn(data)
                if days > 1:
                    s = s.rolling(days, center=True, min_periods=1).mean()
                s = _window(s, date_range)
                ax.plot(s.index, s.values, solid_capstyle="round",
                        solid_joinstyle="round", **styles[name])
            ax.set_ylim(0, 108)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
            ax.grid(axis="y", color=GRID_COLOR, lw=0.9, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylabel(ylabel)
            ax.set_title(f"({chr(97 + k)}) {ptitle}", loc="left", pad=8)
            if key != "storage":
                ax.axhline(100, color=REFERENCE_COLOR, lw=1.2, ls=(0, (1, 2)),
                           zorder=1)
            if key == "diversion":
                _note_empty_storage(ax, runs, labels, date_range)

        axes[-1].set_xlim(start, end)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
        for ax in axes:
            for year in range(start.year, end.year + 1):
                ax.axvline(pd.Timestamp(year=year, month=1, day=1),
                           color=GRID_COLOR, lw=0.9, zorder=0)

        handles = [Line2D([], [], color=styles[n]["color"], lw=styles[n]["lw"],
                          alpha=styles[n]["alpha"], label=labels.get(n, n))
                   for n in runs]
        fig.legend(handles=handles, loc="upper right", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.985, 0.972),
                   handlelength=2.4, columnspacing=2.0)
        if title:
            fig.text(0.075, 0.985, title, ha="left", va="top",
                     fontsize=plt.rcParams["font.size"] + 4)
        if subtitle:
            fig.text(0.075, 0.948, subtitle, ha="left", va="top", color="0.35")
        fig.subplots_adjust(left=0.135, right=0.985, top=0.885, bottom=0.055)
        fig.align_ylabels(axes)
        if output_file is not None:
            save_figure(fig, output_file)
    return fig
