"""policy_dynamics.py - Stacked system-dynamics panels for a set of policies.

A compact, manuscript-style companion to :mod:`historic_timeseries`: the same
derived series (NYC aggregate storage, NYC delivery against its Decree
entitlement, Montague flow) for a few selected policies over a chosen period,
with no metric annotations. Panels are registered in :data:`PANELS` so a
figure can be assembled from any subset or order, and new panels plug in with
the same signature.

Default panels, top to bottom:

    storage     NYC aggregate storage (% of combined capacity)
    diversion   NYC diversion (solid) and shortage below entitlement (dashed)
    montague    Montague flow (solid) and the policy's MRF target (dashed)

The Montague target is the simulated ``mrf_target_delMontague`` series, not
the static Decree constant: the FFMP flow-target factors (and the searched
``mrf_target_scale_montague_*`` decision variables) lower it in drought
stages, so every policy draws its own target.

Inputs come from ``src.policy_simulation.load_policy_runs`` with
:data:`REQUIRED_RESULTS_SETS`. Series derivations are shared with the
objectives (``nyc_storage_pct_daily``) and the full timeseries figure
(``historic_timeseries.nyc_delivery_series``), so nothing is re-derived here.
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
from config import MONTAGUE_DECREE_TARGET_MGD
from src.objectives import nyc_storage_pct_daily
from src.plotting.front_overview import BASELINE_COLOR, POLICY_COLORS
from src.plotting.historic_timeseries import nyc_delivery_series
from src.plotting.style import apply_style, save_figure
from src.policy_simulation import BASELINE_NAME

#: pywrdrb results sets the panels read; pass to ``load_policy_runs``.
REQUIRED_RESULTS_SETS = list(config.RESULTS_SETS) + ["mrf_targets"]

#: Named analysis periods ``name -> (start, end)``; None is the full record.
DROUGHT_PERIODS = OrderedDict([
    ("drought_1960s", ("1961-01-01", "1968-01-01")),
    ("drought_2002", ("2001-01-01", "2003-12-31")),
    ("full", None),
])

#: Line widths and the dashed style shared by the secondary series.
LW_PRIMARY = 1.6
LW_SECONDARY = 1.2
SECONDARY_LS = (0, (3, 2))
#: Montague panel y-axis cap as a multiple of the Decree target, so the
#: drought-stage target band stays legible beneath the wet-season peaks.
MONTAGUE_YLIM_TARGET_MULTIPLE = 3.0


def policy_colors(names, baseline: str = BASELINE_NAME) -> dict:
    """Baseline colour for ``baseline``, the policy palette for the rest."""
    colors, i = {}, 0
    for name in names:
        if name == baseline:
            colors[name] = BASELINE_COLOR
        else:
            colors[name] = POLICY_COLORS[i % len(POLICY_COLORS)]
            i += 1
    return colors


def _window(series: pd.Series, date_range) -> pd.Series:
    """Restrict a series to ``date_range`` (``(start, end)`` or None)."""
    if date_range is None:
        return series
    return series.loc[pd.Timestamp(date_range[0]):pd.Timestamp(date_range[1])]


# --------------------------------------------------------------------------- #
# Panels: (ax, runs, colors, resample, date_range) -> None
# --------------------------------------------------------------------------- #
def panel_storage(ax, runs: dict, colors: dict, resample: str,
                  date_range) -> None:
    """NYC aggregate storage as % of combined capacity."""
    for name, data in runs.items():
        s = _window(nyc_storage_pct_daily(data), date_range).resample(resample).mean()
        ax.plot(s.index, s.values, color=colors[name], lw=LW_PRIMARY)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Storage (% of capacity)")


def panel_diversion_shortage(ax, runs: dict, colors: dict, resample: str,
                             date_range) -> None:
    """NYC diversion (solid) and shortage below the entitlement (dashed)."""
    for name, data in runs.items():
        delivery, entitlement = nyc_delivery_series(data)
        shortage = (entitlement - delivery).clip(lower=0)
        d = _window(delivery, date_range).resample(resample).mean()
        s = _window(shortage, date_range).resample(resample).mean()
        ax.plot(d.index, d.values, color=colors[name], lw=LW_PRIMARY)
        ax.plot(s.index, s.values, color=colors[name], lw=LW_SECONDARY,
                ls=SECONDARY_LS)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("NYC diversion (MGD)")


def panel_montague(ax, runs: dict, colors: dict, resample: str,
                   date_range) -> None:
    """Montague flow (solid) and each policy's simulated MRF target (dashed).

    The y-axis is capped at :data:`MONTAGUE_YLIM_TARGET_MULTIPLE` times the
    Decree target (wet-season peaks are clipped) so the drought-stage target
    band stays legible.
    """
    for name, data in runs.items():
        flow = _window(data["major_flow"]["delMontague"], date_range)
        target = _window(data["mrf_targets"]["mrf_target_delMontague"], date_range)
        f = flow.resample(resample).mean()
        t = target.resample(resample).mean()
        ax.plot(f.index, f.values, color=colors[name], lw=LW_PRIMARY)
        ax.plot(t.index, t.values, color=colors[name], lw=LW_SECONDARY,
                ls=SECONDARY_LS)
    ax.set_ylim(0, MONTAGUE_YLIM_TARGET_MULTIPLE * MONTAGUE_DECREE_TARGET_MGD)
    ax.set_ylabel("Montague flow (MGD)")


#: Panel key -> (title, draw function). Titles name the solid and dashed
#: series so no in-panel legend is needed.
PANELS = OrderedDict([
    ("storage", ("NYC aggregate storage", panel_storage)),
    ("diversion", ("NYC diversion (solid) and shortage (dashed)",
                   panel_diversion_shortage)),
    ("montague", ("Montague flow (solid) and MRF target (dashed)",
                  panel_montague)),
])
DEFAULT_PANELS = tuple(PANELS)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_policy_dynamics(runs: dict, labels: dict, *,
                         baseline: str = BASELINE_NAME,
                         panels=DEFAULT_PANELS, date_range=None,
                         resample: str = "W", figsize=None,
                         output_file: Path | None = None) -> Figure:
    """Stack the registered panels for every policy in ``runs``.

    Args:
        runs: Ordered mapping ``policy name -> results dict`` (from
            ``load_policy_runs`` with :data:`REQUIRED_RESULTS_SETS`).
        labels: ``policy name -> legend label``.
        baseline: Policy drawn in the baseline colour.
        panels: Keys of :data:`PANELS`, top to bottom.
        date_range: ``(start, end)`` date strings, or None for the full record.
        resample: Pandas rule for the plotted means (weekly by default).
        figsize: Figure size; default 11 in wide by 3 in per panel.
        output_file: Optional path stub (no extension) for ``style.save_figure``.

    Returns:
        The matplotlib Figure.
    """
    if not runs:
        raise ValueError("runs is empty; nothing to plot")
    unknown = [p for p in panels if p not in PANELS]
    if unknown:
        raise KeyError(f"unknown panels {unknown}; available: {list(PANELS)}")

    apply_style()
    colors = policy_colors(runs, baseline)
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=figsize or (11.0, 3.0 * n),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, key) in enumerate(zip(axes, panels)):
        title, fn = PANELS[key]
        fn(ax, runs, colors, resample, date_range)
        ax.set_title(f"({chr(97 + k)}) {title}", loc="left")
        ax.grid(True, alpha=0.25)
    if date_range is not None:
        axes[-1].set_xlim(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    handles = [Line2D([], [], color=colors[name], lw=2.2, label=labels.get(name, name))
               for name in runs]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if output_file is not None:
        save_figure(fig, output_file)
    return fig
