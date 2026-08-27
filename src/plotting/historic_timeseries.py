"""historic_timeseries.py - Selected policies vs the FFMP baseline over time.

One stacked-panel figure over the historic simulation period covering the
system states the objectives score:

    NYC aggregate storage          (% of combined capacity)
    Montague flow                  (weekly mean, Decree target, violation weeks)
    Trenton flow                   (weekly mean, Decree target, violation weeks)
    NYC diversion                  (weekly mean, against the Decree entitlement)
    NYC delivery shortage          (weekly mean below the entitlement)

Agreement with the objective values is the point of this figure, so nothing
here re-derives a metric. Violation weeks come from
``src.objectives.weekly_flow_ok`` — the same weekly resample and the same
``FLOW_TARGET_TOL_MGD`` headroom the reliability objectives use — the delivery
entitlement comes from ``src.objectives.delivery_entitlement``, and storage
comes from ``src.objectives.nyc_storage_pct_daily``. The reliability printed in
each flow panel's legend is therefore the objective value, not a lookalike.

The metric window excludes the first ``config.METRIC_EXCLUSION_MONTHS`` months
(the SSI-6 accumulation spin-up, ``objectives.metric_window``); that span is
shaded and labelled on every panel, because nothing inside it was scored.

Model mode: the caller MUST state which model produced the results
(``model_mode``), and it is stamped on the figure. The persisted baseline in
``outputs/baseline/`` was written with the FULL model while search runs the
TRIMMED model, so a figure mixing the two would attribute a model-mode
difference to policy. Simulate every trace in one job, in one mode.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from config import (METRIC_EXCLUSION_MONTHS, MONTAGUE_DECREE_TARGET_MGD,
                    NYC_DECREE_DIVERSION_CAP_MGD, TRENTON_DECREE_TARGET_MGD)
from src.objectives import (delivery_entitlement, metric_window,
                            nyc_storage_pct_daily, weekly_flow_ok)
from src.plotting.front_overview import BASELINE_COLOR, POLICY_COLORS
from src.plotting.style import save_figure

#: Downstream Decree control points and their static flow targets (MGD).
FLOW_TARGETS = [
    ("delMontague", "Montague", MONTAGUE_DECREE_TARGET_MGD),
    ("delTrenton", "Trenton", TRENTON_DECREE_TARGET_MGD),
]

#: Colour of the excluded spin-up shading.
SPINUP_COLOR = "0.55"


def _spinup_cutoff(index) -> pd.Timestamp | None:
    """First timestamp inside the metric window, or None for an empty index."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return None
    return idx[0] + pd.DateOffset(months=METRIC_EXCLUSION_MONTHS)


def _shade_spinup(ax, index, label: bool = False) -> None:
    """Shade the metric-window spin-up that the objectives never scored."""
    idx = pd.DatetimeIndex(index)
    cutoff = _spinup_cutoff(idx)
    if cutoff is None:
        return
    ax.axvspan(idx[0], cutoff, color=SPINUP_COLOR, alpha=0.18, lw=0, zorder=0)
    if label:
        # clip_on: Text is unclipped by default, and an off-screen label in
        # x-data coords would drag the tight bounding box out to it when the
        # x-axis is zoomed away from the start of the record.
        ax.text(cutoff, 0.97, f"  {METRIC_EXCLUSION_MONTHS}-mo spin-up "
                              f"(excluded from every metric)",
                transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=7.5, color="0.35", clip_on=True)


def _nyc_delivery_series(data: dict) -> tuple:
    """NYC daily delivery and its Decree entitlement, full (unwindowed) series.

    The running-average allowance behind the entitlement is path-dependent, so
    it must be reconstructed on the full series before any windowing — exactly
    as ``objectives._nyc_delivery_reliability_weekly`` does.
    """
    delivery = data["ibt_diversions"]["delivery_nyc"]
    entitlement = delivery_entitlement(
        data["ibt_demands"]["demand_nyc"], delivery,
        NYC_DECREE_DIVERSION_CAP_MGD, reset="annual",
    )
    return delivery, entitlement


def _panel_storage(ax, results: dict, colors: dict, resample: str) -> None:
    """NYC combined storage as % of capacity."""
    for label, data in results.items():
        s = nyc_storage_pct_daily(data).resample(resample).mean()
        ax.plot(s.index, s.values, color=colors[label], lw=1.0, alpha=0.9,
                label=label)
    ax.set_ylabel("NYC storage\n(% capacity)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)


def _panel_flow(ax, results: dict, colors: dict, column: str, name: str,
                target: float) -> None:
    """Weekly-mean flow at a Decree control point, with violation weeks marked.

    Violation weeks are the complement of :func:`objectives.weekly_flow_ok` on
    the metric window — the same weekly indicators every reliability objective
    is built from, at either timescale.

    The number in the legend is the WHOLE-TRACE weekly reliability (§1): the
    fraction of metric-window weeks that meet the target. When a scenario
    design is wired the SEARCHED objective is the annual-unit (§2) reduction of
    these same indicators — the fraction of FFMP years with fewer than k
    failing weeks — so the legend value is a different statistic from the
    ``*_reliability_annual`` column in the selection CSV and is labelled as
    such rather than presented as the objective value.
    """
    n = len(results)
    ax.set_yscale("log")   # before any set_ylim: switching scales resets limits
    violations, floor = {}, np.inf
    for label, data in results.items():
        flow = data["major_flow"][column]
        weekly = flow.resample("W").mean()
        ax.plot(weekly.index, weekly.values, color=colors[label], lw=0.8,
                alpha=0.85)
        floor = min(floor, float(np.nanmin(weekly.values)))
        ok = weekly_flow_ok(metric_window(flow), target)
        rel = float(ok.mean()) if len(ok) else float("nan")
        violations[label] = ok.index[~ok.to_numpy()]
        ax.plot([], [], color=colors[label], lw=2,
                label=f"{label} — whole-trace weekly reliability {rel:.3f}, "
                      f"{len(violations[label])} violation weeks")
    # Violation rug: one lane per policy in a reserved band at the axis foot.
    # Drawn INSIDE the axes and clipped — an unclipped rug in x-data coords
    # would drag the tight bounding box out to the full record whenever the
    # x-axis is zoomed, blowing the saved figure up to tens of thousands of px.
    ax.set_ylim(bottom=max(floor, 1e-3) / (2.0 + 0.35 * n))
    for i, (label, bad) in enumerate(violations.items()):
        y = 0.015 + 0.032 * i
        ax.plot(bad, np.full(len(bad), y), linestyle="none", marker="|",
                markersize=3.5, markeredgewidth=0.7, color=colors[label],
                transform=ax.get_xaxis_transform(), clip_on=True)
    ax.axhline(target, color="0.2", ls="--", lw=1.1, zorder=5)
    ax.text(0.002, target, f" Decree target {target:.0f} MGD", fontsize=7,
            color="0.2", va="bottom", ha="left",
            transform=ax.get_yaxis_transform())
    ax.set_ylabel(f"{name} flow\n(weekly mean, MGD)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=max(1, n // 2), framealpha=0.9)


def _panel_diversion(ax, results: dict, colors: dict, resample: str) -> None:
    """NYC diversion against the running-average Decree entitlement."""
    first = True
    for label, data in results.items():
        delivery, entitlement = _nyc_delivery_series(data)
        d = delivery.resample(resample).mean()
        ax.plot(d.index, d.values, color=colors[label], lw=0.8, alpha=0.85,
                label=label)
        if first:
            e = entitlement.resample(resample).mean()
            ax.plot(e.index, e.values, color="0.25", ls=":", lw=0.9,
                    label="entitlement (baseline trace)", zorder=1)
            first = False
    ax.set_ylabel("NYC diversion\n(weekly mean, MGD)")
    ax.grid(True, alpha=0.3)


def _panel_shortage(ax, results: dict, colors: dict, resample: str) -> None:
    """NYC delivery shortfall below the Decree entitlement."""
    for label, data in results.items():
        delivery, entitlement = _nyc_delivery_series(data)
        short = (entitlement - delivery).clip(lower=0).resample(resample).mean()
        ax.plot(short.index, short.values, color=colors[label], lw=0.8,
                alpha=0.85, label=label)
    ax.set_ylabel("NYC shortage\n(weekly mean, MGD)")
    ax.grid(True, alpha=0.3)


def plot_historic_timeseries(
    results: dict,
    output_file: Path | None = None,
    model_mode: str = "unknown",
    date_range: tuple | None = None,
    resample: str = "W",
    baseline_label: str = "FFMP baseline",
    figsize: tuple = (14.0, 12.0),
) -> Figure:
    """Stack selected policies against the baseline over the historic record.

    Args:
        results: Ordered mapping ``label -> results dict`` from
            ``src.simulation.run_simulation_inmemory``. The entry whose key
            equals ``baseline_label`` is drawn in the baseline colour; the rest
            take the representative-policy palette in insertion order.
        output_file: Optional path stub (no extension); saved via
            ``style.save_figure``.
        model_mode: ``"trimmed"`` or ``"full"`` — which Pywr-DRB model produced
            EVERY trace in ``results``. Stamped on the figure, because a
            trimmed candidate plotted against a full-model baseline would
            attribute a model difference to policy.
        date_range: Optional ``(start, end)`` date strings to zoom the x-axis.
            Metrics in the legend are always computed on the FULL metric
            window, never on the zoom, so zooming never changes a reported
            number.
        resample: Pandas resample rule for the storage/diversion/shortage
            panels. The flow panels are always weekly, matching the objective.
        baseline_label: Key in ``results`` treated as the baseline.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results is empty — nothing to plot")

    colors, i = {}, 0
    for label in results:
        if label == baseline_label:
            colors[label] = BASELINE_COLOR
        else:
            colors[label] = POLICY_COLORS[i % len(POLICY_COLORS)]
            i += 1

    index = next(iter(results.values()))["res_storage"].index
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    _panel_storage(axes[0], results, colors, resample)
    for ax, (column, name, target) in zip(axes[1:3], FLOW_TARGETS):
        _panel_flow(ax, results, colors, column, name, target)
    _panel_diversion(axes[3], results, colors, resample)
    _panel_shortage(axes[4], results, colors, resample)

    for k, ax in enumerate(axes):
        _shade_spinup(ax, index, label=(k == 0))
        if date_range is not None:
            ax.set_xlim(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    axes[0].legend(fontsize=8, loc="lower left", ncol=min(4, len(results)),
                   framealpha=0.9)
    axes[-1].set_xlabel("Date")

    span = (f"{pd.Timestamp(index[0]):%Y-%m-%d} to "
            f"{pd.Timestamp(index[-1]):%Y-%m-%d}")
    zoom = "" if date_range is None else f"  |  zoom {date_range[0]}–{date_range[1]}"
    fig.suptitle(f"Selected policies vs FFMP baseline, historic trace ({span})"
                 f"  |  {model_mode} Pywr-DRB model for every trace{zoom}\n"
                 f"Violation weeks are objectives.weekly_flow_ok on the metric "
                 f"window; legend reliabilities are the whole-trace weekly "
                 f"fraction, not the annual-unit search objective",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    if output_file is not None:
        save_figure(fig, output_file)
    return fig
