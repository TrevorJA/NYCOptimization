"""objective_dynamics.py - Operational-dynamics anatomy figures for the 7 active
objectives, on the historical single trace.

One figure per operational quantity (NYC delivery, Montague flow, Trenton flow,
NYC storage, downstream flooding). Each figure shows, for the default FFMP
baseline and one interpretable contrasting policy:

  * the §1 whole-trace dynamics the objective reduces (weekly series vs its
    static-Decree threshold, with sub-threshold failures shaded; or the daily
    storage / gauge-stage series);
  * the statistical reduction that collapses it to a score (CVaR90 deficit tail,
    storage duration curve); and
  * the §2 per-water-year decomposition that Borg actually optimizes on the
    historic trace (N=1 realization over its ~76 water-year units).

Guiding principle: the figures compute nothing the objective functions don't.
Every threshold, weekly-resample basis, metric-window cut and tail rule is imported
from the exact reduction helpers in ``src.objectives`` / ``src.objectives_ensemble``,
and each annotated score is asserted equal (within the objective epsilon) to the
value the driver computed via ``build_objective_set(...).compute(data)``.

This module is a self-contained diagnostic co-located with its outputs under
``outputs/supplemental/objective_dynamics/``; it is imported by the sibling
``make_objective_dynamics_figures.py`` driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Make the project importable when this module is loaded from within the
# outputs/ tree (objective_dynamics -> supplemental -> outputs -> project root).
PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (  # noqa: E402
    MONTAGUE_DECREE_TARGET_MGD,
    NYC_DECREE_DIVERSION_CAP_MGD,
    TRENTON_DECREE_TARGET_MGD,
)
from src.objectives import (  # noqa: E402
    _CVAR_TAIL_FRAC,
    _cvar_worst_mean,
    _delivery_entitlement,
    _DOWNSTREAM_GAUGES,
    _nyc_storage_pct_daily,
    _nyc_weekly_delivery_deficit_pct,
    _metric_window,
    _weekly_delivery_ok,
    _weekly_flow_deficit_pct,
    _weekly_flow_ok,
)
from src.objectives_ensemble import (  # noqa: E402
    _flood_days_minor_annual,
    _montague_deficit_cvar90_annual,
    _montague_failure_weeks_annual,
    _nyc_delivery_deficit_cvar90_annual,
    _nyc_delivery_failure_weeks_annual,
    _nyc_storage_min_annual,
    _trenton_failure_weeks_annual,
    water_year_unit_slices,
)
from src.plotting.style import (  # noqa: E402
    ARCH_COLORS,
    OBJ_AXIS_LABELS,
    label_for,
    save_figure,
)

# ---------------------------------------------------------------------------
# Encoding constants (baseline solid steel-blue; contrast dashed orange; the
# blue/orange pair is CVD-safe and linestyle carries policy identity too).
# ---------------------------------------------------------------------------

BASELINE_COLOR = ARCH_COLORS["ffmp"]      # "steelblue"
CONTRAST_COLOR = "#e0801a"                # muted orange
THRESHOLD_COLOR = "0.35"
FAIL_ALPHA = 0.16

#: Reservoir-tail gauge display names (order matches ``_DOWNSTREAM_GAUGES``).
_GAUGE_LABELS = {
    "01426500": "Hale Eddy (Cannonsville)",
    "01421000": "Fishs Eddy (Pepacton)",
    "01436690": "Bridgeville (Neversink)",
}


class Policy:
    """One policy's simulation data + scores, with its plot encoding.

    Attributes:
        label: Short human-readable name (e.g. "Baseline (FFMP)").
        data: The pywrdrb results dict (results-set name -> daily DataFrame).
        scores: Mapping objective-name -> true value from
            ``build_objective_set(ACTIVE_OBJECTIVES).compute(data)``.
        color: Line colour.
        linestyle: Line style ("-" baseline, "--" contrast).
    """

    def __init__(self, label: str, data: dict, scores: dict,
                 color, linestyle: str):
        self.label = label
        self.data = data
        self.scores = scores
        self.color = color
        self.linestyle = linestyle


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _style_time_axis(ax) -> None:
    """Decade year ticks on a daily/weekly DatetimeIndex axis."""
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)


def _water_year_labels(index: pd.DatetimeIndex):
    """Return (water_years, slices) for the metric-bearing units of a trace.

    Args:
        index: Daily DatetimeIndex of the realization.

    Returns:
        (years, slices): ``years`` is an int array of water-year labels (the
        calendar year of each unit's Sep 30 end); ``slices`` are the positional
        ``water_year_unit_slices``.
    """
    idx = pd.DatetimeIndex(index)
    slices = water_year_unit_slices(idx)
    years = np.array([idx[sl.start].year + 1 for sl in slices], dtype=int)
    return years, slices


def _weekly_mean_metric_window(series: pd.Series) -> pd.Series:
    """Metric-window weekly-mean of a daily series (display basis)."""
    return _metric_window(series).resample("W").mean()


def _score_text(policy: "Policy", name: str, fmt: str) -> str:
    """Legend label: ``"<policy>  (<value>)"`` using the true objective score."""
    return f"{policy.label}   {fmt.format(policy.scores[name])}"


def _check(recomputed: float, policy: "Policy", name: str, eps: float,
           strict: bool) -> None:
    """Assert a figure-recomputed scalar matches the true score within eps."""
    if not strict:
        return
    true = policy.scores[name]
    if not np.isfinite(true):
        return
    assert abs(recomputed - true) <= max(eps, 1e-9), (
        f"{name}: figure recomputed {recomputed:.6g} != score {true:.6g} "
        f"(eps={eps})"
    )


# ---------------------------------------------------------------------------
# Shared panels
# ---------------------------------------------------------------------------

def _flow_threshold_panel(ax, policies, flow_key: str, target: float,
                          obj_name: str, *, strict: bool = True) -> None:
    """Weekly-mean flow vs a static Decree target; sub-threshold weeks shaded.

    Reliability = fraction of weeks weekly-mean flow >= target. The shaded area
    below the target line is exactly the set of failing weeks (fail mask taken
    from ``_weekly_flow_ok``), so the visual proportion is the score.
    """
    for pol in policies:
        flow = pol.data["major_flow"][flow_key]
        weekly = _weekly_mean_metric_window(flow)
        ok = _weekly_flow_ok(_metric_window(flow), target)
        rel = float(ok.sum()) / len(ok) if len(ok) else 0.0
        _check(rel, pol, obj_name, 1e-6, strict)
        ax.plot(weekly.index, weekly.values, lw=0.7, color=pol.color,
                linestyle=pol.linestyle,
                label=_score_text(pol, obj_name, "(rel {:.2f})"))
        ax.fill_between(weekly.index, weekly.values, target,
                        where=(weekly.values < target), color=pol.color,
                        alpha=FAIL_ALPHA, linewidth=0)
    ax.axhline(target, color=THRESHOLD_COLOR, lw=1.0, linestyle=":",
               label=f"Decree target ({target:.0f} MGD)")
    ax.set_yscale("log")
    # Focus on the decision-relevant band: only the low-flow tail near/below the
    # Decree target drives this objective, so clip the (irrelevant) flood peaks.
    ax.set_ylim(target * 0.4, target * 7)
    ax.set_ylabel("Weekly-mean flow (MGD)")
    _style_time_axis(ax)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)


def _tail_panel(ax, policies, deficit_getter, obj_name: str,
                *, eps: float, strict: bool = True) -> None:
    """Rank-sorted weekly deficit % with the CVaR90 tail shaded and marked.

    x = percentage of weeks (worst first); y = weekly deficit %. The worst
    ``ceil(_CVAR_TAIL_FRAC * N)`` weeks are shaded and each policy's CVaR90
    (mean of that tail, identical to ``_cvar_worst_mean``) is drawn as a
    horizontal guide with a dot at the tail boundary — the score.
    """
    tail_pct = 100.0 * _CVAR_TAIL_FRAC
    for pol in policies:
        deficit = np.asarray(deficit_getter(pol), dtype=float)
        deficit = deficit[np.isfinite(deficit)]
        n = deficit.size
        if n == 0:
            continue
        srt = np.sort(deficit)[::-1]
        pct = 100.0 * (np.arange(n) + 0.5) / n
        cvar = _cvar_worst_mean(deficit)
        _check(cvar, pol, obj_name, eps, strict)
        ax.plot(pct, srt, lw=1.1, color=pol.color, linestyle=pol.linestyle,
                label=_score_text(pol, obj_name, "(CVaR90 {:.1f}%)"))
        ax.axhline(cvar, color=pol.color, lw=0.8, linestyle=pol.linestyle,
                   alpha=0.55)
        ax.plot([tail_pct], [cvar], marker="o", ms=6, color=pol.color, zorder=6)
    ax.axvspan(0, tail_pct, color="0.85", alpha=0.6, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Weeks ranked worst → best (%)")
    ax.set_ylabel("Weekly deficit (% of Decree)")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.85", alpha=0.6,
                         label=f"worst {tail_pct:.0f}% (CVaR90 tail; ● = score)"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)


def _annual_strip(ax, policies, annual_getter, index_getter, *, ylabel: str,
                  step: bool = True) -> None:
    """Per-water-year annual metric (the §2 unit the optimizer pools), both policies.

    Args:
        annual_getter: policy -> ndarray of one annual value per water-year unit.
        index_getter: policy -> daily DatetimeIndex used to derive water years.
        ylabel: y-axis label (native metric units).
        step: draw as a mid-step line (True) or markers (False).
    """
    for pol in policies:
        years, _ = _water_year_labels(index_getter(pol))
        vals = np.asarray(annual_getter(pol), dtype=float)
        m = min(len(years), len(vals))
        if step:
            ax.step(years[:m], vals[:m], where="mid", lw=0.9, color=pol.color,
                    linestyle=pol.linestyle, label=pol.label)
        else:
            ax.plot(years[:m], vals[:m], marker="o", ms=2.5, lw=0.0,
                    color=pol.color, label=pol.label)
    ax.set_xlabel("Water year")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)


# ---------------------------------------------------------------------------
# Figure A - NYC delivery (obj 1 reliability + obj 2 CVaR90 deficit)
# ---------------------------------------------------------------------------

def plot_delivery_anatomy(policies, *, output_file=None,
                          figsize=(13, 7.2), strict: bool = True) -> Figure:
    """NYC delivery: reliability time series + CVaR90 deficit tail + §2 strip."""
    rel_name = "nyc_delivery_reliability_weekly"
    cvar_name = "nyc_delivery_deficit_cvar90_pct"
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1.0], hspace=0.32,
                  wspace=0.22)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_tail = fig.add_subplot(gs[1, 0])
    ax_strip = fig.add_subplot(gs[1, 1])

    # (a) weekly-mean delivery vs the shared NYC demand (entitlement ~ demand);
    #     the shortfall of each failing week (delivery < 99% of the running-avg
    #     entitlement) is shaded between delivery and demand.
    demand_ref = _weekly_mean_metric_window(
        policies[0].data["ibt_demands"]["demand_nyc"])
    for pol in policies:
        delivery = pol.data["ibt_diversions"]["delivery_nyc"]
        demand = pol.data["ibt_demands"]["demand_nyc"]
        ent = _delivery_entitlement(demand, delivery,
                                    NYC_DECREE_DIVERSION_CAP_MGD, reset="annual")
        weekly_del = _weekly_mean_metric_window(delivery)
        ok = _weekly_delivery_ok(_metric_window(ent), _metric_window(delivery))
        rel = float(ok.sum()) / len(ok) if len(ok) else 0.0
        _check(rel, pol, rel_name, 0.07, strict)
        ax_ts.plot(weekly_del.index, weekly_del.values, lw=0.7, color=pol.color,
                   linestyle=pol.linestyle,
                   label=_score_text(pol, rel_name, "(rel {:.2f})"))
        # Shade each failing week's shortfall (exact weekly sum-basis mask,
        # drawn on the weekly-mean display series up to the demand cap).
        fail_weeks = (~ok).reindex(weekly_del.index, fill_value=False).values
        dref = demand_ref.reindex(weekly_del.index).values
        ax_ts.fill_between(weekly_del.index, weekly_del.values, dref,
                           where=fail_weeks, color=pol.color, alpha=FAIL_ALPHA,
                           linewidth=0)
    ax_ts.plot(demand_ref.index, demand_ref.values, lw=0.9, color=THRESHOLD_COLOR,
               linestyle=":", label="NYC demand (entitlement cap)")
    ax_ts.set_ylabel("Weekly-mean NYC delivery (MGD)")
    _style_time_axis(ax_ts)
    ax_ts.legend(loc="lower left", framealpha=0.9, fontsize=8, ncol=1)
    ax_ts.set_title("NYC delivery — reliability & deficit "
                    "(shaded = weeks below 99% of running-avg entitlement)",
                    fontsize=10)

    # (b) CVaR90 deficit tail.
    _tail_panel(ax_tail, policies,
                lambda p: _nyc_weekly_delivery_deficit_pct(p.data).values,
                cvar_name, eps=1.5, strict=strict)

    # (c) §2 per-water-year failing-weeks (what Borg minimizes; N=1 over units).
    _annual_strip(ax_strip, policies,
                  lambda p: _nyc_delivery_failure_weeks_annual(p.data),
                  lambda p: p.data["ibt_demands"]["demand_nyc"].index,
                  ylabel="Failing weeks / water-year")
    ax_strip.set_title("Annual-unit view", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure B - Montague flow (obj 3 reliability + obj 4 CVaR90 deficit)
# ---------------------------------------------------------------------------

def plot_montague_anatomy(policies, *, output_file=None,
                          figsize=(13, 7.2), strict: bool = True) -> Figure:
    """Montague flow: reliability time series + CVaR90 deficit tail + §2 strip."""
    rel_name = "montague_flow_reliability_weekly"
    cvar_name = "montague_flow_deficit_cvar90_pct"
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1.0], hspace=0.32,
                  wspace=0.22)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_tail = fig.add_subplot(gs[1, 0])
    ax_strip = fig.add_subplot(gs[1, 1])

    _flow_threshold_panel(ax_ts, policies, "delMontague",
                          MONTAGUE_DECREE_TARGET_MGD, rel_name, strict=strict)
    ax_ts.set_title("Montague flow — reliability & deficit "
                    "(shaded = weeks below the 1131 MGD Decree target)",
                    fontsize=10)

    _tail_panel(
        ax_tail, policies,
        lambda p: _weekly_flow_deficit_pct(
            _metric_window(p.data["major_flow"]["delMontague"]),
            MONTAGUE_DECREE_TARGET_MGD).values,
        cvar_name, eps=1.5, strict=strict)

    _annual_strip(ax_strip, policies,
                  lambda p: _montague_failure_weeks_annual(p.data),
                  lambda p: p.data["major_flow"]["delMontague"].index,
                  ylabel="Failing weeks / water-year")
    ax_strip.set_title("Annual-unit view", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure C - Trenton flow (obj 5 reliability only)
# ---------------------------------------------------------------------------

def plot_trenton_anatomy(policies, *, output_file=None,
                         figsize=(13, 6.4), strict: bool = True) -> Figure:
    """Trenton flow: reliability time series + §2 failing-weeks strip."""
    rel_name = "trenton_flow_reliability_weekly"
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.3, 1.0], hspace=0.35)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0])

    _flow_threshold_panel(ax_ts, policies, "delTrenton",
                          TRENTON_DECREE_TARGET_MGD, rel_name, strict=strict)
    ax_ts.set_title("Trenton flow — reliability "
                    "(shaded = weeks below the 1939 MGD Decree target)",
                    fontsize=10)

    _annual_strip(ax_strip, policies,
                  lambda p: _trenton_failure_weeks_annual(p.data),
                  lambda p: p.data["major_flow"]["delTrenton"].index,
                  ylabel="Failing weeks / water-year")
    ax_strip.set_title("Annual-unit view", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure D - NYC storage (obj 7 p5)
# ---------------------------------------------------------------------------

def plot_storage_anatomy(policies, *, output_file=None,
                         figsize=(13, 7.2), strict: bool = True) -> Figure:
    """NYC storage: daily storage % + p5 line, duration curve, §2 annual-min strip."""
    p5_name = "nyc_storage_p5_pct"
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1.0], hspace=0.32,
                  wspace=0.22)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_dur = fig.add_subplot(gs[1, 0])
    ax_strip = fig.add_subplot(gs[1, 1])

    # (a) daily combined NYC storage % with each policy's p5 line.
    for pol in policies:
        s = _metric_window(_nyc_storage_pct_daily(pol.data))
        p5 = float(np.percentile(s.values, 5))
        _check(p5, pol, p5_name, 1.5, strict)
        ax_ts.plot(s.index, s.values, lw=0.6, color=pol.color,
                   linestyle=pol.linestyle,
                   label=_score_text(pol, p5_name, "(p5 {:.1f}%)"))
        ax_ts.axhline(p5, color=pol.color, lw=0.9, linestyle=pol.linestyle,
                      alpha=0.7)
    ax_ts.set_ylabel("Combined NYC storage (% capacity)")
    ax_ts.set_ylim(0, 100)
    _style_time_axis(ax_ts)
    ax_ts.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax_ts.set_title("NYC storage resilience — daily combined storage "
                    "(horizontal line = 5th percentile)", fontsize=10)

    # (b) storage duration (exceedance) curve with the p5 score marked at 95%.
    for pol in policies:
        s = _metric_window(_nyc_storage_pct_daily(pol.data)).values
        srt = np.sort(s)[::-1]
        exceed = 100.0 * (np.arange(srt.size) + 0.5) / srt.size
        ax_dur.plot(exceed, srt, lw=1.1, color=pol.color,
                    linestyle=pol.linestyle, label=pol.label)
        ax_dur.plot([95], [float(np.percentile(s, 5))], marker="o", ms=6,
                    color=pol.color, zorder=6)
    ax_dur.axvline(95, color=THRESHOLD_COLOR, lw=0.9, linestyle=":")
    ax_dur.set_xlabel("Exceedance (% of days)")
    ax_dur.set_ylabel("Storage (% capacity)")
    ax_dur.set_xlim(0, 100)
    handles, _ = ax_dur.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=THRESHOLD_COLOR, lw=0.9, linestyle=":",
                          label="95% exceedance (● = p5 score)"))
    ax_dur.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)

    # (c) §2 per-water-year annual-minimum storage %.
    _annual_strip(ax_strip, policies,
                  lambda p: _nyc_storage_min_annual(p.data),
                  lambda p: p.data["res_storage"].index,
                  ylabel="Annual-min storage (%)")
    ax_strip.set_title("Annual-unit view", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure E - Downstream flooding (obj 6 minor flood days)
# ---------------------------------------------------------------------------

def plot_flood_anatomy(policies, *, output_file=None,
                       figsize=(13, 7.0), strict: bool = True) -> Figure:
    """Downstream flooding: baseline gauge stages vs minor stage + §2 annual bars.

    Panel (a) shows the three reservoir-tail gauge stages of the BASELINE policy
    against each gauge's NWS minor-flood line (explains what a flood day is);
    panel (b) is the §2 per-water-year flood-day count for both policies (the
    baseline-vs-contrast comparison), annotated with each policy's §1 total.
    """
    obj_name = "downstream_flood_days_minor"
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.0], hspace=0.38)
    ax_stage = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0])

    from pywrdrb.flood_thresholds import flood_stage_thresholds

    # (a) baseline three-gauge daily stage vs each gauge's minor line.
    base = policies[0]
    stage = _metric_window(base.data["flood_stage"][_DOWNSTREAM_GAUGES])
    gauge_colors = ["#3b6bc9", "#4f9648", "#a51d24"]
    for g, c in zip(_DOWNSTREAM_GAUGES, gauge_colors):
        ax_stage.plot(stage.index, stage[g].values, lw=0.5, color=c,
                      label=_GAUGE_LABELS[g])
        minor = flood_stage_thresholds[g]["minor"]
        ax_stage.axhline(minor, color=c, lw=0.8, linestyle=":", alpha=0.8)
    ax_stage.set_ylabel("Gauge stage (ft)")
    _style_time_axis(ax_stage)
    ax_stage.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=3)
    ax_stage.set_title(f"Downstream flooding — {base.label} tail-gauge stage "
                       "(dotted = NWS minor stage; a flood day = any gauge above)",
                       fontsize=10)

    # (b) §2 per-water-year flood-day count, both policies; §1 total annotated.
    for pol in policies:
        years, _ = _water_year_labels(pol.data["flood_stage"].index)
        vals = np.asarray(_flood_days_minor_annual(pol.data), dtype=float)
        m = min(len(years), len(vals))
        total = float(pol.scores[obj_name])
        ax_strip.step(years[:m], vals[:m], where="mid", lw=0.9, color=pol.color,
                      linestyle=pol.linestyle,
                      label=f"{pol.label}   ({total:.0f} flood days total)")
    ax_strip.set_xlabel("Water year")
    ax_strip.set_ylabel("Minor-flood days / water-year")
    ax_strip.margins(x=0.01)
    ax_strip.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax_strip.set_title("Annual-unit view", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure G - objective comparison on parallel axes (baseline vs contrast)
# ---------------------------------------------------------------------------

def _fmt_obj(v: float) -> str:
    """Compact formatting for a raw objective value annotation."""
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def plot_objective_parallel_axes(policies, obj_names, directions, *,
                                 output_file=None, figsize=(12, 5.6),
                                 title=None, ylabel="Relative performance  (↑ better)") -> Figure:
    """Parallel-axis comparison of the two policies' per-axis scores.

    Each axis is native-scaled to the two policies and oriented so "up" is the
    better direction; the leading policy sits at the top of each axis and the
    best/worst raw values are annotated. Crossings make the trade-off legible
    (e.g. baseline higher on flow reliability, contrast higher on storage).

    The same renderer serves both suites, but the caller sets ``title`` to name
    the quantity correctly: the ensemble suite compares the annual-unit
    **objectives** Borg optimizes, whereas the single historic trace compares
    whole-trace **performance metrics** (which are NOT the optimization
    objectives — the optimizer targets the annual-unit versions even on the
    historic trace).

    Args:
        policies: List of :class:`Policy` (uses each ``scores``/colour/style).
        obj_names: Metric/objective names, one per axis (in display order).
        directions: Per-axis direction, +1 (higher better) or -1 (lower better).
        output_file: Optional path stub (no extension); saved via ``save_figure``.
        figsize: Figure size.
        title: Axes title. Defaults to an objective-comparison title; the
            single-trace driver overrides it to a performance-metric title.
        ylabel: y-axis label.

    Returns:
        The matplotlib Figure.
    """
    n = len(obj_names)
    dirs = np.asarray(directions)
    raw = np.array([[pol.scores[name] for name in obj_names] for pol in policies],
                   dtype=float)
    col_min, col_max = raw.min(axis=0), raw.max(axis=0)
    rng = col_max - col_min
    rng[rng == 0] = 1.0
    normed = (raw - col_min) / rng
    for i in range(n):
        if dirs[i] == -1:  # minimize: flip so the low (better) value is on top
            normed[:, i] = 1.0 - normed[:, i]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n)
    for i in x:
        ax.axvline(i, color="0.85", lw=0.8, zorder=0)
    for pol, row in zip(policies, normed):
        ax.plot(x, row, color=pol.color, linestyle=pol.linestyle, lw=2.0,
                marker="o", ms=6, label=pol.label, zorder=5)

    labels = [OBJ_AXIS_LABELS.get(nm, label_for(nm)) for nm in obj_names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.3, n - 0.7)
    ax.set_ylim(-0.16, 1.16)
    ax.set_yticks([])
    ax.set_ylabel(ylabel)

    # Best (top) / worst (bottom) raw value per axis.
    for i in range(n):
        best = col_max[i] if dirs[i] == 1 else col_min[i]
        worst = col_min[i] if dirs[i] == 1 else col_max[i]
        ax.text(i, 1.05, _fmt_obj(best), ha="center", va="bottom", fontsize=7,
                color="0.4")
        ax.text(i, -0.05, _fmt_obj(worst), ha="center", va="top", fontsize=7,
                color="0.4")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=9)
    if title is None:
        title = ("Objective comparison — baseline vs storage-conservative "
                 "(each axis native-scaled to the two policies; top = better)")
    ax.set_title(title, fontsize=10)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig
