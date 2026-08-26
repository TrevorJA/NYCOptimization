"""ensemble_objective_dynamics.py - Ensemble-evaluation anatomy figures.

The ensemble analog of the historical single-trace ``objective_dynamics``
sequence: the same operational quantities (NYC delivery, Montague flow, Trenton
flow, NYC storage, downstream flooding), but scored the way the optimizer scores
them during an ensemble search — the **two-layer annual-unit scheme**
(``objective_definitions.md`` §2, implemented in ``src.objectives_ensemble``):

  Stage (i)  one annual metric per (realization x water-year) unit;
  Stage (ii) a single operator over the ensemble water years (every
             realization's water-year units) POOLED across all realizations
             (failure frequency, pooled P99 / P01, or pooled mean).

Each figure has two rows that mirror the two layers:

  * **Row 1 - seasonal distribution.** The realizations are independent random
    synthetic sequences, so their calendar dates are not comparable and a time
    series would imply a spurious alignment. Instead the underlying dynamic is
    shown as a **distribution across the standard water year** — every
    realization-year pooled by day-of-water-year, drawn as a median line with an
    inter-quartile band per policy, against the static Decree threshold (flood is
    shown as its per-month contribution to the annual objective).
  * **Row 2 - pooled reduction.** The empirical CDF of the pooled annual-unit
    metric for each policy, with the stage-(ii) operator's cut marked (a dot at
    the percentile/mean, or — for a reliability objective — a dot at the failure
    threshold whose height IS the score) and each policy's true score annotated.

Guiding principle (kept from the single-trace suite): the figures compute
nothing the objective functions don't. Row-2 annual metrics come from the
registered stage-(i) functions and the row-2 scores are the registered
stage-(ii) operators applied to the pooled units, each asserted equal (within
the objective epsilon) to the score the driver computed via
``build_ensemble_objective_set(...).compute(data_per_real)``.

Imported by the sibling ``make_ensemble_objective_dynamics_figures.py`` driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Make the project importable when loaded from within the outputs/ tree
# (ensemble_objective_dynamics -> supplemental -> outputs -> project root).
PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (  # noqa: E402
    MONTAGUE_DECREE_TARGET_MGD,
    TRENTON_DECREE_TARGET_MGD,
)
from src.objectives import (  # noqa: E402
    _DOWNSTREAM_GAUGES,
    _flood_over_stage_daily,
    _nyc_storage_pct_daily,
    _metric_window,
)
from src.objectives_ensemble import (  # noqa: E402
    ENSEMBLE_OBJECTIVES,
    FailureFrequencyOp,
    PooledMeanOp,
    PooledPercentileOp,
    ffmp_year_unit_slices,
)
from src.plotting.style import save_figure  # noqa: E402
from src.plotting.style import ARCH_COLORS  # noqa: E402

# ---------------------------------------------------------------------------
# Encoding constants (shared with the single-trace suite: baseline solid
# steel-blue, contrast dashed orange; CVD-safe and linestyle-redundant).
# ---------------------------------------------------------------------------

BASELINE_COLOR = ARCH_COLORS["ffmp"]      # "steelblue"
CONTRAST_COLOR = "#e0801a"                # muted orange
THRESHOLD_COLOR = "0.35"
BAND_ALPHA = 0.18                         # inter-quartile climatology band

#: Day-of-water-year (Oct 1 = 0) at which each month begins (non-leap), for ticks.
_MONTH_START_DOY = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335]
_MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep"]
#: Calendar months in water-year order (Oct .. Sep), for the flood decomposition.
_WY_MONTH_ORDER = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class EnsemblePolicy:
    """One policy's multi-realization simulation + ensemble (§2) scores.

    Attributes:
        label: Short human-readable name (e.g. "Baseline (FFMP)").
        data_per_real: List of per-realization pywrdrb results dicts (each with
            the same shape as a single-trace simulation).
        scores: Mapping annual-objective-name -> true §2 score from
            ``build_ensemble_objective_set(ACTIVE_OBJECTIVES).compute(data_per_real)``.
        color: Line colour.
        linestyle: Line style ("-" baseline, "--" contrast).
    """

    def __init__(self, label: str, data_per_real: list, scores: dict,
                 color, linestyle: str):
        self.label = label
        self.data_per_real = data_per_real
        self.scores = scores
        self.color = color
        self.linestyle = linestyle

    @property
    def n_realizations(self) -> int:
        return len(self.data_per_real)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _check(recomputed: float, true: float, name: str, eps: float,
           strict: bool) -> None:
    """Assert a figure-recomputed score matches the driver's within eps."""
    if not strict or not np.isfinite(true):
        return
    assert abs(recomputed - true) <= max(eps, 1e-9), (
        f"{name}: figure recomputed {recomputed:.6g} != score {true:.6g} "
        f"(eps={eps})"
    )


def _pooled_units(policy: "EnsemblePolicy", annual_metric) -> np.ndarray:
    """Concatenate one policy's stage-(i) annual metrics across realizations."""
    parts = [np.asarray(annual_metric(d), dtype=float).ravel()
             for d in policy.data_per_real]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _policy_proxy(policies, lw: float = 1.8) -> list:
    """One legend handle per policy (median lines carry the identity)."""
    return [Line2D([0], [0], color=p.color, linestyle=p.linestyle, lw=lw,
                   label=p.label) for p in policies]


def _doy_wy(index) -> np.ndarray:
    """Day-of-water-year (Oct 1 = 0 .. Sep 30 = 364/365) for a DatetimeIndex."""
    idx = pd.DatetimeIndex(index)
    start_year = np.where(idx.month >= 10, idx.year, idx.year - 1)
    starts = pd.to_datetime(dict(year=start_year, month=10, day=1))
    return (idx - pd.DatetimeIndex(starts)).days.astype(int)


def _seasonal_stats(policy: "EnsemblePolicy", getter, band):
    """Median + band percentiles of a daily quantity by day-of-water-year.

    Pools every realization-year of ``getter(data)`` (a daily Series) by
    day-of-water-year and returns ``(doy, median, lo, hi)`` arrays over days
    0..364 (the leap day 365 is dropped so all policies share one x-grid).
    """
    doys, vals = [], []
    for d in policy.data_per_real:
        s = getter(d)
        doys.append(_doy_wy(s.index))
        vals.append(np.asarray(s.values, dtype=float))
    df = pd.DataFrame({"doy": np.concatenate(doys), "v": np.concatenate(vals)})
    df = df[(df.doy < 365) & np.isfinite(df.v)]
    g = df.groupby("doy")["v"]
    med = g.median()
    lo = g.quantile(band[0] / 100.0)
    hi = g.quantile(band[1] / 100.0)
    return med.index.values, med.values, lo.values, hi.values


def _units_caption(policies) -> str:
    """`'pooled over R realizations × U water-years = N ensemble water years'`."""
    base = policies[0]
    r = base.n_realizations
    per = _pooled_units(base, ENSEMBLE_OBJECTIVES["nyc_storage_min_p01_pct"].annual_metric)
    n = int(per.size)
    u = n // r if r else 0
    return f"pooled over {r} realizations × {u} water-years = {n} ensemble water years"


# ---------------------------------------------------------------------------
# Row-1 seasonal-distribution panels (climatology across the water year)
# ---------------------------------------------------------------------------

def _month_axis(ax) -> None:
    """Label the day-of-water-year x-axis by month (Oct .. Sep)."""
    ax.set_xticks(_MONTH_START_DOY)
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_xlim(0, 364)
    ax.set_xlabel("Month of water year")


def _seasonal_band(ax, policies, getter, *, ylabel: str, title: str,
                   threshold: "float | None" = None,
                   threshold_label: "str | None" = None,
                   reference_getter=None, reference_label: "str | None" = None,
                   logy: bool = False, ylim=None, band=(25, 75)) -> None:
    """Seasonal climatology (median + IQR band) of a daily quantity, both policies.

    The realizations are independent random sequences, so the annual cycle — not
    calendar time — is the meaningful frame: every realization-year is pooled by
    day-of-water-year, and the policy is drawn as a median line with a shaded
    inter-quartile band. Overlapping bands read as the spread across the random
    traces; the median lines carry the policy comparison.
    """
    handles = []
    for pol in policies:
        doy, med, lo, hi = _seasonal_stats(pol, getter, band)
        ax.fill_between(doy, lo, hi, color=pol.color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(doy, med, color=pol.color, linestyle=pol.linestyle, lw=1.7)
    handles += _policy_proxy(policies)
    if reference_getter is not None:
        doy, med, _, _ = _seasonal_stats(policies[0], reference_getter, band)
        ax.plot(doy, med, color=THRESHOLD_COLOR, lw=1.1, linestyle=":")
        handles.append(Line2D([0], [0], color=THRESHOLD_COLOR, lw=1.1,
                              linestyle=":", label=reference_label))
    if threshold is not None:
        ax.axhline(threshold, color=THRESHOLD_COLOR, lw=1.1, linestyle=":")
        handles.append(Line2D([0], [0], color=THRESHOLD_COLOR, lw=1.1,
                              linestyle=":", label=threshold_label))
    handles.append(Patch(facecolor="0.6", alpha=BAND_ALPHA,
                         label=f"inter-quartile band ({band[0]}–{band[1]}%)"))
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    _month_axis(ax)
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8, ncol=2)
    ax.set_title(title, fontsize=10)


def _flood_seasonality(ax, policies) -> None:
    """Per-month contribution to the mean annual minor-flood-day objective.

    For each policy, the pooled minor-flood days are attributed to their calendar
    month and divided by the number of pooled ensemble water years, so the bars are
    *mean flood days per month per year* and their total equals the row-2
    objective (mean annual flood days). This shows the seasonal timing of the
    flood cost and the policy difference without a spurious calendar time series.
    """
    xm = np.arange(12)
    width = 0.38
    offsets = np.linspace(-width / 2, width / 2, len(policies))
    for pol, off in zip(policies, offsets):
        total = np.zeros(12)
        n_units = 0
        for d in pol.data_per_real:
            over = _flood_over_stage_daily(
                d["flood_stage"][_DOWNSTREAM_GAUGES], "minor")
            for sl in ffmp_year_unit_slices(over.index):
                n_units += 1
                seg = over.iloc[sl]
                by_month = seg.groupby(seg.index.month).sum()
                for m, c in by_month.items():
                    total[_WY_MONTH_ORDER.index(int(m))] += float(c)
        mean_by_month = total / n_units if n_units else total
        ax.bar(xm + off, mean_by_month, width=width, color=pol.color,
               alpha=0.85, label=f"{pol.label}   (mean {mean_by_month.sum():.2f} d/yr)")
    ax.set_xticks(xm)
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_xlabel("Month of water year")
    ax.set_ylabel("Mean minor-flood days / year")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.set_title("Seasonal contribution to expected annual flood days "
                 "(bars sum to the objective)", fontsize=10)


# ---------------------------------------------------------------------------
# Row-2 pooled-reduction panel (empirical CDF of the pooled ensemble water years + the
# stage-(ii) operator's cut + the true score, marked)
# ---------------------------------------------------------------------------

def _pooled_reduction_panel(ax, policies, annual_name: str, *, xlabel: str,
                            score_fmt: str, strict: bool = True) -> None:
    """ECDF of the pooled annual-unit metric with the §2 operator's cut marked.

    The marked "cut" is dispatched on the objective's stage-(ii) operator, so the
    score is always a labelled point on the figure:

      * pooled percentile — a dot where the ECDF crosses the operator's
        percentile (P99 / P01); the score is that dot's **x-value**, with a
        horizontal reference at the percentile;
      * pooled mean — a dot on the ECDF at the mean; the score is its x-value;
      * failure frequency — a dot at the failure threshold ``k`` whose **height**
        is the score (reliability = fraction of ensemble water years left of ``k``), with a
        horizontal guide at that height and the reliable region shaded.

    The objective is fetched from ``ENSEMBLE_OBJECTIVES`` (the same registry the
    driver's objective set is built from), so the operator here is identical to
    the one that produced ``policy.scores``.
    """
    obj = ENSEMBLE_OBJECTIVES[annual_name]
    op = obj.unit_operator
    is_freq = isinstance(op, FailureFrequencyOp)

    for pol in policies:
        pooled = _pooled_units(pol, obj.annual_metric)
        score = op(pooled)                       # exactly the driver's reduction
        _check(score, pol.scores.get(annual_name, np.nan), annual_name,
               1e-6, strict)
        finite = pooled[np.isfinite(pooled)]
        if finite.size == 0:
            continue
        srt = np.sort(finite)
        y = (np.arange(srt.size) + 1) / srt.size
        ax.step(srt, y, where="post", color=pol.color, linestyle=pol.linestyle,
                lw=1.5, label=f"{pol.label}   ({score_fmt.format(score)})")
        if isinstance(op, PooledPercentileOp):
            ax.axvline(score, color=pol.color, linestyle=pol.linestyle, lw=0.8,
                       alpha=0.55)
            ax.plot([score], [op.q / 100.0], marker="o", ms=6, color=pol.color,
                    zorder=6)
        elif isinstance(op, PooledMeanOp):
            f_at = float(np.mean(finite <= score))
            ax.axvline(score, color=pol.color, linestyle=pol.linestyle, lw=0.8,
                       alpha=0.55)
            ax.plot([score], [f_at], marker="o", ms=6, color=pol.color, zorder=6)
        elif is_freq:
            # The objective is the CDF HEIGHT at the failure threshold k, not a
            # point along x: mark it with a horizontal guide and a dot at (k, score).
            ax.axhline(score, color=pol.color, linestyle=pol.linestyle, lw=0.8,
                       alpha=0.55)
            ax.plot([op.k], [score], marker="o", ms=7, color=pol.color, zorder=6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction of ensemble water years")
    ax.set_ylim(0, 1.02)

    if is_freq:
        # The reliability score is the marked dot's HEIGHT (the CDF at the failure
        # threshold k); the threshold line and reliable-region shading are labelled
        # in the legend rather than with free-standing annotations.
        x0 = ax.get_xlim()[0]
        ax.axvline(op.k, color=THRESHOLD_COLOR, lw=1.1, linestyle=":")
        ax.axvspan(x0, op.k, color="#4f9648", alpha=0.08, zorder=0)
        handles, _ = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], color=THRESHOLD_COLOR, lw=1.1, linestyle=":",
                   label=f"failure threshold (k = {op.k})"),
            Patch(facecolor="#4f9648", alpha=0.18,
                  label=f"reliable years (< {op.k} failing wk)"),
        ]
        ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=8)
    else:
        if isinstance(op, PooledPercentileOp):
            ax.axhline(op.q / 100.0, color=THRESHOLD_COLOR, lw=0.9, linestyle=":")
            ax.text(ax.get_xlim()[0], op.q / 100.0, f" P{op.q:g} (objective) ",
                    va="bottom", ha="left", fontsize=8, color="0.4")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8)


# ---------------------------------------------------------------------------
# Figure A — NYC delivery (reliability frequency + deficit P99)
# ---------------------------------------------------------------------------

def plot_delivery_anatomy(policies, *, output_file=None,
                          figsize=(13, 7.6), strict: bool = True) -> Figure:
    """NYC delivery: seasonal distribution + pooled reliability & deficit reductions."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.42,
                  wspace=0.22)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_rel = fig.add_subplot(gs[1, 0])
    ax_def = fig.add_subplot(gs[1, 1])

    _seasonal_band(
        ax_ts, policies,
        getter=lambda d: _metric_window(d["ibt_diversions"]["delivery_nyc"]),
        ylabel="NYC delivery (MGD)",
        reference_getter=lambda d: _metric_window(d["ibt_demands"]["demand_nyc"]),
        reference_label="NYC demand (median)",
        title="NYC delivery — seasonal distribution across the water year")
    _pooled_reduction_panel(
        ax_rel, policies, "nyc_delivery_reliability_annual",
        xlabel="Failing weeks per ensemble water year", score_fmt="rel {:.2f}",
        strict=strict)
    ax_rel.set_title("Reliability — no-failing-week fraction (● = score)", fontsize=9)
    _pooled_reduction_panel(
        ax_def, policies, "nyc_delivery_deficit_p99_pct",
        xlabel="Within-year CVaR90 delivery deficit (% of Decree)",
        score_fmt="P99 {:.1f}%", strict=strict)
    ax_def.set_title("Deficit — pooled within-year CVaR90", fontsize=9)
    fig.suptitle(f"NYC delivery under the ensemble  ({_units_caption(policies)})",
                 fontsize=11, y=0.995)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure B — Montague flow (reliability frequency + deficit P99)
# ---------------------------------------------------------------------------

def plot_montague_anatomy(policies, *, output_file=None,
                          figsize=(13, 7.6), strict: bool = True) -> Figure:
    """Montague flow: seasonal distribution + pooled reliability & deficit reductions."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.42,
                  wspace=0.22)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_rel = fig.add_subplot(gs[1, 0])
    ax_def = fig.add_subplot(gs[1, 1])

    _seasonal_band(
        ax_ts, policies,
        getter=lambda d: _metric_window(d["major_flow"]["delMontague"]),
        ylabel="Daily flow (MGD)",
        threshold=MONTAGUE_DECREE_TARGET_MGD,
        threshold_label=f"Decree target ({MONTAGUE_DECREE_TARGET_MGD:.0f} MGD)",
        logy=True, ylim=(MONTAGUE_DECREE_TARGET_MGD * 0.35,
                         MONTAGUE_DECREE_TARGET_MGD * 8),
        title="Montague flow — seasonal distribution across the water year")
    _pooled_reduction_panel(
        ax_rel, policies, "montague_flow_reliability_annual",
        xlabel="Failing weeks per ensemble water year", score_fmt="rel {:.2f}",
        strict=strict)
    ax_rel.set_title("Reliability — no-failing-week fraction (● = score)", fontsize=9)
    _pooled_reduction_panel(
        ax_def, policies, "montague_flow_deficit_p99_pct",
        xlabel="Within-year CVaR90 flow deficit (% of Decree)",
        score_fmt="P99 {:.1f}%", strict=strict)
    ax_def.set_title("Deficit — pooled within-year CVaR90", fontsize=9)
    fig.suptitle(f"Montague flow under the ensemble  ({_units_caption(policies)})",
                 fontsize=11, y=0.995)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure C — Trenton flow (reliability frequency only)
# ---------------------------------------------------------------------------

def plot_trenton_anatomy(policies, *, output_file=None,
                         figsize=(13, 6.8), strict: bool = True) -> Figure:
    """Trenton flow: seasonal distribution + pooled reliability reduction."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.25, 1.0], hspace=0.40)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_rel = fig.add_subplot(gs[1, 0])

    _seasonal_band(
        ax_ts, policies,
        getter=lambda d: _metric_window(d["major_flow"]["delTrenton"]),
        ylabel="Daily flow (MGD)",
        threshold=TRENTON_DECREE_TARGET_MGD,
        threshold_label=f"Decree target ({TRENTON_DECREE_TARGET_MGD:.0f} MGD)",
        logy=True, ylim=(TRENTON_DECREE_TARGET_MGD * 0.35,
                         TRENTON_DECREE_TARGET_MGD * 8),
        title="Trenton flow — seasonal distribution across the water year")
    _pooled_reduction_panel(
        ax_rel, policies, "trenton_flow_reliability_annual",
        xlabel="Failing weeks per ensemble water year", score_fmt="rel {:.2f}",
        strict=strict)
    ax_rel.set_title(f"Reliability — no-failing-week fraction (● = score)  "
                     f"({_units_caption(policies)})", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure D — NYC storage (annual-minimum-storage P01)
# ---------------------------------------------------------------------------

def plot_storage_anatomy(policies, *, output_file=None,
                         figsize=(13, 6.8), strict: bool = True) -> Figure:
    """NYC storage: seasonal distribution + pooled annual-minimum-storage P01."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.0], hspace=0.40)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_p01 = fig.add_subplot(gs[1, 0])

    _seasonal_band(
        ax_ts, policies,
        getter=lambda d: _metric_window(_nyc_storage_pct_daily(d)),
        ylabel="Combined NYC storage (% capacity)",
        ylim=(0, 100),
        title="NYC storage — seasonal distribution across the water year")
    _pooled_reduction_panel(
        ax_p01, policies, "nyc_storage_min_p01_pct",
        xlabel="Annual-minimum combined NYC storage (% capacity)",
        score_fmt="P01 {:.1f}%", strict=strict)
    ax_p01.set_title(f"Vulnerability — pooled annual-minimum storage  "
                     f"({_units_caption(policies)})", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig


# ---------------------------------------------------------------------------
# Figure E — Downstream flooding (mean annual minor-flood days)
# ---------------------------------------------------------------------------

def plot_flood_anatomy(policies, *, output_file=None,
                       figsize=(13, 6.8), strict: bool = True) -> Figure:
    """Downstream flooding: seasonal flood-day decomposition + pooled mean reduction."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.1, 1.0], hspace=0.42)
    ax_seas = fig.add_subplot(gs[0, 0])
    ax_flood = fig.add_subplot(gs[1, 0])

    _flood_seasonality(ax_seas, policies)
    _pooled_reduction_panel(
        ax_flood, policies, "downstream_flood_days_annual",
        xlabel="Minor-flood days per ensemble water year",
        score_fmt="mean {:.2f} d/yr", strict=strict)
    ax_flood.set_title(f"Expected annual flood days — pooled ensemble water years  "
                       f"({_units_caption(policies)})", fontsize=9)

    if output_file is not None:
        save_figure(fig, output_file)
    return fig
