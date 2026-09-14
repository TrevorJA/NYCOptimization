"""
hazard_examples.py - The hazard metrics illustrated on example HF realizations.

A methods figure for Section 3.1.3: what the six selection axes measure, shown
on the streamflow sequences they summarize. The left panel is the Hazard
Filling search ensemble's hazard characteristics, every member in gray and a
few example realizations in color, drawn either as a 3-D scatter over a hazard
triple or as parallel axes over all six selection axes. The right column shows
each example's sequence on the scored metric window: the SSI-6 series with the
largest drought event filled, and the annual peak discharge as bars hanging
from the top with the year of the critical flood pulse filled, so the drought
metrics read off the bottom of each panel and the flood metrics off the top.

Examples are chosen by percentile targets on the selection axes: a target
names the ensemble percentile wanted on some axes, and the member nearest that
target in rank space is taken (:func:`select_examples`). The targets, the 3-D
triple and the output tree are settings of ``supplemental_config.py``
(``HEX_*``); the driver is ``scripts/supplemental/hazard_examples_figures.py``.

Data contracts: the staged HF ensemble's ``hazard_image.npz`` (pool image plus
``selected_rows``) and ``catchment_inflow_mgd.hdf5`` (the N selected
realizations, in ``selected_rows`` order). The SSI-6 fit, the flood threshold
and the normalizing mean are the pool's own, refitted on the historical record
through ``scengen.hazard_metrics.get_reference_fits``, and each example's
recomputed hazard vector is checked against its stored image row so the
sequence drawn is the one the metrics scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator

import config
from src.plotting.ensemble_composition import (DEFAULT_SCATTER_TRIPLE,
                                               SCATTER_3D_VIEW, _style_cube)

#: Symbol per selection axis (manuscript Section 3.1.3, Equations 1-5).
HAZARD_SYMBOLS: dict[str, str] = {
    "drought_magnitude":     r"$M$",
    "drought_severity":      r"$S$",
    "drought_onset_rate":    r"$R_\mathrm{on}$",
    "drought_recovery_rate": r"$R_\mathrm{rec}$",
    "flood_peak_discharge":  r"$D$",
    "flood_pulse_duration":  r"$T_P$",
}

#: Unit per selection axis ("" for the dimensionless peak discharge).
HAZARD_UNITS: dict[str, str] = {
    "drought_magnitude":     "deficit-months",
    "drought_severity":      "s.d.",
    "drought_onset_rate":    "s.d. month$^{-1}$",
    "drought_recovery_rate": "s.d. month$^{-1}$",
    "flood_peak_discharge":  "",
    "flood_pulse_duration":  "days",
}

#: Example identity, assigned in fixed order: the Okabe-Ito hues not reserved
#: for a scenario design (CVD-validated), paired with a marker shape so identity
#: never rests on hue alone.
EXAMPLE_COLORS: tuple[str, ...] = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7")
EXAMPLE_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")

#: The N ensemble members behind the examples.
ENSEMBLE_COLOR = "0.70"

#: Neutral tokens of the sequence panels: the SSI line, deficit fill and
#: non-critical flood bars. Only the scored event and pulse take the example
#: color, so the color marks what the metrics measure.
SSI_LINE = "0.15"
DEFICIT_FILL = "0.84"
BAR_FILL = "0.80"
GUIDE = "0.40"

#: Share of each sequence panel's height the tallest flood bar may hang down.
BAR_DEPTH_FRAC = 0.40

#: Type size for this figure (above the 12 pt manuscript floor; the sequence
#: panels are read at a steep reduction).
EXAMPLES_FONTSIZE: int = 16


def axis_label(metric: str) -> str:
    """Symbol plus unit for one hazard metric, e.g. ``$M$ (deficit-months)``."""
    symbol = HAZARD_SYMBOLS.get(metric, metric)
    unit = HAZARD_UNITS.get(metric, "")
    return f"{symbol} ({unit})" if unit else symbol


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ExampleSequence:
    """One example realization's scored sequence, in years since its start.

    Attributes:
        index: Row of the example in the HF ensemble (its HDF5 column).
        t: Month stamps of the SSI-6 series (years).
        ssi: SSI-6 values on the metric window.
        event: Boolean mask of the largest drought event over ``t`` (all False
            when the realization has no qualifying event).
        year_mid: Mid-points of the FFMP-years of the metric window (years).
        annual_peak: Annual maximum daily inflow over the reference mean.
        critical_year: Position in ``year_mid`` of the year holding the
            realization's maximum daily flow (the critical pulse).
        H: The realization's recomputed hazard vector (candidate axes).
    """

    index: int
    t: np.ndarray
    ssi: np.ndarray
    event: np.ndarray
    year_mid: np.ndarray
    annual_peak: np.ndarray
    critical_year: int
    H: np.ndarray


def hf_ensemble_slug(draw: int | None = None) -> str:
    """Staged slug of the HF search ensemble for ``draw`` (default: the env draw)."""
    from src.scenario_designs import SCENARIO_DESIGNS

    if draw is None:
        draw = config.SCENARIO_ENSEMBLE_DRAW
    return SCENARIO_DESIGNS["hazard_filling_stationary"].search_ensemble_slug(draw)


def required_files(slug: str | None = None) -> list[Path]:
    """The staged files this figure needs."""
    d = config.STAGED_ENSEMBLE_DIR / (slug or hf_ensemble_slug())
    return [d / "hazard_image.npz", d / "catchment_inflow_mgd.hdf5"]


def validate_targets(targets: list[dict[str, float]]) -> list[dict[str, float]]:
    """Check example targets: selection axes only, percentiles in [0, 1], and no
    more targets than there are example identities.

    Args:
        targets: ``{axis: percentile}`` per example.

    Returns:
        ``targets`` unchanged.
    """
    if len(targets) > len(EXAMPLE_COLORS):
        raise ValueError(
            f"at most {len(EXAMPLE_COLORS)} examples carry a distinct identity; "
            f"got {len(targets)} targets")
    for t in targets:
        bad = [a for a in t if a not in config.HAZARD_SELECTION_AXES]
        if bad:
            raise ValueError(f"example target names non-selection axes {bad}")
        if any(not 0.0 <= p <= 1.0 for p in t.values()):
            raise ValueError(f"example percentiles must lie in [0, 1]: {t}")
    return targets


def ensemble_percentile(H: np.ndarray, axes: list[str], metric: str) -> np.ndarray:
    """Mid-rank ensemble percentile of every member on ``metric`` (ties averaged)."""
    from scipy.stats import rankdata

    return (rankdata(H[:, axes.index(metric)]) - 0.5) / H.shape[0]


def select_examples(H: np.ndarray, axes: list[str],
                    targets: list[dict[str, float]]) -> list[int]:
    """The ensemble member nearest each percentile target, in rank space.

    Every axis named by a target is converted to its ensemble percentile
    (mid-ranks, ties averaged, so a zero-inflated drought axis places its
    no-event atom at one shared percentile); the member minimizing the
    Euclidean distance to the target over the axes it names is chosen, each
    member at most once.

    Args:
        H: ``(n, m)`` ensemble hazard image.
        axes: Axis names aligned with the columns of ``H``.
        targets: Percentile targets, ``{axis: percentile}`` each.

    Returns:
        Row indices into ``H``, one per target, in target order.
    """
    n = H.shape[0]
    pct = {a: ensemble_percentile(H, axes, a) for t in targets for a in t}
    chosen: list[int] = []
    for t in targets:
        d = np.zeros(n)
        for a, p in t.items():
            d += (pct[a] - p) ** 2
        d[chosen] = np.inf
        chosen.append(int(np.argmin(d)))
    return chosen


def sequence_of(daily: pd.Series, index: int, reference: tuple, n_years: int) -> ExampleSequence:
    """Score one realization exactly as the pool did and keep its sequence.

    Mirrors ``src.ensemble_generation._hazard_block``: the trailing partial
    FFMP-year is cut from the daily and monthly inputs, the wet axes exclude
    the leading ``config.METRIC_EXCLUSION_MONTHS`` by date, and the SSI-6
    series keeps its leading months as accumulation input before the same
    cut is applied to it.

    Args:
        daily: Daily aggregate NYC inflow of the realization (DatetimeIndex
            starting on the realization epoch).
        index: The realization's row in the HF ensemble.
        reference: ``(reference_monthly, reference_daily)`` historical arrays.
        n_years: Realization length L.

    Returns:
        The realization's sequence and recomputed hazard vector.
    """
    from synhydro.droughts.ssi import get_drought_metrics

    from scengen.hazard_metrics import (DRY_EVENT_METRICS,
                                        compute_candidate_hazard_image,
                                        critical_event_descriptors,
                                        flows_to_series, get_reference_fits)

    excl = config.METRIC_EXCLUSION_MONTHS
    idx = pd.DatetimeIndex(daily.index)
    t0 = idx[0]
    metric_start = t0 + pd.DateOffset(months=excl)
    metric_end = metric_start + pd.DateOffset(years=n_years - 1)
    daily = daily.loc[idx < metric_end].astype(float)
    idx = pd.DatetimeIndex(daily.index)
    monthly = daily.resample("MS").mean()
    cut = int((idx < metric_start).sum())

    ref_m, ref_d = reference
    H_row, _ = compute_candidate_hazard_image(
        monthly.to_numpy()[None, :], daily.to_numpy()[None, :], ref_m, ref_d,
        wet_exclusion_days=cut,
    )
    dry_calc, _threshold, ref_mean = get_reference_fits(ref_m, ref_d)

    # SynHydro returns only the months on which SSI-6 is defined, so the
    # output is stamped from the tail of the monthly index; the leading cut is
    # the one ``compute_candidate_hazard_image`` applies, and the descriptor
    # check below fails loudly if the two ever drift apart.
    ssi = dry_calc.transform(flows_to_series(monthly.to_numpy(), freq="MS"))
    ssi = pd.Series(ssi.to_numpy(dtype=float), index=monthly.index[-len(ssi):])
    ssi = ssi.iloc[excl:]
    dry = critical_event_descriptors(ssi)
    dry_row = [dry[m.removeprefix("drought_")] for m in DRY_EVENT_METRICS]
    if not np.allclose(dry_row, H_row[0, :len(DRY_EVENT_METRICS)], rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"the drawn SSI-6 series scores {dry_row} but the hazard image scored "
            f"{H_row[0, :len(DRY_EVENT_METRICS)]}: the series cut no longer mirrors "
            f"scengen.hazard_metrics.compute_candidate_hazard_image.")
    event = np.zeros(len(ssi), dtype=bool)
    events = get_drought_metrics(ssi, end_drought_threshold_months=3)
    if len(events):
        largest = events.loc[events["magnitude"].astype(float).abs().idxmax()]
        event = (ssi.index >= largest["start"]) & (ssi.index <= largest["end"])

    scored = daily.loc[idx >= metric_start]
    ffmp_year = scored.index.year - (scored.index.month < 6)
    annual_max = scored.groupby(ffmp_year).max()
    year_start = pd.DatetimeIndex([pd.Timestamp(y, 6, 1) for y in annual_max.index])

    def years(stamps) -> np.ndarray:
        return (pd.DatetimeIndex(stamps) - t0).days.to_numpy() / 365.25

    return ExampleSequence(
        index=index,
        t=years(ssi.index),
        ssi=ssi.to_numpy(),
        event=event,
        year_mid=years(year_start) + 0.5,
        annual_peak=annual_max.to_numpy(dtype=float) / ref_mean,
        critical_year=int(np.argmax(annual_max.to_numpy())),
        H=H_row[0],
    )


def load_examples(slug: str, targets: list[dict[str, float]]) -> dict:
    """The HF ensemble's hazard image, the chosen examples and their sequences.

    Args:
        slug: Staged HF ensemble slug.
        targets: Percentile targets (see :func:`select_examples`).

    Returns:
        Dict with ``H`` ``(N, m_all)``, ``axes``, ``chosen`` (rows of ``H``),
        ``global_ids`` (pool ids of the chosen rows) and ``sequences`` (one
        :class:`ExampleSequence` per target).

    Raises:
        ValueError: If an example's recomputed hazard vector disagrees with
            its stored image row (the drawn sequence would not be the scored one).
    """
    from synhydro.core.ensemble import Ensemble

    from scengen.diagnostics import load_hazard_image
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
    from scripts.main.compute_etest_hazard_image import _reference_series
    from scripts.main.compute_historic_hazard_windows import FLOWTYPE

    validate_targets(targets)
    d = config.STAGED_ENSEMBLE_DIR / slug
    haz = load_hazard_image(d / "hazard_image.npz")
    rows = haz["selected_rows"]
    H = haz["H"][rows] if len(rows) else haz["H"]
    axes = list(haz["hazard_axes"])
    chosen = select_examples(H, axes, targets)

    meta = json.loads((d / "_meta.json").read_text())
    n_years = int(meta.get("realization_years") or config.SCENARIO_YEARS)
    global_ids = [int(meta["global_realization_ids"][i]) for i in chosen]

    reference = _reference_series(FLOWTYPE)
    frames = Ensemble.from_hdf5(
        str(d / "catchment_inflow_mgd.hdf5"), realization_subset=chosen
    ).data_by_realization
    sequences = []
    for k, i in enumerate(chosen):
        agg = frames[k].loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
        seq = sequence_of(agg, i, reference, n_years)
        if not np.allclose(seq.H, H[i], rtol=1e-4, atol=1e-6):
            raise ValueError(
                f"example row {i} of '{slug}' rescored to {seq.H} but the stored "
                f"hazard image holds {H[i]}; the staged daily traces and the "
                f"hazard image disagree.")
        sequences.append(seq)
    return {"H": H, "axes": axes, "chosen": chosen, "global_ids": global_ids,
            "sequences": sequences}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _limits(v: np.ndarray) -> tuple[float, float]:
    """Padded display limits of one axis, never below zero."""
    lo, hi = float(v.min()), float(v.max())
    pad = 0.04 * (hi - lo or 1.0)
    return max(0.0, lo - pad), hi + pad


def draw_scatter_3d(ax, H: np.ndarray, axes: list[str], triple: tuple[str, str, str],
                    sequences: list[ExampleSequence]) -> None:
    """The ensemble in a hazard triple, the examples as large colored markers.

    Each example carries a drop line to the floor and a floor shadow, which
    fix its position in the cube without a per-member clutter of lines.

    Args:
        ax: Target axes, created with ``projection="3d"``.
        H: ``(N, m_all)`` ensemble hazard image.
        axes: Axis names aligned with the columns of ``H``.
        triple: ``(x, y, z)`` hazard metrics (y is the depth axis).
        sequences: The examples, in identity order.
    """
    cols = [H[:, axes.index(a)] for a in triple]
    lims = [_limits(c) for c in cols]
    elev, azim = SCATTER_3D_VIEW
    ax.view_init(elev=elev, azim=azim)
    ax.computed_zorder = False
    _style_cube(ax, [axis_label(a) for a in triple], lims)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))

    x, y, z = cols
    ax.scatter(x, y, z, s=18, c=ENSEMBLE_COLOR, alpha=0.8, linewidths=0,
               depthshade=False, zorder=2)
    floor = lims[2][0]
    for k, seq in enumerate(sequences):
        color, marker = EXAMPLE_COLORS[k], EXAMPLE_MARKERS[k]
        xi, yi, zi = x[seq.index], y[seq.index], z[seq.index]
        ax.plot([xi, xi], [yi, yi], [floor, zi], color=color, lw=1.4, alpha=0.9,
                zorder=4)
        ax.scatter([xi], [yi], [floor], s=60, c=color, marker=marker, alpha=0.45,
                   linewidths=0, depthshade=False, zorder=3)
        ax.scatter([xi], [yi], [zi], s=190, c=color, marker=marker,
                   edgecolors="white", linewidths=1.4, depthshade=False, zorder=6)


def _fmt(v: float) -> str:
    """Axis-end value to three significant figures."""
    return f"{v:.3g}"


def draw_parallel_axes(ax, H: np.ndarray, axes: list[str], plot_axes: list[str],
                       sequences: list[ExampleSequence]) -> None:
    """The ensemble on parallel axes, every member gray, the examples in color.

    Each axis runs from the ensemble minimum (bottom) to maximum (top) of
    its metric, so up is more hazardous on every axis; the raw end values and
    the metric's symbol and unit are the only annotation.

    Args:
        ax: Target axes (its frame is hidden).
        H: ``(N, m_all)`` ensemble hazard image.
        axes: Axis names aligned with the columns of ``H``.
        plot_axes: Metrics drawn, left to right.
        sequences: The examples, in identity order.
    """
    fs = plt.rcParams["font.size"]
    V = H[:, [axes.index(a) for a in plot_axes]]
    lo, hi = V.min(axis=0), V.max(axis=0)
    U = (V - lo) / np.where(hi > lo, hi - lo, 1.0)
    m = len(plot_axes)
    xs = np.arange(m)

    for row in U:
        ax.plot(xs, row, color=ENSEMBLE_COLOR, lw=0.8, alpha=0.5, zorder=1)
    for j in xs:
        ax.plot([j, j], [0, 1], color="0.25", lw=1.3, zorder=2)
        ax.text(j, -0.03, _fmt(lo[j]), ha="center", va="top")
        ax.text(j, 1.03, _fmt(hi[j]), ha="center", va="bottom")
        ax.text(j, 1.12, HAZARD_SYMBOLS.get(plot_axes[j], plot_axes[j]),
                ha="center", va="bottom", fontsize=fs + 4)
        unit = HAZARD_UNITS.get(plot_axes[j], "")
        if unit:
            # Two-word units break at the space so neighbouring axes never touch.
            ax.text(j, -0.11, "(" + unit.replace(" ", "\n") + ")", ha="center",
                    va="top", fontsize=fs - 4, color="0.35", linespacing=1.1)
    for k, seq in enumerate(sequences):
        color, marker = EXAMPLE_COLORS[k], EXAMPLE_MARKERS[k]
        ax.plot(xs, U[seq.index], color=color, lw=3.2, zorder=4,
                path_effects=[withStroke(linewidth=5.6, foreground="white")])
        ax.plot(xs, U[seq.index], ls="none", marker=marker, ms=11, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.set_xlim(-0.45, m - 0.55)
    ax.set_ylim(-0.22, 1.28)
    ax.axis("off")


def draw_sequence_panel(ax, seq: ExampleSequence, color: str, marker: str, *,
                        ssi_lim: float, d_max: float, bottom: bool) -> None:
    """One example's SSI-6 series (left axis) and annual peak discharge (right).

    The flood bars hang from the top edge on an inverted right axis, the
    hyetograph convention, so the two tails occupy opposite halves of the
    panel: deficits fill downward from zero and the largest drought event
    takes the example color; the bar of the year holding the critical pulse
    takes it likewise. A dotted guide marks the drought retention threshold
    (SSI-6 = -1).

    Args:
        ax: Target axes.
        seq: The example's sequence.
        color: Example color.
        marker: Example marker (drawn as the panel's identity badge above its
            top-left corner).
        ssi_lim: Symmetric SSI-6 limit shared across rows.
        d_max: Largest annual peak discharge across rows (bar-depth scale).
        bottom: Whether this is the bottom row (x tick labels and label).
    """
    t, ssi = seq.t, seq.ssi
    ax.axhline(0.0, color=GUIDE, lw=0.9, zorder=2)
    ax.axhline(-1.0, color=GUIDE, lw=0.9, ls=(0, (1, 3)), zorder=2)
    ax.fill_between(t, ssi, 0.0, where=ssi < 0, color=DEFICIT_FILL, lw=0,
                    interpolate=True, zorder=1)
    if seq.event.any():
        ax.fill_between(t, ssi, 0.0, where=seq.event & (ssi < 0), color=color,
                        lw=0, interpolate=True, zorder=3)
    ax.plot(t, ssi, color=SSI_LINE, lw=1.6, zorder=4)
    ax.set_ylim(-ssi_lim, ssi_lim)
    ax.set_yticks([-ssi_lim, 0, ssi_lim])
    ax.set_ylabel("SSI-6")

    ax2 = ax.twinx()
    ax2.bar(seq.year_mid, seq.annual_peak, width=0.8, color=BAR_FILL, lw=0, zorder=1)
    c = seq.critical_year
    ax2.bar([seq.year_mid[c]], [seq.annual_peak[c]], width=0.8, color=color, lw=0,
            zorder=3)
    ax2.set_ylim(d_max / BAR_DEPTH_FRAC, 0.0)
    ax2.set_yticks([0.0, 10.0 * np.ceil(d_max / 10.0)])
    ax2.set_ylabel(HAZARD_SYMBOLS["flood_peak_discharge"], rotation=0,
                   ha="left", va="center", labelpad=10)
    ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # The bars live on the twin axes, which are painted after the host; put
    # the SSI series back on top and let the bars show through the host.
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    t_lo, t_hi = seq.year_mid[0] - 0.5, seq.year_mid[-1] + 0.5
    ax.set_xlim(t_lo, t_hi)
    ax.set_xticks(np.arange(np.ceil(t_lo), np.floor(t_hi) + 1))
    ax.tick_params(labelbottom=bottom)
    if bottom:
        ax.set_xlabel("Year")
    ax.plot([0.0], [1.09], transform=ax.transAxes, ls="none", marker=marker,
            ms=13, color=color, markeredgecolor="white", markeredgewidth=1.5,
            clip_on=False, zorder=10)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

@plt.rc_context({
    "font.size":       EXAMPLES_FONTSIZE,
    "axes.labelsize":  EXAMPLES_FONTSIZE,
    "xtick.labelsize": EXAMPLES_FONTSIZE,
    "ytick.labelsize": EXAMPLES_FONTSIZE,
    "axes.labelpad":   3.0,
})
def build_hazard_examples_figure(
    H: np.ndarray, axes: list[str], sequences: list[ExampleSequence], *,
    left: str = "3d",
    triple: tuple[str, str, str] = DEFAULT_SCATTER_TRIPLE,
    parallel_axes: list[str] | None = None,
):
    """Build the figure: the hazard panel (left) and one sequence row per example.

    Args:
        H: ``(N, m_all)`` HF ensemble hazard image.
        axes: Axis names aligned with the columns of ``H``.
        sequences: The examples, in identity order (one row each).
        left: ``"3d"`` (scatter over ``triple``) or ``"parallel"`` (parallel
            axes over ``parallel_axes``).
        triple: ``(x, y, z)`` metrics of the 3-D panel.
        parallel_axes: Metrics of the parallel-axes panel (default: the
            campaign selection set).

    Returns:
        The matplotlib figure.
    """
    if left not in ("3d", "parallel"):
        raise ValueError(f"left must be '3d' or 'parallel', got {left!r}")
    k = len(sequences)
    three_d = left == "3d"
    fig = plt.figure(figsize=(14.0, max(7.0, 2.15 * k + 1.0)))
    gs = fig.add_gridspec(
        k, 2, width_ratios=(1.15, 1.0) if three_d else (1.3, 1.0),
        left=0.115 if three_d else 0.05, right=0.94, top=0.955, bottom=0.09,
        hspace=0.32, wspace=0.10 if three_d else 0.16,
    )
    ax_left = fig.add_subplot(gs[:, 0], projection="3d" if three_d else None)
    if three_d:
        draw_scatter_3d(ax_left, H, axes, triple, sequences)
        ax_left.set_box_aspect((1.0, 1.0, 1.1), zoom=1.25)
    else:
        draw_parallel_axes(ax_left, H, axes,
                           list(parallel_axes or config.HAZARD_SELECTION_AXES),
                           sequences)

    ssi_lim = float(np.ceil(max(3.0, max(np.abs(s.ssi).max() for s in sequences)) * 2) / 2)
    d_max = max(float(s.annual_peak.max()) for s in sequences)
    for i, seq in enumerate(sequences):
        ax = fig.add_subplot(gs[i, 1])
        draw_sequence_panel(
            ax, seq, EXAMPLE_COLORS[i], EXAMPLE_MARKERS[i], ssi_lim=ssi_lim,
            d_max=d_max, bottom=(i == k - 1),
        )
    return fig
