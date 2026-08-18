"""
ensemble_composition.py - Manuscript figure 4: realized composition of the search ensembles.

Section 4.1's evidence base: the realized hazard-space composition of the two
matched search ensembles (PS and HF) against the candidate pool they share a
population law with, and against the historical record. Two scatter panels show
each ensemble over the pool's joint density in the drought magnitude-intensity
plane; six marginal panels show every campaign selection axis, with the pool as
a filled density, the two ensembles as kernel-density curves, and the historical
record's disjoint 10-yr windows as tick markers rather than a distribution
(seven windows cannot support one).

Data contracts:
    * Pool / ensemble layers: ``hazard_image.npz`` in each staged directory
      (``scengen.diagnostics.save_hazard_image`` format; a non-empty
      ``selected_rows`` marks the ensemble within a pool image, an empty one
      means every row IS the ensemble).
    * Historical layer: the cached window image of
      ``scripts/main/compute_historic_hazard_windows.py`` (computed on first
      use).

Configuration is via environment variables (no CLI value flags):

    NYCOPT_COMPOSITION_POOL_SLUG  staged candidate-pool slug
                                  (default statpool_10yr_n1000000_d0)
    NYCOPT_ENSEMBLE_DRAW          ensemble draw whose realized composition is
                                  shown (default 0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
from src.plotting import style

#: Manuscript display labels for the six selection axes (Section 3.1.3 symbols).
HAZARD_METRIC_LABELS: dict[str, str] = {
    "drought_magnitude":      "Drought magnitude, $M$ (SSI-months)",
    "drought_severity":       "Drought severity, $S$ (s.d.)",
    "drought_onset_rate":     "Onset rate, $R_\\mathrm{on}$ (s.d. month$^{-1}$)",
    "drought_recovery_rate":  "Recovery rate, $R_\\mathrm{rec}$ (s.d. month$^{-1}$)",
    "flood_peak_discharge":   "Peak discharge, $D$ (-)",
    "flood_pulse_duration":   "Pulse duration, $T_P$ (days)",
}

#: Non-design grayscale tokens: the pool density fill/field and the historical
#: tick markers. The pool is a magnitude job (light grays); the historical
#: reference must stay separable from it, so its marks are near-black rather
#: than the DESIGN_STYLE historic gray, which vanishes on the pool fill.
POOL_FILL = "0.88"
POOL_EDGE = "0.72"
HISTORIC_MARK = "0.15"

#: Default 2-D scatter pair ``(x, y)``: the drought severity-magnitude plane.
DEFAULT_SCATTER_PAIR: tuple[str, str] = ("drought_severity", "drought_magnitude")

DEFAULT_POOL_SLUG = "statpool_10yr_n1000000_d0"

#: Display cap percentile for the marginal panels: the pool's extreme right
#: tail (flood peak max ~6x its p99) would otherwise crush every distribution
#: into the left tenth of the panel. Never applied to the data, only to limits.
MARGINAL_CAP_PCT = 99.9


@dataclass
class CompositionLayer:
    """One realized search ensemble for the composition figure.

    Attributes:
        name: Legend label.
        H: ``(n, m_all)`` hazard coordinates of the ensemble members.
        axes: Axis names aligned with columns of ``H``.
        color: Entity-stable hex color (``style.DESIGN_STYLE``).
    """

    name: str
    H: np.ndarray
    axes: list[str]
    color: str


def _col(H: np.ndarray, axes: list[str], name: str) -> np.ndarray:
    return H[:, list(axes).index(name)]


def load_ensemble_layer(slug: str, name: str, color: str) -> CompositionLayer:
    """Load a staged ensemble's hazard coordinates as a figure layer.

    Args:
        slug: Staged ensemble directory name.
        name: Legend label for the layer.
        color: Entity-stable color.

    Returns:
        The layer, resolving the ``selected_rows`` convention (non-empty =
        the ensemble within a pool image; empty = every row is the ensemble).
    """
    from scengen.diagnostics import load_hazard_image

    from src.ensembles import staged_ensemble_dir

    haz = load_hazard_image(staged_ensemble_dir(slug) / "hazard_image.npz")
    rows = haz["selected_rows"]
    H = haz["H"][rows] if len(rows) else haz["H"]
    return CompositionLayer(name=name, H=H, axes=list(haz["hazard_axes"]), color=color)


def _kde_density(
    values: np.ndarray, grid: np.ndarray,
    reflect_lo: float | None, reflect_hi: float | None = None,
) -> np.ndarray:
    """Gaussian KDE on ``grid``, boundary-corrected by reflection where needed.

    Args:
        values: Sample values.
        grid: Evaluation grid.
        reflect_lo: Lower support boundary; reflecting the sample about it
            removes the KDE mass leak below a hard zero (all six metrics are
            non-negative and several concentrate near zero).
        reflect_hi: Upper support boundary (the SSI clip puts an atom at a
            metric's hard maximum); None for unbounded upper support.

    Returns:
        Density values on ``grid``.
    """
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float)
    kde = gaussian_kde(values)
    dens = kde(grid)
    if reflect_lo is not None:
        dens = dens + kde(2.0 * reflect_lo - grid)
        dens[grid < reflect_lo] = 0.0
    if reflect_hi is not None:
        dens = dens + kde(2.0 * reflect_hi - grid)
        dens[grid > reflect_hi] = 0.0
    return dens


def _clip_atom(values: np.ndarray, frac: float = 0.02) -> float | None:
    """The sample maximum, when an atom (> ``frac`` of the sample) sits on it."""
    values = np.asarray(values, dtype=float)
    vmax = float(values.max())
    return vmax if np.mean(values >= vmax - 1e-9) > frac else None


def _pool_cmap() -> ListedColormap:
    """Truncated Greys so even single-count pool cells stay visible on white."""
    base = plt.get_cmap("Greys")
    return ListedColormap(base(np.linspace(0.18, 0.78, 256)))


def draw_scatter_panel(
    ax,
    pool_H: np.ndarray, pool_axes: list[str],
    layer: CompositionLayer,
    ax_x: str, ax_y: str,
    *,
    xlim: tuple[float, float], ylim: tuple[float, float],
    historic_H: np.ndarray | None = None, historic_axes: list[str] | None = None,
    grid_bins: int = 140,
    scatter_size: float = 34.0,
) -> None:
    """One ensemble's members over the pool's joint density in a hazard pair.

    Args:
        ax: Target axes.
        pool_H: Candidate-pool hazard image.
        pool_axes: Pool axis names.
        layer: The ensemble scatter layer.
        ax_x: Hazard metric on x.
        ax_y: Hazard metric on y.
        xlim: Panel x limits (also the density-field range).
        ylim: Panel y limits (also the density-field range).
        historic_H: Optional historical window coordinates (marker overlay).
        historic_axes: Axis names for ``historic_H``.
        grid_bins: 2-D histogram resolution for the pool density field.
        scatter_size: Ensemble marker size.
    """
    hist, xe, ye = np.histogram2d(
        _col(pool_H, pool_axes, ax_x), _col(pool_H, pool_axes, ax_y),
        bins=grid_bins, range=[xlim, ylim],
    )
    ax.pcolormesh(
        xe, ye, np.ma.masked_equal(hist.T, 0), cmap=_pool_cmap(),
        norm=LogNorm(vmin=1, vmax=max(hist.max(), 2)), rasterized=True, zorder=1,
    )
    if historic_H is not None:
        ax.scatter(
            _col(historic_H, historic_axes, ax_x),
            _col(historic_H, historic_axes, ax_y),
            marker="X", s=110, facecolor=HISTORIC_MARK, edgecolor="white",
            linewidths=0.9, zorder=3,
        )
    ax.scatter(
        _col(layer.H, layer.axes, ax_x), _col(layer.H, layer.axes, ax_y),
        s=scatter_size, color=layer.color, edgecolors="white", linewidths=0.5,
        alpha=0.95, zorder=4,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def draw_marginal_panel(
    ax,
    metric: str,
    pool_v: np.ndarray,
    layers: list[CompositionLayer],
    historic_v: np.ndarray | None,
    *,
    xlim: tuple[float, float],
    show_pool: bool = True,
    pool_bins: int = 90,
) -> None:
    """One metric's marginal composition: pool fill, ensemble KDEs, historical ticks.

    Args:
        ax: Target axes.
        metric: Hazard metric name (column of every layer).
        pool_v: Pool values of the metric.
        layers: Ensemble layers (KDE curves, in draw order).
        historic_v: Historical window values (tick markers), or None.
        xlim: Panel x limits; densities are evaluated on this support.
        show_pool: Draw the pool as a filled histogram density.
        pool_bins: Pool histogram resolution.
    """
    grid = np.linspace(*xlim, 400)
    reflect_lo = 0.0 if xlim[0] <= 0.05 * (xlim[1] - xlim[0]) else None
    if show_pool:
        integer_valued = np.all(pool_v == np.round(pool_v))
        if integer_valued:
            edges = np.arange(np.floor(xlim[0]) - 0.5, np.ceil(xlim[1]) + 1.5)
            dens, edges = np.histogram(pool_v, bins=edges, density=True)
        else:
            dens, edges = np.histogram(pool_v, bins=pool_bins, range=xlim, density=True)
        ax.stairs(dens, edges, fill=True, color=POOL_FILL, zorder=1)
        ax.stairs(dens, edges, color=POOL_EDGE, lw=0.8, zorder=2)
    for layer in layers:
        v = _col(layer.H, layer.axes, metric)
        ax.plot(grid, _kde_density(v, grid, reflect_lo, _clip_atom(v)),
                color=layer.color, lw=2.0, zorder=4)
    if historic_v is not None:
        ax.plot(historic_v, np.zeros_like(historic_v), ls="", marker="X",
                markersize=8, markerfacecolor=HISTORIC_MARK,
                markeredgecolor="white", markeredgewidth=0.7,
                clip_on=False, zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(HAZARD_METRIC_LABELS.get(metric, metric))


def _marginal_limits(
    metric: str,
    pool_v: np.ndarray,
    layers: list[CompositionLayer],
    historic_v: np.ndarray | None,
    range_layers: list[CompositionLayer] | None = None,
) -> tuple[float, float]:
    """Display limits for one metric's marginal panel.

    The upper limit is the span of ``range_layers`` (plus the historical
    markers, which must never fall out of frame) when given -- the campaign
    default trims every panel to the HF-sampled range so the pool's extreme
    tail (flood peak max ~6x its p99) cannot crush the distributions. With
    ``range_layers=None`` the cap falls back to the pool's
    ``MARGINAL_CAP_PCT`` percentile joined with every layer's span.
    """
    cap = range_layers or layers
    hi = max(
        *(float(_col(l.H, l.axes, metric).max()) for l in cap),
        float(historic_v.max()) if historic_v is not None else -np.inf,
    )
    if range_layers is None:
        hi = max(hi, float(np.percentile(pool_v, MARGINAL_CAP_PCT)))
    lo = min(
        float(pool_v.min()),
        *(float(_col(l.H, l.axes, metric).min()) for l in layers),
        float(historic_v.min()) if historic_v is not None else np.inf,
    )
    pad = 0.03 * (hi - lo or 1.0)
    return max(0.0, lo - pad), hi + pad


def build_composition_figure(
    pool_H: np.ndarray, pool_axes: list[str],
    layers: list[CompositionLayer],
    historic_H: np.ndarray | None, historic_axes: list[str] | None,
    *,
    scatter_pair: tuple[str, str] = DEFAULT_SCATTER_PAIR,
    marginal_axes: list[str] | None = None,
    show_pool_marginal: bool = True,
    show_historic_scatter: bool = True,
    range_layers: list[CompositionLayer] | None = None,
    pool_label: str = "Candidate pool",
    historic_label: str = "Historical record (10-yr windows)",
):
    """Build the realized-composition figure (2 scatter + 6 marginal panels).

    Args:
        pool_H: ``(P, m_all)`` candidate-pool hazard image.
        pool_axes: Pool axis names.
        layers: One layer per matched ensemble (a scatter panel each, and a
            KDE curve in every marginal panel), in display order.
        historic_H: Historical window coordinates, or None to omit the layer.
        historic_axes: Axis names for ``historic_H``.
        scatter_pair: ``(x, y)`` hazard metrics of the scatter panels.
        marginal_axes: Metrics of the marginal panels (default: the campaign
            selection set, ``config.HAZARD_SELECTION_AXES``).
        show_pool_marginal: Include the pool density in the marginal panels.
        show_historic_scatter: Overlay the historical windows on the scatter
            panels.
        range_layers: Layers whose sampled span sets every panel's upper
            display limit (historical markers always stay in frame). None
            shows the pool's full span in the scatter panels and its
            ``MARGINAL_CAP_PCT`` cap in the marginals.
        pool_label: Legend label for the pool layers.
        historic_label: Legend label for the historical markers.

    Returns:
        The matplotlib figure.
    """
    marginal_axes = list(marginal_axes or config.HAZARD_SELECTION_AXES)
    n_marg = len(marginal_axes)
    ax_x, ax_y = scatter_pair

    fig = plt.figure(figsize=(12.0, 1.55 * n_marg))
    gs = fig.add_gridspec(
        n_marg, 2, width_ratios=(1.08, 1.0),
        left=0.06, right=0.985, top=0.965, bottom=0.14,
        hspace=0.95, wspace=0.16,
    )
    half = n_marg // 2
    scatter_axes = [
        fig.add_subplot(gs[:half, 0]),
        fig.add_subplot(gs[half:, 0]),
    ]
    marg_axes = [fig.add_subplot(gs[i, 1]) for i in range(n_marg)]

    # Shared scatter limits. With range_layers set, the upper limit is that
    # span (the campaign default trims to the HF-sampled range); otherwise the
    # pool's full reachable span. The lower limit always includes the pool so
    # the no-event atom at the origin stays in frame.
    hist_pair = None
    if show_historic_scatter and historic_H is not None:
        hist_pair = historic_H
    lims = {}
    for a in (ax_x, ax_y):
        hi_src = [_col(l.H, l.axes, a) for l in (range_layers or [])] or [
            _col(pool_H, pool_axes, a)]
        if hist_pair is not None:
            hi_src.append(_col(hist_pair, historic_axes, a))
        hi = max(float(v.max()) for v in hi_src)
        lo = min(float(_col(pool_H, pool_axes, a).min()),
                 *(float(v.min()) for v in hi_src))
        pad = 0.03 * (hi - lo or 1.0)
        lims[a] = (max(0.0, lo - pad), hi + pad)

    letters = iter("abcdefghijklmn")
    for ax, layer in zip(scatter_axes, layers):
        draw_scatter_panel(
            ax, pool_H, pool_axes, layer, ax_x, ax_y,
            xlim=lims[ax_x], ylim=lims[ax_y],
            historic_H=hist_pair, historic_axes=historic_axes,
        )
        ax.set_title(f"({next(letters)}) {layer.name}", loc="left")
        ax.set_ylabel(HAZARD_METRIC_LABELS.get(ax_y, ax_y))
    scatter_axes[0].tick_params(labelbottom=False)
    scatter_axes[1].set_xlabel(HAZARD_METRIC_LABELS.get(ax_x, ax_x))

    for ax, metric in zip(marg_axes, marginal_axes):
        pool_v = _col(pool_H, pool_axes, metric)
        hist_v = (_col(historic_H, historic_axes, metric)
                  if historic_H is not None else None)
        xlim = _marginal_limits(metric, pool_v, layers, hist_v, range_layers)
        draw_marginal_panel(
            ax, metric, pool_v, layers, hist_v,
            xlim=xlim, show_pool=show_pool_marginal,
        )
        ax.text(0.985, 0.86, f"({next(letters)})", transform=ax.transAxes,
                ha="right", va="top")
    marg_axes[n_marg // 2].set_ylabel("Density", labelpad=8.0)

    handles = []
    if show_pool_marginal:
        handles.append(Patch(facecolor=POOL_FILL, edgecolor=POOL_EDGE,
                             label=pool_label))
    handles += [
        Line2D([], [], color=layer.color, lw=2.0, marker="o", markersize=6,
               markeredgecolor="white", label=layer.name)
        for layer in layers
    ]
    if historic_H is not None:
        handles.append(Line2D([], [], color=HISTORIC_MARK, lw=0, marker="X",
                              markersize=8, markeredgecolor="white",
                              label=historic_label))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    return fig


def required_hazard_images(pool_slug: str | None = None) -> list[Path]:
    """The staged hazard-image files figure 4 needs (for the render-pass gate)."""
    from src.scenario_designs import SCENARIO_DESIGNS

    draw = config.SCENARIO_ENSEMBLE_DRAW
    pool_slug = pool_slug or os.environ.get(
        "NYCOPT_COMPOSITION_POOL_SLUG", DEFAULT_POOL_SLUG)
    slugs = [
        pool_slug,
        SCENARIO_DESIGNS["fixed_probabilistic"].search_ensemble_slug(draw),
        SCENARIO_DESIGNS["hazard_filling_stationary"].search_ensemble_slug(draw),
    ]
    return [config.STAGED_ENSEMBLE_DIR / s / "hazard_image.npz" for s in slugs]


def fig_ensemble_composition(ctx, out_stub: Path, table_dir: Path) -> list[Path]:
    """Figure 4: realized hazard-space composition of the search ensembles.

    Loads the staged candidate pool and both matched search ensembles (draw
    ``NYCOPT_ENSEMBLE_DRAW``), computes the historical window layer on first
    use, renders the figure, and writes the companion CSVs (per-metric
    composition summary + the historical window coordinates).
    """
    import pandas as pd

    from scengen.diagnostics import load_hazard_image

    from scripts.main.compute_historic_hazard_windows import historic_hazard_windows
    from src.ensembles import staged_ensemble_dir
    from src.scenario_designs import SCENARIO_DESIGNS

    draw = config.SCENARIO_ENSEMBLE_DRAW
    pool_path, ps_path, hf_path = required_hazard_images()
    for p in (pool_path, ps_path, hf_path):
        if not p.exists():
            raise FileNotFoundError(
                f"staged hazard image not found: {p} (workflow steps 02-03; "
                f"fixed_probabilistic is scored post hoc by "
                f"scripts/supplemental/compute_staged_hazard_image.py)"
            )

    pool = load_hazard_image(pool_path)
    ps = load_ensemble_layer(
        SCENARIO_DESIGNS["fixed_probabilistic"].search_ensemble_slug(draw),
        "PS ensemble", style.design_color("fixed_probabilistic"))
    hf = load_ensemble_layer(
        SCENARIO_DESIGNS["hazard_filling_stationary"].search_ensemble_slug(draw),
        "HF ensemble", style.design_color("hazard_filling_stationary"))
    hist_H, hist_axes, window_starts = historic_hazard_windows()

    fig = build_composition_figure(
        pool["H"], list(pool["hazard_axes"]), [ps, hf], hist_H, hist_axes,
        range_layers=[hf],
    )
    written = style.save_manuscript_figure(fig, out_stub)
    plt.close(fig)

    metrics = list(config.HAZARD_SELECTION_AXES)
    pd.DataFrame(
        hist_H[:, [hist_axes.index(m) for m in metrics]],
        index=pd.Index([str(s.date()) for s in window_starts], name="window_start"),
        columns=metrics,
    ).to_csv(table_dir / "historic_windows.csv")

    rows = []
    for m in metrics:
        pool_v = _col(pool["H"], list(pool["hazard_axes"]), m)
        row = {"metric": m,
               "pool_p1": np.percentile(pool_v, 1),
               "pool_p50": np.percentile(pool_v, 50),
               "pool_p99": np.percentile(pool_v, 99)}
        for layer, key in ((ps, "ps"), (hf, "hf")):
            v = _col(layer.H, layer.axes, m)
            row.update({f"{key}_min": v.min(), f"{key}_p50": np.median(v),
                        f"{key}_max": v.max()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(table_dir / "composition_summary.csv", index=False)

    print(f"[fig04] pool P={pool['H'].shape[0]:,}, PS n={ps.H.shape[0]}, "
          f"HF n={hf.H.shape[0]}, historic windows={hist_H.shape[0]} "
          f"(draw {draw})")
    return written
