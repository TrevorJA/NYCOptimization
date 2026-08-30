"""
ensemble_composition.py - Manuscript figure 4: realized composition of the search ensembles.

Section 4.1's evidence base: the realized hazard-space composition of the two
matched search ensembles (MC and HF) against the candidate pool they share a
population law with, and against the historical record. Two scatter panels show
each ensemble over the pool's joint density in the drought magnitude-intensity
plane; six marginal panels show every campaign selection axis, with the pool as
a filled density, the two ensembles as kernel-density curves, and the historical
record's disjoint 10-yr windows as tick markers rather than a distribution
(seven windows cannot support one).

The scatter panels have two geometries. The default 3-D one plots each
ensemble in a drought-drought-flood triple, with the pool projected as
highest-density bands on the three cube walls instead of drawn as a point
cloud, so the N sampled members stay the subject of the panel. The 2-D one
is the original severity-magnitude plane over the pool's density field, which
the 3-D cube keeps intact as its back wall.

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
    NYCOPT_COMPOSITION_SCATTER    scatter-panel geometry, ``3d`` (default) or
                                  ``2d``
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

#: Default 3-D scatter triple ``(x, y, z)``. The x-z pair is the 2-D mode's
#: severity-magnitude plane, kept intact on the back wall of the cube; the
#: depth axis opens it out along a flood metric, so the 3-D panel is a
#: superset of the 2-D one rather than a different view.
DEFAULT_SCATTER_TRIPLE: tuple[str, str, str] = (
    "drought_severity", "flood_peak_discharge", "drought_magnitude")

#: Scatter-panel geometry: ``"3d"`` (cube with pool wall projections) or
#: ``"2d"`` (the pool-density plane). Override with NYCOPT_COMPOSITION_SCATTER.
DEFAULT_SCATTER_MODE = "3d"

#: Pool mass enclosed by each shaded band of the 3-D wall projections
#: (highest-density regions), with their fills in the same ascending-density
#: order that ``contourf`` consumes: outermost band lightest, core darkest.
#: Bands rather than a continuous field keep the pool legible as a backdrop;
#: a point cloud or a full density image at P=1e6 buries the N sampled members.
POOL_BAND_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99)
POOL_BAND_FILLS: tuple[str, ...] = ("0.945", "0.875", "0.78")
POOL_BAND_EDGE = "0.62"

#: Camera for the 3-D panels: ``(elevation, azimuth)`` in degrees. The azimuth
#: puts the severity-magnitude wall square-on behind the cloud.
SCATTER_3D_VIEW: tuple[float, float] = (13.0, -55.0)

#: Type floor for this figure, above the 12 pt manuscript minimum
#: (:data:`style.MANUSCRIPT_MIN_FONTSIZE`): the 3-D panels carry three axes of
#: ticks each, so their labels are read at a steeper reduction than a 2-D
#: panel's. Applied as an rc override around the build, not globally, so the
#: other manuscript figures keep the shared 12 pt style.
COMPOSITION_FONTSIZE: int = 14

#: Gap between an axis and its label, in points (all panels).
LABELPAD: float = 2.0

#: Marker-size depth cue for the 3-D scatter: ``base * (lo + span * nearness)``.
#: Size (not matplotlib's ``depthshade`` alpha fade) carries depth, so every
#: member keeps its full entity color.
DEPTH_SIZE_RANGE: tuple[float, float] = (0.55, 0.85)

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


def _hdr_levels(field: np.ndarray, quantiles) -> list[float]:
    """Density thresholds enclosing ``quantiles`` of a 2-D field's mass.

    Args:
        field: Non-negative density (or count) field.
        quantiles: Enclosed mass fractions, e.g. ``(0.5, 0.9, 0.99)``.

    Returns:
        One threshold per quantile: the level whose super-level set holds that
        fraction of the field's total mass (a highest-density region).
    """
    flat = np.sort(field.ravel())[::-1]
    cum = np.cumsum(flat) / (flat.sum() or 1.0)
    return [float(flat[min(np.searchsorted(cum, q), flat.size - 1)])
            for q in quantiles]


def _draw_wall_density(
    ax, pool_H: np.ndarray, pool_axes: list[str],
    a1: str, a2: str, zdir: str, offset: float,
    lim1: tuple[float, float], lim2: tuple[float, float],
    *, bins: int = 140, smooth: float = 2.0,
) -> None:
    """Project the pool's joint density for ``(a1, a2)`` onto one cube wall.

    Args:
        ax: 3-D axes.
        pool_H: Candidate-pool hazard image.
        pool_axes: Pool axis names.
        a1: Hazard metric on the wall's first in-plane axis.
        a2: Hazard metric on the wall's second in-plane axis.
        zdir: Wall normal (``"x"``, ``"y"`` or ``"z"``).
        offset: Position of the wall along ``zdir``.
        lim1: Display limits of ``a1`` (also the histogram range).
        lim2: Display limits of ``a2``.
        bins: 2-D histogram resolution per axis.
        smooth: Gaussian smoothing width in bins, so the band edges read as
            density contours rather than histogram staircases.
    """
    from scipy.ndimage import gaussian_filter

    h, e1, e2 = np.histogram2d(
        _col(pool_H, pool_axes, a1), _col(pool_H, pool_axes, a2),
        bins=bins, range=[lim1, lim2],
    )
    h = gaussian_filter(h, smooth)
    levels = sorted(_hdr_levels(h, POOL_BAND_QUANTILES)) + [h.max() * 1.001]
    c1 = 0.5 * (e1[:-1] + e1[1:])
    c2 = 0.5 * (e2[:-1] + e2[1:])
    G1, G2 = np.meshgrid(c1, c2, indexing="ij")
    fill = dict(levels=levels, colors=list(POOL_BAND_FILLS), zdir=zdir,
                offset=offset, zorder=0)
    line = dict(levels=levels[:-1], colors=POOL_BAND_EDGE, linewidths=0.7,
                zdir=zdir, offset=offset, zorder=0.1)
    args = {"z": (G1, G2, h), "y": (G1, h, G2), "x": (h, G1, G2)}[zdir]
    ax.contourf(*args, **fill)
    ax.contour(*args, **line)


def _depth_cue(coords: list[np.ndarray], lims: list[tuple[float, float]],
               elev: float, azim: float) -> np.ndarray:
    """Per-point nearness to the camera on ``[0, 1]`` (1 = nearest).

    Args:
        coords: ``(x, y, z)`` sample coordinates.
        lims: Display limits per axis, which set the cube's aspect.
        elev: Camera elevation (degrees).
        azim: Camera azimuth (degrees).

    Returns:
        Nearness per point, from the projection of its cube-normalized
        position onto the view direction.
    """
    el, az = np.deg2rad(elev), np.deg2rad(azim)
    view = (np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el))
    d = -sum(
        (np.asarray(c, float) - lo) / ((hi - lo) or 1.0) * v
        for c, (lo, hi), v in zip(coords, lims, view)
    )
    return (d - d.min()) / (np.ptp(d) or 1.0)


def _style_cube(ax, labels: list[str], lims: list[tuple[float, float]]) -> None:
    """White panes, dotted gridlines and 5-tick axes for a 3-D scatter panel."""
    from matplotlib.ticker import MaxNLocator

    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.55")
        axis.pane.set_alpha(1.0)
        axis.pane.set_linewidth(0.9)
        axis._axinfo["grid"].update(
            color="0.80", linestyle=(0, (1, 4)), linewidth=1.0)
        axis.line.set_color("0.35")
        axis.set_major_locator(MaxNLocator(5))
        axis.set_tick_params(colors="0.25", pad=0)
    # Vertical axis on the cube's left-hand edge: the marginal column sits
    # immediately to the right of these panels, so a right-hand z axis spends
    # the gap between them on ticks instead of on the cube.
    ax.zaxis.set_ticks_position("lower")
    ax.zaxis.set_label_position("lower")
    ax.set_xlabel(labels[0], labelpad=6)
    ax.set_ylabel(labels[1], labelpad=6)
    ax.set_zlabel(labels[2], labelpad=0)


def draw_scatter_panel_3d(
    ax,
    pool_H: np.ndarray, pool_axes: list[str],
    layer: CompositionLayer,
    ax_x: str, ax_y: str, ax_z: str,
    *,
    xlim: tuple[float, float], ylim: tuple[float, float],
    zlim: tuple[float, float],
    historic_H: np.ndarray | None = None, historic_axes: list[str] | None = None,
    scatter_size: float = 36.0,
    show_wall_shadows: bool = True,
    view: tuple[float, float] = SCATTER_3D_VIEW,
) -> None:
    """One ensemble in a hazard triple, over the pool projected on the walls.

    The pool appears as highest-density bands on the three cube walls (the
    three pairwise marginals of ``(ax_x, ax_y, ax_z)``) rather than as a point
    cloud inside the cube: at P=1e6 an in-cube pool hides the N members the
    panel exists to show. Members carry a marker-size depth cue and a faint
    shadow on each of the three walls -- their own 2-D marginals, read directly
    against the pool bands they sit on -- which together fix their position in
    the cube without the per-point drop lines that would swamp the panel.

    Args:
        ax: Target axes, created with ``projection="3d"``.
        pool_H: Candidate-pool hazard image.
        pool_axes: Pool axis names.
        layer: The ensemble scatter layer.
        ax_x: Hazard metric on x.
        ax_y: Hazard metric on y (the depth axis).
        ax_z: Hazard metric on z (vertical).
        xlim: Panel x limits (also the wall-density range).
        ylim: Panel y limits.
        zlim: Panel z limits.
        historic_H: Optional historical window coordinates (marker overlay).
        historic_axes: Axis names for ``historic_H``.
        scatter_size: Ensemble marker size at mid depth.
        show_wall_shadows: Project a faint gray copy of each member onto all
            three walls, alongside the pool bands.
        view: ``(elevation, azimuth)`` camera angles in degrees.
    """
    elev, azim = view
    ax.view_init(elev=elev, azim=azim)
    # Explicit zorder: matplotlib depth-sorts 3-D artists by mean depth, which
    # would put the full-wall density fields in front of near-camera points.
    ax.computed_zorder = False
    lims = [xlim, ylim, zlim]
    _style_cube(ax, [HAZARD_METRIC_LABELS.get(a, a) for a in (ax_x, ax_y, ax_z)],
                lims)

    _draw_wall_density(ax, pool_H, pool_axes, ax_x, ax_z, "y", ylim[1],
                       xlim, zlim)
    _draw_wall_density(ax, pool_H, pool_axes, ax_y, ax_z, "x", xlim[0],
                       ylim, zlim)
    _draw_wall_density(ax, pool_H, pool_axes, ax_x, ax_y, "z", zlim[0],
                       xlim, ylim)

    x, y, z = (_col(layer.H, layer.axes, a) for a in (ax_x, ax_y, ax_z))
    if show_wall_shadows:
        shadow = dict(s=scatter_size * 0.30, c="0.45", alpha=0.30, linewidths=0,
                      depthshade=False, zorder=1)
        ax.scatter(x, y, np.full_like(z, zlim[0]), **shadow)   # floor
        ax.scatter(x, np.full_like(y, ylim[1]), z, **shadow)   # back wall
        ax.scatter(np.full_like(x, xlim[0]), y, z, **shadow)   # side wall
    lo, span = DEPTH_SIZE_RANGE
    ax.scatter(
        x, y, z, s=scatter_size * (lo + span * _depth_cue([x, y, z], lims, elev, azim)),
        color=layer.color, edgecolors="white", linewidths=0.6, alpha=0.95,
        depthshade=False, zorder=5,
    )
    if historic_H is not None:
        hx, hy, hz = (_col(historic_H, historic_axes, a)
                      for a in (ax_x, ax_y, ax_z))
        ax.scatter(hx, hy, hz, marker="X", s=115, c=HISTORIC_MARK,
                   edgecolors="white", linewidths=1.0, depthshade=False,
                   zorder=6)


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


@plt.rc_context({
    "font.size":             COMPOSITION_FONTSIZE,
    "axes.titlesize":        COMPOSITION_FONTSIZE + 1,
    "axes.labelsize":        COMPOSITION_FONTSIZE,
    "xtick.labelsize":       COMPOSITION_FONTSIZE,
    "ytick.labelsize":       COMPOSITION_FONTSIZE,
    "legend.fontsize":       COMPOSITION_FONTSIZE,
    "legend.title_fontsize": COMPOSITION_FONTSIZE,
    "axes.labelpad":         LABELPAD,
})
def build_composition_figure(
    pool_H: np.ndarray, pool_axes: list[str],
    layers: list[CompositionLayer],
    historic_H: np.ndarray | None, historic_axes: list[str] | None,
    *,
    scatter_mode: str = DEFAULT_SCATTER_MODE,
    scatter_pair: tuple[str, str] = DEFAULT_SCATTER_PAIR,
    scatter_triple: tuple[str, str, str] = DEFAULT_SCATTER_TRIPLE,
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
        scatter_mode: ``"3d"`` for cube panels with the pool projected on the
            walls, ``"2d"`` for the pool-density plane.
        scatter_pair: ``(x, y)`` hazard metrics of the 2-D scatter panels.
        scatter_triple: ``(x, y, z)`` hazard metrics of the 3-D scatter panels
            (y is the depth axis, z the vertical one).
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
    three_d = scatter_mode == "3d"
    scatter_metrics = list(scatter_triple if three_d else scatter_pair)

    # The 3-D panels need a squarer, taller box than the 2-D plane: a cube
    # drawn into a wide-short axes loses most of its usable area to the
    # projected corners.
    if three_d:
        fig = plt.figure(figsize=(13.5, 2.4 * n_marg))
        gs = fig.add_gridspec(
            n_marg, 2, width_ratios=(1.30, 1.0),
            left=0.085, right=0.995, top=0.98, bottom=0.115,
            hspace=0.95, wspace=0.02,
        )
    else:
        fig = plt.figure(figsize=(12.0, 1.55 * n_marg))
        gs = fig.add_gridspec(
            n_marg, 2, width_ratios=(1.08, 1.0),
            left=0.06, right=0.985, top=0.965, bottom=0.14,
            hspace=0.95, wspace=0.16,
        )
    half = n_marg // 2
    proj = "3d" if three_d else None
    scatter_axes = [
        fig.add_subplot(gs[:half, 0], projection=proj),
        fig.add_subplot(gs[half:, 0], projection=proj),
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
    for a in scatter_metrics:
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
        if three_d:
            ax_x, ax_y, ax_z = scatter_metrics
            draw_scatter_panel_3d(
                ax, pool_H, pool_axes, layer, ax_x, ax_y, ax_z,
                xlim=lims[ax_x], ylim=lims[ax_y], zlim=lims[ax_z],
                historic_H=hist_pair, historic_axes=historic_axes,
            )
            ax.set_box_aspect((1.0, 1.0, 1.12), zoom=1.15)
            ax.set_title(f"({next(letters)}) {layer.name}", loc="left", y=0.94)
        else:
            ax_x, ax_y = scatter_metrics
            draw_scatter_panel(
                ax, pool_H, pool_axes, layer, ax_x, ax_y,
                xlim=lims[ax_x], ylim=lims[ax_y],
                historic_H=hist_pair, historic_axes=historic_axes,
            )
            ax.set_title(f"({next(letters)}) {layer.name}", loc="left")
            ax.set_ylabel(HAZARD_METRIC_LABELS.get(ax_y, ax_y))
    if not three_d:
        scatter_axes[0].tick_params(labelbottom=False)
        scatter_axes[1].set_xlabel(
            HAZARD_METRIC_LABELS.get(scatter_metrics[0], scatter_metrics[0]))

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
        band_pct = "/".join(f"{q:.0%}" for q in POOL_BAND_QUANTILES)
        handles.append(Patch(
            facecolor=POOL_BAND_FILLS[1] if three_d else POOL_FILL,
            edgecolor=POOL_BAND_EDGE if three_d else POOL_EDGE,
            label=f"{pool_label} ({band_pct} density)" if three_d else pool_label))
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
        SCENARIO_DESIGNS["monte_carlo"].search_ensemble_slug(draw),
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
    pool_path, mc_path, hf_path = required_hazard_images()
    for p in (pool_path, mc_path, hf_path):
        if not p.exists():
            raise FileNotFoundError(
                f"staged hazard image not found: {p} (workflow steps 02-03; "
                f"monte_carlo is scored post hoc by "
                f"scripts/supplemental/compute_staged_hazard_image.py)"
            )

    pool = load_hazard_image(pool_path)
    mc = load_ensemble_layer(
        SCENARIO_DESIGNS["monte_carlo"].search_ensemble_slug(draw),
        "MC ensemble", style.design_color("monte_carlo"))
    hf = load_ensemble_layer(
        SCENARIO_DESIGNS["hazard_filling_stationary"].search_ensemble_slug(draw),
        "HF ensemble", style.design_color("hazard_filling_stationary"))
    hist_H, hist_axes, window_starts = historic_hazard_windows()

    mode = os.environ.get("NYCOPT_COMPOSITION_SCATTER", DEFAULT_SCATTER_MODE)
    if mode not in ("2d", "3d"):
        raise ValueError(
            f"NYCOPT_COMPOSITION_SCATTER must be '2d' or '3d', got {mode!r}")
    fig = build_composition_figure(
        pool["H"], list(pool["hazard_axes"]), [mc, hf], hist_H, hist_axes,
        scatter_mode=mode, range_layers=[hf],
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
        for layer, key in ((mc, "mc"), (hf, "hf")):
            v = _col(layer.H, layer.axes, m)
            row.update({f"{key}_min": v.min(), f"{key}_p50": np.median(v),
                        f"{key}_max": v.max()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(table_dir / "composition_summary.csv", index=False)

    print(f"[fig04] pool P={pool['H'].shape[0]:,}, MC n={mc.H.shape[0]}, "
          f"HF n={hf.H.shape[0]}, historic windows={hist_H.shape[0]} "
          f"(draw {draw}, scatter {mode})")
    return written
