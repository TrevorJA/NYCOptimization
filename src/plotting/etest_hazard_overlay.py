"""
src/plotting/etest_hazard_overlay.py - E_test vs pool vs search-ensemble hazard overlay.

Corner-style pairwise figure over the campaign hazard selection axes: the candidate
pool as a grayscale log-density field, the robust p1/p99 selection box, each realized
search ensemble as categorical scatter layers, and E_test's sub-window cloud as
density contours. Answers whether E_test's severe events occupy the same hazard
coordinates as the pool's natural-variability corners the selector enriches.

Data contracts:
    * Pool / search layers: ``hazard_image.npz`` written by ``scengen.diagnostics
      .save_hazard_image`` (search layers use ``selected_rows`` when non-empty,
      otherwise all rows are the ensemble).
    * E_test layer: ``hazard_image_subwindows.npz`` written by
      ``scripts/main/compute_etest_hazard_image.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

#: Display labels for the candidate hazard axes (dry SSI-6 run-theory; wet POT).
HAZARD_AXIS_LABELS: dict[str, str] = {
    "drought_deficit_volume": "Deficit volume\n(|ΣSSI|)",
    "drought_duration":       "Drought duration\n(months)",
    "drought_peak_depth":     "Peak depth\n(|min SSI|)",
    "drought_onset_rate":     "Onset rate\n(SSI/month)",
    "drought_recovery_rate":  "Recovery rate\n(SSI/month)",
    "flood_peak_magnitude":   "Flood peak\n(x ref. mean)",
    "flood_pulse_duration":   "Pulse duration\n(days)",
    "flood_rise_rate":        "Rise rate\n(x ref. mean/day)",
}

#: Fixed categorical hue per overlay layer (identity job; never cycled). The pool
#: density field is grayscale (magnitude job) and E_test is the reserved purple.
LAYER_COLORS: dict[str, str] = {
    "hazard_filling_stationary": "#d95f02",
    "fixed_probabilistic":       "#1f77b4",
    "etest":                     "#7a4fa3",
}
_FALLBACK_COLORS = ("#2a9d8f", "#c44536", "#8a8a3b")

#: E_test contour levels: fraction of sub-window mass enclosed, outermost first.
ETEST_MASS_LEVELS = (0.99, 0.90, 0.50)

#: Coarser grid + Gaussian smoothing for the E_test contour field only, so mass
#: contours stay closed curves instead of per-cell speckle on sparse histograms.
ETEST_CONTOUR_BINS = 60
ETEST_CONTOUR_SMOOTH_SIGMA = 1.2


@dataclass
class OverlayLayer:
    """One scatter layer (a realized search ensemble) for the overlay figure.

    Attributes:
        name: Design/draw label used in the legend.
        H: ``(n, m_all)`` hazard coordinates of the ensemble members.
        axes: Axis names aligned with columns of ``H``.
        color: Hex color; assigned from ``LAYER_COLORS`` by the driver.
    """

    name: str
    H: np.ndarray
    axes: list[str] = field(default_factory=list)
    color: str = "#2a9d8f"


def _col(H: np.ndarray, axes: list[str], name: str) -> np.ndarray:
    return H[:, list(axes).index(name)]


def _mass_level_thresholds(hist: np.ndarray, fractions: tuple[float, ...]) -> list[float]:
    """Density thresholds whose superlevel sets enclose the given mass fractions."""
    flat = np.sort(hist.ravel())[::-1]
    csum = np.cumsum(flat)
    total = csum[-1]
    out = []
    for f in fractions:
        i = int(np.searchsorted(csum, f * total))
        out.append(float(flat[min(i, len(flat) - 1)]))
    return out


def overlap_stats(
    pool_H: np.ndarray, pool_axes: list[str],
    etest_H: np.ndarray, etest_axes: list[str],
    axes_names: list[str], lo_pct: float, hi_pct: float,
) -> dict:
    """Per-axis containment of E_test's sub-window cloud in the pool's hazard span.

    Args:
        pool_H: Candidate-pool hazard image.
        pool_axes: Pool axis names.
        etest_H: E_test sub-window hazard image.
        etest_axes: E_test axis names.
        axes_names: Axes to report (the campaign selection set).
        lo_pct: Robust lower selection-bound percentile (campaign p1).
        hi_pct: Robust upper selection-bound percentile (campaign p99).

    Returns:
        ``{axis: {pool_lo, pool_hi, pool_min, pool_max, etest_frac_below_lo,
        etest_frac_above_hi, etest_frac_outside_hull, etest_p50, pool_p50}}``.
    """
    out: dict[str, dict] = {}
    for a in axes_names:
        p = _col(pool_H, pool_axes, a)
        e = _col(etest_H, etest_axes, a)
        lo, hi = np.percentile(p, [lo_pct, hi_pct])
        out[a] = {
            "pool_lo": float(lo), "pool_hi": float(hi),
            "pool_min": float(p.min()), "pool_max": float(p.max()),
            "pool_p50": float(np.median(p)), "etest_p50": float(np.median(e)),
            "etest_frac_below_lo": float((e < lo).mean()),
            "etest_frac_above_hi": float((e > hi).mean()),
            "etest_frac_outside_hull": float(((e < p.min()) | (e > p.max())).mean()),
        }
    return out


def build_overlay_figure(
    pool_H: np.ndarray, pool_axes: list[str],
    layers: list[OverlayLayer],
    etest_H: np.ndarray, etest_axes: list[str],
    axes_names: list[str],
    *, lo_pct: float = 1.0, hi_pct: float = 99.0, grid_bins: int = 120,
):
    """Build the corner-style pairwise hazard-space overlay figure.

    Args:
        pool_H: ``(P, m_all)`` candidate-pool hazard image.
        pool_axes: Pool axis names (columns of ``pool_H``).
        layers: Realized search-ensemble layers, in legend order.
        etest_H: ``(n_sub, m_all)`` E_test sub-window hazard image.
        etest_axes: E_test axis names.
        axes_names: The m axes to plot (campaign selection set); panel (i, j) with
            ``i > j`` shows ``axes_names[j]`` (x) vs ``axes_names[i]`` (y), and the
            diagonal shows per-axis marginals.
        lo_pct: Robust lower selection-bound percentile for the box overlay.
        hi_pct: Robust upper selection-bound percentile for the box overlay.
        grid_bins: 2-D histogram resolution for the pool field and E_test contours.

    Returns:
        The matplotlib figure.
    """
    m = len(axes_names)
    fig, axarr = plt.subplots(m, m, figsize=(2.1 * m + 1.2, 2.1 * m + 0.8))
    etest_color = LAYER_COLORS["etest"]

    # Panel limits: span of pool and E_test jointly, per axis, padded 3%.
    lims = {}
    for a in axes_names:
        v = np.concatenate([_col(pool_H, pool_axes, a), _col(etest_H, etest_axes, a)])
        lo, hi = float(v.min()), float(v.max())
        pad = 0.03 * (hi - lo or 1.0)
        lims[a] = (lo - pad, hi + pad)
    box = {a: np.percentile(_col(pool_H, pool_axes, a), [lo_pct, hi_pct]) for a in axes_names}

    for i in range(m):
        for j in range(m):
            ax = axarr[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            ax_y, ax_x = axes_names[i], axes_names[j]
            if i == j:
                # Marginals: pool filled (gray), E_test line (purple), layers rug.
                edges = np.linspace(*lims[ax_x], grid_bins + 1)
                pv = _col(pool_H, pool_axes, ax_x)
                ev = _col(etest_H, etest_axes, ax_x)
                ax.hist(pv, bins=edges, density=True, color="0.82", zorder=1)
                ax.hist(ev, bins=edges, density=True, histtype="step",
                        color=etest_color, lw=1.6, zorder=3)
                for layer in layers:
                    lv = _col(layer.H, layer.axes, ax_x)
                    ax.plot(lv, np.full_like(lv, -0.04 * ax.get_ylim()[1]), "|",
                            color=layer.color, ms=5, alpha=0.8, clip_on=False, zorder=4)
                for b in box[ax_x]:
                    ax.axvline(b, color="0.25", lw=0.8, ls="--", zorder=2)
                ax.set_xlim(*lims[ax_x])
                ax.set_yticks([])
            else:
                x_p = _col(pool_H, pool_axes, ax_x)
                y_p = _col(pool_H, pool_axes, ax_y)
                hist, xe, ye = np.histogram2d(
                    x_p, y_p, bins=grid_bins, range=[lims[ax_x], lims[ax_y]]
                )
                ax.pcolormesh(
                    xe, ye, np.ma.masked_equal(hist.T, 0), cmap="Greys",
                    norm=LogNorm(vmin=1, vmax=max(hist.max(), 2)),
                    rasterized=True, zorder=1,
                )
                (x0, x1), (y0, y1) = box[ax_x], box[ax_y]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="0.25", lw=0.9, ls="--", zorder=2))
                eh, exe, eye = np.histogram2d(
                    _col(etest_H, etest_axes, ax_x), _col(etest_H, etest_axes, ax_y),
                    bins=ETEST_CONTOUR_BINS, range=[lims[ax_x], lims[ax_y]],
                )
                eh_s = gaussian_filter(eh, ETEST_CONTOUR_SMOOTH_SIGMA)
                levels = sorted(set(_mass_level_thresholds(eh_s, ETEST_MASS_LEVELS)))
                levels = [lv for lv in levels if lv > 0]
                if levels:
                    xc, yc = (exe[:-1] + exe[1:]) / 2, (eye[:-1] + eye[1:]) / 2
                    ax.contour(xc, yc, eh_s.T, levels=levels, colors=etest_color,
                               linewidths=1.3, zorder=3)
                for layer in layers:
                    ax.scatter(
                        _col(layer.H, layer.axes, ax_x), _col(layer.H, layer.axes, ax_y),
                        s=11, color=layer.color, edgecolors="white", linewidths=0.4,
                        alpha=0.9, zorder=4,
                    )
                ax.set_xlim(*lims[ax_x])
                ax.set_ylim(*lims[ax_y])
            if i < m - 1:
                ax.set_xticklabels([])
            if j > 0 and i != j:
                ax.set_yticklabels([])
            if i == m - 1:
                ax.set_xlabel(HAZARD_AXIS_LABELS.get(ax_x, ax_x), fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(HAZARD_AXIS_LABELS.get(ax_y, ax_y), fontsize=8)
            ax.tick_params(labelsize=7)

    handles = [
        Line2D([], [], marker="s", ls="", color="0.7", label="candidate pool (log density)"),
        Line2D([], [], color="0.25", ls="--", lw=0.9,
               label=f"robust p{lo_pct:g}/p{hi_pct:g} selection box"),
        Line2D([], [], color=etest_color, lw=1.6,
               label="E_test sub-windows (50/90/99% mass)"),
    ] + [
        Line2D([], [], marker="o", ls="", color=layer.color, markersize=6,
               markeredgecolor="white", label=layer.name)
        for layer in layers
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98),
               frameon=False, fontsize=9)
    fig.suptitle("E_test hazard-space coverage vs candidate pool and search ensembles",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
