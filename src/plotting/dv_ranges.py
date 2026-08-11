"""
dv_ranges.py - Bound-normalized decision-variable ranges under objective criteria.

The pre-re-evaluation link between rule structure and performance: for each
decision variable (normalized to its search bounds) one panel per objective
criterion shows the FULL front's min-max range as a light grey bar with the
criterion-satisfying subset overlaid in colour (min-max whisker + interquartile
box + median tick) and the default FFMP policy marked. Where a subset's box
sits away from the grey bar, that variable is doing the work of meeting the
criterion; where it spans the bar, the criterion leaves it unconstrained.

:func:`default_criteria` builds the standard criterion list (baseline dominance
and NYC-storage floors) shared by the post-optimization figure scripts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.formulations import get_baseline_values, get_bounds, get_var_names
from src.plotting.style import save_figure

#: Reference-marker colour for the FFMP baseline, matching front_overview.
BASELINE_COLOR = "firebrick"

#: Fill for the full-front min-max range bars.
FULL_RANGE_COLOR = "0.88"

#: Per-criterion highlight colours (CVD-safe, matching front_overview).
CRITERIA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9467bd"]

#: Display names for DV families (grouped by leading name token).
_FAMILY_LABELS = {
    "nyc":   "NYC alloc.",
    "nj":    "NJ alloc.",
    "zone":  "Storage-zone boundary shifts",
    "flood": "Flood release scales",
    "mrf":   "MRF profile / target scales",
}


def default_criteria(
    natural: np.ndarray,
    obj_names: list,
    directions,
    baseline: np.ndarray | None = None,
    dom_mask: np.ndarray | None = None,
) -> list:
    """Standard ``(label, mask)`` criteria for the DV-ranges figure.

    Included when computable: dominance of the FFMP baseline (from
    ``dom_mask``, or derived from ``baseline``) and NYC-storage floors at 10%
    and at the 26% FFMP drought-emergency goalpost
    (``objectives_ensemble`` satisficing threshold).

    Args:
        natural: ``(n_solutions, n_objs)`` objectives in natural units.
        obj_names: Objective names aligned to the columns.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        baseline: Optional baseline objective vector (used only when
            ``dom_mask`` is None).
        dom_mask: Optional precomputed baseline-dominance mask.

    Returns:
        List of ``(label, boolean mask)`` tuples; may be empty.
    """
    criteria = []
    if dom_mask is None and baseline is not None:
        from src.solution_selection import dominance_mask
        dom_mask = dominance_mask(np.atleast_2d(natural), baseline, directions)
    if dom_mask is not None:
        criteria.append(("Dominates the FFMP baseline", np.asarray(dom_mask)))
    if "nyc_storage_min_p01_pct" in obj_names:
        sv = np.atleast_2d(natural)[:, obj_names.index("nyc_storage_min_p01_pct")]
        criteria.append(("NYC storage (P1 of ann. min) ≥ 10% of capacity",
                         sv >= 10.0))
        criteria.append(("NYC storage (P1 of ann. min) ≥ 26% "
                         "(FFMP drought-emergency floor)", sv >= 26.0))
    return criteria


def plot_dv_ranges(
    dv: np.ndarray,
    formulation: str,
    criteria: list,
    output_file: Path | None = None,
    show_baseline: bool = True,
    figsize: tuple | None = None,
) -> Figure:
    """DV ranges of criterion-satisfying subsets against the full front.

    Args:
        dv: ``(n_solutions, n_vars)`` decision variables of the full front, in
            native units (normalized internally by the formulation bounds).
        formulation: Formulation name (bounds, variable names, baseline DVs).
        criteria: ``(label, mask)`` tuples, one stacked panel each; masks are
            boolean arrays aligned to the ``dv`` rows (see
            :func:`default_criteria`).
        output_file: Optional path stub (no extension); saved via
            ``style.save_figure``.
        show_baseline: Draw the default FFMP policy's DVs as markers.
        figsize: Figure size; a default is derived from the panel/DV counts.

    Returns:
        The matplotlib Figure.
    """
    dv = np.atleast_2d(np.asarray(dv, dtype=float))
    lo, hi = get_bounds(formulation)
    names = get_var_names(formulation)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    nd = (dv - lo) / span
    base_nd = ((get_baseline_values(formulation) - lo) / span
               if show_baseline else None)
    n_sol, n_vars = nd.shape
    x = np.arange(n_vars)
    full_min, full_max = nd.min(axis=0), nd.max(axis=0)

    # DV family boundaries (change of leading name token).
    fams = [n.split("_")[0] for n in names]
    breaks = [k for k in range(1, n_vars) if fams[k] != fams[k - 1]]

    n_panels = len(criteria)
    if figsize is None:
        figsize = (max(11.0, 0.33 * n_vars + 2.0), 2.5 * n_panels + 2.0)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                             sharey=True)
    axes = np.atleast_1d(axes)

    for k, (ax, (label, mask)) in enumerate(zip(axes, criteria)):
        color = CRITERIA_COLORS[k % len(CRITERIA_COLORS)]
        mask = np.asarray(mask, dtype=bool)
        sub = nd[mask]
        ax.bar(x, full_max - full_min, bottom=full_min, width=0.72,
               color=FULL_RANGE_COLOR, edgecolor="0.75", lw=0.4, zorder=1)
        if sub.shape[0]:
            q1, q2, q3 = np.percentile(sub, [25, 50, 75], axis=0)
            ax.vlines(x, sub.min(axis=0), sub.max(axis=0), color=color,
                      lw=1.3, alpha=0.9, zorder=3)
            ax.bar(x, q3 - q1, bottom=q1, width=0.45, color=color, alpha=0.55,
                   edgecolor=color, lw=0.6, zorder=4)
            ax.plot(x, q2, ls="none", marker="_", markersize=7,
                    markeredgewidth=1.6, color=color, zorder=5)
        else:
            ax.text(0.5, 0.5, "0 solutions satisfy this criterion",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=color,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=color, alpha=0.9), zorder=10)
        if base_nd is not None:
            ax.plot(x, base_nd, ls="none", marker="D", markersize=4,
                    color=BASELINE_COLOR, zorder=6)
        for b in breaks:
            ax.axvline(b - 0.5, color="0.7", lw=0.8, ls=":", zorder=2)
        # Extra pad on the top panel keeps its title clear of the family labels.
        ax.set_title(f"{label}  —  {int(mask.sum())} of {n_sol} solutions",
                     fontsize=9, loc="left", pad=24 if k == 0 else 6)
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel("DV value\n(fraction of bounds)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlim(-0.8, n_vars - 0.2)

        handles = [
            Patch(facecolor=FULL_RANGE_COLOR, edgecolor="0.75",
                  label="full front min–max"),
            Patch(facecolor=color, alpha=0.55, edgecolor=color,
                  label="subset IQR (median tick)"),
            Line2D([0], [0], color=color, lw=1.3, label="subset min–max"),
        ]
        if base_nd is not None:
            handles.append(Line2D([0], [0], ls="none", marker="D",
                                  markersize=4, color=BASELINE_COLOR,
                                  label="FFMP baseline"))
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.005, 1.0), fontsize=6.5, frameon=False)

    # Family group labels above the top panel, under its padded title.
    edges = [0] + breaks + [n_vars]
    for a, b in zip(edges[:-1], edges[1:]):
        axes[0].text((a + b - 1) / 2.0, 1.02, _FAMILY_LABELS.get(fams[a], fams[a]),
                     transform=axes[0].get_xaxis_transform(), ha="center",
                     va="bottom", fontsize=7.5, color="0.35")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(names, rotation=90, fontsize=6.5)
    fig.suptitle("Decision-variable ranges of criterion-satisfying subsets "
                 f"({formulation}, front n={n_sol})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.90, 0.93))
    if output_file is not None:
        save_figure(fig, output_file)
    return fig
