"""
tradeoff_scatter.py - Pairwise objective-space scatter views of a Pareto set.

Parallel axes show per-objective ranges but hide the SHAPE of two-way tradeoffs
(knees, gaps, compromise regions). These figures complement them with scatter
panels in natural units, all pre-re-evaluation:

  :func:`plot_key_tradeoffs`
      A small grid of the headline objective pairs (``style.SCATTER_PAIRS``),
      points colored by a third objective, the FFMP baseline starred, and a
      grey arrow pointing into each panel's ideal corner.
  :func:`plot_scatter_matrix`
      The full lower-triangle matrix over every objective pair — the
      exhaustive companion for finding structure the curated pairs miss.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.plotting.style import (SCATTER_PAIRS, label_for, save_figure,
                                short_label_for)

#: Reference-marker colour for the FFMP baseline, matching front_overview.
BASELINE_COLOR = "firebrick"

#: Point colour when no coloring objective is given.
POINT_COLOR = "steelblue"

#: Colormap for coloring points by a third objective.
COLOR_MAP = "viridis"


def _color_values(natural, obj_names, color_by):
    """Resolve ``color_by`` to (values, colormap-normalized label) or None."""
    if color_by is None:
        return None, None
    idx = obj_names.index(color_by) if isinstance(color_by, str) else int(color_by)
    return natural[:, idx], label_for(obj_names[idx])


def _ideal_arrow(ax, dir_x: int, dir_y: int, fontsize: int = 7) -> None:
    """Small grey arrow pointing into the panel's ideal corner (axes fraction)."""
    cx = 0.95 if dir_x == 1 else 0.05
    cy = 0.95 if dir_y == 1 else 0.05
    sx = cx - 0.10 * (1 if dir_x == 1 else -1)
    sy = cy - 0.10 * (1 if dir_y == 1 else -1)
    ax.annotate("", xy=(cx, cy), xytext=(sx, sy), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.2),
                zorder=3)
    ax.annotate("ideal", xy=(sx, sy), xycoords="axes fraction",
                ha="right" if dir_x == 1 else "left",
                va="top" if dir_y == 1 else "bottom",
                fontsize=fontsize, color="0.45", zorder=3)


def _baseline_handle():
    return Line2D([0], [0], ls="none", marker="*", markersize=12,
                  markerfacecolor=BASELINE_COLOR, markeredgecolor="white",
                  markeredgewidth=0.6, label="FFMP baseline")


def plot_key_tradeoffs(
    natural: np.ndarray,
    obj_names: list,
    directions,
    pairs: list | None = None,
    color_by: str | int | None = None,
    baseline: np.ndarray | None = None,
    output_file: Path | None = None,
    figsize: tuple | None = None,
) -> Figure:
    """Grid of headline pairwise tradeoff scatters in natural units.

    Args:
        natural: ``(n_solutions, n_objs)`` objectives in natural units.
        obj_names: Objective names aligned to the columns.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        pairs: ``(i, j)`` column-index pairs, one panel each (default: the
            ``style.SCATTER_PAIRS`` that fit the objective count).
        color_by: Objective (name or index) coloring every point through a
            shared colorbar; None for single-colour points.
        baseline: Optional ``(n_objs,)`` baseline vector, drawn as a star.
        output_file: Optional path stub (no extension); saved via
            ``style.save_figure``.
        figsize: Figure size; a default is derived from the panel count.

    Returns:
        The matplotlib Figure.
    """
    natural = np.atleast_2d(np.asarray(natural, dtype=float))
    directions = np.asarray(directions)
    n_objs = len(obj_names)
    if pairs is None:
        pairs = [p for p in SCATTER_PAIRS if max(p) < n_objs]
    cvals, clabel = _color_values(natural, obj_names, color_by)

    ncol = min(3, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    if figsize is None:
        figsize = (4.3 * ncol, 3.6 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    sc = None
    for ax, (ix, iy) in zip(axes, pairs):
        if cvals is not None:
            sc = ax.scatter(natural[:, ix], natural[:, iy], c=cvals,
                            cmap=COLOR_MAP, s=12, alpha=0.65, lw=0, zorder=2)
        else:
            ax.scatter(natural[:, ix], natural[:, iy], color=POINT_COLOR,
                       s=12, alpha=0.5, lw=0, zorder=2)
        if baseline is not None:
            ax.scatter(baseline[ix], baseline[iy], marker="*", s=230,
                       color=BASELINE_COLOR, edgecolor="white", lw=0.7,
                       zorder=10)
        _ideal_arrow(ax, int(directions[ix]), int(directions[iy]))
        ax.set_xlabel(label_for(obj_names[ix]), fontsize=8)
        ax.set_ylabel(label_for(obj_names[iy]), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    if baseline is not None:
        axes[0].legend(handles=[_baseline_handle()], loc="best", fontsize=7)
    fig.suptitle(f"Pairwise objective tradeoffs across the front "
                 f"(n={natural.shape[0]})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if cvals is not None:
        cb = fig.colorbar(sc, ax=axes[:len(pairs)].tolist(), shrink=0.7,
                          pad=0.02)
        cb.set_label(clabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if output_file is not None:
        save_figure(fig, output_file)
    return fig


def plot_scatter_matrix(
    natural: np.ndarray,
    obj_names: list,
    directions,
    color_by: str | int | None = None,
    baseline: np.ndarray | None = None,
    output_file: Path | None = None,
    figsize: tuple | None = None,
) -> Figure:
    """Lower-triangle scatter matrix over every objective pair.

    Edge labels carry the preference direction (↑ maximize, ↓ minimize); the
    colorbar and baseline-star legend sit in the empty upper triangle.

    Args:
        natural: ``(n_solutions, n_objs)`` objectives in natural units.
        obj_names: Objective names aligned to the columns.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        color_by: Objective (name or index) coloring every point; None for
            single-colour points.
        baseline: Optional ``(n_objs,)`` baseline vector, drawn as a star.
        output_file: Optional path stub (no extension).
        figsize: Figure size; a default is derived from the objective count.

    Returns:
        The matplotlib Figure.
    """
    natural = np.atleast_2d(np.asarray(natural, dtype=float))
    directions = np.asarray(directions)
    n_objs = len(obj_names)
    m = n_objs - 1
    cvals, clabel = _color_values(natural, obj_names, color_by)
    if figsize is None:
        figsize = (1.55 * m + 1.6, 1.45 * m + 1.4)

    def _edge_label(k: int) -> str:
        arrow = "↑" if directions[k] == 1 else "↓"
        return f"{short_label_for(obj_names[k])} {arrow}"

    fig, axes = plt.subplots(m, m, figsize=figsize, sharex="col", sharey="row")
    axes = np.atleast_2d(axes)
    sc = None
    for row in range(m):          # panel row = objective row + 1
        i = row + 1
        for col in range(m):      # panel col = objective col
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if cvals is not None:
                sc = ax.scatter(natural[:, col], natural[:, i], c=cvals,
                                cmap=COLOR_MAP, s=6, alpha=0.5, lw=0,
                                rasterized=True, zorder=2)
            else:
                ax.scatter(natural[:, col], natural[:, i], color=POINT_COLOR,
                           s=6, alpha=0.4, lw=0, rasterized=True, zorder=2)
            if baseline is not None:
                ax.scatter(baseline[col], baseline[i], marker="*", s=110,
                           color=BASELINE_COLOR, edgecolor="white", lw=0.5,
                           zorder=10)
            ax.tick_params(labelsize=6)
            ax.locator_params(nbins=3)
            ax.grid(True, alpha=0.25)
            if row == m - 1:
                ax.set_xlabel(_edge_label(col), fontsize=7)
                ax.tick_params(axis="x", rotation=40)
            if col == 0:
                ax.set_ylabel(_edge_label(i), fontsize=7)

    fig.suptitle(f"Objective-space scatter matrix (n={natural.shape[0]})",
                 fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.11, top=0.94,
                        wspace=0.10, hspace=0.10)
    # Colorbar + legend live in the empty upper-right triangle.
    if cvals is not None and sc is not None:
        cax = fig.add_axes([0.58, 0.80, 0.30, 0.02])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label(clabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if baseline is not None:
        fig.legend(handles=[_baseline_handle()], loc="upper right",
                   bbox_to_anchor=(0.90, 0.92), fontsize=8, frameon=False)
    if output_file is not None:
        save_figure(fig, output_file)
    return fig
