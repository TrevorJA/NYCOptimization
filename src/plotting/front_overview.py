"""front_overview.py - Whole-front views of a Pareto-approximate set.

Three composable figures over one reference set, each taking objectives in
NATURAL units (see :func:`src.pareto_filter.to_natural`) and returning a
``Figure``:

  :func:`plot_baseline_dominance`
      Parallel axes with the baseline-dominating subset picked out of the
      front and the FFMP baseline drawn as a reference line.
  :func:`plot_selected_policies`
      Parallel axes with named representatives highlighted against the front.
  :func:`plot_objective_tradeoffs`
      Spearman rank-correlation heatmap in PREFERENCE orientation, so the sign
      reads directly as "improve together" vs "trade off".

Axis convention follows the rest of the repo (``search_diagnostics``,
``parallel_coordinates``): every axis is min-max normalized and flipped so UP
is the preferred direction, with the raw best/worst values annotated at the
top/bottom of each axis and the direction carried in the tick label from
``style.OBJ_AXIS_LABELS``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plotting.style import (FIGSIZE_WIDE, OBJ_AXIS_LABELS, annotated_corr_heatmap,
                                label_for, save_figure)
from src.solution_selection import dominance_mask, orient_maximize

#: Reference-line colour for the FFMP baseline, matching search_diagnostics.
BASELINE_COLOR = "firebrick"

#: Background colour for the full front.
FRONT_COLOR = "0.72"

#: Highlight colour for a single called-out subset.
HIGHLIGHT_COLOR = "#2a78d6"

#: Qualitative colours for named representative policies (CVD-safe set).
POLICY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9467bd",
                 "#17becf"]


def _format_value(v: float) -> str:
    """Format a raw objective value for an axis end annotation."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


class _ParallelAxes:
    """Shared normalization + axis furniture for the parallel-axes figures.

    Holds the min-max frame spanning every row that will be drawn (front plus
    any reference vector) so highlighted lines and the baseline share one
    scale, and exposes :meth:`normalize` for callers to map raw natural
    vectors onto it.
    """

    def __init__(self, natural_obj: np.ndarray, directions, obj_names,
                 extra_rows=(), figsize: tuple = FIGSIZE_WIDE):
        self.directions = np.asarray(directions)
        self.obj_names = list(obj_names)
        self.n_objs = len(self.obj_names)
        self.x = np.arange(self.n_objs)

        rows = [np.atleast_2d(np.asarray(natural_obj, dtype=float))]
        rows += [np.atleast_2d(np.asarray(r, dtype=float))
                 for r in extra_rows if r is not None]
        stacked = np.vstack([r for r in rows if r.size])
        if stacked.size == 0:
            stacked = np.zeros((1, self.n_objs))
        self.lo = stacked.min(axis=0)
        self.hi = stacked.max(axis=0)
        self._span = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)

        self.fig, self.ax = plt.subplots(figsize=figsize)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Map natural values onto [0, 1] with up = preferred."""
        n = (np.atleast_2d(np.asarray(values, dtype=float)) - self.lo) / self._span
        n[:, self.directions == -1] = 1.0 - n[:, self.directions == -1]
        return n

    def draw(self, values: np.ndarray, **kwargs) -> None:
        """Plot one or more polylines from natural values."""
        for row in self.normalize(values):
            self.ax.plot(self.x, row, **kwargs)

    def finish(self, title: str, legend_loc: str = "lower right") -> Figure:
        """Apply ticks, annotations, title and legend; return the figure."""
        ax = self.ax
        ax.set_xticks(self.x)
        ax.set_xticklabels([OBJ_AXIS_LABELS.get(n, n) for n in self.obj_names],
                           fontsize=8)
        ax.set_ylabel("Preference direction  (↑ better)")
        ax.set_title(title)
        ax.set_ylim(-0.16, 1.14)
        ax.grid(True, alpha=0.3, axis="x")
        for i in range(self.n_objs):
            best, worst = ((self.hi[i], self.lo[i]) if self.directions[i] == 1
                           else (self.lo[i], self.hi[i]))
            ax.text(i, 1.04, _format_value(best), ha="center", va="bottom",
                    fontsize=7, color="0.3")
            ax.text(i, -0.05, _format_value(worst), ha="center", va="top",
                    fontsize=7, color="0.3")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc, fontsize=8, framealpha=0.9)
        self.fig.tight_layout()
        return self.fig


def plot_baseline_dominance(
    natural_obj: np.ndarray,
    directions,
    obj_names,
    baseline: np.ndarray,
    output_file: Path | None = None,
    tol: float = 0.0,
    title: str | None = None,
    figsize: tuple = FIGSIZE_WIDE,
) -> Figure:
    """Parallel axes highlighting the front members that dominate a baseline.

    Dominance is the standard weak definition from
    :func:`src.solution_selection.dominance_mask` — no worse on every
    objective and strictly better on at least one — so ties on an axis do not
    disqualify a solution.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        obj_names: Objective names aligned to the columns.
        baseline: ``(n_objs,)`` reference vector in natural units, aligned to
            ``obj_names`` BY NAME by the caller.
        output_file: Optional path stub (no extension); saved via
            ``style.save_figure``.
        tol: Dominance comparison slack (see ``solution_selection.dominates``).
        title: Figure title; a default naming the counts is built when None.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure. An empty dominating subset is not an error —
        the figure renders with an explicit "0 solutions" annotation.
    """
    natural_obj = np.atleast_2d(np.asarray(natural_obj, dtype=float))
    baseline = np.asarray(baseline, dtype=float).ravel()
    mask = dominance_mask(natural_obj, baseline, directions, tol=tol)
    n_dom, n_total = int(mask.sum()), natural_obj.shape[0]

    pa = _ParallelAxes(natural_obj, directions, obj_names,
                       extra_rows=[baseline], figsize=figsize)
    pa.draw(natural_obj[~mask], color=FRONT_COLOR, alpha=0.10, lw=0.6, zorder=1)
    pa.ax.plot([], [], color=FRONT_COLOR, lw=2,
               label=f"does not dominate baseline (n={n_total - n_dom})")
    if n_dom:
        pa.draw(natural_obj[mask], color=HIGHLIGHT_COLOR, alpha=0.35, lw=0.9,
                zorder=3)
    pa.ax.plot([], [], color=HIGHLIGHT_COLOR, lw=2,
               label=f"dominates baseline (n={n_dom})")
    pa.draw(baseline, color=BASELINE_COLOR, lw=2.5, marker="o", markersize=5,
            zorder=10, label="FFMP baseline")

    if n_dom == 0:
        pa.ax.text(0.5, 0.5, "0 solutions dominate the baseline",
                   transform=pa.ax.transAxes, ha="center", va="center",
                   fontsize=12, color=BASELINE_COLOR,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                             edgecolor=BASELINE_COLOR, alpha=0.9), zorder=20)

    if title is None:
        pct = 100.0 * n_dom / max(1, n_total)
        title = (f"Pareto-approximate set vs FFMP baseline — "
                 f"{n_dom}/{n_total} ({pct:.1f}%) dominate the baseline")
    # Legend below the axes: the front fills every corner, and an in-axes
    # legend covers the worst-value annotations at the axis feet.
    return _finish_and_save(pa, title, output_file, legend_loc="upper center",
                            legend_bbox=(0.5, -0.16))


def plot_selected_policies(
    natural_obj: np.ndarray,
    directions,
    obj_names,
    selections,
    baseline: np.ndarray | None = None,
    output_file: Path | None = None,
    title: str | None = None,
    figsize: tuple = FIGSIZE_WIDE,
) -> Figure:
    """Parallel axes with named representative policies drawn over the front.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        obj_names: Objective names aligned to the columns.
        selections: Sequence of ``src.solution_selection.Selection`` records
            (or anything with ``.label`` and ``.index``). Drawn in order, each
            in its own colour, labelled with its reference-set row index so the
            figure stays traceable back to the ``.set`` file.
        baseline: Optional ``(n_objs,)`` reference vector, drawn bold.
        output_file: Optional path stub (no extension).
        title: Figure title; a default is built when None.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure. An empty ``selections`` renders the bare front
        with an explicit annotation.
    """
    natural_obj = np.atleast_2d(np.asarray(natural_obj, dtype=float))
    selections = list(selections)

    pa = _ParallelAxes(natural_obj, directions, obj_names,
                       extra_rows=[baseline], figsize=figsize)
    pa.draw(natural_obj, color=FRONT_COLOR, alpha=0.10, lw=0.6, zorder=1)
    pa.ax.plot([], [], color=FRONT_COLOR, lw=2,
               label=f"Pareto-approximate set (n={natural_obj.shape[0]})")
    if baseline is not None:
        pa.draw(np.asarray(baseline, dtype=float), color=BASELINE_COLOR,
                lw=2.5, marker="o", markersize=5, zorder=9,
                label="FFMP baseline")
    for i, sel in enumerate(selections):
        pa.draw(natural_obj[sel.index], color=POLICY_COLORS[i % len(POLICY_COLORS)],
                lw=2.0, marker="o", markersize=4, alpha=0.95, zorder=10,
                label=f"{sel.label} (row {sel.index})")
    if not selections:
        pa.ax.text(0.5, 0.5, "0 solutions selected", transform=pa.ax.transAxes,
                   ha="center", va="center", fontsize=12, color=BASELINE_COLOR,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                             edgecolor=BASELINE_COLOR, alpha=0.9), zorder=20)

    if title is None:
        title = (f"Representative policies on the Pareto-approximate set "
                 f"({len(selections)} selected of {natural_obj.shape[0]})")
    return _finish_and_save(pa, title, output_file,
                            legend_loc="upper center",
                            legend_bbox=(0.5, -0.16))


def plot_objective_tradeoffs(
    natural_obj: np.ndarray,
    directions,
    obj_names,
    output_file: Path | None = None,
    flag_threshold: float = 0.7,
    figsize: tuple = (7.6, 6.6),
) -> Figure:
    """Spearman rank-correlation heatmap across the front, in preference units.

    Both objectives of each pair are first oriented so larger is better
    (:func:`src.solution_selection.orient_maximize`), so the sign of rho reads
    directly: **positive = the two objectives improve together across the
    front, negative = they trade off**. Reading correlations in raw natural
    units instead would make the sign depend on whether each objective happens
    to be a reliability or a deficit.

    Only variation ACROSS the front is measured, so a constant objective (no
    rank variation) renders as a grey NaN row rather than a spurious value.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        obj_names: Objective names aligned to the columns.
        output_file: Optional path stub (no extension).
        flag_threshold: ``|rho|`` above which an off-diagonal cell is outlined.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    oriented = np.atleast_2d(orient_maximize(natural_obj, directions))
    df = pd.DataFrame(oriented, columns=list(obj_names))
    rho = df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=figsize)
    im = annotated_corr_heatmap(ax, rho.to_numpy(), list(obj_names),
                                label_fn=label_for,
                                box_threshold=flag_threshold, fontsize=7)
    ax.set_title("Objective trade-off structure across the front\n"
                 "Spearman rho in preference units "
                 "(+ improve together, − trade off)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()
    if output_file is not None:
        save_figure(fig, output_file)
    return fig


def _finish_and_save(pa: _ParallelAxes, title: str, output_file,
                     legend_loc: str = "lower right",
                     legend_bbox: tuple | None = None) -> Figure:
    """Finish a parallel-axes figure, optionally parking the legend outside."""
    fig = pa.finish(title, legend_loc=legend_loc)
    if legend_bbox is not None and pa.ax.get_legend() is not None:
        handles, labels = pa.ax.get_legend_handles_labels()
        pa.ax.legend(handles, labels, loc=legend_loc, bbox_to_anchor=legend_bbox,
                     ncol=min(4, max(1, len(labels))), fontsize=8,
                     frameon=False)
        fig.tight_layout()
    if output_file is not None:
        save_figure(fig, output_file)
    return fig
