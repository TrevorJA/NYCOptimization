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

The parallel-axis figures are drawn by the project's single renderer,
:func:`src.plotting.parallel_coordinates.custom_parallel_coordinates` (the
same one behind the manuscript Pareto-set figure), so every parallel-axis
figure in the repo shares one style: axes normalized with UP preferred, raw
best/worst values at the axis ends, a preference arrow instead of a y-axis,
and no spines.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plotting.parallel_coordinates import (custom_parallel_coordinates,
                                               minmaxs_from_directions,
                                               wrap_axis_labels)
from src.plotting.style import (annotated_corr_heatmap, font_rc, label_for,
                                save_figure)
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

#: Parallel-axis figure size (eight axes across).
PARALLEL_FIGSIZE = (13.5, 5.5)


@plt.rc_context(font_rc(13))
def _parallel_axes(natural_obj, directions, obj_names, figsize, title,
                   output_file, **kwargs) -> Figure:
    """Draw one parallel-axis figure through the shared renderer and save it."""
    frame = pd.DataFrame(natural_obj, columns=list(obj_names))
    fig, ax = custom_parallel_coordinates(
        frame, columns_axes=list(obj_names),
        axis_labels=wrap_axis_labels(obj_names),
        minmaxs=minmaxs_from_directions(directions),
        figsize=figsize, **kwargs)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    if output_file is not None:
        save_figure(fig, output_file)
    return fig


def plot_baseline_dominance(
    natural_obj: np.ndarray,
    directions,
    obj_names,
    baseline: np.ndarray,
    output_file: Path | None = None,
    tol: float = 0.0,
    title: str | None = None,
    figsize: tuple = PARALLEL_FIGSIZE,
) -> Figure:
    """Parallel axes highlighting the front members that dominate a baseline.

    Dominance is the standard weak definition from
    :func:`src.solution_selection.dominance_mask` (no worse on every
    objective and strictly better on at least one), so ties on an axis do not
    disqualify a solution. Non-dominating members are drawn faint grey
    underneath, the way every screened parallel-axis figure in the repo does.

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
        The matplotlib Figure. An empty dominating subset is not an error; the
        legend then reports n = 0.
    """
    natural_obj = np.atleast_2d(np.asarray(natural_obj, dtype=float))
    baseline = np.asarray(baseline, dtype=float).ravel()
    mask = dominance_mask(natural_obj, baseline, directions, tol=tol)
    n_dom, n_total = int(mask.sum()), natural_obj.shape[0]
    if title is None:
        title = (f"Pareto-approximate set vs FFMP baseline "
                 f"({n_dom} of {n_total} dominate the baseline)")
    return _parallel_axes(
        natural_obj, directions, obj_names, figsize, title, output_file,
        highlight_mask=mask, single_color=HIGHLIGHT_COLOR,
        highlight_label="dominates baseline",
        exclude_label="does not dominate baseline",
        alpha_base=0.35, alpha_brush=0.10, lw_base=0.9,
        baseline=baseline, baseline_label="FFMP baseline")


def plot_selected_policies(
    natural_obj: np.ndarray,
    directions,
    obj_names,
    selections,
    baseline: np.ndarray | None = None,
    output_file: Path | None = None,
    title: str | None = None,
    figsize: tuple = PARALLEL_FIGSIZE,
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
        The matplotlib Figure. An empty ``selections`` renders the bare front.
    """
    natural_obj = np.atleast_2d(np.asarray(natural_obj, dtype=float))
    selections = list(selections)
    overlays = [{"values": natural_obj[sel.index],
                 "label": f"{sel.label} (row {sel.index})",
                 "color": POLICY_COLORS[i % len(POLICY_COLORS)]}
                for i, sel in enumerate(selections)]
    if title is None:
        title = (f"Representative policies on the Pareto-approximate set "
                 f"({len(selections)} selected of {natural_obj.shape[0]})")
    return _parallel_axes(
        natural_obj, directions, obj_names, figsize, title, output_file,
        single_color=FRONT_COLOR,
        single_label=f"Pareto-approximate set (n={natural_obj.shape[0]})",
        alpha_base=0.10, lw_base=0.6, overlays=overlays,
        baseline=None if baseline is None else np.asarray(baseline, float),
        baseline_label="FFMP baseline")


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
                                box_threshold=flag_threshold)
    ax.set_title("Objective trade-off structure across the front\n"
                 "Spearman rho in preference units "
                 "(+ improve together, − trade off)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()
    if output_file is not None:
        save_figure(fig, output_file)
    return fig
