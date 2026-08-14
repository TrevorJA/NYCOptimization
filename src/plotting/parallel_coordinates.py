"""
parallel_coordinates.py - Customizable parallel-coordinates plots of Pareto sets.

Core renderer adapted from the Reed Group Figure Library parallel-coordinates
recipe (https://reedgroup.github.io/FigureLibrary/ParallelCoordinatesPlots.html),
extended for this project's conventions:

* axes normalized per objective with the RAW best/worst values annotated at the
  axis ends, oriented so the ideal direction is shared by every axis;
* continuous (colormap + colorbar) or categorical (color dict + legend) line
  coloring, and z-order layering by any axis;
* brushing by per-axis thresholds (grey rectangles mark the excluded region)
  and/or an arbitrary ``highlight_mask`` (e.g. epsilon-archive membership or a
  stakeholder screen), with screened-out rows drawn faint grey underneath;
* the FFMP baseline drawn as a bold reference polyline.

:func:`plot_parallel_coordinates` is the project-facing entry point (loads a
``.set``/``.ref`` file, un-negates Borg objectives, applies the registry labels);
:func:`custom_parallel_coordinates` is the shared renderer for callers that
already hold natural-unit objective arrays.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.formulations import get_obj_names, get_obj_directions, get_n_vars, get_n_objs
from src.load.reference_set import load_reference_set

#: Style for rows screened out by brushing / ``highlight_mask``: faint grey
#: underneath the surviving lines, matching the rest of the repo's figures.
EXCLUDED_COLOR = "0.62"

#: Reference-line colour for the FFMP baseline (matches ``front_overview``).
BASELINE_COLOR = "firebrick"


def minmaxs_from_directions(directions) -> list:
    """Map per-objective direction ints (+1 maximize, -1 minimize) to 'max'/'min'."""
    return ["max" if int(d) == 1 else "min" for d in np.asarray(directions)]


def _format_value(v: float) -> str:
    """Format a raw objective value for an axis-end annotation."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    elif abs(v) >= 1:
        return f"{v:.2f}"
    else:
        return f"{v:.4f}"


def _axis_index(col, columns_axes) -> int:
    """Resolve a column identifier (name or positional index) to an axis index."""
    if isinstance(col, (int, np.integer)):
        return int(col)
    return list(columns_axes).index(col)


def _brush_mask(objs: pd.DataFrame, columns_axes, brushing_dict) -> np.ndarray:
    """Boolean mask of rows satisfying every ``{axis: (threshold, op)}`` criterion."""
    ops = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}
    satisfice = np.ones(len(objs), dtype=bool)
    for col, (threshold, op) in brushing_dict.items():
        if op not in ops:
            raise ValueError(f"Unknown brushing operator {op!r}; use one of {list(ops)}")
        vals = objs[columns_axes[_axis_index(col, columns_axes)]].to_numpy(dtype=float)
        satisfice &= ops[op](vals, threshold)
    return satisfice


def custom_parallel_coordinates(
    objs,
    columns_axes=None,
    axis_labels=None,
    ideal_direction: str = "top",
    minmaxs=None,
    color_by_continuous=None,
    color_palette_continuous: str = "viridis",
    color_by_categorical=None,
    color_dict_categorical: dict = None,
    colorbar_ticks_continuous=None,
    zorder_by=None,
    zorder_num_classes: int = 10,
    zorder_direction: str = "ascending",
    alpha_base: float = 0.45,
    brushing_dict: dict = None,
    alpha_brush: float = 0.08,
    lw_base: float = 1.0,
    fontsize: int = 9,
    figsize: tuple = (13, 5.5),
    single_color: str = "steelblue",
    baseline=None,
    baseline_label: str = "FFMP baseline",
    highlight_mask=None,
    highlight_label: str = "highlighted",
    exclude_label: str = "excluded",
    legend_loc: str = "upper center",
    legend_bbox: tuple = (0.5, -0.02),
    legend_ncol: int = None,
    title: str = None,
    ax: plt.Axes = None,
    save_fig_filename=None,
    axis_ranges=None,
    add_colorbar: bool = True,
):
    """Render a customizable parallel-coordinates plot of natural-unit objectives.

    Every axis is min-max normalized over the rows drawn (baseline included) and
    oriented so ``ideal_direction`` is preferred on all axes, with the raw
    best/worst values annotated at the axis ends. Rows failing ``brushing_dict``
    and/or ``highlight_mask`` are drawn faint grey underneath the survivors.

    Args:
        objs: ``(n_rows, n_cols)`` DataFrame (or array) of RAW natural-unit
            objective values — NOT Borg-negated.
        columns_axes: Columns to draw as axes, in order (default: all columns).
        axis_labels: Display label per axis (default: column names).
        ideal_direction: Preferred end of every axis, ``'top'`` or ``'bottom'``.
        minmaxs: Per-axis optimization direction, ``'max'``/``'min'`` per column
            (default all ``'max'``). See :func:`minmaxs_from_directions`.
        color_by_continuous: Axis (name or index) whose value colors each line
            through ``color_palette_continuous``; adds a horizontal colorbar in
            raw units. Mutually exclusive with ``color_by_categorical``.
        color_palette_continuous: Matplotlib colormap name for continuous coloring.
        color_by_categorical: Column name in ``objs`` (need not be an axis) or a
            per-row array of category labels; colored via ``color_dict_categorical``
            with one legend entry per category.
        color_dict_categorical: ``{category: color}`` for categorical coloring.
        colorbar_ticks_continuous: Optional explicit colorbar ticks (raw units).
        zorder_by: Axis (name or index) that stacks lines by value so the
            best/worst draw on top.
        zorder_num_classes: Number of z-order bins for ``zorder_by``.
        zorder_direction: ``'ascending'`` draws high values of ``zorder_by`` on
            top; ``'descending'`` the reverse.
        alpha_base: Line alpha for rows passing the brush/mask.
        brushing_dict: ``{axis: (threshold, op)}`` with ``op`` in
            ``'<' '<=' '>' '>='`` — rows failing any criterion are greyed out and
            the excluded region of each brushed axis is shaded.
        alpha_brush: Line alpha for greyed-out rows.
        lw_base: Line width for rows passing the brush/mask.
        fontsize: Base font size (labels; annotations use ``fontsize - 2``).
        figsize: Figure size in inches (ignored when ``ax`` is given).
        single_color: Line colour when no coloring scheme is given.
        baseline: Optional ``(n_axes,)`` raw reference vector aligned to
            ``columns_axes``, drawn as a bold marked polyline and included in the
            axis ranges.
        baseline_label: Legend label for the baseline line.
        highlight_mask: Optional per-row boolean array; ``False`` rows are greyed
            out exactly like brush failures (combines with ``brushing_dict`` by AND).
        highlight_label: Legend label for rows passing the mask/brush.
        exclude_label: Legend label for the greyed-out rows.
        legend_loc: Matplotlib legend location.
        legend_bbox: ``bbox_to_anchor`` in axes fraction; the default parks the
            legend below the axis labels. ``None`` keeps the legend inside.
        legend_ncol: Legend columns; default lays entries out horizontally
            (up to 4). Pass 1 for a vertical legend parked outside-right.
        title: Optional axes title.
        ax: Optional existing axes to draw on (for panel figures).
        save_fig_filename: If given, save (300 dpi, tight) and close; otherwise
            return the figure for further composition.
        axis_ranges: Optional ``(2, n_axes)`` raw (lo, hi) per axis, widened to
            include the drawn rows -- pass the same array across several calls
            to give panel figures identical axis (and colorbar) scales.
        add_colorbar: Set False to suppress the per-call colorbar when a panel
            figure shares one colorbar across axes (continuous coloring only).

    Returns:
        ``(fig, ax)`` when not saving, else ``None``.
    """
    assert ideal_direction in ("top", "bottom")
    assert zorder_direction in ("ascending", "descending")
    assert color_by_continuous is None or color_by_categorical is None
    if minmaxs is not None:
        assert all(mm in ("max", "min") for mm in minmaxs)

    objs = objs if isinstance(objs, pd.DataFrame) else pd.DataFrame(np.atleast_2d(objs))
    if columns_axes is None:
        columns_axes = list(objs.columns)
    if axis_labels is None:
        axis_labels = [str(c) for c in columns_axes]
    data = objs[columns_axes].to_numpy(dtype=float)
    n_rows, n_axes = data.shape
    if minmaxs is None:
        minmaxs = ["max"] * n_axes
    if n_rows == 0:
        print("Empty solution set — nothing to plot.")
        return None

    # --- normalization: [0, 1] per axis, ideal_direction preferred on every axis
    baseline = None if baseline is None else np.asarray(baseline, dtype=float).ravel()
    all_rows = data if baseline is None else np.vstack([data, baseline])
    col_min = np.nanmin(all_rows, axis=0)
    col_max = np.nanmax(all_rows, axis=0)
    if axis_ranges is not None:
        ar = np.asarray(axis_ranges, dtype=float)
        col_min = np.minimum(col_min, ar[0])
        col_max = np.maximum(col_max, ar[1])
    col_rng = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    base_norm = (data - col_min) / col_rng  # unflipped: 0 = raw min, 1 = raw max
    flip = np.array([(ideal_direction == "top") != (mm == "max") for mm in minmaxs])
    normed = np.where(flip, 1.0 - base_norm, base_norm)
    tops = np.where(flip, col_min, col_max)      # raw value rendered at y = 1
    bottoms = np.where(flip, col_max, col_min)   # raw value rendered at y = 0
    if baseline is not None:
        b_norm = (baseline - col_min) / col_rng
        b_norm = np.where(flip, 1.0 - b_norm, b_norm)

    # --- survivors of the brush and/or explicit mask
    satisfice = np.ones(n_rows, dtype=bool)
    if highlight_mask is not None:
        satisfice &= np.asarray(highlight_mask, dtype=bool)
    if brushing_dict is not None:
        satisfice &= _brush_mask(objs, columns_axes, brushing_dict)

    # --- per-row colors and z-orders
    if color_by_continuous is not None:
        ci = _axis_index(color_by_continuous, columns_axes)
        cmap = colormaps.get_cmap(color_palette_continuous)
        row_colors = cmap(base_norm[:, ci])  # unflipped, so the colorbar reads raw
    elif color_by_categorical is not None:
        cats = (objs[color_by_categorical].to_numpy()
                if isinstance(color_by_categorical, str)
                and color_by_categorical in objs.columns
                else np.asarray(color_by_categorical))
        row_colors = [color_dict_categorical[c] for c in cats]
    else:
        row_colors = [single_color] * n_rows

    if zorder_by is None:
        zorders = np.full(n_rows, 4)
    else:
        zi = _axis_index(zorder_by, columns_axes)
        xgrid = np.arange(0, 1.001, 1 / zorder_num_classes)
        cmp_vals = base_norm[:, zi][:, None]
        if zorder_direction == "ascending":
            zorders = 4 + (cmp_vals > xgrid[None, :]).sum(axis=1)
        else:
            zorders = 4 + (cmp_vals < xgrid[None, :]).sum(axis=1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    x = np.arange(n_axes)

    # --- shade the excluded region of each brushed axis
    if brushing_dict is not None:
        for col, (threshold, op) in brushing_dict.items():
            j = _axis_index(col, columns_axes)
            t = np.clip((threshold - col_min[j]) / col_rng[j], 0.0, 1.0)
            if flip[j]:
                t = 1.0 - t
            # excluded raw side: high for '<'/'<=', low for '>'/'>='
            excluded_top_end = (op in ("<", "<=")) != bool(flip[j])
            y0, y1 = (t, 1.0) if excluded_top_end else (0.0, t)
            ax.add_patch(Rectangle((j - 0.07, y0), 0.14, y1 - y0,
                                   facecolor="0.82", edgecolor="0.4",
                                   lw=0.8, alpha=0.9, zorder=3))

    # --- solution polylines: excluded faint grey underneath, survivors on top
    for i in range(n_rows):
        if satisfice[i]:
            ax.plot(x, normed[i], c=row_colors[i], alpha=alpha_base,
                    lw=lw_base, zorder=zorders[i])
        else:
            ax.plot(x, normed[i], c=EXCLUDED_COLOR, alpha=alpha_brush,
                    lw=0.8, zorder=2)

    # --- baseline reference polyline
    if baseline is not None:
        ax.plot(x, b_norm, c=BASELINE_COLOR, lw=2.5, marker="o",
                markersize=5, zorder=4 + zorder_num_classes + 2)

    # --- axis furniture: vertical axes, raw end values, labels
    for j in range(n_axes):
        ax.plot([j, j], [0, 1], c="0.25", lw=1.0, zorder=3)
        ax.annotate(_format_value(tops[j]), (j, 1.02), ha="center", va="bottom",
                    fontsize=fontsize - 2, color="0.3", zorder=5)
        ax.annotate(_format_value(bottoms[j]), (j, -0.02), ha="center", va="top",
                    fontsize=fontsize - 2, color="0.3", zorder=5)
        ax.annotate(axis_labels[j], (j, -0.09), ha="center", va="top",
                    fontsize=fontsize - 1, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- direction-of-preference arrow, left of the first axis
    y_from, y_to = (0.15, 0.85) if ideal_direction == "top" else (0.85, 0.15)
    ax.annotate("", xy=(-0.55, y_to), xytext=(-0.55, y_from),
                arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.5))
    ax.text(-0.72, 0.5, "Direction of preference", rotation=90,
            ha="center", va="center", fontsize=fontsize - 1, color="0.2")
    ax.set_xlim(-0.95, n_axes - 1 + 0.4)
    ax.set_ylim(-0.42, 1.12)

    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2)

    # --- legend / colorbar
    handles = []
    n_keep, n_drop = int(satisfice.sum()), int((~satisfice).sum())
    screened = highlight_mask is not None or brushing_dict is not None
    if color_by_categorical is not None:
        handles += [Line2D([0], [0], color=c, lw=3, alpha=min(1.0, alpha_base + 0.2),
                           label=str(lab))
                    for lab, c in color_dict_categorical.items()]
    elif screened and color_by_continuous is None:
        handles.append(Line2D([0], [0], color=single_color, lw=3,
                              label=f"{highlight_label} (n={n_keep})"))
    if screened:
        handles.append(Line2D([0], [0], color=EXCLUDED_COLOR, lw=3,
                              label=f"{exclude_label} (n={n_drop})"))
    if baseline is not None:
        handles.append(Line2D([0], [0], color=BASELINE_COLOR, lw=2.5, marker="o",
                              markersize=5, label=baseline_label))
    if handles:
        kwargs = {"loc": legend_loc, "fontsize": fontsize - 1, "frameon": False,
                  "ncol": legend_ncol or min(4, len(handles))}
        if legend_bbox is not None:
            kwargs["bbox_to_anchor"] = legend_bbox
        ax.legend(handles=handles, **kwargs)

    if color_by_continuous is not None and add_colorbar:
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_clim(vmin=col_min[ci], vmax=col_max[ci])
        cb = fig.colorbar(mappable, ax=ax, orientation="horizontal",
                          shrink=0.35, pad=0.02)
        cb.set_label(axis_labels[ci].replace("\n", " "), fontsize=fontsize - 1)
        if colorbar_ticks_continuous is not None:
            cb.set_ticks(colorbar_ticks_continuous)
        cb.ax.tick_params(labelsize=fontsize - 2)

    if save_fig_filename is not None:
        fig.savefig(save_fig_filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_fig_filename}")
        plt.close(fig)
        return None
    return fig, ax


def plot_parallel_coordinates(
    set_file: Path,
    formulation: str,
    output_file: Path = None,
    baseline_objs: np.ndarray = None,
    figsize: tuple = (13, 5.5),
    keep_mask: np.ndarray = None,
    **kwargs,
):
    """Parallel-coordinates view of a Pareto-approximate ``.set``/``.ref`` file.

    Loads the file, un-negates the Borg-minimized objectives back to natural
    units, applies the registry axis labels, and renders through
    :func:`custom_parallel_coordinates` — so every customization (continuous /
    categorical coloring, ``zorder_by``, ``brushing_dict``, ...) is available
    as a passthrough keyword.

    Args:
        set_file: Path to .set or .ref file (vars + objs, whitespace-delimited).
        formulation: Formulation name (for n_vars and title).
        output_file: Path to save figure. If None, displays interactively.
        baseline_objs: Optional array of baseline objective values (raw, not
            Borg-negated), drawn as a bold reference line.
        figsize: Figure size.
        keep_mask: Optional boolean array aligned to the set rows. Rows where
            False are drawn faint grey (screened out by a stakeholder floor);
            the axis ranges still span every solution so the screen's effect is
            visible. None = all solutions drawn alike.
        **kwargs: Forwarded to :func:`custom_parallel_coordinates`.
    """
    from src.plotting.style import OBJ_AXIS_LABELS, label_for

    n_vars = get_n_vars(formulation)
    _, obj_data = load_reference_set(set_file, n_vars, n_objs=get_n_objs())
    obj_names = get_obj_names()
    directions = get_obj_directions()

    if obj_data.shape[0] == 0:
        print("Empty solution set — nothing to plot.")
        return None

    # Un-negate maximization objectives (Borg stores all-minimized)
    natural = obj_data * np.where(np.asarray(directions) == 1, -1.0, 1.0)

    kwargs.setdefault("highlight_label", "acceptable")
    kwargs.setdefault("exclude_label", "screened out")
    # Adaptive alpha: large epsilon-archives saturate at a fixed alpha, hiding
    # the density structure of the cloud.
    kwargs.setdefault("alpha_base",
                      float(np.clip(300.0 / natural.shape[0], 0.10, 0.60)))
    kwargs.setdefault("title", f"Pareto-approximate set "
                               f"({formulation}, {obj_data.shape[0]} solutions)")
    result = custom_parallel_coordinates(
        pd.DataFrame(natural, columns=obj_names),
        axis_labels=[OBJ_AXIS_LABELS.get(n, label_for(n)) for n in obj_names],
        minmaxs=minmaxs_from_directions(directions),
        baseline=baseline_objs,
        highlight_mask=keep_mask,
        figsize=figsize,
        save_fig_filename=output_file,
        **kwargs,
    )
    if output_file is None and result is not None:
        plt.show()
    return result
