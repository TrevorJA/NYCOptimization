"""
pareto_parallel.py - Manuscript figure 5: per-design Pareto-approximate sets
on parallel axes, colored by ONE search objective.

The objective-results companion to the robustness figures downstream: three
stacked parallel-axis panels (one per search design) drawing every
epsilon-refiltered Pareto policy over the eight search objectives in natural
units, with IDENTICAL axis ranges across panels and the FFMP incumbent as
the bold reference polyline. Lines are colored by a single objective --
:data:`COLOR_BY_OBJECTIVE`, overridable via ``NYCOPT_PARALLEL_COLOR_OBJ`` --
not by robustness; robustness enters the sequence in later figures.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.plotting.layout import shared_legend
from src.plotting.parallel_coordinates import custom_parallel_coordinates
from src.plotting.robustness_comparison import (
    _incumbent_search_vector,
    _natural_front,
)
from src.plotting.satisficing_diagnostics import _designs
from src.plotting.style import (
    INCUMBENT_COLOR,
    label_for,
    save_figure,
    short_label_for,
)

#: The objective whose value colors (and z-orders) every polyline. Edit here
#: or set ``NYCOPT_PARALLEL_COLOR_OBJ`` to recolor without a code change.
COLOR_BY_OBJECTIVE = os.environ.get("NYCOPT_PARALLEL_COLOR_OBJ",
                                    "nyc_storage_min_p01_pct")

#: Base size handed to the renderer, which draws its smallest text (axis-end
#: values) at ``FONTSIZE - 2`` -- so 16 keeps every character at >= 14 pt.
FONTSIZE = 16

#: Panel width in inches. Eight axes at 14 pt annotations need more room than
#: the column-true 7.48 in; sized for legibility during review rounds.
PANEL_WIDTH = 13.5

#: Panel titles use the manuscript's scenario-design names (Table 1); the
#: parenthetical carries the Pareto-set size instead of the acronym.
DESIGN_TITLES = {
    "monte_carlo": "Monte Carlo Sampling",
    "hazard_filling_stationary": "Hazard Filling",
    "historic": "Historical",
}

#: Legend entry for the status-quo polyline: the incumbent is scored on each
#: panel's own search ensemble, so it is scenario-matched to that panel.
INCUMBENT_LEGEND = "Current FFMP policy evaluated in each corresponding scenario"

#: Character width the axis labels wrap to (eight axes across the panel).
_AXIS_LABEL_WRAP = 11


def _axis_label(name: str) -> str:
    """Wrapped abbreviation; the preference arrow already gives direction."""
    return textwrap.fill(short_label_for(name), _AXIS_LABEL_WRAP)


def fig_pareto_parallel_axes(results: dict, out_stub: Path,
                             table_dir: Path) -> dict:
    """Each design's Pareto set on parallel axes, colored by one objective.

    Axes are the eight search objectives (natural units, shared ranges across
    the design panels); the colorbar reads the raw value of
    :data:`COLOR_BY_OBJECTIVE`, and z-order stacks the best values of that
    objective on top. The firebrick polyline is the scenario-matched FFMP
    incumbent.
    """
    designs = _designs(results)
    first = results[designs[0]].raw
    obj_names = list(first.obj_names)
    if COLOR_BY_OBJECTIVE not in obj_names:
        raise KeyError(f"color objective {COLOR_BY_OBJECTIVE!r} is not a "
                       f"search objective; expected one of {obj_names}")
    minmaxs = ["max" if first.directions[n] == "maximize" else "min"
               for n in obj_names]
    labels = [_axis_label(n) for n in obj_names]
    # Best values of the color objective draw on top either way.
    zorder_direction = ("ascending"
                        if first.directions[COLOR_BY_OBJECTIVE] == "maximize"
                        else "descending")

    # Per-design frames + incumbent vectors, then SHARED axis ranges.
    frames, baselines = {}, {}
    for d in designs:
        res = results[d]
        frames[d] = pd.DataFrame(_natural_front(res), columns=obj_names)
        baselines[d] = _incumbent_search_vector(d, obj_names)
    stacked = np.vstack([np.vstack([frames[d].to_numpy(), baselines[d]])
                         for d in designs])
    axis_ranges = np.vstack([np.nanmin(stacked, axis=0),
                             np.nanmax(stacked, axis=0)])

    fig, axes = plt.subplots(
        len(designs), 1, figsize=(PANEL_WIDTH, 3.9 * len(designs)))
    rows = []
    for letter, ax, d in zip("abcdefg", np.atleast_1d(axes), designs):
        n = len(frames[d])
        custom_parallel_coordinates(
            frames[d], columns_axes=obj_names, axis_labels=labels,
            minmaxs=minmaxs,
            color_by_continuous=COLOR_BY_OBJECTIVE,
            zorder_by=COLOR_BY_OBJECTIVE,
            zorder_direction=zorder_direction,
            zorder_num_classes=40,
            alpha_base=float(np.clip(300.0 / n, 0.10, 0.60)),
            lw_base=2.4,
            fontsize=FONTSIZE,
            baseline=baselines[d], baseline_label=INCUMBENT_LEGEND,
            ax=ax, axis_ranges=axis_ranges,
            # The shared colorbar is drawn at FIGURE level below: an
            # ax-attached colorbar steals height from the last panel and
            # overprints its axis labels.
            add_colorbar=False,
            add_legend=False,
        )
        ax.set_title(f"({letter}) Search & evaluation scenario: "
                     f"{DESIGN_TITLES.get(d, d)} "
                     f"(n = {n} Pareto-approximate policies)",
                     loc="left", fontsize=FONTSIZE)
        # Frame flush with the normalized axis extent (first axis to last,
        # 0 to 1); the end-value annotations sit just outside it.
        ax.add_patch(Rectangle((0, 0), len(obj_names) - 1, 1, fill=False,
                               edgecolor="black", lw=1.2, zorder=60,
                               clip_on=False))
        for sid, vec in zip(results[d].raw.solution_ids,
                            frames[d].to_numpy()):
            rows.append({"design": d, "solution_id": int(sid),
                         **dict(zip(obj_names, map(float, vec)))})
        rows.append({"design": f"{d}__incumbent", "solution_id": -1,
                     **dict(zip(obj_names, map(float, baselines[d])))})
    fig.tight_layout()
    # Shared colorbar and legend stack below the panels, anchored in figure
    # coordinates (savefig bbox='tight' keeps them in frame).
    ci = obj_names.index(COLOR_BY_OBJECTIVE)
    mappable = cm.ScalarMappable(cmap=colormaps.get_cmap("viridis"))
    mappable.set_clim(vmin=axis_ranges[0][ci], vmax=axis_ranges[1][ci])
    cax = fig.add_axes([0.36, -0.045, 0.28, 0.012])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label_for(COLOR_BY_OBJECTIVE), fontsize=FONTSIZE - 1)
    cb.ax.tick_params(labelsize=FONTSIZE - 2)
    shared_legend(fig, [Line2D([], [], color=INCUMBENT_COLOR, lw=2.5,
                               marker="o", markersize=5,
                               label=INCUMBENT_LEGEND)],
                  y=-0.105, fontsize=FONTSIZE - 1)

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        table_dir / f"pareto_parallel_axes_{COLOR_BY_OBJECTIVE}.csv",
        index=False)
    return {"color_by": COLOR_BY_OBJECTIVE}
