"""
dv_distributions.py - Distribution of every decision variable across a policy set.

One row per decision variable, scaled to its search bounds so all rows share
one horizontal axis (lower bound at the left, upper bound at the right), and
grouped by the rule the variables parameterize (the panel titles of
``policy_rules_compact``). Each row shows:

* a mirrored histogram of the whole set (grey; widths scaled per variable, so
  a row reads as the SHAPE of that variable's distribution);
* optionally, a highlighted subset as individual dots (e.g. the policies that
  dominate the current FFMP policy, the highlighted lines of the dominance
  parallel-axis figure), stacked where they coincide, over a thin line
  spanning their range;
* the current FFMP policy's value as a vertical tick.

The native-unit search bounds of each variable are listed at the right of its
row, so the scaled position can be read back to a value.

Where the dots of a row sit together, the highlighted policies agree on that
variable; where they spread over the grey shape, the variable is free.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.formulations import get_baseline_values, get_bounds, get_var_names
from src.plotting.style import (AXIS_COLOR, INCUMBENT_COLOR, save_figure,
                                style_axes)

#: Highlighted-subset colour: the highlight colour of the parallel-axis
#: figures (``parallel_coordinates.custom_parallel_coordinates``), so the same
#: policies wear the same colour in objective space and in decision space.
HIGHLIGHT_COLOR = "steelblue"

#: Fill of the whole-set histograms.
ALL_COLOR = "0.80"

#: Muted ink of the bounds column and the bound guides.
MUTED_COLOR = "0.45"
GUIDE_COLOR = "0.88"

#: Rule groups in display order: ``key -> (title, what the variables are)``.
#: Titles match ``policy_rules_compact.TITLES``.
GROUPS = {
    "div":     ("Diversion curtailments",
                "added reduction per drought level (fraction of allocation)"),
    "zones":   ("Storage-zone boundary curves",
                "plateau shift (fraction of capacity); timing shift (days)"),
    "spill":   ("Spill-mitigation releases",
                "multiplier on the flood-zone release schedule"),
    "release": ("NYC reservoir minimum releases",
                "seasonal multiplier on the release schedule"),
    "tgt":     ("Flow-target reductions",
                "multiplier on the drought-level flow-target factor"),
    "other":   ("Other decision variables", ""),
}

#: Groups drawn in the left column; the rest stack in the right column.
LEFT_GROUPS = ("div", "zones")

_PARTY = {"nyc": "NYC", "nj": "NJ"}
_PLATEAU = {"lower": "low plateau", "upper": "high plateau"}


def _zone(level: str) -> str:
    """'1b' / 'level1b' / 'l1b' -> 'L1b'."""
    return "L" + level.lower().removeprefix("level").removeprefix("l")


def describe_variable(name: str) -> tuple:
    """Rule group and short row label of a decision variable.

    Args:
        name: Decision-variable name of a formulation.

    Returns:
        ``(group_key, row_label, unit_suffix)``; an unrecognized name falls in
        the ``other`` group under its raw name.
    """
    if m := re.fullmatch(r"(nyc|nj)_allocation_reduction_(L\w+)", name):
        return "div", f"{_PARTY[m[1]]}, {_zone(m[2])}", ""
    if m := re.fullmatch(r"zone_vshift_(level\w+?)_(lower|upper)", name):
        return "zones", f"{_zone(m[1])} {_PLATEAU[m[2]]}", ""
    if m := re.fullmatch(r"zone_tshift_(level\w+)", name):
        return "zones", f"{_zone(m[1])} timing", " d"
    if m := re.fullmatch(r"flood_release_scale_(l\w+?)_(\w+)", name):
        return "spill", f"{_zone(m[1])}, {m[2].capitalize()}", ""
    if m := re.fullmatch(r"mrf_profile_scale_(\w+)", name):
        return "release", m[1].capitalize(), ""
    if m := re.fullmatch(r"mrf_target_scale_(\w+?)_(level\w+)", name):
        return "tgt", f"{m[1].capitalize()}, {_zone(m[2])}", ""
    return "other", name, ""


def _bounds_label(lo: float, hi: float, unit: str) -> str:
    """'[lo, hi]' in native units, with a true minus sign."""
    return f"[{lo:g}, {hi:g}]{unit}".replace("-", "\N{MINUS SIGN}")


def _stack_offsets(x_in: np.ndarray, step: float, reach: float) -> np.ndarray:
    """Vertical offsets (inches) that unstack dots lying close together.

    Dots are placed left to right; each takes the level nearest the row's
    centre line that is at least ``reach`` from every dot already placed.
    Each run of mutually close dots is then recentred on the row.

    Args:
        x_in: Dot centres along the row (inches).
        step: Vertical spacing between stack levels (inches).
        reach: Centre-to-centre distance below which two dots collide (inches).

    Returns:
        Offsets from the row's centre line, aligned to ``x_in``.
    """
    order = np.argsort(x_in, kind="stable")
    levels = [0]
    for k in range(1, len(x_in) + 1):
        levels += [k, -k]
    off = np.zeros(len(x_in))
    placed: list[int] = []
    for i in order:
        for lev in levels:
            y = lev * step
            if all(np.hypot(x_in[i] - x_in[j], y - off[j]) >= reach for j in placed):
                off[i] = y
                break
        placed.append(i)
    # Recentre each run of neighbouring dots.
    run = [order[0]] if len(order) else []
    for a, b in zip(order[:-1], order[1:]):
        if x_in[b] - x_in[a] < reach:
            run.append(b)
            continue
        off[run] -= (off[run].max() + off[run].min()) / 2.0
        run = [b]
    if run:
        off[run] -= (off[run].max() + off[run].min()) / 2.0
    return off


def plot_dv_distributions(
    dv: np.ndarray,
    formulation: str = "ffmp",
    highlight_mask: np.ndarray | None = None,
    *,
    all_label: str = "all policies",
    highlight_label: str = "highlighted",
    baseline_label: str = "Current FFMP policy",
    show_baseline: bool = True,
    title: str | None = None,
    n_bins: int = 25,
    output_file: Path | None = None,
) -> Figure:
    """Per-variable distributions of a policy set, with an optional highlighted subset.

    Args:
        dv: ``(n_policies, n_vars)`` decision variables in native units.
        formulation: Formulation name (variable names, bounds, FFMP values).
        highlight_mask: Optional boolean array aligned to the ``dv`` rows; the
            flagged policies are drawn as individual dots over the histograms.
        all_label: Legend label of the whole set (its size is appended).
        highlight_label: Legend label of the highlighted subset (size appended).
        baseline_label: Legend label of the current FFMP policy's tick.
        show_baseline: Mark the current FFMP policy's value on each row.
        title: Optional left-aligned figure title.
        n_bins: Histogram bins across each variable's search bounds.
        output_file: Optional path stub (no extension); saved via
            ``style.save_figure``.

    Returns:
        The matplotlib Figure.
    """
    dv = np.atleast_2d(np.asarray(dv, dtype=float))
    names = get_var_names(formulation)
    lo, hi = get_bounds(formulation)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    nd = (dv - lo) / span
    base_nd = (get_baseline_values(formulation) - lo) / span if show_baseline else None
    mask = (np.asarray(highlight_mask, dtype=bool) if highlight_mask is not None
            else np.zeros(len(nd), dtype=bool))

    # Rows of each rule group, in formulation order.
    rows: dict[str, list] = {key: [] for key in GROUPS}
    for j, name in enumerate(names):
        key, label, unit = describe_variable(name)
        rows[key].append((j, label, _bounds_label(lo[j], hi[j], unit)))
    columns = [[k for k in GROUPS if rows[k] and k in LEFT_GROUPS],
               [k for k in GROUPS if rows[k] and k not in LEFT_GROUPS]]
    columns = [c for c in columns if c]

    # --- layout, in inches: equal row pitch in every group and column
    fs = plt.rcParams["font.size"]
    pitch = 0.36                      # height of one variable's row
    head = 4.2 * fs / 72.0            # title + description band above a group
    strip_w = 4.3                     # width of the scaled-value axis
    label_w = 0.62 * fs / 72.0 * max(len(r[1]) for g in rows.values() for r in g)
    bounds_w = 0.62 * fs / 72.0 * max(len(r[2]) for g in rows.values() for r in g)
    col_w = label_w + strip_w + bounds_w
    gap, margin = 0.55, 0.15
    title_pt = plt.rcParams["figure.titlesize"]
    top = ((title.count("\n") + 1) * 1.35 * title_pt / 72.0 + 0.10) if title else 0.0
    xaxis_h = 4.6 * fs / 72.0         # x ticks, two-line tick labels, axis label
    n_legend = 1 + int(mask.any()) + int(base_nd is not None)
    legend_h = (1.5 * n_legend + 1.0) * fs / 72.0
    col_h = [sum(head + pitch * len(rows[k]) for k in col) for col in columns]
    # The legend goes under the last column where that column is the shorter
    # one (filling the space beside the taller column), else under the figure.
    legend_in_column = len(columns) > 1 and col_h[-1] < max(col_h)
    if legend_in_column:
        bottom = max(xaxis_h, col_h[-1] + xaxis_h + legend_h - max(col_h))
    else:
        bottom = xaxis_h + legend_h
    fig_w = 2 * margin + len(columns) * col_w + (len(columns) - 1) * gap
    fig_h = top + max(col_h) + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    half = 0.42                       # histogram half-height, in row units
    dot_d = 5.8 / 72.0                # highlighted dot diameter (inches)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, width = (edges[:-1] + edges[1:]) / 2.0, edges[1] - edges[0]
    last_axes = []

    for c, col in enumerate(columns):
        x0 = margin + c * (col_w + gap) + label_w
        y_top = fig_h - top
        for g, key in enumerate(col):
            group = rows[key]
            h = pitch * len(group)
            y_top -= head
            ax = fig.add_axes([x0 / fig_w, (y_top - h) / fig_h,
                               strip_w / fig_w, h / fig_h])
            y_top -= h
            style_axes(ax)
            ypos = np.arange(len(group))

            for g_x in (0.0, 0.5, 1.0):
                ax.axvline(g_x, color=GUIDE_COLOR, lw=0.9,
                           ls="-" if g_x != 0.5 else (0, (2, 3)), zorder=0)
            ax.hlines(ypos, 0.0, 1.0, color=ALL_COLOR, lw=0.7, zorder=1)
            for y, (j, _, _) in zip(ypos, group):
                counts, _ = np.histogram(np.clip(nd[:, j], 0.0, 1.0), bins=edges)
                hh = half * counts / max(counts.max(), 1)
                ax.bar(centers, 2 * hh, bottom=y - hh, width=width,
                       color=ALL_COLOR, lw=0, zorder=1)
                if base_nd is not None:
                    ax.plot([base_nd[j]] * 2, [y - half, y + half],
                            color=INCUMBENT_COLOR, lw=2.2, solid_capstyle="butt",
                            zorder=3)
                if mask.any():
                    xs = nd[mask, j]
                    ax.plot([xs.min(), xs.max()], [y, y], color=HIGHLIGHT_COLOR,
                            lw=1.3, alpha=0.55, solid_capstyle="round", zorder=4)
                    off = _stack_offsets(xs * strip_w, step=0.55 * dot_d,
                                         reach=0.55 * dot_d)
                    # The y axis is inverted, so a positive offset is plotted up.
                    ax.scatter(xs, y - off / pitch, s=(dot_d * 72.0) ** 2,
                               facecolor=HIGHLIGHT_COLOR, edgecolor="white",
                               linewidth=0.8, zorder=5, clip_on=False)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(len(group) - 0.5, -0.5)
            ax.set_yticks(ypos)
            ax.set_yticklabels([r[1] for r in group])
            ax.tick_params(axis="y", length=0, pad=6)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.tick_params(axis="x", length=0, labelbottom=False)

            # Native-unit bounds as the tick labels of a right-hand axis.
            axr = ax.twinx()
            axr.set_ylim(ax.get_ylim())
            axr.set_yticks(ypos)
            axr.set_yticklabels([r[2] for r in group], color=MUTED_COLOR)
            axr.tick_params(axis="y", length=0, pad=6)
            for spine in axr.spines.values():
                spine.set_visible(False)

            title_text, what = GROUPS[key]
            ax.set_title(f"{title_text}\n", loc="left", color="0.1", pad=6)
            if what:
                ax.annotate(what, xy=(0.0, 1.0), xycoords="axes fraction",
                            xytext=(0, 4), textcoords="offset points",
                            ha="left", va="bottom", color=MUTED_COLOR)
            if g == 0:
                # Header of the bounds column, on the group title's line.
                axr.annotate("Search bounds", xy=(1.0, 1.0),
                             xycoords="axes fraction",
                             xytext=(6, 6 + 1.05 * fs), textcoords="offset points",
                             ha="left", va="bottom", color=MUTED_COLOR)
        last_axes.append(ax)

    # --- x axis on the lowest group of each column
    for ax in last_axes:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_position(("outward", 4))
        ax.tick_params(axis="x", length=3.5, labelbottom=True, colors=AXIS_COLOR)
        ax.set_xticklabels(["lower\nbound", "midpoint", "upper\nbound"])
        ax.set_xlabel("Value within the variable's search bounds", color="0.1")

    # --- legend and title
    handles = [Patch(facecolor=ALL_COLOR, edgecolor="none",
                     label=f"{all_label} (n={len(nd)}; histogram, "
                           "width scaled per variable)")]
    if mask.any():
        handles.append(Line2D([0], [0], color=HIGHLIGHT_COLOR, lw=1.3, alpha=1.0,
                              marker="o", markersize=dot_d * 72.0,
                              markerfacecolor=HIGHLIGHT_COLOR,
                              markeredgecolor="white", markeredgewidth=0.8,
                              label=f"{highlight_label} (n={int(mask.sum())}; "
                                    "line spans their range)"))
    if base_nd is not None:
        handles.append(Line2D([0], [0], ls="none", marker="|", markersize=13,
                              markeredgewidth=2.2, color=INCUMBENT_COLOR,
                              label=baseline_label))
    if legend_in_column:
        x_leg = margin + (len(columns) - 1) * (col_w + gap)
        y_leg = fig_h - top - col_h[-1] - xaxis_h - 0.4 * fs / 72.0
        fig.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                   bbox_to_anchor=(x_leg / fig_w, y_leg / fig_h),
                   borderaxespad=0.0, handletextpad=0.6)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=2.2,
                   handletextpad=0.6)
    if title:
        fig.suptitle(title, x=margin / fig_w, y=1.0, ha="left", va="top")

    if output_file is not None:
        save_figure(fig, output_file)
    return fig
