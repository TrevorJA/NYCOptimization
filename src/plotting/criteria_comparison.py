"""
criteria_comparison.py - Robustness under the named satisficing criterion sets.

Recomputes joint/decomposed satisficing under the criterion sets of
``src.satisficing_criteria`` (pure post-processing on the per-SOW cubes):
Quinn et al. (2017)-style subsets that each threshold only 1-3 member axes
and leave every other axis unconstrained, plus the all-axes reference
conjunction (always last, visually separated). The spread across sets shows
how much the cross-design robustness landscape -- including the design
ranking -- depends on the stakeholder framing. Every figure carries the
provenance footer plus one explicit bulleted threshold block PER criterion
set (project rule: the exact criteria are always stated on the figure;
``style.criteria_lines`` collapses the unconstrained axes into one line).
Layouts size themselves to however many criterion sets are registered.

Figure conventions follow ``satisficing_diagnostics``: Okabe-Ito design
colors, firebrick incumbent, solid = best policy / open marker = median
policy, no in-panel numeric annotations, one companion CSV per figure written
to the figure-tables tree (never under outputs/figures/).
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src import results_data as rd
from src.satisficing_criteria import ALL_SETS
from src.plotting.satisficing_diagnostics import (
    INCUMBENT_LABEL,
    _designs,
    _design_legend,
    _policies_line,
)
from src.plotting.style import (
    ETEST,
    INCUMBENT_COLOR,
    add_figure_footer,
    criteria_lines,
    design_color,
    save_figure,
    short_label_for,
)


def _criteria_footer(results: dict, fig, *, y_policies: float,
                     y_criteria: float) -> None:
    """Provenance box plus one explicit bulleted criteria box per set."""
    res0 = results[_designs(results)[0]]
    first = res0.raw
    add_figure_footer(fig, [_policies_line(results)], y=y_policies)
    n = len(ALL_SETS)
    fontsize = 6.4 if n <= 4 else 5.9
    for i, cset in enumerate(ALL_SETS):
        thr = rd.criterion_thresholds(res0, cset)
        lines = criteria_lines(thr, first.kinds, obj_order=first.obj_names,
                               header=f"{cset.label}:")
        add_figure_footer(fig, lines, x=(i + 0.5) / n, y=y_criteria,
                          fontsize=fontsize)


def _criterion_grid(n: int, panel_w: float, panel_h: float, ncols: int = 3):
    """A grid of ``n`` shared-axis panels; unused cells hidden.

    Returns ``(fig, panels)`` where ``panels[i]`` is the axes for criterion
    ``i``. Panels with no used panel below them get bottom tick labels.
    """
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             sharex=True, sharey=True, squeeze=False)
    panels = []
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i < n:
            panels.append(ax)
            if i + ncols >= n:                    # nothing used below
                ax.tick_params(labelbottom=True)
        else:
            ax.set_visible(False)
    return fig, panels


###############################################################################
# P2.1 -- criterion x design robustness matrix
###############################################################################

def fig_criterion_robustness_matrix(results: dict, out_stub: Path,
                                    table_dir: Path) -> dict:
    """Joint satisficing under each criterion set's conjunction, by design.

    For each criterion set: the best and median Pareto policy's joint SOW
    fraction per design, plus the incumbent's. The horizontal spread across
    criterion sets shows how much the robustness landscape -- including the
    design ranking -- depends on the criterion framing. The all-axes
    reference set sits last, behind a divider.
    """
    designs = _designs(results)
    xs = np.arange(len(ALL_SETS))
    offsets = np.linspace(-0.25, 0.25, num=len(designs))

    rows = []
    fig, ax = plt.subplots(figsize=(2.4 * len(ALL_SETS) + 1.5, 5.2))
    for di, d in enumerate(designs):
        color = design_color(d)
        res = results[d]
        for xi, cset in enumerate(ALL_SETS):
            thr = rd.criterion_thresholds(res, cset)
            jf = rd.joint_fraction(rd.satisfaction(res.raw, thresholds=thr))
            inc = rd.incumbent_satisfaction(res, thresholds=thr)
            incj = float(inc.all(axis=1).mean()) if inc is not None else np.nan
            x = xs[xi] + offsets[di]
            ax.plot([x, x], [float(np.median(jf)), float(jf.max())],
                    color=color, lw=1.6, alpha=0.5, zorder=2)
            ax.plot(x, float(jf.max()), "o", ms=7, color=color, zorder=4)
            ax.plot(x, float(np.median(jf)), "o", ms=7, mfc="white",
                    mec=color, mew=1.5, zorder=4)
            ax.plot(x, incj, "D", ms=6, color=INCUMBENT_COLOR, zorder=5)
            rows.append({"criterion": cset.key, "design": d,
                         "best_policy": float(jf.max()),
                         "median_policy": float(np.median(jf)),
                         "incumbent": incj,
                         "n_policies_nonzero": int((jf > 0).sum()),
                         "rationale": cset.rationale})

    # The reference conjunction is a diagnostic, not a framing peer: divide it
    # off from the named sets.
    ax.axvline(xs[-1] - 0.5, color="0.55", ls="--", lw=1.0)
    ax.axvspan(xs[-1] - 0.5, xs[-1] + 0.5, color="0.94", zorder=0)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["\n".join(textwrap.wrap(c.label, 16))
                        for c in ALL_SETS], fontsize=8.5)
    ax.set_ylabel(f"Fraction of {ETEST} SOWs meeting\n"
                  "all criteria in the set jointly")
    ax.set_ylim(-0.012, max(0.32, 1.08 * max(r["best_policy"] for r in rows)))
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)

    handles = _design_legend(results, incumbent=False)
    handles += [
        Line2D([], [], marker="o", ls="none", ms=7, color="0.25",
               label="best policy"),
        Line2D([], [], marker="o", ls="none", ms=7, mfc="white", mec="0.25",
               mew=1.5, label="median policy"),
        Line2D([], [], marker="D", ls="none", ms=6, color=INCUMBENT_COLOR,
               label=INCUMBENT_LABEL),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    _criteria_footer(results, fig, y_policies=-0.05, y_criteria=-0.16)

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / "criterion_robustness_matrix.csv",
                              index=False)
    return {"criteria": [c.key for c in ALL_SETS]}


###############################################################################
# P2.2 -- conjunction collapse per criterion set
###############################################################################

def fig_criterion_collapse(results: dict, out_stub: Path,
                           table_dir: Path) -> dict:
    """Conjunction-collapse curves under each criterion set (one panel each).

    Same fixed conjunction order as the phase-1 collapse figure, so panels are
    directly comparable: what changes between panels is only WHERE each
    criterion's thresholds sit, not the order axes are conjoined.
    """
    designs = _designs(results)
    order = list(rd.COLLAPSE_ORDER)
    depths = np.arange(1, len(order) + 1)
    obj_names = results[designs[0]].raw.obj_names

    tables = []
    fig, panels = _criterion_grid(len(ALL_SETS), 5.0, 3.9)
    for ax, cset in zip(panels, ALL_SETS):
        for d in designs:
            res = results[d]
            thr = rd.criterion_thresholds(res, cset)
            sat = rd.satisfaction(res.raw, thresholds=thr)
            curve = rd.collapse_curve(sat, obj_names, order)
            curve.insert(0, "design", d)
            curve.insert(0, "criterion", cset.key)
            tables.append(curve)
            color = design_color(d)
            ax.plot(depths, curve["best_policy"], "-o", ms=3.5, color=color,
                    zorder=4)
            ax.plot(depths, curve["any_policy"], "--", lw=1.2, color=color,
                    alpha=0.85, zorder=3)
            if d == designs[0]:
                inc = rd.incumbent_satisfaction(res, thresholds=thr)
                if inc is not None:
                    icurve = rd.collapse_curve(inc[np.newaxis], obj_names, order)
                    icurve.insert(0, "design", "ffmp_incumbent")
                    icurve.insert(0, "criterion", cset.key)
                    tables.append(icurve)
                    ax.plot(depths, icurve["best_policy"], "-D", ms=3.5,
                            color=INCUMBENT_COLOR, lw=1.6, zorder=5)
        ax.set_title(cset.label, fontsize=10,
                     fontstyle="italic" if cset.reference else "normal")
        if cset.reference:
            ax.set_facecolor("0.965")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color="0.92", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xticks(depths)
        ax.set_xticklabels([short_label_for(n) for n in order],
                           rotation=35, ha="right", fontsize=7)
        if panels.index(ax) % 3 == 0:
            ax.set_ylabel("Fraction of SOWs meeting\nall conjoined criteria")

    handles = _design_legend(results, incumbent=False)
    handles += [
        Line2D([], [], color="0.25", ls="-", marker="o", ms=3.5,
               label="best single policy"),
        Line2D([], [], color="0.25", ls="--", label="any policy (ceiling)"),
        Line2D([], [], color=INCUMBENT_COLOR, ls="-", marker="D", ms=3.5,
               label=INCUMBENT_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    _criteria_footer(results, fig, y_policies=-0.08, y_criteria=-0.15)

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.concat(tables, ignore_index=True).to_csv(
        table_dir / "criterion_collapse.csv", index=False)
    return {"criteria": [c.key for c in ALL_SETS]}


###############################################################################
# P2.3 -- the wet-dry pincer: low-flow-side vs flood-side satisficing
###############################################################################

def fig_drought_flood_split(results: dict, out_stub: Path,
                            table_dir: Path) -> dict:
    """Each policy's supply/low-flow-side vs flood-side satisficing.

    x: joint SOW fraction over the SET'S non-flood member criteria
    (deliveries, Decree flows, storage -- non-member axes are non-binding);
    y: SOW fraction meeting the flood criterion alone. For a set with no
    flood member axis, y is scored at the adopted flood threshold instead --
    a diagnostic overlay, not a member criterion, flagged on the panel. The
    empty upper-right corner IS the structural wet-dry pincer: policies can
    buy low-flow robustness or flood robustness, and the frontier shows the
    exchange rate under each criterion framing.
    """
    designs = _designs(results)
    obj_names = results[designs[0]].raw.obj_names
    flood = "downstream_flood_exceedance_annual"
    k_flood = obj_names.index(flood)
    dry_idx = [i for i, n in enumerate(obj_names) if n != flood]

    rows = []
    fig, panels = _criterion_grid(len(ALL_SETS), 4.4, 4.0)
    for ax, cset in zip(panels, ALL_SETS):
        flood_in_set = cset.reference or flood in cset.axes
        for d in designs:
            res = results[d]
            thr = rd.criterion_thresholds(res, cset)
            if not flood_in_set:
                # y-axis diagnostic overlay: the adopted flood threshold.
                thr[flood] = res.raw.thresholds[flood]
            sat = rd.satisfaction(res.raw, thresholds=thr)
            # dry_idx spans every non-flood axis, but non-member axes are
            # non-binding, so this is the SET'S non-flood conjunction.
            x = sat[:, :, dry_idx].all(axis=2).mean(axis=1)
            y = sat[:, :, k_flood].mean(axis=1)
            ax.scatter(x, y, s=14, color=design_color(d), alpha=0.45, lw=0,
                       zorder=3)
            rows += [{"criterion": cset.key, "design": d, "solution_id": sid,
                      "lowflow_joint": float(xv), "flood_frac": float(yv),
                      "flood_axis_in_set": flood_in_set}
                     for sid, xv, yv in zip(res.raw.solution_ids, x, y)]
            if d == designs[0]:
                inc = rd.incumbent_satisfaction(res, thresholds=thr)
                if inc is not None:
                    ix = float(inc[:, dry_idx].all(axis=1).mean())
                    iy = float(inc[:, k_flood].mean())
                    ax.plot(ix, iy, "D", ms=7, color=INCUMBENT_COLOR, zorder=5)
                    rows.append({"criterion": cset.key,
                                 "design": "ffmp_incumbent", "solution_id": -1,
                                 "lowflow_joint": ix, "flood_frac": iy,
                                 "flood_axis_in_set": flood_in_set})
        ax.set_title(cset.label, fontsize=10,
                     fontstyle="italic" if cset.reference else "normal")
        if cset.reference:
            ax.set_facecolor("0.965")
        if not flood_in_set:
            ax.text(0.02, 0.02, "flood axis not in set:\ny at the adopted "
                    "threshold (diagnostic)", transform=ax.transAxes,
                    fontsize=6.6, color="0.35", va="bottom")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color="0.92", lw=0.7)
        ax.set_axisbelow(True)
        if panels.index(ax) % 3 == 0:
            ax.set_ylabel("Fraction of SOWs meeting\nthe flood criterion")
        if panels.index(ax) + 3 >= len(panels):
            ax.set_xlabel("Fraction of SOWs meeting the set's\n"
                          "non-flood criteria jointly")

    handles = _design_legend(results)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()
    _criteria_footer(results, fig, y_policies=-0.075, y_criteria=-0.14)

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / "drought_flood_split.csv", index=False)
    return {"n_rows": len(rows)}
