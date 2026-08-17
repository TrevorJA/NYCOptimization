"""
robustness_comparison.py - Phase-3 results figures: cross-design policy
robustness under the FOCAL satisficing criterion.

The focal criterion is env-selected: ``NYCOPT_FOCAL_CRITERION`` (default
``compromise``), resolved through
``src.satisficing_criteria.focal_criterion`` -- the whole tranche
re-parameterizes if the focal choice changes, and output filenames carry the
criterion key so runs under different criteria coexist. The footer on every
figure states the focal thresholds explicitly (non-member axes are
unconstrained and collapse to one footer line).

Figures:
  * parallel coordinates of each design's Pareto set (search objectives + a
    ninth robustness axis that also drives the line coloring), incumbent
    polyline, IDENTICAL axis scales across the design panels;
  * robustness exceedance curves (joint Starr under the focal criterion, and
    the smooth mean-fraction-of-criteria secondary score) on shared axes;
  * the regret-vs-robustness plane: focal satisficing against the tolerance-
    laddered no-harm frequency vs the incumbent, with per-design frontiers.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from src import results_data as rd
from src.load.reference_set import load_reference_set
from src.formulations import get_n_vars
from src.satisficing_criteria import focal_criterion
from src.plotting.parallel_coordinates import custom_parallel_coordinates
from src.plotting.regret_summary import pareto_frontier
from src.plotting.layout import WIDTH_DOUBLE_COL, shared_legend
from src.plotting.satisficing_diagnostics import (
    INCUMBENT_LABEL,
    _add_footer,
    _designs,
    _design_legend,
)
from src.plotting.style import (
    INCUMBENT_COLOR,
    axis_label_for,
    design_color,
    design_label,
    save_figure,
    short_label_for,
)

#: Suffix of the merged Pareto ``.set`` file the parallel-coordinates panels
#: read their search-objective vectors from (row index == reeval solution_id).
SET_SUFFIX = os.environ.get("NYCOPT_RESULTS_SET_SUFFIX", "_merged_eps20260812")

#: Axis / colorbar label for the appended robustness axis.
ROBUSTNESS_AXIS = "focal_robustness"
ROBUSTNESS_AXIS_LABEL = "Robustness\n(focal set)\n(max)"

#: Colorbar label for the same quantity. The colorbar is not an axis, so it
#: carries no "(max)" direction marker.
ROBUSTNESS_CBAR_LABEL = "Robustness: fraction of SOWs meeting all focal criteria"

#: Character width the parallel-axis labels wrap to. Nine axes across a
#: double-column figure leave ~0.8 in each, so the long-form label -- and even
#: the unwrapped abbreviation -- overprints its neighbours. This is the
#: project's ABBREVIATION convention (``short_label_for``), wrapped; it is not
#: a third naming scheme.
_AXIS_LABEL_WRAP = 11


def _parallel_axis_label(name: str, direction: str) -> str:
    """Wrapped abbreviation + direction marker, for a dense parallel axis."""
    body = textwrap.fill(short_label_for(name), _AXIS_LABEL_WRAP)
    return f"{body}\n({'max' if direction == 'maximize' else 'min'})"


def _focal_header(focal) -> str:
    return f"Focal satisficing criteria — {focal.label} (all must hold):"


def _natural_front(res: rd.DesignResults) -> np.ndarray:
    """The design's Pareto search objectives in natural units, cube-aligned.

    Loads the merged ``.set`` (objectives stored all-minimized), un-negates
    maximize objectives via the cube's own direction snapshot, and selects the
    rows re-evaluated in the cube (row index == ``solution_id``).

    The slug comes from the loaded result's own path
    (``outputs/{design}/{slug}/reeval/{tag}``), so the reference set can never
    be read from a different run than the cube being plotted.
    """
    slug = res.path.parents[1].name
    set_file = (config.OUTPUTS_DIR / res.design / slug / "sets"
                / f"{slug}{SET_SUFFIX}.set")
    _, obj = load_reference_set(set_file, get_n_vars("ffmp"),
                                n_objs=len(res.raw.obj_names))
    signs = res.raw.direction_signs()          # +1 maximize, -1 minimize
    natural = obj * np.where(signs > 0, -1.0, 1.0)
    if natural.shape[0] < len(res.raw.solution_ids):
        raise ValueError(f"{res.design}: set file has {natural.shape[0]} rows "
                         f"< {len(res.raw.solution_ids)} re-evaluated solutions")
    return natural[np.asarray(res.raw.solution_ids, dtype=int)]


def _incumbent_search_vector(design: str, obj_names) -> np.ndarray:
    """The incumbent's scenario-matched search-objective vector (natural units)."""
    df = pd.read_csv(config.baseline_objectives_csv("ffmp", design))
    return df.iloc[0].reindex(obj_names).to_numpy(dtype=float)


###############################################################################
# P3.1 -- parallel coordinates with a robustness axis
###############################################################################

def fig_parallel_coords_focal(results: dict, out_stub: Path,
                              table_dir: Path) -> dict:
    """Each design's Pareto set on parallel axes, colored by focal robustness.

    Axes are the eight search objectives plus a ninth robustness axis (the
    focal-criterion joint SOW fraction), which also drives the line coloring
    -- so the visual question "which trade-off positions are robust?" is read
    directly. All panels share identical axis ranges; the firebrick polyline
    is the incumbent scored on the same E_test SOWs (scenario-matched search
    objectives + its focal robustness).
    """
    focal = focal_criterion()
    designs = _designs(results)
    first = results[designs[0]].raw
    obj_names = first.obj_names
    cols = list(obj_names) + [ROBUSTNESS_AXIS]
    minmaxs = ["max" if first.directions[n] == "maximize" else "min"
               for n in obj_names] + ["max"]
    labels = [_parallel_axis_label(n, first.directions[n]) for n in obj_names]
    labels += [ROBUSTNESS_AXIS_LABEL]

    # Assemble per-design frames + incumbent vectors, then shared axis ranges.
    frames, baselines = {}, {}
    for d in designs:
        res = results[d]
        thr = rd.criterion_thresholds(res, focal)
        jf = rd.joint_fraction(rd.satisfaction(res.raw, thresholds=thr))
        frames[d] = pd.DataFrame(
            np.column_stack([_natural_front(res), jf]), columns=cols)
        inc_sat = rd.incumbent_satisfaction(res, thresholds=thr)
        inc_j = float(inc_sat.all(axis=1).mean()) if inc_sat is not None else np.nan
        baselines[d] = np.append(
            _incumbent_search_vector(d, obj_names), inc_j)
    stacked = np.vstack([np.vstack([frames[d].to_numpy(), baselines[d]])
                         for d in designs])
    axis_ranges = np.vstack([np.nanmin(stacked, axis=0),
                             np.nanmax(stacked, axis=0)])

    # Column-TRUE width: built at 13.5 in and printed at 7.48 in, every
    # annotation shrank by ~1.8x, which is why the axis end-values were
    # unreadable on the page. Panels are sized here at the width they will
    # actually be reproduced at, so what is legible on screen is legible in
    # print.
    fig, axes = plt.subplots(
        len(designs), 1,
        figsize=(WIDTH_DOUBLE_COL, 2.95 * len(designs)))
    rows = []
    for ax, d in zip(np.atleast_1d(axes), designs):
        n = len(frames[d])
        custom_parallel_coordinates(
            frames[d], columns_axes=cols, axis_labels=labels, minmaxs=minmaxs,
            color_by_continuous=ROBUSTNESS_AXIS,
            zorder_by=ROBUSTNESS_AXIS,
            # Finer z-binning: with ~10 bins the high-robustness lines that
            # carry the message sat inside a bin with far more low-robustness
            # lines and were overdrawn by them.
            zorder_num_classes=40,
            alpha_base=float(np.clip(300.0 / n, 0.10, 0.60)),
            fontsize=9,
            baseline=baselines[d], baseline_label=INCUMBENT_LABEL,
            title=f"{design_label(d)}, n = {n}",
            ax=ax, axis_ranges=axis_ranges,
            add_colorbar=(d == designs[-1]),
            colorbar_label=ROBUSTNESS_CBAR_LABEL,
            # ONE figure-level legend below; the per-panel legend otherwise
            # repeats the incumbent entry in every inter-panel gap.
            add_legend=False,
        )
        rows += [{"design": d, "solution_id": sid, "focal_robustness": v}
                 for sid, v in zip(results[d].raw.solution_ids,
                                   frames[d][ROBUSTNESS_AXIS])]
        rows.append({"design": f"{d}__incumbent", "solution_id": -1,
                     "focal_robustness": float(baselines[d][-1])})
    fig.tight_layout()
    shared_legend(fig, [Line2D([], [], color=INCUMBENT_COLOR, lw=2.5,
                               marker="o", markersize=5,
                               label=INCUMBENT_LABEL)], y=-0.005)
    _add_footer(results, fig, y=-0.05,
                criteria=rd.criterion_thresholds(results[designs[0]], focal),
                criteria_header=_focal_header(focal))

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / f"parallel_coords_{focal.key}.csv",
                              index=False)
    return {"criterion": focal.key}


###############################################################################
# P3.2 -- robustness exceedance curves
###############################################################################

def fig_robustness_cdf_focal(results: dict, out_stub: Path,
                             table_dir: Path) -> dict:
    """Exceedance curves of focal joint satisficing per design, shared axes.

    The strict focal-criterion joint SOW fraction (Starr) across each design's
    Pareto policies: a design whose curve sits to the upper-right yields more
    robust policies at every rank. Firebrick vertical: the incumbent.
    """
    focal = focal_criterion()
    designs = _designs(results)

    rows = []
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for d in designs:
        res = results[d]
        thr = rd.criterion_thresholds(res, focal)
        joint = rd.joint_fraction(rd.satisfaction(res.raw, thresholds=thr))
        xs = np.sort(joint)
        exceed = 1.0 - np.arange(len(xs)) / len(xs)
        ax.step(xs, exceed, where="post", color=design_color(d), lw=1.9)
        rows += [{"design": d, "solution_id": sid, "joint_starr": float(j)}
                 for sid, j in zip(res.raw.solution_ids, joint)]
        if d == designs[0]:
            inc = rd.incumbent_satisfaction(res, thresholds=thr)
            if inc is not None:
                ij = float(inc.all(axis=1).mean())
                ax.axvline(ij, color=INCUMBENT_COLOR, lw=1.6)
                rows.append({"design": "ffmp_incumbent", "solution_id": -1,
                             "joint_starr": ij})

    ax.set_xlabel("Fraction of SOWs meeting all criteria in the focal set jointly")
    ax.set_ylabel("Fraction of the design's policies\nat or above the score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=-0.005)
    ax.grid(color="0.92", lw=0.7)
    ax.set_axisbelow(True)

    fig.legend(handles=_design_legend(results), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _add_footer(results, fig, y=-0.17,
                criteria=rd.criterion_thresholds(results[designs[0]], focal),
                criteria_header=_focal_header(focal))

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / f"robustness_cdf_{focal.key}.csv",
                              index=False)
    return {"criterion": focal.key}


###############################################################################
# P3.3 -- regret vs robustness plane
###############################################################################

def fig_regret_robustness_plane_focal(results: dict, out_stub: Path,
                                      table_dir: Path) -> dict:
    """Focal satisficing robustness against no-harm frequency vs the incumbent.

    x: the focal-criterion joint SOW fraction. y: ``no_harm_freq_tau`` from
    the robustness scorecard -- the fraction of SOWs where NO objective is
    degraded relative to the incumbent beyond the epsilon-anchored tolerance
    ladder tau. Per-design non-dominated frontiers (both axes maximized) trace
    the achievable trade-off; the upper-right corner is jointly robust AND
    harm-free relative to the status quo.
    """
    focal = focal_criterion()
    designs = _designs(results)

    rows = []
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    for d in designs:
        res = results[d]
        thr = rd.criterion_thresholds(res, focal)
        joint = rd.joint_fraction(rd.satisfaction(res.raw, thresholds=thr))
        no_harm = res.scorecard["no_harm_freq_tau"].reindex(
            res.raw.solution_ids).to_numpy(dtype=float)
        color = design_color(d)
        ax.scatter(joint, no_harm, s=16, color=color, alpha=0.4, lw=0, zorder=3)
        keep = np.isfinite(joint) & np.isfinite(no_harm)
        idx = np.flatnonzero(keep)[pareto_frontier(joint[keep], no_harm[keep])]
        order = np.argsort(joint[idx])
        # Marked line: a frontier can degenerate to a single dominating point.
        ax.plot(joint[idx][order], no_harm[idx][order], "-o", color=color,
                lw=2.0, ms=7, mec="0.15", mew=0.9, zorder=4)
        rows += [{"design": d, "solution_id": sid, "joint_starr": float(j),
                  "no_harm_freq_tau": float(h),
                  "on_frontier": bool(i in set(idx))}
                 for i, (sid, j, h) in enumerate(
                     zip(res.raw.solution_ids, joint, no_harm))]

    ax.set_xlabel("Fraction of SOWs meeting all criteria in the focal set jointly")
    ax.set_ylabel("Fraction of SOWs with no objective degraded\n"
                  "vs the FFMP incumbent beyond tolerance τ")
    ax.set_xlim(-0.01, None)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color="0.92", lw=0.7)
    ax.set_axisbelow(True)

    handles = _design_legend(results, incumbent=False)
    handles.append(Line2D([], [], color="0.25", lw=2.0, marker="o", ms=7,
                          mec="0.15", mew=0.9,
                          label="non-dominated frontier (per design)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    _add_footer(results, fig, y=-0.13,
                criteria=rd.criterion_thresholds(results[designs[0]], focal),
                criteria_header=_focal_header(focal))

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        table_dir / f"regret_robustness_plane_{focal.key}.csv", index=False)
    return {"criterion": focal.key}
