"""
robustness_comparison.py - Cross-design policy robustness under the FOCAL
satisficing criterion.

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

#: Colorbar label for the same quantity. The colorbar is not an axis, so it
#: carries no "(max)" direction marker.

#: Character width the parallel-axis labels wrap to. Nine axes across a
#: double-column figure leave ~0.8 in each, so the long-form label -- and even
#: the unwrapped abbreviation -- overprints its neighbours. This is the
#: project's ABBREVIATION convention (``short_label_for``), wrapped; it is not
#: a third naming scheme.
_AXIS_LABEL_WRAP = 11


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

