"""reeval_robustness_parallel_figures.py - Parallel axes of one design's
Pareto set with an E_test robustness axis that also colors the lines.

Axes: the eight objectives (natural units) plus a final robustness axis =
the focal criterion set's joint satisficing fraction across the E_test SOWs
(``sat_set__<criterion>`` in ``robustness_scorecard_criteria.csv``). Lines
are colored and z-ordered by robustness. Reuses the project renderer
(:func:`src.plotting.parallel_coordinates.custom_parallel_coordinates`) and
dominance test (:func:`src.solution_selection.dominance_mask`).

Objective sources (``--demand``):
    (none)        the search scores, from ``sets/{slug}{--set-suffix}.set``;
                  incumbent = the scenario-matched baseline objectives.
    constant_max  the historic trace re-run under a constant 800 MGD NYC
                  demand (``reeval/historic_nyc_demand_constant_max/``,
                  from constant_demand_reeval.py); incumbent = its
                  ``baseline`` row. Robustness stays the E_test value.

``--dominates-baseline`` colors only policies that weakly Pareto-dominate the
incumbent in the plotted objective space; the rest are faint grey.

Outputs (figure + companion CSV of the plotted values) go to
``outputs/figures/{design}/{slug}/parallel_coords/robustness_{tag}/``:
    parallel_robustness_{criterion}[_{demand}][_dominating].png / .csv

Usage (repo root, venv active; light, one core):
    python scripts/supplemental/reeval_robustness_parallel_figures.py --all
    python scripts/supplemental/reeval_robustness_parallel_figures.py \
        --demand constant_max --dominates-baseline [--criterion compromise]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src import results_data as rd  # noqa: E402
from src.formulations import get_n_vars  # noqa: E402
from src.load.reference_set import load_reference_set  # noqa: E402
from src.plotting.parallel_coordinates import (  # noqa: E402
    custom_parallel_coordinates, wrap_axis_labels)
from src.plotting.robustness_comparison import _incumbent_search_vector  # noqa: E402
from src.plotting.style import INCUMBENT_COLOR, font_rc, save_figure  # noqa: E402
from src.satisficing_criteria import criterion_by_key  # noqa: E402
from src.solution_selection import dominance_mask  # noqa: E402


def search_front(res, slug: str, set_suffix: str) -> np.ndarray:
    """Search objectives (natural units) of the re-evaluated rows of the set."""
    set_file = (config.OUTPUTS_DIR / res.design / slug / "sets"
                / f"{slug}{set_suffix}.set")
    _, obj = load_reference_set(set_file, get_n_vars("ffmp"),
                                n_objs=len(res.raw.obj_names))
    natural = obj * np.where(res.raw.direction_signs() > 0, -1.0, 1.0)
    return natural[np.asarray(res.raw.solution_ids, dtype=int)]


def render(res, a, demand: str | None, dominating: bool, out_dir: Path) -> None:
    obj_names = list(res.raw.obj_names)
    cset = criterion_by_key(a.criterion)
    sol_ids = np.asarray(res.raw.solution_ids, int)

    if demand:
        cd_dir = res.path.parent / f"historic_nyc_demand_{demand}"
        cd = pd.read_csv(cd_dir / "sets" / f"{a.slug}_merged.csv",
                         index_col="set_row")          # set_row == solution_id
        objs = cd.loc[sol_ids, [f"{o}__{demand}" for o in obj_names]].to_numpy(float)
        pol = pd.read_csv(cd_dir / "policies.csv")
        inc_obj = pol.loc[pol.policy_id == "baseline", obj_names].to_numpy(float)[0]
        source = f"historic trace re-run with NYC demand = {demand}"
    else:
        objs = search_front(res, a.slug, a.set_suffix)
        inc_obj = _incumbent_search_vector(res.design, obj_names)
        source = "search objectives"

    front = pd.DataFrame(objs, columns=obj_names)
    crit = pd.read_csv(res.path / "robustness_scorecard_criteria.csv",
                       index_col="solution_id")
    rob_col = f"sat_set__{a.criterion}"
    front[rob_col] = crit.loc[sol_ids, rob_col].to_numpy()

    axes = obj_names + [rob_col]
    minmaxs = ["max" if res.raw.directions[n] == "maximize" else "min"
               for n in obj_names] + ["max"]
    labels = wrap_axis_labels(obj_names) + ["Robust-\nness"]

    baseline = None
    inc_sat = rd.incumbent_satisfaction(res, rd.criterion_thresholds(res, cset))
    if inc_sat is not None:
        baseline = np.append(inc_obj, float(np.nanmean(inc_sat.all(axis=1))))

    mask = None
    if dominating:
        dirs = np.where(np.array(minmaxs[:-1]) == "max", 1, -1)
        mask = dominance_mask(front[obj_names].to_numpy(float), inc_obj, dirs)

    stem = (f"parallel_robustness_{a.criterion}"
            + (f"_{demand}" if demand else "")
            + ("_dominating" if dominating else ""))
    n_sow = len(res.raw.sow_labels)

    with plt.rc_context(font_rc(13)):
        fig, ax = plt.subplots(figsize=(15, 6))
        custom_parallel_coordinates(
            front, columns_axes=axes, axis_labels=labels, minmaxs=minmaxs,
            color_by_continuous=rob_col, zorder_by=rob_col,
            zorder_direction="ascending", zorder_num_classes=40,
            alpha_base=0.6, lw_base=1.8, frame=True, ax=ax,
            baseline=baseline, highlight_mask=mask, add_legend=False,
            colorbar_label=(f"Fraction of {n_sow} E_test SOWs satisfying "
                            f"'{cset.label}'"),
        )
        if baseline is not None:
            ax.legend(handles=[Line2D([], [], color=INCUMBENT_COLOR, lw=2.5,
                                      marker="o", ms=5,
                                      label="Current FFMP policy")],
                      loc="lower right", bbox_to_anchor=(1.0, 1.08),
                      frameon=False)
        if mask is None:
            head = f"{res.design}: {len(front)} Pareto-approximate policies"
        else:
            head = (f"{res.design}: {int(mask.sum())} of {len(front)} policies "
                    f"dominate the current FFMP policy (others grey)")
        ax.set_title(f"{head}\nObjective axes: {source}; color/last axis: "
                     f"E_test robustness ({a.tag})", loc="left")
        save_figure(fig, out_dir / stem)
        plt.close(fig)

    table = front.copy()
    table.insert(0, "solution_id", sol_ids)
    if mask is not None:
        table["dominates_incumbent"] = mask
    if baseline is not None:
        inc_row = dict(zip(axes, baseline), solution_id=-1)
        if mask is not None:
            inc_row["dominates_incumbent"] = False
        table = pd.concat([table, pd.DataFrame([inc_row])], ignore_index=True)
    table.to_csv(out_dir / f"{stem}.csv", index=False)

    r = front[rob_col]
    msg = (f"[{stem}] {len(front)} policies, robustness "
           f"{r.min():.3f}/{r.median():.3f}/{r.max():.3f} (min/med/max)")
    if baseline is not None:
        msg += f"; incumbent {baseline[-1]:.3f}"
    if mask is not None:
        msg += (f"; {int(mask.sum())} dominate the incumbent: "
                f"{sol_ids[mask].tolist()}")
    print(msg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--design", default="historic")
    p.add_argument("--slug", default="ffmp_obj8")
    p.add_argument("--tag", default="etest_kn_50yr_n12500")
    p.add_argument("--criterion", default="compromise",
                   help="nyc_supply | downstream_flows | flood | compromise | "
                        "reference_all8")
    p.add_argument("--set-suffix", default="_merged",
                   help="merged .set the re-eval scored (row == solution_id)")
    p.add_argument("--demand", default=None,
                   help="e.g. constant_max; omit for the search objectives")
    p.add_argument("--dominates-baseline", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="render search + constant_max, each all/dominating")
    a = p.parse_args()

    res = rd.load_design_results(a.tag, slug=a.slug, designs=[a.design])[a.design]
    out_dir = config.figure_dir_for(a.design, a.slug, "parallel_coords") / f"robustness_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ([(d, dom) for d in (None, "constant_max") for dom in (False, True)]
                if a.all else [(a.demand, a.dominates_baseline)])
    for demand, dom in variants:
        render(res, a, demand, dom, out_dir)
    print(f"figures + tables -> {out_dir}")


if __name__ == "__main__":
    main()
