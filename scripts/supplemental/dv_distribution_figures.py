"""dv_distribution_figures.py - Decision-variable distributions of the merged
set, with the policies that dominate the current FFMP policy highlighted.

The decision-space companion of ``constant_demand_figures.py``'s dominance
parallel-axis figure: every decision variable of the merged set as one row
scaled to its search bounds (:func:`src.plotting.dv_distributions.plot_dv_distributions`),
the whole set as a histogram, the current FFMP policy as a tick, and the
policies that dominate it on the historic trace re-run under ``--demand``
as dots in the highlight colour of that parallel-axis figure.

Reads the merged ``.set`` of the search
(``outputs/{design}/{slug}/sets/{slug}_merged.set``; row == ``set_row``) for
the decision variables, and the output of ``constant_demand_reeval.py``
(``outputs/{design}/{slug}/reeval/historic_nyc_demand_{demand}/``) for the
dominance test: the same one as the parallel-axis figure
(``src.solution_selection.dominance_mask``: weak Pareto dominance, exact
comparison, on the full-precision objectives of ``sets/{slug}_merged.csv``
against the ``baseline`` row of ``policies.csv``).

Outputs, in ``outputs/figures/{design}/{slug}/decision_variables/``:

    historic_nyc_demand_{demand}_merged_dv_distributions_dominance.png
    historic_nyc_demand_{demand}_merged_dv_distributions_dominance.csv
        the plotted values: one row per decision variable with its bounds,
        the FFMP value, the set's quartiles and the dominating policies' values

Usage (repo root, venv active; light, one core):
    python scripts/supplemental/dv_distribution_figures.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src.formulations import (get_baseline_values, get_bounds,  # noqa: E402
                              get_obj_directions, get_obj_names, get_var_names)
from src.plotting import style  # noqa: E402
from src.plotting.dv_distributions import plot_dv_distributions  # noqa: E402
from src.solution_selection import dominance_mask, load_natural_front  # noqa: E402

DEMAND_TITLES = {
    "constant_max": "constant 800 MGD NYC demand",
    "historical": "time-varying NYC demand (control)",
}


def dominating_rows(reeval_dir: Path, slug: str, demand: str, n_rows: int) -> np.ndarray:
    """Mask over the merged set's rows: dominates the FFMP policy under ``demand``.

    Args:
        reeval_dir: ``.../reeval/historic_nyc_demand_{demand}/``.
        slug: The moea slug (names the re-scored set).
        demand: NYC demand of the historic re-run.
        n_rows: Row count of the merged ``.set``.

    Returns:
        Boolean array of length ``n_rows`` (``set_row`` order); a policy whose
        re-run has a missing objective is not flagged.
    """
    obj_names = get_obj_names()
    rows = pd.read_csv(reeval_dir / "sets" / f"{slug}_merged.csv", index_col="set_row")
    pol = pd.read_csv(reeval_dir / "policies.csv")
    base = pol.loc[pol.policy_id == "baseline", obj_names]
    if base.empty or base.isna().any(axis=None):
        sys.exit(f"[dv_distributions] no scored baseline row in {reeval_dir / 'policies.csv'}")
    if sorted(rows.index) != list(range(n_rows)):
        sys.exit(f"[dv_distributions] {reeval_dir.name} re-scores {len(rows)} rows "
                 f"but the merged set has {n_rows}; re-run constant_demand_reeval.")
    objs = rows.loc[range(n_rows), [f"{o}__{demand}" for o in obj_names]].to_numpy(float)
    ok = ~np.isnan(objs).any(axis=1)
    mask = np.zeros(n_rows, dtype=bool)
    mask[ok] = dominance_mask(objs[ok], base.to_numpy(float)[0],
                              np.asarray(get_obj_directions()))
    return mask


def summary_table(dv: np.ndarray, mask: np.ndarray, formulation: str) -> pd.DataFrame:
    """The plotted values, one row per decision variable (native units)."""
    lo, hi = get_bounds(formulation)
    q25, q50, q75 = np.percentile(dv, [25, 50, 75], axis=0)
    table = pd.DataFrame({
        "decision_variable": get_var_names(formulation),
        "lower_bound": lo, "upper_bound": hi,
        "ffmp_value": get_baseline_values(formulation),
        "set_min": dv.min(axis=0), "set_q25": q25, "set_median": q50,
        "set_q75": q75, "set_max": dv.max(axis=0),
    })
    for row in np.flatnonzero(mask):
        table[f"dominating_s{row:04d}"] = dv[row]
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--design", default="historic")
    p.add_argument("--slug", default="ffmp_obj8")
    p.add_argument("--formulation", default="ffmp")
    p.add_argument("--demand", choices=tuple(DEMAND_TITLES), default="constant_max",
                   help="NYC demand of the historic re-run behind the dominance test")
    a = p.parse_args()

    run_dir = config.OUTPUTS_DIR / a.design / a.slug
    set_file = run_dir / "sets" / f"{a.slug}_merged.set"
    reeval_dir = run_dir / "reeval" / f"historic_nyc_demand_{a.demand}"
    for path in (set_file, reeval_dir / "policies.csv"):
        if not path.exists():
            sys.exit(f"[dv_distributions] not found: {path}")

    dv, *_ = load_natural_front(set_file, a.formulation)
    dom = dominating_rows(reeval_dir, a.slug, a.demand, len(dv))
    print(f"[dominance] {int(dom.sum())}/{len(dv)} policies dominate the FFMP "
          f"policy under {DEMAND_TITLES[a.demand]}: rows {np.flatnonzero(dom).tolist()}")

    out = (config.figure_dir_for(a.design, a.slug, "decision_variables")
           / f"historic_nyc_demand_{a.demand}_merged_dv_distributions_dominance")
    style.apply_style()
    fig = plot_dv_distributions(
        dv, a.formulation, dom,
        highlight_label="dominates current FFMP policy",
        title=(f"Decision variables of the merged set ({a.slug}, {len(dv)} policies)\n"
               f"{int(dom.sum())} policies dominate the current FFMP policy on the "
               f"historic trace, {DEMAND_TITLES[a.demand]}"),
        output_file=out)
    plt.close(fig)
    summary_table(dv, dom, a.formulation).to_csv(out.with_suffix(".csv"), index=False)
    print(f"[figure] -> {out}.png\n[table]  -> {out}.csv")


if __name__ == "__main__":
    main()
