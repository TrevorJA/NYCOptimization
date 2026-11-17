"""constant_demand_figures.py - Parallel-axis plots of the merged set re-scored
under a constant 800 MGD NYC demand.

Reads the output of ``constant_demand_reeval.py``
(``outputs/{scenario}/{slug}/reeval/historic_nyc_demand_{demand}/``): the
re-scored merged set (natural units at full precision from
``sets/{slug}_merged.csv``) and the ``baseline`` row of ``policies.csv`` (the
incumbent FFMP policy under the same demand), and draws them with :func:`src.plotting.parallel_coordinates.custom_parallel_coordinates`
(the renderer behind ``plot_parallel_coordinates``) using the short wrapped
axis labels, so the eight axis names stay legible.
The re-scored rows are every policy of the searched set, not a Pareto filter
under the new demand.

Two figures, in ``outputs/figures/{scenario}/{slug}/parallel_coords/``:

    historic_nyc_demand_{demand}_merged_parallel_coords.png
        every policy alike
    historic_nyc_demand_{demand}_merged_parallel_coords_dominance.png
        policies that dominate the baseline under the same demand highlighted,
        the rest faint grey (``src.solution_selection.dominance_mask``: weak
        Pareto dominance, exact comparison, as in explore_results)

Both read full-precision values: the ``.set`` file rounds to 6 significant
digits, which an exact dominance test against the full-precision baseline
would mistake for real differences.

Usage (from repo root, with the historic env file sourced; light, one core):
    srun -n1 python3 scripts/supplemental/constant_demand_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from config import active_scenario_name, derive_slug, figure_dir_for, run_output_dir  # noqa: E402
from src.formulations import get_obj_directions, get_obj_names  # noqa: E402
from src.plotting.parallel_coordinates import (  # noqa: E402
    custom_parallel_coordinates, minmaxs_from_directions, wrap_axis_labels,
)
from src.plotting.style import font_rc, save_figure  # noqa: E402
from src.solution_selection import dominance_mask, n_objectives_beaten  # noqa: E402

DEMAND_TITLES = {
    "constant_max": "constant 800 MGD NYC demand",
    "historical": "time-varying NYC demand (control)",
}


def _render(objs: pd.DataFrame, title: str, out: Path, **kw) -> None:
    """Draw one panel in the shared parallel-axis style and save it."""
    with matplotlib.pyplot.rc_context(font_rc(13)):
        fig, ax = matplotlib.pyplot.subplots(figsize=(15, 6))
        custom_parallel_coordinates(objs, ax=ax, **kw)
        ax.set_title(title, loc="left")
        save_figure(fig, out.with_suffix(""))
        matplotlib.pyplot.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--formulation", default="ffmp")
    ap.add_argument("--demand", choices=tuple(DEMAND_TITLES), default="constant_max")
    args = ap.parse_args()

    scenario = active_scenario_name()
    slug = derive_slug(args.formulation)
    reeval_dir = run_output_dir(scenario, slug, "reeval") / f"historic_nyc_demand_{args.demand}"
    set_csv = reeval_dir / "sets" / f"{slug}_merged.csv"
    if not set_csv.exists():
        raise SystemExit(f"re-scored set not found: {set_csv}; run constant_demand_reeval first")

    pol = pd.read_csv(reeval_dir / "policies.csv")
    base = pol[pol["policy_id"] == "baseline"]
    obj_names = get_obj_names()
    baseline = (base[obj_names].to_numpy(float)[0]
                if len(base) and base[obj_names].notna().all(axis=None) else None)

    directions = np.asarray(get_obj_directions())
    rows = pd.read_csv(set_csv)
    natural = rows[[f"{o}__{args.demand}" for o in obj_names]].to_numpy(float)
    ok = ~np.isnan(natural).any(axis=1)
    natural = natural[ok]
    fig_dir = figure_dir_for(scenario, slug, "parallel_coords")
    stem = f"historic_nyc_demand_{args.demand}_merged_parallel_coords"
    title = (f"Merged set ({slug}, {len(natural)} policies) re-evaluated on the "
             f"historic trace, {DEMAND_TITLES[args.demand]}")
    common = dict(
        axis_labels=wrap_axis_labels(obj_names),
        minmaxs=minmaxs_from_directions(directions),
        baseline=baseline,
        alpha_base=float(np.clip(300.0 / len(natural), 0.10, 0.60)),
        # Style shared with reeval_robustness_parallel_figures.py.
        lw_base=1.8, frame=True, baseline_label="Current FFMP policy",
    )

    out = fig_dir / f"{stem}.png"
    # Single-color panel: lighter lines so 240 overlapping polylines stay readable.
    _render(pd.DataFrame(natural, columns=obj_names), title, out,
            **{**common, "alpha_base": 0.22})
    print(f"[figure] -> {out}")

    if baseline is None:
        print("[figure] no baseline row: dominance figure skipped")
        return
    dom = dominance_mask(natural, baseline, directions)
    beaten = n_objectives_beaten(natural, baseline, directions)
    print(f"[dominance] {int(dom.sum())}/{len(natural)} policies dominate the FFMP "
          f"baseline under {DEMAND_TITLES[args.demand]}; objectives-beaten "
          f"distribution {np.bincount(beaten, minlength=len(obj_names) + 1).tolist()}")
    out = fig_dir / f"{stem}_dominance.png"
    _render(pd.DataFrame(natural, columns=obj_names),
            f"{title}\n{int(dom.sum())} of {len(natural)} policies dominate the "
            f"current FFMP policy (others grey)", out,
            highlight_mask=dom,
            highlight_label="dominates current FFMP policy",
            exclude_label="does not dominate",
            **common)
    print(f"[figure] -> {out}")


if __name__ == "__main__":
    main()
