"""Render the cross-design results-figure sequence from the E_test re-evaluation.

The ground-up replacement for the retired ``outputs/figures/comparison/*/
robustness`` set. Phase 1 (this file's initial registry) diagnoses the adopted
joint satisficing criterion: which thresholds bind, how satisficing decomposes
by objective and design, how the joint fraction collapses under conjunction,
and why SOWs are unattainable. Later phases (alternative criterion sets,
cross-design robustness comparisons, factor maps) extend the same registry.

Everything is post-processing on the persisted per-SOW cubes -- no simulation.
The :data:`FIGURES` registry is the source of truth: one builder + one entry
per figure, each writing its PNG and a companion CSV with the exact numbers
(no in-panel annotations, per repo convention) into
``outputs/figures/comparison/{slug}/{kind}/``.

Run through srun/sbatch, never on a login node::

    sbatch workflow/14_results_figures.sh
    python3 -m scripts.main.results_figures --list
    python3 -m scripts.main.results_figures --figure conjunction_collapse

Settings come from the environment, never CLI value flags (repo convention)::

    NYCOPT_REEVAL_TAG      # E_test re-eval tag; defaults to the campaign spec's
                           # tag, so the interim run must set it explicitly
    NYCOPT_RESULTS_SLUG    # moea slug shared by the campaign runs (ffmp_obj8)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src import results_data as rd  # noqa: E402
from src.plotting import criteria_comparison as critcmp  # noqa: E402
from src.plotting import factor_maps as fmaps  # noqa: E402
from src.plotting import robustness_comparison as robcmp  # noqa: E402
from src.plotting import satisficing_diagnostics as satdiag  # noqa: E402
from src.plotting import style  # noqa: E402
from src.reeval_core import reeval_tag as reeval_tag_of  # noqa: E402

#: Figure name -> (builder, figure kind). The kind picks the output subdir via
#: ``config.figure_dir_for("comparison", slug, kind)``.
FIGURES: dict[str, tuple[Callable, str]] = {
    # Phase 1 -- adopted-criterion diagnostics
    "satisficing_decomposition": (satdiag.fig_satisficing_decomposition, "satisficing"),
    "conjunction_collapse":      (satdiag.fig_conjunction_collapse, "satisficing"),
    "threshold_response":        (satdiag.fig_threshold_response, "satisficing"),
    "attainability_blockers":    (satdiag.fig_attainability_blockers, "satisficing"),
    "pairwise_cosatisficing":    (satdiag.fig_pairwise_cosatisficing, "satisficing"),
    # Phase 2 -- alternative criterion sets (src/satisficing_criteria.py)
    "criterion_robustness_matrix": (critcmp.fig_criterion_robustness_matrix, "criteria"),
    "criterion_collapse":          (critcmp.fig_criterion_collapse, "criteria"),
    "drought_flood_split":         (critcmp.fig_drought_flood_split, "criteria"),
    # Phase 3 -- policy robustness under the focal criterion
    # (NYCOPT_FOCAL_CRITERION, default "downstream"; filenames carry the key)
    "parallel_coords_focal":        (robcmp.fig_parallel_coords_focal, "parallel_coords"),
    "robustness_cdf_focal":         (robcmp.fig_robustness_cdf_focal, "robustness_cdf"),
    "regret_robustness_plane_focal": (robcmp.fig_regret_robustness_plane_focal, "robustness"),
    # Phase 4 -- factor maps over the theta DU forcing space
    "factor_maps_theta_focal":      (fmaps.fig_factor_maps_theta_focal, "factor_maps"),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--figure", action="append", choices=sorted(FIGURES),
                   help="render one figure (repeatable); default is --all")
    p.add_argument("--all", action="store_true", help="render every figure")
    p.add_argument("--list", action="store_true", help="list the sequence and exit")
    args = p.parse_args(argv)

    if args.list:
        for name, (fn, kind) in sorted(FIGURES.items()):
            print(f"{name:28s} [{kind}] {(fn.__doc__ or '').splitlines()[0]}")
        return 0

    names = sorted(FIGURES) if (args.all or not args.figure) else args.figure
    tag = os.environ.get("NYCOPT_REEVAL_TAG") or reeval_tag_of(config.REEVAL_ENSEMBLE_SPEC)
    slug = os.environ.get("NYCOPT_RESULTS_SLUG", "ffmp_obj8")

    print(f"[results_figures] slug={slug} tag={tag}")
    results = rd.load_design_results(tag, slug=slug)
    for d, res in results.items():
        s, g, m = res.raw.cube.shape
        print(f"[results_figures]   {d}: {s} solutions x {g} SOWs x {m} "
              f"objectives; incumbent={'yes' if res.incumbent is not None else 'NO'}")

    style.apply_style()
    for name in names:
        fn, kind = FIGURES[name]
        out_dir = config.figure_dir_for("comparison", slug, kind)
        # Companion tables live in the (tag-scoped) comparison tree, never
        # under outputs/figures/ (project rule: figures dirs hold PNGs only).
        table_dir = (config.OUTPUTS_DIR / "comparison" / slug / tag
                     / "figure_tables" / kind)
        table_dir.mkdir(parents=True, exist_ok=True)
        print(f"[results_figures] building {name} -> {out_dir}")
        info = fn(results, out_dir, table_dir)
        print(f"[results_figures]   done {name}: {info} (tables -> {table_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
