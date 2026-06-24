"""
reevaluate.py - Re-simulate Pareto-optimal solutions with the full Pywr-DRB
model and save per-solution HDF5 outputs + objective summary CSV.

Runs independent solutions in parallel via multiprocessing.Pool (spawn
context, so each worker builds its own model instance). On HPC this should
be invoked from a single node; for multi-node re-evaluation, split by seed
and launch one job per seed.

Example
-------
    python -m src.reevaluate --formulation ffmp --seed 1 --njobs 16
    python -m src.reevaluate --formulation ffmp_10 --njobs 32 --max 50
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    OUTPUTS_DIR, OUTPUT_REFERENCE_SETS_DIR, REEVAL_ENSEMBLE_SPEC,
    derive_slug, get_n_vars,
    active_scenario_name, run_output_dir,
)
from src.load.reference_set import load_reference_set
from src.reeval_core import (
    evaluate_solution, reeval_obj_names, reeval_output_dir, reeval_tag,
)


def _evaluate_one(task: tuple) -> tuple[int, list[float] | None, str | None]:
    """Worker: re-evaluate one solution on the common re-eval ensemble.

    Args:
        task: (solution_id, dv_vector, formulation, out_path).

    Returns:
        (solution_id, objectives or None, error message or None).
    """
    solution_id, dv_vector, formulation, out_path = task
    return evaluate_solution(solution_id, dv_vector, formulation, out_path)


def reevaluate(formulation: str,
               seed: int | None = None,
               max_solutions: int = 0,
               njobs: int = 1) -> Path:
    """Re-simulate Pareto solutions for a formulation.

    Args:
        formulation: Formulation / architecture name.
        seed: Optional seed number. If provided, outputs go to a seed
            subdirectory to avoid collision across multi-seed reruns.
        max_solutions: Cap on number of solutions (0 = all).
        njobs: Parallel worker count.

    Returns:
        Path to the summary CSV.
    """
    scenario = active_scenario_name()
    slug = derive_slug(formulation)
    # Reference set: prefer the merged Pareto set written by diagnostics, then
    # the Borg per-seed solution set written directly by the optimizer, then a
    # curated reference_sets/ entry and the legacy formulation-keyed path, so
    # re-evals work whether or not run_diagnostics has merged a reference set.
    sets_dir = run_output_dir(scenario, slug, "sets")
    set_seed = seed if seed is not None else 1
    candidate_refs = [
        sets_dir / f"{slug}_merged.set",
        sets_dir / f"seed_{set_seed:02d}_{slug}.set",
        OUTPUT_REFERENCE_SETS_DIR / f"{slug}.ref",
        OUTPUTS_DIR / "reference_sets" / f"{formulation}.ref",
    ]
    ref_file = next((p for p in candidate_refs if p.exists()), candidate_refs[0])

    # Per-ensemble re-eval output dir, so re-evals on alternative common
    # ensembles never clobber each other.
    reeval_dir = reeval_output_dir(scenario, slug, REEVAL_ENSEMBLE_SPEC, seed)

    n_vars = get_n_vars(formulation)
    dv_data, _ = load_reference_set(ref_file, n_vars)
    n_solutions = dv_data.shape[0]
    if max_solutions > 0:
        n_solutions = min(n_solutions, max_solutions)

    tasks = [
        (i, dv_data[i, :], formulation, reeval_dir / f"solution_{i:04d}.hdf5")
        for i in range(n_solutions)
    ]
    print(f"Re-evaluating {n_solutions} solutions for '{formulation}' "
          f"(scenario='{scenario}') on common ensemble "
          f"'{reeval_tag(REEVAL_ENSEMBLE_SPEC)}' using {njobs} worker(s)...")
    print(f"  reference set: {ref_file}")
    print(f"  outputs:       {reeval_dir}")

    results: list[tuple[int, list[float] | None, str | None]] = []
    if njobs <= 1:
        results = [_evaluate_one(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(njobs) as pool:
            for r in pool.imap_unordered(_evaluate_one, tasks):
                sid, objs, err = r
                if err:
                    print(f"  [FAIL] solution {sid}: {err}")
                else:
                    print(f"  [ok]   solution {sid}")
                results.append(r)

    results.sort(key=lambda r: r[0])
    obj_names = reeval_obj_names()
    rows = []
    for sid, objs, err in results:
        if objs is None:
            rows.append([np.nan] * len(obj_names))
        else:
            rows.append(objs)
    summary_df = pd.DataFrame(rows, columns=obj_names)
    summary_df.index.name = "solution_id"
    summary_csv = reeval_dir / "objectives_summary.csv"
    summary_df.to_csv(summary_csv)
    print(f"Summary -> {summary_csv}")
    return summary_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed number (for output subdir). Omit for combined.")
    parser.add_argument("--max", type=int, default=0,
                        help="Max solutions to evaluate (0 = all).")
    parser.add_argument("--njobs", type=int, default=1,
                        help="Parallel workers (multiprocessing.Pool).")
    args = parser.parse_args()
    reevaluate(args.formulation, args.seed, args.max, args.njobs)


if __name__ == "__main__":
    main()
