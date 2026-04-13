"""
reevaluate.py - Re-simulate Pareto-optimal solutions with the full Pywr-DRB
model and save per-solution HDF5 outputs + objective summary CSV.

Runs independent solutions in parallel via multiprocessing.Pool (spawn
context, so each worker builds its own model instance). On HPC this should
be invoked from a single node; for multi-node re-evaluation, split by seed
and launch one job per seed.

Example
-------
    python -m src.reevaluate --formulation rbf --seed 1 --njobs 16
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

from config import OUTPUTS_DIR, get_n_vars, get_obj_names
from src.load.reference_set import load_reference_set
from src.simulation import dvs_to_config, run_simulation_to_disk
from src.objectives import DEFAULT_OBJECTIVES


def _evaluate_one(task: tuple) -> tuple[int, list[float] | None, str | None]:
    """Worker: simulate one solution and compute objectives.

    Args:
        task: (solution_id, dv_vector, formulation, out_path).

    Returns:
        (solution_id, objectives or None, error message or None).
    """
    solution_id, dv_vector, formulation, out_path = task
    try:
        cfg = dvs_to_config(dv_vector, formulation)
        data = run_simulation_to_disk(cfg, out_path)
        objs = list(DEFAULT_OBJECTIVES.compute(data))
        return solution_id, objs, None
    except Exception as e:
        return solution_id, None, f"{type(e).__name__}: {e}"


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
    ref_file = OUTPUTS_DIR / "reference_sets" / f"{formulation}.ref"
    reeval_dir = OUTPUTS_DIR / "reevaluation" / formulation
    if seed is not None:
        reeval_dir = reeval_dir / f"seed_{seed:02d}"
    reeval_dir.mkdir(parents=True, exist_ok=True)

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
          f"using {njobs} worker(s)...")

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
    obj_names = get_obj_names()
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
