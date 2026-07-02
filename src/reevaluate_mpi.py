"""
reevaluate_mpi.py - MPI-based re-simulation of Pareto-optimal solutions.

Each MPI rank gets a slice of solutions (via numpy.array_split), runs each
to a per-solution HDF5 file, and computes objectives. Rank 0 gathers the
per-rank results into a single objectives_summary.csv.

Why a separate module from src.reevaluate:
    src.reevaluate uses multiprocessing.Pool, which is single-node only.
    This module uses mpi4py for multi-node scaling on Anvil/Hopper. The
    single-node module is preserved as the fallback path so the simpler
    code path stays maintainable for interactive use.

Realization scaffolding:
    `realization_ids` is accepted but ignored in Phase 1 (deterministic
    single-trace re-eval). Phase 3 will scatter (solution_id, realization_id)
    pairs and combine per-realization HDF5s per solution.

Example
-------
    # Single-rank (mpirun -np 1 or no mpirun at all):
    python -m src.reevaluate_mpi --formulation ffmp --max 8

    # Multi-rank on Anvil:
    mpirun -np 64 --mca pml ob1 --mca btl self,vader,tcp \\
        python -m src.reevaluate_mpi --formulation ffmp
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    OUTPUTS_DIR, OUTPUT_REFERENCE_SETS_DIR, REEVAL_ENSEMBLE_SPEC,
    derive_slug, get_n_vars,
    active_scenario_name, run_output_dir,
)
from src.load.reference_set import load_reference_set
from src.reeval_core import (
    evaluate_solution_raw, persist_reeval_raw, reeval_output_dir, reeval_tag,
)


###############################################################################
# MPI context
###############################################################################

def _get_mpi_context():
    """Return (comm, rank, size). Falls back to (None, 0, 1) without mpi4py."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


###############################################################################
# Per-rank evaluation
###############################################################################

def _evaluate_one(solution_id: int, dv_vector: np.ndarray, formulation: str):
    """Re-evaluate one solution and return its raw per-realization matrix.

    Returns:
        (solution_id, base_matrix | None, base_names | None, error | None).
    """
    return evaluate_solution_raw(solution_id, dv_vector, formulation)


def _resolve_ref_file(slug: str, formulation: str, scenario: str,
                      seed: int = 1) -> Path:
    """Locate the reference set file.

    Prefers the merged Pareto set written by diagnostics, then the Borg
    per-seed solution set written directly by the optimizer, then a curated
    reference_sets/ entry, then the legacy formulation-keyed path.
    """
    sets_dir = run_output_dir(scenario, slug, "sets")
    candidates = [
        sets_dir / f"{slug}_merged.set",
        sets_dir / f"seed_{seed:02d}_{slug}.set",
        OUTPUT_REFERENCE_SETS_DIR / f"{slug}.ref",
        OUTPUTS_DIR / "reference_sets" / f"{formulation}.ref",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


###############################################################################
# Driver
###############################################################################

def reevaluate_mpi(
    formulation: str,
    seed: Optional[int] = None,
    max_solutions: int = 0,
    realization_ids: Optional[List[int]] = None,
) -> Optional[Path]:
    """Re-simulate Pareto solutions across MPI ranks.

    Args:
        formulation: Formulation name (e.g. "ffmp", "ffmp_10").
        seed: Optional seed number. If provided, outputs land under a
            seed_NN subdir to avoid collision across multi-seed reruns.
        max_solutions: Cap on number of solutions (0 = all).
        realization_ids: Phase 3 scaffold; ignored in Phase 1. When set,
            each (solution_id, realization_id) pair becomes a unit of work.

    Returns:
        Path to objectives_summary.csv on rank 0; None on other ranks.
    """
    comm, rank, size = _get_mpi_context()
    is_root = rank == 0

    # Phase 1 ignores realizations; warn if caller passed non-trivial input.
    if realization_ids is not None and is_root:
        print("[reevaluate_mpi] WARN: realization_ids ignored in Phase 1 "
              f"(received {len(realization_ids)} ids).")

    scenario = active_scenario_name()
    slug = derive_slug(formulation)

    # ---- Rank 0: locate reference set, load DVs, set up output dir ----
    if is_root:
        ref_file = _resolve_ref_file(slug, formulation, scenario,
                                     seed=seed if seed is not None else 1)
        if not ref_file.exists():
            raise FileNotFoundError(
                f"Reference set not found for formulation '{formulation}' "
                f"(scenario='{scenario}', slug='{slug}'). Looked at: {ref_file}"
            )
        n_vars = get_n_vars(formulation)
        dv_data, _ = load_reference_set(ref_file, n_vars)
        n_solutions = dv_data.shape[0]
        if max_solutions > 0:
            n_solutions = min(n_solutions, max_solutions)
            dv_data = dv_data[:n_solutions]

        # Per-ensemble re-eval output dir (re-evals on alternative common
        # ensembles never clobber each other).
        reeval_dir = reeval_output_dir(scenario, slug, REEVAL_ENSEMBLE_SPEC, seed)

        print(f"[reevaluate_mpi] scenario={scenario} formulation={formulation} "
              f"slug={slug} n_solutions={n_solutions} ranks={size}")
        print(f"[reevaluate_mpi] common ensemble: "
              f"{reeval_tag(REEVAL_ENSEMBLE_SPEC)}")
        print(f"[reevaluate_mpi] reference: {ref_file}")
        print(f"[reevaluate_mpi] outputs:   {reeval_dir}")
        payload = (dv_data, n_solutions, str(reeval_dir))
    else:
        payload = None

    # ---- Broadcast (DVs + count + output dir) to every rank ----
    if comm is not None:
        payload = comm.bcast(payload, root=0)
    dv_data, n_solutions, reeval_dir_str = payload
    reeval_dir = Path(reeval_dir_str)

    if n_solutions == 0:
        if is_root:
            print("[reevaluate_mpi] reference set is empty; nothing to do.")
        return None

    # Each rank computes its slice via array_split (matches MOEA-FIND).
    all_ids = np.arange(n_solutions, dtype=int)
    rank_ids = list(np.array_split(all_ids, size)[rank])

    # ---- Per-rank evaluation ----
    rank_results: list = []
    t0 = time.time()
    for sid in rank_ids:
        result = _evaluate_one(int(sid), dv_data[sid], formulation)
        rank_results.append(result)
        err = result[3]
        tag = "FAIL" if err else "ok"
        print(f"  [rank {rank:>3} {tag}] solution {sid:04d}"
              + (f"  ({err})" if err else ""), flush=True)
    print(f"  [rank {rank:>3}] done in {time.time() - t0:.1f}s "
          f"({len(rank_ids)} solutions)", flush=True)

    # ---- Gather results to rank 0 ----
    if comm is not None:
        gathered = comm.gather(rank_results, root=0)
    else:
        gathered = [rank_results]

    if not is_root:
        return None

    # Flatten and sort. Each result is (sid, base_matrix|None, base_names|None,
    # err); rows carry their own solution_id so no positional stitching needed.
    flat: list = []
    for chunk in gathered:
        flat.extend(chunk)
    flat.sort(key=lambda r: r[0])

    fail_count = sum(1 for r in flat if r[1] is None)
    summary_csv, raw_path, meta_path = persist_reeval_raw(
        reeval_dir, flat, formulation, n_solutions, seed,
    )

    print(f"[reevaluate_mpi] {n_solutions - fail_count}/{n_solutions} "
          f"solutions ok; {fail_count} failed.")
    print(f"[reevaluate_mpi] raw matrix -> {raw_path}")
    print(f"[reevaluate_mpi] meta       -> {meta_path}")
    print(f"[reevaluate_mpi] summary    -> {summary_csv}")
    return summary_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed number (for output subdir). Omit for combined.")
    parser.add_argument("--max", type=int, default=0,
                        help="Max solutions to evaluate (0 = all).")
    # Accepted for parity with src.reevaluate's CLI; ignored under MPI.
    parser.add_argument("--njobs", type=int, default=1,
                        help="(ignored under MPI; accepted for CLI parity)")
    args = parser.parse_args()
    reevaluate_mpi(args.formulation, seed=args.seed,
                   max_solutions=args.max)


if __name__ == "__main__":
    main()
