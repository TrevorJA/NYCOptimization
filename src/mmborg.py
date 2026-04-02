"""
mmborg.py - Multi-Master Borg MOEA optimization driver.

Uses the passNFE_ALH_PyCheckpoint branch of MMBorgMOEA with the
dict-based Borg constructor and NFE-aware objective function signature.

Compilation (from MMBorgMOEA repo, passNFE_ALH_PyCheckpoint branch):
    mpicc -shared -fPIC -O3 -o libborgmm.so borgmm.c mt19937ar.c -lm

Required files in borg/ directory:
    - borg.py      (Python wrapper from passNFE_ALH_PyCheckpoint branch)
    - libborgmm.so (compiled MM Borg shared library)
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# borg.py lives at the project root (NYCOptimization/borg.py); no borg/ subdir needed

from config import (
    get_n_vars,
    get_n_objs,
    get_bounds,
    get_epsilons,
    get_var_names,
    get_obj_names,
    BORG_SETTINGS,
    OUTPUTS_DIR,
)
from src.simulation import evaluate


def run_mmborg(
    formulation_name: str,
    seed: int,
    n_islands: int,
    max_evaluations: int = None,
    max_time: int = None,
    runtime_frequency: int = None,
    checkpoint_base: str = None,
    restore_checkpoint: str = None,
):
    """Execute Multi-Master Borg optimization.

    Called by all MPI ranks. Only the controller (rank 0) receives results.

    Args:
        formulation_name: Problem formulation name.
        seed: Random seed number (1-indexed).
        n_islands: Number of MM Borg islands.
        max_evaluations: Max NFE per island (default from config).
        max_time: Max wall time in seconds (overrides NFE if set).
        runtime_frequency: NFE interval for runtime snapshots.
        checkpoint_base: Path base for new checkpoint files.
        restore_checkpoint: Path to existing checkpoint file to restore from.
    """
    from borg import Borg, Configuration

    n_vars = get_n_vars(formulation_name)
    n_objs = get_n_objs()

    print(f"[MM-Borg] Pre-MPI setup: {n_vars} vars, {n_objs} objs, "
          f"{n_islands} islands, {max_evaluations} NFE/island", flush=True)

    if max_evaluations is None:
        max_evaluations = BORG_SETTINGS["max_evaluations"]
    if runtime_frequency is None:
        runtime_frequency = BORG_SETTINGS["runtime_frequency"]

    # --- Objective function (passNFE branch passes NFE as second arg) ---
    # IMPORTANT: borg.py's innerFunction only catches KeyboardInterrupt; any
    # other Python exception propagates through the ctypes C→Python boundary
    # and corrupts the GIL state, causing a fatal Python error on the worker.
    # We catch all exceptions here and return a large penalty so Borg keeps
    # running even if a specific DV combination causes a simulation failure.
    _penalty = [1e10] * n_objs

    def objective(vars, NFE):
        try:
            return evaluate(np.array(vars), formulation_name=formulation_name)
        except Exception as e:
            try:
                from mpi4py import MPI
                rank = MPI.COMM_WORLD.Get_rank()
            except Exception:
                rank = -1
            sys.stderr.write(
                f"[Rank {rank}] WARNING: eval exception (returning penalty): "
                f"{type(e).__name__}: {e}\n"
            )
            sys.stderr.flush()
            return _penalty

    # --- Borg instance (dict constructor from passNFE_ALH_PyCheckpoint) ---
    lower, upper = get_bounds(formulation_name)
    borg = Borg(
        numberOfVariables=n_vars,
        numberOfObjectives=n_objs,
        numberOfConstraints=0,
        function=objective,
        epsilons=list(get_epsilons()),
        bounds=[[lo, hi] for lo, hi in zip(lower, upper)],
        seed=seed,
    )

    # --- Output paths ---
    opt_dir = OUTPUTS_DIR / "optimization" / formulation_name
    runtime_dir = opt_dir / "runtime"
    sets_dir = opt_dir / "sets"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)

    # %d is replaced by island index by MM Borg
    runtime_path = str(
        runtime_dir / f"seed_{seed:02d}_{formulation_name}_%d.runtime"
    )

    # --- MPI lifecycle ---
    Configuration.startMPI()
    t_start = time.time()
    print(f"[MM-Borg] MPI started, launching solveMPI...", flush=True)

    solve_kwargs = {
        "maxEvaluations": max_evaluations,
        "islands": n_islands,
        "runtime": runtime_path,
        "frequency": runtime_frequency,
    }
    if max_time is not None:
        solve_kwargs["maxTime"] = max_time
    if checkpoint_base is not None:
        solve_kwargs["newCheckpointFileBase"] = checkpoint_base
    if restore_checkpoint is not None:
        solve_kwargs["oldCheckpointFile"] = restore_checkpoint

    result = borg.solveMPI(**solve_kwargs)

    t_end = time.time()
    if result is not None:
        set_file = sets_dir / f"seed_{seed:02d}_{formulation_name}.set"
        _write_set_file(result, set_file, formulation_name, seed)
        print(f"[Seed {seed}] {result.size()} solutions -> {set_file}", flush=True)
        print(f"[MM-Borg] Total wall time: {t_end - t_start:.1f}s", flush=True)
    else:
        print(f"[MM-Borg] solveMPI completed (worker rank, no result). "
              f"Wall time: {t_end - t_start:.1f}s", flush=True)

    Configuration.stopMPI()


def _write_set_file(result, set_file: Path, formulation_name: str, seed: int):
    """Write Borg archive to a whitespace-delimited .set file."""
    var_names = get_var_names(formulation_name)
    obj_names = get_obj_names()

    with open(set_file, "w") as f:
        f.write(f"# Formulation: {formulation_name}, Seed: {seed}\n")
        f.write(f"# Variables: {','.join(var_names)}\n")
        f.write(f"# Objectives: {','.join(obj_names)}\n")

        for solution in result:
            variables = solution.getVariables()
            objectives = solution.getObjectives()
            line = " ".join(
                [f"{v:.6e}" for v in variables]
                + [f"{o:.6e}" for o in objectives]
            )
            f.write(line + "\n")
