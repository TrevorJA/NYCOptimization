"""
mmborg.py - Multi-Master Borg MOEA optimization driver.

Uses the passNFE_ALH_PyCheckpoint branch of MMBorgMOEA with the
dict-based Borg constructor and NFE-aware objective function signature.

Compilation (from MMBorgMOEA repo, passNFE_ALH_PyCheckpoint branch):
    mpicc -shared -fPIC -O3 -o lib/borg/libborgmm.so \\
        lib/borg/borgmm.c lib/borg/mt19937ar.c -lm

Required files in lib/borg/:
    - borg.py      (Python wrapper from passNFE_ALH_PyCheckpoint branch)
    - libborg.so   (serial Borg shared library)
    - libborgmm.so (compiled MM Borg shared library)
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BORG_DIR, get_epsilons, run_output_dir
from src.moea_config import MOEAConfig
from src.formulations import (
    get_n_vars,
    get_n_objs,
    get_n_constrs,
    get_bounds,
    get_var_names,
    get_obj_names,
    make_objective_function,
    make_constraint_function,
)


def run_mmborg(
    formulation_name: str,
    seed: int,
    scenario: str,
    moea_config: MOEAConfig,
    slug: str,
    checkpoint_base: str = None,
    restore_checkpoint: str = None,
):
    """Execute Multi-Master Borg optimization.

    Called by all MPI ranks. Only the controller (rank 0) receives results.

    All algorithm settings (islands, NFE, runtime frequency, wall time) come
    from ``moea_config`` — there are no value-carrying overrides. Outputs land
    under ``outputs/{scenario}/{slug}/``.

    Args:
        formulation_name: Problem formulation name (drives DV bounds & objectives).
        seed: Random seed number (1-indexed).
        scenario: Scenario-design name (top-level output partition).
        moea_config: Algorithm-settings bundle (see src/moea_config.py).
        slug: The moea slug (inner output partition + filename prefix), from
            ``config.derive_slug(formulation)``.
        checkpoint_base: Path base for new checkpoint files.
        restore_checkpoint: Path to existing checkpoint file to restore from.

    Raises:
        ValueError: If ``moea_config`` leaves a required algorithm setting unset
            (e.g. the ``production`` config before its numbers are decided).
    """
    n_islands = moea_config.n_islands
    max_evaluations = moea_config.max_evaluations
    runtime_frequency = moea_config.runtime_frequency
    max_time = moea_config.max_time_seconds

    missing = [
        n for n, v in (
            ("n_islands", n_islands),
            ("max_evaluations", max_evaluations),
            ("runtime_frequency", runtime_frequency),
        ) if v is None
    ]
    if missing:
        raise ValueError(
            f"MOEA config '{moea_config.name}' is missing required settings "
            f"{missing}. Select a fully-specified config (e.g. 'smoke') or fill "
            f"in the TBD values in src/moea_config.py."
        )

    # Scenario design must have a wired search ensemble to optimize.
    from config import SEARCH_ENSEMBLE_SPEC
    if SEARCH_ENSEMBLE_SPEC is None:
        raise ValueError(
            f"Scenario design '{scenario}' did not resolve a search ensemble, so "
            f"optimization cannot run under it. Every ensemble design needs its "
            f"staged data first: run workflow step 02 (forcing master; also the "
            f"resolution step for the fixed/resampled probabilistic and "
            f"input_stratified designs) and, for hazard_filling*, step 03 "
            f"(reduced subsample). Only 'historic' and the supplemental "
            f"'scaling_stationary' design resolve without staging."
        )

    # borg.py loads ./libborg.so and ./libborgmm.so relative to CWD, so
    # cd into lib/borg/ for the import + MPI initialization, then restore.
    _saved_cwd = os.getcwd()
    os.chdir(str(BORG_DIR))
    from borg import Borg, Configuration
    Configuration.startMPI()
    os.chdir(_saved_cwd)
    print(f"[MM-Borg] MPI started", flush=True)

    n_vars = get_n_vars(formulation_name)
    n_objs = get_n_objs()
    n_constrs = get_n_constrs()

    print(f"[MM-Borg] Pre-MPI setup: scenario={scenario}, slug={slug}, "
          f"formulation={formulation_name}, {n_vars} vars, {n_objs} objs, "
          f"{n_constrs} constrs, {n_islands} islands, "
          f"{max_evaluations} NFE/island", flush=True)

    # --- Objective function (passNFE branch passes NFE as second arg) ---
    # Uses make_objective_function() for "ffmp" / "ffmp_N" formulations,
    # returning (objectives, constraints) tuples. Constraint violations are
    # pure DV arithmetic computed BEFORE any Pywr-DRB simulation; infeasible
    # vectors skip simulation entirely and return penalty objectives —
    # constraint-dominance precedes Pareto dominance in Borg, so their
    # objective values are never consulted against feasible solutions.
    # Constraints must be plain Python lists (borg.py truth-tests them).
    #
    # IMPORTANT: borg.py's innerFunction only catches KeyboardInterrupt; any
    # other Python exception propagates through the ctypes C→Python boundary
    # and corrupts the GIL state, causing a fatal Python error on the worker.
    # We catch all exceptions here and return a large penalty so Borg keeps
    # running even if a specific DV combination causes a simulation failure.
    _eval_fn = make_objective_function(formulation_name)
    _constraint_fn = make_constraint_function(formulation_name)
    _penalty = [1e10] * n_objs

    def objective(vars, NFE):
        try:
            cons = _constraint_fn(np.array(vars))
            if any(c > 0.0 for c in cons):
                return (_penalty, cons)
            return (_eval_fn(np.array(vars)), cons)
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
            return (_penalty, [0.0] * n_constrs)

    # --- Borg instance (dict constructor from passNFE_ALH_PyCheckpoint) ---
    lower, upper = get_bounds(formulation_name)
    borg = Borg(
        numberOfVariables=n_vars,
        numberOfObjectives=n_objs,
        numberOfConstraints=n_constrs,
        function=objective,
        epsilons=list(get_epsilons()),
        bounds=[[lo, hi] for lo, hi in zip(lower, upper)],
        seed=seed,
    )

    # --- Output paths: outputs/{scenario}/{slug}/{runtime,sets}/ ---
    runtime_dir = run_output_dir(scenario, slug, "runtime")
    sets_dir = run_output_dir(scenario, slug, "sets")

    # %d is replaced by island index by MM Borg
    runtime_path = str(
        runtime_dir / f"seed_{seed:02d}_{slug}_%d.runtime"
    )

    # --- solveMPI ---
    t_start = time.time()
    print(f"[MM-Borg] launching solveMPI...", flush=True)

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
        set_file = sets_dir / f"seed_{seed:02d}_{slug}.set"
        _write_set_file(result, set_file, formulation_name, seed)
        print(f"[Seed {seed}] {result.size()} solutions -> {set_file}", flush=True)
        print(f"[MM-Borg] Total wall time: {t_end - t_start:.1f}s", flush=True)
    else:
        print(f"[MM-Borg] solveMPI completed (worker rank, no result). "
              f"Wall time: {t_end - t_start:.1f}s", flush=True)

    Configuration.stopMPI()


def _write_set_file(result, set_file: Path, formulation_name: str, seed: int):
    """Write Borg archive to a whitespace-delimited .set file.

    Rows are variables + objectives, feasible solutions only — mirroring the
    C runtime writer (BORG_Archive_append skips constraint violators and
    never writes constraint columns). An infeasible solution can appear in
    the final archive only if the run found no feasible solution at all.
    """
    var_names = get_var_names(formulation_name)
    obj_names = get_obj_names()

    with open(set_file, "w") as f:
        f.write(f"# Formulation: {formulation_name}, Seed: {seed}\n")
        f.write(f"# Variables: {','.join(var_names)}\n")
        f.write(f"# Objectives: {','.join(obj_names)}\n")

        for solution in result:
            if solution.violatesConstraints():
                continue
            variables = solution.getVariables()
            objectives = solution.getObjectives()
            line = " ".join(
                [f"{v:.6e}" for v in variables]
                + [f"{o:.6e}" for o in objectives]
            )
            f.write(line + "\n")
