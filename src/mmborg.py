"""
mmborg.py - Multi-Master Borg MOEA optimization driver.

Provides the setup and execution functions for running MMBorg
optimization of NYC reservoir operations. Designed for HPC
execution with MPI.

Compilation (from MMBorgMOEA repo, passNFE_ALH_PyCheckpoint branch):
    mpicc -shared -fPIC -O3 -o libborgmm.so borgmm.c mt19937ar.c -lm

Required files in borg/ directory:
    - borg.py      (revised Python wrapper from BorgTraining repo)
    - libborgmm.so (compiled MM Borg shared library)

References:
    - WaterProgramming: "Everything You Need to Run Borg MOEA" Parts 1-2
    - WaterProgramming: "MM Borg MOEA Python Wrapper" (Aug 2025)
    - WaterProgramming: "MM Borg Training" Parts 1-2
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "borg"))

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
    checkpoint_file: str = None,
    restore_file: str = None,
):
    """Execute Multi-Master Borg optimization.

    The borg.py wrapper's solveMPI() handles MPI communication internally.
    This function should be called by all MPI ranks. Only the controller
    node (rank 0) receives the result.

    Args:
        formulation_name: Problem formulation name.
        seed: Random seed number (1-indexed).
        n_islands: Number of MM Borg islands (masters).
        max_evaluations: Maximum NFE (default from config).
        max_time: Maximum wall time in seconds (optional, overrides NFE).
        runtime_frequency: NFE interval for runtime output (default from config).
        checkpoint_file: Path base for checkpoint files (enables restart).
        restore_file: Path to checkpoint file to restore from.
    """
    import borg as bg

    n_vars = get_n_vars(formulation_name)
    n_objs = get_n_objs()
    n_constrs = 0

    if max_evaluations is None:
        max_evaluations = BORG_SETTINGS["max_evaluations"]
    if runtime_frequency is None:
        runtime_frequency = BORG_SETTINGS["runtime_frequency"]

    # --- Objective function closure ---
    def borg_objective(*vars):
        dv_vector = np.array(vars)
        return evaluate(dv_vector, formulation_name=formulation_name)

    # --- Create Borg instance ---
    borg_instance = bg.Borg(n_vars, n_objs, n_constrs, borg_objective)

    # Set bounds per variable
    lower, upper = get_bounds(formulation_name)
    for i in range(n_vars):
        borg_instance.setBounds(i, lower[i], upper[i])

    # Set epsilon values
    borg_instance.setEpsilons(*get_epsilons())

    # --- Output paths ---
    opt_dir = OUTPUTS_DIR / "optimization" / formulation_name
    runtime_dir = opt_dir / "runtime"
    sets_dir = opt_dir / "sets"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)

    # Runtime file: %d is replaced by island index by MM Borg
    runtime_path = str(
        runtime_dir / f"seed_{seed:02d}_{formulation_name}_%d.runtime"
    )

    # --- Start MPI ---
    bg.Configuration.startMPI()

    # --- Build solveMPI kwargs ---
    solve_kwargs = {
        "maxEvaluations": max_evaluations,
        "islands": n_islands,
        "runtime": runtime_path,
        "frequency": runtime_frequency,
        "runtimeformat": "borg",
    }

    # Override with wall-time limit if provided
    if max_time is not None:
        solve_kwargs["maxTime"] = max_time

    # Checkpointing
    if checkpoint_file is not None:
        solve_kwargs["newCheckpointFileBase"] = checkpoint_file
    if restore_file is not None:
        solve_kwargs["restoreCheckpointFileBase"] = restore_file

    # --- Solve ---
    result = borg_instance.solveMPI(**solve_kwargs)

    # --- Save results (controller node only) ---
    if result is not None:
        set_file = sets_dir / f"seed_{seed:02d}_{formulation_name}.set"
        _write_set_file(result, set_file, formulation_name, seed)

        print(f"[Seed {seed}] Complete. {len(result)} solutions in archive.")
        print(f"[Seed {seed}] Runtime: {runtime_path}")
        print(f"[Seed {seed}] Set: {set_file}")

    # --- Stop MPI ---
    bg.Configuration.stopMPI()


def _write_set_file(result, set_file: Path, formulation_name: str, seed: int):
    """Write Borg solution archive to a .set file."""
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
