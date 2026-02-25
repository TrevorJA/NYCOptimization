"""
run_mmborg.py - Multi-Master Borg MOEA optimization driver.

Runs MMBorg optimization of NYC reservoir operations using the
Pywr-DRB simulation model. Designed for HPC execution with MPI.

Usage (via SLURM, see submit_mmborg.sh):
    mpirun -np <total_procs> python run_mmborg.py \
        --formulation ffmp \
        --seed 1 \
        --max_time 86400

The MMBorg Python wrapper uses:
    - libborgmm.so (compiled from MMBorgMOEA repo, passNFE_ALH_PyCheckpoint branch)
    - borg.py (revised wrapper from BorgTraining repo)

File naming conventions:
    Runtime:  outputs/optimization/<formulation>/runtime/seed_<S>_<formulation>.runtime
    Sets:     outputs/optimization/<formulation>/sets/seed_<S>_<formulation>.set
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "borg"))

from config import (
    get_n_vars,
    get_n_objs,
    get_bounds,
    get_epsilons,
    get_var_names,
    get_obj_names,
    BORG_SETTINGS,
    MMBORG_SETTINGS,
    OUTPUTS_DIR,
)
from src.simulation import evaluate


def setup_borg(formulation_name, seed, max_time):
    """Configure and return the Borg MOEA instance for MMBorg.

    Uses borg.py wrapper with libborgmm.so for multi-master parallelization.
    """
    # Import the borg wrapper (must be on sys.path)
    import borg as bg

    n_vars = get_n_vars(formulation_name)
    n_objs = get_n_objs()
    n_constrs = 0

    # Define the objective function closure
    def borg_objective(*vars):
        """Borg calls this with decision variables as positional args."""
        dv_vector = np.array(vars)
        objs = evaluate(dv_vector, formulation_name=formulation_name)
        return objs

    # Create Borg instance
    borg_instance = bg.Borg(
        n_vars,
        n_objs,
        n_constrs,
        borg_objective,
    )

    # Set bounds
    lower, upper = get_bounds(formulation_name)
    for i in range(n_vars):
        borg_instance.setBounds(i, lower[i], upper[i])

    # Set epsilons
    epsilons = get_epsilons()
    borg_instance.setEpsilons(*epsilons)

    # Setup output directories
    opt_dir = OUTPUTS_DIR / "optimization" / formulation_name
    runtime_dir = opt_dir / "runtime"
    sets_dir = opt_dir / "sets"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)

    # Runtime file path
    runtime_file = runtime_dir / f"seed_{seed:02d}_{formulation_name}.runtime"

    return borg_instance, runtime_file, max_time


def run_optimization(formulation_name, seed, max_time):
    """Execute MMBorg optimization."""
    import borg as bg

    borg_instance, runtime_file, max_time = setup_borg(
        formulation_name, seed, max_time
    )

    n_vars = get_n_vars(formulation_name)
    n_objs = get_n_objs()
    runtime_freq = BORG_SETTINGS["runtime_frequency"]

    # Start MPI
    bg.Configuration.startMPI()

    # Solve using multi-master Borg
    result = borg_instance.solveMPI(
        maxTime=max_time,
        runtime=str(runtime_file),
        frequency=runtime_freq,
        runtimeformat="borg",
    )

    # Only the controller node (rank 0) gets a result
    if result is not None:
        # Save final solution set
        opt_dir = OUTPUTS_DIR / "optimization" / formulation_name
        sets_dir = opt_dir / "sets"
        set_file = sets_dir / f"seed_{seed:02d}_{formulation_name}.set"

        with open(set_file, "w") as f:
            # Header comment
            f.write(f"# Formulation: {formulation_name}\n")
            f.write(f"# Seed: {seed}\n")
            f.write(f"# Variables: {','.join(get_var_names(formulation_name))}\n")
            f.write(f"# Objectives: {','.join(get_obj_names())}\n")

            for solution in result:
                variables = solution.getVariables()
                objectives = solution.getObjectives()
                line = " ".join(
                    [f"{v:.6e}" for v in variables]
                    + [f"{o:.6e}" for o in objectives]
                )
                f.write(line + "\n")

        print(f"[Seed {seed}] Optimization complete. "
              f"{len(result)} solutions in archive.")
        print(f"[Seed {seed}] Runtime: {runtime_file}")
        print(f"[Seed {seed}] Set: {set_file}")

    # Stop MPI
    bg.Configuration.stopMPI()


def main():
    parser = argparse.ArgumentParser(
        description="Run MMBorg optimization of NYC reservoir operations."
    )
    parser.add_argument(
        "--formulation", type=str, default="ffmp",
        help="Problem formulation name (default: ffmp)",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Random seed number (1-indexed)",
    )
    parser.add_argument(
        "--max_time", type=int,
        default=MMBORG_SETTINGS["max_time_hours"] * 3600,
        help="Maximum optimization time in seconds (default: 24h)",
    )
    args = parser.parse_args()

    run_optimization(args.formulation, args.seed, args.max_time)


if __name__ == "__main__":
    main()
