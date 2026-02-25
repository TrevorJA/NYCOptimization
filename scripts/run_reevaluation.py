"""
run_reevaluation.py - Re-evaluate Pareto-approximate policies under uncertainty.

Reads the reference set from optimization, then re-evaluates each policy
across an ensemble of stochastic streamflow realizations. Designed for
MPI-parallel execution on HPC.

Usage:
    mpirun -np <N> python run_reevaluation.py \
        --formulation ffmp \
        --reference_set outputs/reference_sets/ffmp.ref

Output:
    outputs/reevaluation/<formulation>/
        objectives_realization_<RRRR>.csv   (one per realization)
        robustness_metrics.csv              (aggregated across realizations)
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    get_n_vars,
    get_n_objs,
    get_var_names,
    get_obj_names,
    get_obj_directions,
    OUTPUTS_DIR,
    REEVALUATION_SETTINGS,
)
from src.simulation import dvs_to_config, run_simulation
from src.objectives import compute_all_objectives


def load_reference_set(ref_file: Path, formulation_name: str) -> np.ndarray:
    """Load reference set file and extract decision variable vectors.

    Reference set files contain lines of:
        var1 var2 ... varN obj1 obj2 ... objM

    Returns:
        2D array of shape (n_solutions, n_vars).
    """
    n_vars = get_n_vars(formulation_name)

    solutions = []
    with open(ref_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            # Extract only decision variables (first n_vars columns)
            solutions.append(values[:n_vars])

    return np.array(solutions)


def evaluate_policy_on_realization(
    dv_vector: np.ndarray,
    formulation_name: str,
    realization_inflow_type: str,
) -> list:
    """Evaluate a single policy on a single stochastic realization.

    Args:
        dv_vector: Decision variable vector.
        formulation_name: Formulation name.
        realization_inflow_type: Inflow type string pointing to the
            stochastic realization data.

    Returns:
        List of raw objective values (not Borg-negated).
    """
    config = dvs_to_config(dv_vector, formulation_name)
    data = run_simulation(config)
    return compute_all_objectives(data)


def compute_robustness(
    objectives_all: np.ndarray,
    thresholds: dict,
) -> dict:
    """Compute robustness metrics across realizations for one policy.

    Args:
        objectives_all: Array of shape (n_realizations, n_objs).
        thresholds: Dict mapping objective name to satisficing threshold.

    Returns:
        Dict of robustness metrics.
    """
    obj_names = get_obj_names()
    directions = get_obj_directions()
    n_real = objectives_all.shape[0]

    # Satisficing: fraction of realizations meeting ALL thresholds
    meets_all = np.ones(n_real, dtype=bool)
    for i, name in enumerate(obj_names):
        threshold = thresholds.get(name)
        if threshold is None:
            continue
        if directions[i] == 1:  # maximize
            meets_all &= (objectives_all[:, i] >= threshold)
        else:  # minimize
            meets_all &= (objectives_all[:, i] <= threshold)

    satisficing = float(meets_all.sum()) / n_real

    # Per-objective statistics
    stats = {"satisficing": satisficing}
    for i, name in enumerate(obj_names):
        vals = objectives_all[:, i]
        stats[f"{name}_mean"] = float(np.mean(vals))
        stats[f"{name}_std"] = float(np.std(vals))
        stats[f"{name}_p10"] = float(np.percentile(vals, 10))
        stats[f"{name}_p50"] = float(np.percentile(vals, 50))
        stats[f"{name}_p90"] = float(np.percentile(vals, 90))

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate Pareto-approximate policies under uncertainty."
    )
    parser.add_argument(
        "--formulation", type=str, default="ffmp",
    )
    parser.add_argument(
        "--reference_set", type=str, required=True,
        help="Path to reference set file (.ref)",
    )
    parser.add_argument(
        "--n_realizations", type=int, default=None,
        help="Number of stochastic realizations (overrides config)",
    )
    args = parser.parse_args()

    formulation = args.formulation
    ref_file = Path(args.reference_set)
    n_real = args.n_realizations or REEVALUATION_SETTINGS["n_realizations"]

    if n_real is None:
        raise ValueError("n_realizations must be specified (in config or --n_realizations)")

    # Setup output directory
    reeval_dir = OUTPUTS_DIR / "reevaluation" / formulation
    reeval_dir.mkdir(parents=True, exist_ok=True)

    # Load reference set
    solutions = load_reference_set(ref_file, formulation)
    n_solutions = solutions.shape[0]
    print(f"Loaded {n_solutions} solutions from {ref_file}")

    # --- MPI distribution ---
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        rank = 0
        size = 1

    # Distribute work: each rank handles a subset of (solution, realization) pairs
    all_work = [
        (sol_idx, real_idx)
        for sol_idx in range(n_solutions)
        for real_idx in range(n_real)
    ]
    my_work = all_work[rank::size]

    if rank == 0:
        print(f"Total evaluations: {len(all_work)}")
        print(f"MPI ranks: {size}")
        print(f"Evaluations per rank: ~{len(all_work) // size}")

    # --- Evaluate ---
    obj_names = get_obj_names()
    local_results = []

    for sol_idx, real_idx in my_work:
        dv_vector = solutions[sol_idx]
        # TODO: modify inflow_type per realization once stochastic
        # ensemble generation is implemented. For now, this evaluates
        # on the reference historical scenario.
        objs = evaluate_policy_on_realization(
            dv_vector, formulation, "reference"
        )
        local_results.append({
            "solution_idx": sol_idx,
            "realization_idx": real_idx,
            **{name: val for name, val in zip(obj_names, objs)},
        })

    # --- Gather results ---
    if size > 1:
        all_results = comm.gather(local_results, root=0)
    else:
        all_results = [local_results]

    if rank == 0:
        # Flatten
        flat_results = [r for sublist in all_results for r in sublist]
        df = pd.DataFrame(flat_results)
        df.to_csv(reeval_dir / "reevaluation_results.csv", index=False)
        print(f"Results saved to {reeval_dir / 'reevaluation_results.csv'}")

        # Compute per-solution robustness
        # TODO: Define satisficing thresholds based on baseline performance
        # For now, placeholder thresholds
        thresholds = {}  # Will be populated after baseline analysis

        robustness_rows = []
        for sol_idx in range(n_solutions):
            sol_df = df[df["solution_idx"] == sol_idx]
            obj_array = sol_df[obj_names].values
            rob = compute_robustness(obj_array, thresholds)
            rob["solution_idx"] = sol_idx
            robustness_rows.append(rob)

        rob_df = pd.DataFrame(robustness_rows)
        rob_df.to_csv(reeval_dir / "robustness_metrics.csv", index=False)
        print(f"Robustness saved to {reeval_dir / 'robustness_metrics.csv'}")


if __name__ == "__main__":
    main()
