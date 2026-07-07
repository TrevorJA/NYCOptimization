#!/bin/bash
# ensemble_objective_sensitivity_prep.sh — one-time staging of the experiment's
# fixed probabilistic (Kirsch-Nowak) ensemble: generate it (serial on rank 0)
# and stage the pywrdrb HDF5 inputs (flood / STARFIT-release / predicted-inflow
# preprocessors, MPI-parallel across realizations).
#
# All settings (ensemble size, realization length, seed, prediction modes,
# output paths) live in `supplemental_config.py` — this script carries NO value
# flags and sources NO env file. Run once per ensemble before the DV sweep.
#
# Usage (from repo root):
#   sbatch workflow/supplemental/ensemble_objective_sensitivity_prep.sh
#
#SBATCH --job-name=ens_prep
#SBATCH --partition=wholenode
#SBATCH --nodes=5
#SBATCH --ntasks=200
#SBATCH --time=03:00:00
#SBATCH --output=logs/ensemble_prep_%j.out
#SBATCH --error=logs/ensemble_prep_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_pin_threads

# Use the TCP libfabric provider: the preprocessors' point-to-point MPI gathers
# hang under some OpenMPI + libfabric default providers on this cluster.
export FI_PROVIDER=tcp

NTASKS_MPI="${SLURM_NTASKS:-16}"

echo "=== Staging ensemble inputs (ranks=${NTASKS_MPI}) ==="
mpirun -np "${NTASKS_MPI}" python3 -u \
    scripts/supplemental/ensemble_objective_sensitivity_prep.py
echo "=== Completed: $(date) ==="
