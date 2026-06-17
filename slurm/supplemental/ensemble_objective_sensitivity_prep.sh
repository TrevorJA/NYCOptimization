#!/bin/bash
# ensemble_objective_sensitivity_prep.sh — one-time staging of the experiment's
# fixed probabilistic (Kirsch-Nowak) ensemble: generate it (Step 1, serial on
# rank 0) and stage the pywrdrb HDF5 inputs (Step 3, flood / STARFIT-release /
# predicted-inflow preprocessors, MPI-parallel across realizations).
#
# All settings (ensemble size, realization length, seed, prediction modes,
# output paths) live in `supplemental_config.py` — this script carries NO value
# flags and sources NO env file. Run once per ensemble before the DV sweep.
#
# Usage:
#   sbatch slurm/supplemental/ensemble_objective_sensitivity_prep.sh
#
#SBATCH --job-name=ens_prep
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/ensemble_prep_%j.out
#SBATCH --error=logs/ensemble_prep_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.5 || true
source venv/bin/activate

# Thread pinning so each rank's BLAS doesn't oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Use the TCP libfabric provider: the preprocessors' point-to-point MPI gathers
# hang under some OpenMPI + libfabric default providers on this cluster.
export FI_PROVIDER=tcp

NTASKS_MPI="${SLURM_NTASKS:-16}"

echo "=== Staging ensemble inputs (ranks=${NTASKS_MPI}) ==="
mpirun -np "${NTASKS_MPI}" python3 -u \
    scripts/supplemental/ensemble_objective_sensitivity_prep.py
echo "=== Completed: $(date) ==="
