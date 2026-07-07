#!/bin/bash
# ensemble_objective_sensitivity.sh — MPI-parallel ensemble objective-sensitivity
# DV sweep. Evaluates (baseline + N random DV vectors) over ONE fixed
# probabilistic ensemble and stores the per-realization base-metric matrix; each
# MPI rank simulates a numpy.array_split slice of the DV vectors. Figures are
# then generated (serial) from the stored matrix — no re-simulation.
#
# PREREQUISITE: run slurm/supplemental/ensemble_objective_sensitivity_prep.sh
# once first to generate + stage the ensemble.
#
# All settings (ensemble, DV count, seed, formulation, objective list, batch
# size, K-grid, operators, thresholds, output paths) live in
# `supplemental_config.py` — this script carries NO value flags. Set
# ENS_SMOKE=False there for the full campaign.
#
# Sizing (full scale): ENS_SMOKE=False -> ~200 DVs, each one ensemble simulation
# over N=256 x 20-yr realizations (batched). Size --ntasks to the DV count you
# can afford in parallel; e.g. 4 nodes x 32 ranks clears 200 DVs in ~2 waves.
#
# Usage:
#   sbatch slurm/supplemental/ensemble_objective_sensitivity.sh
#
#SBATCH --job-name=ens_objsens
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=33
#SBATCH --time=05:00:00
#SBATCH --output=logs/ensemble_objective_sensitivity_%j.out
#SBATCH --error=logs/ensemble_objective_sensitivity_%j.err

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

# TCP libfabric provider: filesystem-barrier shard combine avoids comm.gather,
# but keep this set so any incidental MPI collective stays on a stable provider.
export FI_PROVIDER=tcp

NTASKS_MPI="${SLURM_NTASKS:-32}"

echo "=== Ensemble objective-sensitivity sweep (ranks=${NTASKS_MPI}) ==="
mpirun -np "${NTASKS_MPI}" python3 -u \
    scripts/supplemental/ensemble_objective_sensitivity_run.py
echo "=== Sweep complete; generating figures ==="
python3 -u scripts/supplemental/ensemble_objective_sensitivity_figures.py
echo "=== Completed: $(date) ==="
