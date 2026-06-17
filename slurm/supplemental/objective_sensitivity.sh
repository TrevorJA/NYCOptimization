#!/bin/bash
# objective_sensitivity.sh — MPI-parallel random-DV objective-sensitivity
# diagnostic. Runs (baseline + N random DV vectors) through Pywr-DRB on a
# single historical reference trace and records every evaluated objective.
# Each MPI rank simulates a numpy.array_split slice of the samples.
#
# All settings (sample count, seed, formulation, objective set, salinity LSTM,
# simulation window, output paths) live in `supplemental_config.py` — this
# script carries NO value flags and sources NO env file. Set SMOKE=False there
# for the full campaign.
#
# Sizing (full scale): SMOKE=False -> N≈500 sims at ~150–300 s each with the
# salinity LSTM in-loop. They batch across whatever ranks SLURM allocates, so
# size --ntasks to (cores you can afford); e.g. 4 nodes × 32 ranks ≈ 128 ranks
# clears 500 sims in ~4 waves (~20 min). Raise --time accordingly.
#
# Usage:
#   sbatch slurm/supplemental/objective_sensitivity.sh
#
#SBATCH --job-name=obj_sens
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --time=00:30:00
#SBATCH --output=logs/objective_sensitivity_%j.out
#SBATCH --error=logs/objective_sensitivity_%j.err

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

NTASKS_MPI="${SLURM_NTASKS:-11}"

echo "=== Launching objective-sensitivity diagnostic (ranks=${NTASKS_MPI}) ==="
mpirun -np "${NTASKS_MPI}" python3 -u \
    scripts/supplemental/objective_sensitivity_run.py
echo "=== Run complete; generating figures ==="
python3 -u scripts/supplemental/objective_sensitivity_figures.py
echo "=== Completed: $(date) ==="
