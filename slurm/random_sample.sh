#!/bin/bash
# random_sample.sh — MPI-parallel random-sample diagnostic.
# Runs baseline + N random DV vectors through Pywr-DRB and reports the
# objective-value spread. Each MPI rank simulates ~1 DV vector, so total
# wall time is ~1 sim duration regardless of N (within node capacity).
#
# Usage:
#   sbatch slurm/random_sample.sh                        # default N=10, seed=42
#   sbatch --export=ALL,N_SAMPLES=20,SEED=7 slurm/random_sample.sh
#
#SBATCH --job-name=randsample
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --time=00:30:00
#SBATCH --output=logs/random_sample_%j.out
#SBATCH --error=logs/random_sample_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.5 || true
source venv/bin/activate

# Defaults — override via --export when submitting.
N_SAMPLES="${N_SAMPLES:-10}"
SEED="${SEED:-42}"
FORMULATION="${FORMULATION:-ffmp}"
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ffmp_obj7_sal.env}"

if [[ -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[random_sample] sourced env file: ${NYCOPT_ENV_FILE}"
fi

# Thread pinning so each rank's BLAS doesn't oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# MPI rank count = N+1 (baseline + N samples). If SLURM allocated fewer
# tasks than that, fall back to whatever was allocated (sims are batched
# across ranks via array_split).
NTASKS_MPI="${SLURM_NTASKS:-$((N_SAMPLES + 1))}"

echo "=== Launching MPI random sample (N=${N_SAMPLES}, seed=${SEED}, formulation=${FORMULATION}) ==="
echo "    ranks=${NTASKS_MPI}"
mpirun -np "${NTASKS_MPI}" python3 -u scripts/supplemental/random_sample_mpi.py \
    --n "${N_SAMPLES}" --seed "${SEED}" --formulation "${FORMULATION}"
echo "=== Completed: $(date) ==="
