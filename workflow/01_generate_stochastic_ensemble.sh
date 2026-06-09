#!/bin/bash
# Step 1: Generate a Kirsch-Nowak synthetic streamflow ensemble.
#
# Configuration is entirely env-driven. Source one of slurm/envs/ensemble_kn_*.env
# (or set NYCOPT_ENSEMBLE_KN_* directly) and submit:
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ensemble_kn_long.env \
#          workflow/01_generate_stochastic_ensemble.sh
#
# Output: outputs/synthetic_ensembles/kn_{Y}yr_n{N}/
#
#SBATCH --job-name=gen_stoch_ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_stoch_ensemble_%j.out
#SBATCH --error=logs/gen_stoch_ensemble_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.5 || true
source venv/bin/activate

NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ensemble_kn_long.env}"
if [[ -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[01_generate_stochastic_ensemble] sourced env file: ${NYCOPT_ENV_FILE}"
fi

# Generator is single-process; let BLAS use the full allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python3 -u scripts/main/generate_stochastic_ensemble.py
