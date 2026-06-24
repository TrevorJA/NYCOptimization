#!/bin/bash
# subsample_ensemble.sh — Step 2 (hazard-filling) on a compute node.
# Runs scripts/main/subsample_hazard_filling.py for the active hazard-filling
# design (NYCOPT_SCENARIO_DESIGN ∈ {hazard_filling, hazard_filling_absolute}),
# slicing the staged master pool (kn_5yr_n1000) into the reduced N=64 ensemble.
# Serial (the selector is single-process); loads the pool HDF5 into memory.
#
# Submit:
#   sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling \
#          slurm/main/subsample_ensemble.sh
#
#SBATCH --job-name=subsample_ens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/subsample_ens_%j.out
#SBATCH --error=logs/subsample_ens_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.5 || true
source venv/bin/activate

# Optional per-experiment env file (the design can also come straight from
# --export=ALL,NYCOPT_SCENARIO_DESIGN=...).
if [[ -n "${NYCOPT_ENV_FILE:-}" && -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a; source "${NYCOPT_ENV_FILE}"; set +a
    echo "[subsample_ens] sourced env file: ${NYCOPT_ENV_FILE}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "[subsample_ens] design=${NYCOPT_SCENARIO_DESIGN:-<default>}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/subsample_hazard_filling.py
echo "[subsample_ens] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
