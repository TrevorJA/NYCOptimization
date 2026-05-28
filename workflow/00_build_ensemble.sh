#!/bin/bash
# Step 0: Stage the multi-realization streamflow ensemble named by the
# active env file (NYCOPT_ENSEMBLE_PRESET) so pywrdrb can find it during
# optimization and re-evaluation.
#
# Idempotent: if the ensemble HDF5s already exist for the resolved preset,
# this step is a fast no-op. Pass --force to regenerate.
#
# `historic_single` (the default legacy preset) does nothing — the legacy
# single-trace inflow_type is bundled with pywrdrb and needs no staging.
#
# Usage:
#   bash workflow/00_build_ensemble.sh
#   bash workflow/00_build_ensemble.sh --force
#   NYCOPT_ENV_FILE=slurm/envs/wcu_obj6_sal_n5.env bash workflow/00_build_ensemble.sh
#   sbatch workflow/00_build_ensemble.sh
#
#SBATCH --job-name=build_ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_ensemble.out
#SBATCH --error=logs/build_ensemble.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}/..}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

# Source per-experiment env file if provided. NYCOPT_ENSEMBLE_PRESET is read
# at config import time inside scripts/build_ensemble.py; set it here so the
# Python process inherits it.
if [[ -n "${NYCOPT_ENV_FILE:-}" && -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[00_build_ensemble] sourced env file: ${NYCOPT_ENV_FILE}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Unbuffered stdout/stderr so SLURM-redirected logs flush in real time
# (otherwise long-running steps look hung — they're not, just block-buffered).
python3 -u scripts/build_ensemble.py "$@"
