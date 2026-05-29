#!/bin/bash
# Step 4: Evaluate the default FFMP policy (no optimization) and save
# baseline objective values for comparison.
#
# Usage:
#   bash workflow/04_run_baseline.sh
#   sbatch workflow/04_run_baseline.sh
#   NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7_sal.env bash workflow/04_run_baseline.sh
#
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/baseline.out
#SBATCH --error=logs/baseline.err
set -euo pipefail

# `cd ..` so we land at the project root regardless of whether the script
# was invoked from project root or workflow/. workflow/01_run_baseline.sh
# now mirrors slurm/_common.sh in supporting NYCOPT_ENV_FILE.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}/..}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

# Source per-experiment env file if provided (NYCOPT_* knobs apply at config
# import time inside scripts/run_baseline.py).
if [[ -n "${NYCOPT_ENV_FILE:-}" && -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[04_run_baseline] sourced env file: ${NYCOPT_ENV_FILE}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 scripts/run_baseline.py "$@"
