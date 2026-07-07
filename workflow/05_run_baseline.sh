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
# now mirrors slurm/main/_common.sh in supporting NYCOPT_ENV_FILE.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}/..}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

# Source per-experiment env file if provided (NYCOPT_* knobs apply at config
# import time inside scripts/main/run_baseline.py).
if [[ -n "${NYCOPT_ENV_FILE:-}" && -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[04_run_baseline] sourced env file: ${NYCOPT_ENV_FILE}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 scripts/main/run_baseline.py "$@"

# Also persist the baseline policy's re-eval matrix on the common re-eval
# ensemble (raw per-realization base metrics), so step 7's robustness scoring can
# compute regret-from-baseline (auto-detected at <reeval_dir>/baseline). Opt out
# with NYCOPT_BASELINE_SKIP_REEVAL=1.
if [[ "${NYCOPT_BASELINE_SKIP_REEVAL:-0}" != "1" ]]; then
    echo "[04_run_baseline] persisting baseline re-eval matrix (regret-from-baseline)"
    python3 scripts/main/run_baseline.py "$@" --reeval
fi
