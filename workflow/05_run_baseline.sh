#!/bin/bash
# Step 5: Evaluate the default FFMP policy (no optimization) and save
# baseline objective values — the comparison anchor for optimized Pareto sets.
#
# Usage (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200 \
#          workflow/05_run_baseline.sh
#   NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env bash workflow/05_run_baseline.sh
#
#SBATCH --job-name=baseline
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/baseline.out
#SBATCH --error=logs/baseline.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

python3 scripts/main/run_baseline.py "$@"

# Also persist the baseline policy's re-eval matrix on the common re-eval
# ensemble (raw per-realization base metrics), so step 08's robustness scoring
# can compute regret-from-baseline (auto-detected at <reeval_dir>/baseline).
# Opt out with NYCOPT_BASELINE_SKIP_REEVAL=1.
if [[ "${NYCOPT_BASELINE_SKIP_REEVAL:-0}" != "1" ]]; then
    echo "[05_run_baseline] persisting baseline re-eval matrix (regret-from-baseline)"
    python3 scripts/main/run_baseline.py "$@" --reeval
fi
