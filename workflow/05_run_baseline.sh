#!/bin/bash
# Step 5: Evaluate the default FFMP policy (no optimization) and save
# baseline objective values — the comparison anchor for optimized Pareto sets.
# Runs once per design on its searched draw (d0); d1-d2 are staged only for the
# SI draw-sensitivity re-evaluation (docs/notes/methods/campaign_design.md).
#
# Usage (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
#          workflow/05_run_baseline.sh
#   NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env bash workflow/05_run_baseline.sh
#
#SBATCH --job-name=baseline
#SBATCH --account=ees260021
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
# ensemble (per-SOW annual-unit objective values), so step 08's robustness
# scoring can compute the incumbent-relative regret family (auto-detected at
# <reeval_dir>/baseline). Opt out with NYCOPT_BASELINE_SKIP_REEVAL=1.
if [[ "${NYCOPT_BASELINE_SKIP_REEVAL:-0}" != "1" ]]; then
    echo "[05_run_baseline] persisting baseline re-eval matrix (improvement-vs-baseline)"
    python3 scripts/main/run_baseline.py "$@" --reeval
fi
