#!/bin/bash
# Build QC after any ensemble is staged: the staged flows' monthly climatology,
# grouped by true index month, must align with the historic record's at
# circular shift zero (scripts/supplemental/validate_staged_seasonality.py,
# SI Text S2). Exits nonzero on a rotation.
#
# Env inputs:
#   NYCOPT_SEASVAL_SLUG      staged slug to validate; default: the active
#                            design's search ensemble at NYCOPT_ENSEMBLE_DRAW
#                            (so the design's env file selects it)
#   NYCOPT_SEASVAL_MAX_REAL  realizations pooled for the cycle (default 50)
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env \
#          workflow/supplemental/checks/validate_staged_seasonality.sh
#   sbatch --export=ALL,NYCOPT_SEASVAL_SLUG=etest_kn_50yr_n12500 \
#          workflow/supplemental/checks/validate_staged_seasonality.sh
#
#SBATCH --job-name=validate_seasonality
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/validate_seasonality_%j.out
#SBATCH --error=logs/validate_seasonality_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
export NYCOPT_ENSEMBLE_DRAW="${DRAW:-${NYCOPT_ENSEMBLE_DRAW:-0}}"
nycopt_pin_threads

echo "[validate_seasonality] slug=${NYCOPT_SEASVAL_SLUG:-<active design, draw ${NYCOPT_ENSEMBLE_DRAW}>}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/validate_staged_seasonality.py
echo "[validate_seasonality] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
