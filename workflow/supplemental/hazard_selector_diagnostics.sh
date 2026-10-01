#!/bin/bash
# Production selector + axis-set diagnostic on one staged candidate pool
# (diagnose_hazard_selectors.py, full battery; design and findings in
# docs/notes/methods/hazard_selector_diagnostics.md). Selection-level only:
# reads hazard_image.npz and the historical record named in the pool's
# _meta.json, no simulation. The array index is the pool draw k; the pool slug
# is statpool_10yr_n{P}_d{k} unless NYCOPT_SELDIAG_POOL_SLUG is exported.
#
# Measured on one workstation at P = 1e6, N = 300: 45 min, 3.2 GB peak.
#
# Submit (from repo root; one task per draw):
#   sbatch --array=0-1 --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_SELDIAG_N=300 \
#          workflow/supplemental/hazard_selector_diagnostics.sh
#
# Output: outputs/supplemental/hazard_selector_diagnostics/{pool_slug}/
#
#SBATCH --job-name=hazard_seldiag
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0
#SBATCH --output=logs/hazard_seldiag_%A_%a.out
#SBATCH --error=logs/hazard_seldiag_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_SCENARIO_DESIGN="${NYCOPT_SCENARIO_DESIGN:-hazard_filling_stationary}"
export NYCOPT_CANDIDATE_POOL_N="${NYCOPT_CANDIDATE_POOL_N:-1000000}"
export NYCOPT_ENSEMBLE_DRAW="${SLURM_ARRAY_TASK_ID:-${NYCOPT_ENSEMBLE_DRAW:-0}}"
export NYCOPT_SELDIAG_POOL_SLUG="${NYCOPT_SELDIAG_POOL_SLUG:-statpool_10yr_n${NYCOPT_CANDIDATE_POOL_N}_d${NYCOPT_ENSEMBLE_DRAW}}"
export NYCOPT_SELDIAG_N="${NYCOPT_SELDIAG_N:-300}"

echo "[hazard_seldiag] pool=${NYCOPT_SELDIAG_POOL_SLUG} N=${NYCOPT_SELDIAG_N} draw=${NYCOPT_ENSEMBLE_DRAW}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/diagnose_hazard_selectors.py
echo "[hazard_seldiag] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
