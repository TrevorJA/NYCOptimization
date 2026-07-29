#!/bin/bash
# Nested-P saturation diagnostic, stage 1b: driver self-check (minutes).
#
# Exercises both new code paths of diagnose_hazard_selectors.py on the small
# P=2,000 smoke pool BEFORE the long pool generation commits: prefix mode with
# the full battery, and prefix + lean saturation mode. Outputs land under
# throwaway ``statpool_10yr_n2000_d0_prefix*`` slugs (removed after review).
#
#SBATCH --job-name=nestedp_check
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nestedp_check_%j.out
#SBATCH --error=logs/nestedp_check_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[nestedp_check] full battery, prefix 1000  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
NYCOPT_SELDIAG_POOL_SLUG=statpool_10yr_n2000_d0 \
NYCOPT_SELDIAG_PREFIX_P=1000 \
python3 -u scripts/supplemental/diagnose_hazard_selectors.py

echo "[nestedp_check] saturation mode, prefix 800  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
NYCOPT_SELDIAG_POOL_SLUG=statpool_10yr_n2000_d0 \
NYCOPT_SELDIAG_PREFIX_P=800 \
NYCOPT_SELDIAG_SATURATION=1 \
python3 -u scripts/supplemental/diagnose_hazard_selectors.py

echo "[nestedp_check] done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
