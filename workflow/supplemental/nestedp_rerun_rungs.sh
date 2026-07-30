#!/bin/bash
# Rerun the nested-P prefix rungs only (no cross-rung analysis, which would
# overwrite the annotated results file): refreshes the per-rung outputs so the
# committed campaign selection set (config.HAZARD_SELECTION_AXES, m = 6) is
# scored alongside m4/m6/full at every rung. The image is not regenerated.
#
#SBATCH --job-name=nestedp_rungs
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nestedp_rungs_%j.out
#SBATCH --error=logs/nestedp_rungs_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

: "${NYCOPT_NESTEDP_POOL_SLUG:?set NYCOPT_NESTEDP_POOL_SLUG via --export}"
: "${NYCOPT_NESTEDP_RUNGS:?set NYCOPT_NESTEDP_RUNGS via --export}"
NYCOPT_NESTEDP_FULL_MAX="${NYCOPT_NESTEDP_FULL_MAX:-5000}"

for P in ${NYCOPT_NESTEDP_RUNGS}; do
    echo "[nestedp_rungs] rung P'=${P}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if (( P <= NYCOPT_NESTEDP_FULL_MAX )); then
        NYCOPT_SELDIAG_POOL_SLUG="${NYCOPT_NESTEDP_POOL_SLUG}" \
        NYCOPT_SELDIAG_PREFIX_P="${P}" \
        python3 -u scripts/supplemental/diagnose_hazard_selectors.py
    else
        NYCOPT_SELDIAG_POOL_SLUG="${NYCOPT_NESTEDP_POOL_SLUG}" \
        NYCOPT_SELDIAG_PREFIX_P="${P}" \
        NYCOPT_SELDIAG_SATURATION=1 \
        python3 -u scripts/supplemental/diagnose_hazard_selectors.py
    fi
done
echo "[nestedp_rungs] done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
