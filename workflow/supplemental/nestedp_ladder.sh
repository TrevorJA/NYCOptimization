#!/bin/bash
# Nested-P saturation diagnostic, stage 2: the analysis ladder.
#
# Runs `diagnose_hazard_selectors.py` on nested prefixes of ONE staged
# stream-only candidate pool (full battery at the two smallest rungs — the
# continuity check against the laptop numbers — lean saturation mode above),
# then the cross-rung fit + results file + figure. Everything is
# selection-level: no simulation, no optimization.
#
# Before the rungs, verifies the prefix-identity contract on which the design
# rests: the first rows of the large image must be bit-identical to the
# standalone P=2,000 smoke pool generated from the same seed domain.
#
# Submit with the pool slug + rungs (spaces in the rung list -> quote it):
#   sbatch --export=ALL,NYCOPT_NESTEDP_POOL_SLUG=statpool_10yr_n1000000_d0,\
# NYCOPT_NESTEDP_RUNGS="2000 5000 20000 100000 300000 1000000",\
# NYCOPT_NESTEDP_GEN_SU=<su> workflow/supplemental/nestedp_ladder.sh
#
#SBATCH --job-name=nestedp_ladder
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/nestedp_ladder_%j.out
#SBATCH --error=logs/nestedp_ladder_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

: "${NYCOPT_NESTEDP_POOL_SLUG:?set NYCOPT_NESTEDP_POOL_SLUG via --export}"
: "${NYCOPT_NESTEDP_RUNGS:?set NYCOPT_NESTEDP_RUNGS via --export}"
#: Rungs at or below this size run the full battery (laptop-continuity check).
NYCOPT_NESTEDP_FULL_MAX="${NYCOPT_NESTEDP_FULL_MAX:-5000}"

# Generation SU for the results file: allocated core-hours of the generation
# jobs (space-separated ids in NYCOPT_NESTEDP_GEN_JOBIDS), unless the caller
# supplied NYCOPT_NESTEDP_GEN_SU directly.
if [[ -z "${NYCOPT_NESTEDP_GEN_SU:-}" && -n "${NYCOPT_NESTEDP_GEN_JOBIDS:-}" ]]; then
    _su=$(sacct -j "${NYCOPT_NESTEDP_GEN_JOBIDS// /,}" -n -X --format=CPUTimeRAW 2>/dev/null \
          | awk '{s+=$1} END {if (s > 0) printf "%.0f", s/3600}')
    if [[ -n "${_su}" ]]; then
        export NYCOPT_NESTEDP_GEN_SU="~${_su} SU (allocated core-hours, sacct, jobs ${NYCOPT_NESTEDP_GEN_JOBIDS})"
    fi
fi

echo "[nestedp_ladder] pool=${NYCOPT_NESTEDP_POOL_SLUG} rungs='${NYCOPT_NESTEDP_RUNGS}'  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[nestedp_ladder] verifying prefix identity vs the P=2000 smoke pool"
python3 -u scripts/supplemental/verify_prefix_identity.py

for P in ${NYCOPT_NESTEDP_RUNGS}; do
    echo "[nestedp_ladder] rung P'=${P}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

echo "[nestedp_ladder] cross-rung analysis  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/nestedp_saturation_analysis.py

echo "[nestedp_ladder] done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
