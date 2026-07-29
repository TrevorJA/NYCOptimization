#!/bin/bash
# Nested-P saturation diagnostic, stage 1: test suites + P=2,000 smoke pool.
#
# Runs the scengen test suite (must be green — aborts otherwise), records the
# NYCOptimization suite result, then generates the P=2,000 stationary
# stream-only candidate pool via the step-02 code path with wall-clock timing.
# The smoke pool is the time-calibration point for projecting the P=10^6
# generation cost against the documented ~3,000 SU bound
# (docs/notes/methods/scenario_design_methods.md §6); its first rows are also
# bit-identical to the large pool's prefix (global-index child streams), which
# the analysis job verifies.
#
#SBATCH --job-name=nestedp_smoke
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nestedp_smoke_%j.out
#SBATCH --error=logs/nestedp_smoke_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[nestedp_smoke] scengen pytest  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! (cd ../NYCOptimization_scenario_generation && python3 -m pytest -q); then
    echo "[nestedp_smoke] FATAL: scengen test suite not green — aborting." >&2
    exit 1
fi

echo "[nestedp_smoke] NYCOptimization pytest (-m 'not slow')  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -m pytest -q -m "not slow" || echo "[nestedp_smoke] NYCOptimization pytest rc=$? (recorded; Pywr-DRB import failures are a known branch issue)"

echo "[nestedp_smoke] generating P=2000 stationary stream-only pool  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
export NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary
export NYCOPT_CANDIDATE_POOL_N=2000
export NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1
export NYCOPT_ENSEMBLE_DRAW=0
t0=$(date +%s)
python3 -u scripts/main/generate_stochastic_ensemble.py --draw 0
t1=$(date +%s)
echo "[nestedp_smoke] P=2000 generation wall seconds: $((t1 - t0))"
echo "[nestedp_smoke] done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
