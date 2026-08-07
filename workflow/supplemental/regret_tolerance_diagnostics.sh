#!/bin/bash
#SBATCH --job-name=rtol_diag
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/regret_tolerance_diagnostics_%j.out
#SBATCH --error=logs/regret_tolerance_diagnostics_%j.err

# Regret-tolerance diagnostics
# (docs/notes/methods/regret_tolerance_diagnostics.md)
#
# Fixes the two free parameters of the incumbent-relative regret comparison --
# the no-harm tolerance tau_i = k * eps_i and the non-inferiority margin delta --
# from anchors that cannot bias the answer. Zero simulation.
#
# TWO PASSES, and the order matters.
#
#   Pass A needs ONLY the step-05 incumbent cube, so run it as soon as that
#   lands and BEFORE any search finishes. It measures the estimator noise floor
#   and picks the headline tolerance rung. Running it early is the point: a
#   tolerance chosen after seeing the campaign contrast is not a pre-registered
#   tolerance, whatever it is set to.
#
#   Pass B needs the re-evaluated policy sets and reports the discrimination
#   band, the empirical nulls, and the assay-sensitivity check. It computes the
#   margin only once RTOL_ADOPTED_K is filled in from pass A.
#
# Prerequisites:
#   pass A   outputs/historic/ffmp_obj8_mm_full/reeval/etest_kn_50yr_n25000/baseline/
#            (step 05 run_baseline.py --reeval on the test ensemble)
#   pass B   step 08 re-evaluation + `python -m src.robustness` per design,
#            each with its own baseline/ subdir staged from the SAME preset
#
# All settings live in supplemental_config.py (RTOL_ section) -- no CLI value
# flags. Pass B is skipped with a printed reason when its inputs are absent, so
# this wrapper is safe to run at either stage.
#
# Sizing: pandas/numpy over a 200k-row parquet plus RTOL_SPLIT_HALF_REPS
# partitions of the incumbent cube (~1-2 min); the paired bootstrap over
# RTOL_BOOTSTRAP_N resamples dominates pass B when RTOL_ADOPTED_K is set.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[rtol] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/regret_tolerance_diagnostics.py
echo "[rtol] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
