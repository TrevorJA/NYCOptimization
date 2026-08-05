#!/bin/bash
#SBATCH --job-name=rtd_diag
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/robustness_threshold_diagnostics_%j.out
#SBATCH --error=logs/robustness_threshold_diagnostics_%j.err

# Robustness satisficing-threshold diagnostics
# (docs/notes/methods/robustness_threshold_diagnostics.md)
#
# Zero simulation: reduces the persisted baseline-FFMP E_test re-eval cube,
# the E_test forcing profiles, and one cached historic-anchor recompute from
# outputs/baseline/ffmp_baseline.hdf5 into the SI threshold-placement
# tables + figures under outputs/supplemental/robustness_threshold_diagnostics/.
#
# Prerequisites:
#   outputs/historic/ffmp_obj8_mm_full/reeval/etest_kn_50yr_n25000/baseline/
#       (step 05 run_baseline.py --reeval on E_test)
#   outputs/baseline/ffmp_baseline.hdf5          (run_baseline.py, historic)
#   outputs/synthetic_ensembles/etest_kn_50yr_n25000/forcing_profiles.npz
#
# All settings live in supplemental_config.py (RTD_ section) — no CLI value
# flags. Pass 2 (after filling RTD_RECOMMENDED_THRESHOLDS) reruns this same
# wrapper; the anchor JSON cache makes the first step near-instant
# (NYCOPT_RTD_REFRESH=1 forces the recompute).
#
# Sizing: the anchor step is one pywrdrb.Data() load of a ~133 MB HDF5 plus 8
# base-metric reductions (~1-2 min); the figures step is pandas/numpy over a
# 200k-row parquet (~1 min).

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[rtd] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/robustness_threshold_anchor.py
python3 -u scripts/supplemental/robustness_threshold_figures.py
echo "[rtd] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
