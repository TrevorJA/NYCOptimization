#!/bin/bash
#SBATCH --job-name=hsd_diag
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/hazard_support_decomposition_%j.out
#SBATCH --error=logs/hazard_support_decomposition_%j.err

# Hazard-support decomposition of the E_test design contrast
# (docs/notes/methods/hazard_support_decomposition.md; SI Text S10)
#
# Zero simulation. Stage A reduces the E_test sub-window hazard image
# (125,000 x 8) against the three P=1e6 candidate-pool images on the six
# campaign selection axes; stage B (auto-gated on both matched designs having
# a re-eval leaf on HSD_REEVAL_TAG — by default the interim first10ch subset
# the go/no-go sets were scored on) re-scores the design contrast per support
# stratum from those persisted per-SOW cubes. This is a PRE-CAMPAIGN decision
# instrument; it is not meant to wait for the production campaign. Figures
# regenerate from tables.
#
# Prerequisites (stage A):
#   outputs/synthetic_ensembles/etest_kn_50yr_n25000/hazard_image_subwindows.npz
#   outputs/synthetic_ensembles/etest_kn_50yr_n25000/{hazard_image,forcing_profiles}.npz
#   outputs/synthetic_ensembles/statpool_10yr_n1000000_d{0,1,2}/hazard_image.npz
# Stage B additionally needs re-eval leaves for both matched designs on the tag
# and a sourced production env file (NYCOPT_REGRET_TAU, NYCOPT_CRITERIA_VARIANT):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env \
#          workflow/supplemental/hazard_support_decomposition.sh
#
# All settings live in supplemental_config.py (HSD_ section) — no CLI value
# flags. Smoke: submit with --export=ALL,NYCOPT_HSD_SMOKE=1 (P=2,000 pools,
# first 200 SOWs; artifacts prefixed smoke_). NYCOPT_HSD_REFRESH=1 forces the
# stage-A recompute.
#
# Sizing: memory is dominated by one 1e6 x 8 pool image plus its cKDTree
# (< 2 GB resident; pools load sequentially). The pool self-NN query
# (1e6 points, k=2, 6-D) is the long pole at a few minutes per pool with
# 8 threaded workers; everything else is seconds. 24 GB / 8 cpus / 90 min is
# generous headroom.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[hsd] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (smoke=${NYCOPT_HSD_SMOKE:-0})"
python3 -u scripts/supplemental/hazard_support_run.py
python3 -u scripts/supplemental/hazard_support_figures.py
echo "[hsd] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
