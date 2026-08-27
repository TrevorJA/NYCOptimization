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
# (docs/notes/methods/hazard_support_decomposition.md; SI Text S10). Zero
# simulation. Stage A scores the E_test sub-window hazard image (125,000 x 8)
# against the three P=1e6 candidate-pool images on the six campaign selection
# axes and labels support strata; stage B (run when both matched designs have
# a re-eval leaf on HSD_REEVAL_TAG, default the campaign re-eval preset)
# re-scores the design contrast per stratum from the persisted per-SOW cubes.
# Figures regenerate from tables.
#
# Prerequisites (stage A):
#   outputs/synthetic_ensembles/etest_kn_50yr_n25000/hazard_image_subwindows.npz
#   outputs/synthetic_ensembles/etest_kn_50yr_n25000/{hazard_image,forcing_profiles}.npz
#   outputs/synthetic_ensembles/statpool_10yr_n1000000_d{0,1,2}/hazard_image.npz
# Stage B additionally needs the matched designs' re-eval leaves on the tag and
# a sourced production env file (NYCOPT_REGRET_TAU, NYCOPT_CRITERIA_VARIANT).
#
# Submit:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env \
#          workflow/supplemental/hazard_support_decomposition.sh
# Switches: NYCOPT_HSD_SMOKE=1 (P=2,000 pools, first 200 SOWs, smoke_ prefix),
# NYCOPT_HSD_REFRESH=1 (stage-A recompute), NYCOPT_HSD_FIGURES_ONLY=1 (redraw
# from persisted tables). Settings in supplemental_config.py (HSD_ section).
#
# Outputs: outputs/supplemental/hazard_support_decomposition/{tables,figures}.
# Sizing: one 1e6 x 8 pool image + cKDTree resident at a time (< 2 GB); the
# pool self-NN query is a few minutes per pool on 8 threads.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[hsd] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (smoke=${NYCOPT_HSD_SMOKE:-0}, figures_only=${NYCOPT_HSD_FIGURES_ONLY:-0})"
if [[ "${NYCOPT_HSD_FIGURES_ONLY:-0}" != "1" ]]; then
    python3 -u scripts/supplemental/hazard_support_run.py
fi
python3 -u scripts/supplemental/hazard_support_figures.py
echo "[hsd] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
