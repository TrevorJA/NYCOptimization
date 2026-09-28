#!/bin/bash
# Hazard image of an already-staged L=SCENARIO_YEARS search ensemble.
#
# Wraps scripts/supplemental/compute_staged_hazard_image.py, the post-hoc scorer
# for ensembles whose generation path never writes a hazard image: `monte_carlo`
# stages flows only (no selection happens, so step 02 skips the SSI/POT pass),
# yet the realized-composition diagnostics — manuscript figure 4
# (src/plotting/ensemble_composition.py) above all — need its coordinates on the
# same 8-axis convention as the candidate pool. The hazard-filling designs get
# their image from step 03 and do not need this script.
#
# Refuses to recompute over an existing hazard_image.npz (the scorer returns
# early); delete the file to force a rebuild.
#
# Env inputs:
#   NYCOPT_HAZIMG_SLUG   staged slug to score. Default: the active design's
#                        search_ensemble_slug(NYCOPT_ENSEMBLE_DRAW), so the
#                        design's env file plus NYCOPT_SEARCH_N resolves it.
#   NYCOPT_ENV_FILE      optional; supplies NYCOPT_SCENARIO_DESIGN et al.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_mc_production.env \
#          workflow/supplemental/staged_hazard_image.sh
#   sbatch --export=ALL,NYCOPT_HAZIMG_SLUG=fixprob_10yr_n300_d0 \
#          workflow/supplemental/staged_hazard_image.sh
#
# Sizing: single-process SSI-6 + POT scoring of N realizations x L yr (one
# window each) — minutes at campaign N, next to nothing beside the E_test image.
#
#SBATCH --job-name=staged_hazimg
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/staged_hazimg_%j.out
#SBATCH --error=logs/staged_hazimg_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

# Scorer is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_ENSEMBLE_DRAW="${NYCOPT_ENSEMBLE_DRAW:-0}"

echo "[staged_hazimg] slug=${NYCOPT_HAZIMG_SLUG:-<active design draw ${NYCOPT_ENSEMBLE_DRAW}>} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/compute_staged_hazard_image.py
echo "[staged_hazimg] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
