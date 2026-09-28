#!/bin/bash
#SBATCH --job-name=hazard_examples
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/hazard_examples_%j.out
#SBATCH --error=logs/hazard_examples_%j.err

# The hazard metrics illustrated on example HF realizations (manuscript
# Section 3.1.3): the HF search ensemble's hazard characteristics with a few
# example realizations highlighted, next to each example's SSI-6 series and
# annual peak discharge. A figure driver only: reads the staged HF ensemble
# (hazard image + daily traces of the N selected realizations); no simulation,
# no pool. Both left-panel geometries (3-D scatter, parallel axes) are drawn.
#
# Prerequisites (workflow steps 02-03, draw NYCOPT_ENSEMBLE_DRAW, default 0):
#   outputs/synthetic_ensembles/hazfill_stat_abs_10yr_n300_d0/hazard_image.npz
#   outputs/synthetic_ensembles/hazfill_stat_abs_10yr_n300_d0/catchment_inflow_mgd.hdf5
#
# Submit (from repo root):
#   sbatch workflow/supplemental/hazard_examples.sh
#   sbatch --export=ALL,NYCOPT_ENSEMBLE_DRAW=1 workflow/supplemental/hazard_examples.sh
# Settings (example targets, 3-D triple, geometries): supplemental_config.py, HEX_ section.
#
# Outputs: outputs/supplemental/hazard_examples/{figures,tables}.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[hex] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (draw=${NYCOPT_ENSEMBLE_DRAW:-0})"
python3 -u scripts/supplemental/hazard_examples_figures.py
echo "[hex] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
