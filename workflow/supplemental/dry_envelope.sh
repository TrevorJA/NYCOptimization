#!/bin/bash
#SBATCH --job-name=dry_envelope
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dry_envelope_%j.out
#SBATCH --error=logs/dry_envelope_%j.err

# Dry end of the E_test forcing box (docs/notes/methods/campaign_design.md §5,
# forcing_parameterization.md; SI Text S6). Stream-only, no Pywr-DRB run.
#
# Local leg (default; also runnable on a laptop, ~45 min): paired 10-yr windows
# at fixed annual-volume multipliers spanning the E_test lower bound at the
# current and candidate widening margins, scored under the current hazard
# rules against a stationary sample and the historical record's windows.
#
# Production leg (NYCOPT_DRYENV_PRODUCTION=1): bins the staged E_test SOWs on
# their volume multiplier and scores their sub-window drought tails against
# the P=1e6 pool image. Requires images scored under the CURRENT rules (the
# reader refuses any other), i.e. run AFTER the June 1 recompute of
#   outputs/synthetic_ensembles/etest_kn_50yr_n25000/hazard_image_subwindows.npz
#   outputs/synthetic_ensembles/statpool_10yr_n1000000_d0/hazard_image.npz
#
# Submit:
#   sbatch workflow/supplemental/dry_envelope.sh
#   sbatch --export=ALL,NYCOPT_DRYENV_PRODUCTION=1 workflow/supplemental/dry_envelope.sh
# Settings in supplemental_config.py (DRYENV_ section); tables and figures
# under outputs/supplemental/dry_envelope/.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[dryenv] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (production=${NYCOPT_DRYENV_PRODUCTION:-0})"
python3 -u scripts/supplemental/dry_envelope_run.py
python3 -u scripts/supplemental/dry_envelope_figures.py
echo "[dryenv] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
