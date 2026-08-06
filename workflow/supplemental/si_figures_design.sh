#!/bin/bash
#SBATCH --job-name=si_figures_design
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/si_figures_design_%j.out
#SBATCH --error=logs/si_figures_design_%j.err

# Regenerate the Supporting Information figures that justify the EXPERIMENTAL
# DESIGN. These all rest on pre-campaign experiments, so this driver is
# runnable today and does not wait on the search campaign.
#
# It is a FIGURE driver: it re-renders from each study's already-computed
# results and never re-runs the underlying experiment. The compute stages keep
# their own per-study drivers (workflow/supplemental/epsilon_calibration.sh,
# objective_sensitivity.sh, satisfaction_factor.sh, ...); this script only
# dispatches to their *_figures.py counterparts, so nothing is duplicated.
#
# A study whose results are absent is skipped with a message, not an error.
#
# Companion drivers:
#   workflow/13_main_figures.sh                  main-manuscript figure sequence
#   workflow/supplemental/si_figures_results.sh  SI figures needing campaign output
#
# Environment:
#   SI_ONLY   optional comma-separated subset, e.g. SI_ONLY=forcing_parameterization
#
# Submit (from repo root):
#   sbatch workflow/supplemental/si_figures_design.sh
#   sbatch --export=ALL,SI_ONLY=forcing_parameterization \
#          workflow/supplemental/si_figures_design.sh

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

SUP="outputs/supplemental"
CMIP6_CSV="$(python3 -c 'import config; print(config.ENSEMBLE_FORCING_MEAN_FRAC_CSV)')"

echo "[si-design] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# name | prerequisite path | command
nycopt_run_figure_studies si-design <<EOF
forcing_parameterization|${CMIP6_CSV}|python3 -u scripts/supplemental/figures_forcing_parameterization.py
epsilon_calibration|${SUP}/epsilon_calibration|python3 -u scripts/supplemental/epsilon_calibration_figures.py
objective_sensitivity|${SUP}/objective_sensitivity|python3 -u scripts/supplemental/objective_sensitivity_figures.py
satisfaction_factor|${SUP}/satisfaction_factor|python3 -u scripts/supplemental/satisfaction_factor_figures.py
flood_objective|${SUP}/flood_objective|python3 -u scripts/supplemental/flood_objective_figures.py
flood_exceedance_baseline|${SUP}/flood_objective|python3 -u scripts/supplemental/flood_exceedance_baseline_figures.py
EOF

echo "[si-design] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
