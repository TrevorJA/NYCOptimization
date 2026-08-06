#!/bin/bash
#SBATCH --job-name=si_figures_results
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/si_figures_results_%j.out
#SBATCH --error=logs/si_figures_results_%j.err

# Regenerate the Supporting Information figures that depend on CAMPAIGN
# OUTPUTS — the staged E_test hazard image, the re-evaluation cubes, and the
# baseline re-eval. Safe to run before the campaign finishes: every study whose
# prerequisites are absent is skipped with a message rather than failing.
#
# As with its design-side companion, this is a FIGURE driver that dispatches to
# existing per-study scripts; the compute stages keep their own drivers.
#
# NOT covered here (they are per-run and parameterized by an env file, so they
# stay with their own workflow steps):
#   workflow/07_run_diagnostics.sh   search runtime diagnostics
#   workflow/10_compare_designs.sh   between-design comparison figures
#   workflow/11_scenario_discovery.sh  hazard-space mechanism test
#
# Companion drivers:
#   workflow/13_main_figures.sh                 main-manuscript figure sequence
#   workflow/supplemental/si_figures_design.sh  pre-campaign design-support SI
#
# Environment:
#   SI_ONLY                        optional comma-separated subset
#   NYCOPT_REEVAL_ENSEMBLE_PRESET  E_test slug, for the hazard overlay
#
# Submit (from repo root):
#   sbatch workflow/supplemental/si_figures_results.sh

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

SUP="outputs/supplemental"
ETEST_SLUG="$(python3 -c 'from src import etest as e; print(e.E_TEST_VARIANTS[e.E_TEST_VARIANT].slug)')"
ETEST_HAZARD="outputs/synthetic_ensembles/${ETEST_SLUG}/hazard_image.npz"

echo "[si-results] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[si-results] E_test = ${ETEST_SLUG}"

# name | prerequisite path | command
# robustness_threshold runs its anchor step first; the anchor JSON is cached, so
# a repeat pass is near-instant (NYCOPT_RTD_REFRESH=1 forces the recompute).
nycopt_run_figure_studies si-results <<EOF
robustness_threshold|${SUP}/robustness_threshold_diagnostics|python3 -u scripts/supplemental/robustness_threshold_anchor.py && python3 -u scripts/supplemental/robustness_threshold_figures.py
etest_hazard_overlay|${ETEST_HAZARD}|python3 -u scripts/main/plot_etest_hazard_overlay.py
hazard_selector_diagnostics|${SUP}/hazard_selector_diagnostics|python3 -u scripts/supplemental/diagnose_hazard_selectors.py
cv_axis_footprint|${SUP}/cv_axis_footprint|python3 -u scripts/supplemental/diagnose_cv_axis_footprint.py
EOF

echo "[si-results] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
