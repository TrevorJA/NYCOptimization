#!/bin/bash
# Step 14: Render the SI + exploratory figure tiers from the unified registry.
#
# The figure sequence lives in src/figures/registry.py (one FigureSpec per
# figure, tiered manuscript / si / exploratory) and renders through the single
# driver scripts/main/figures.py. This step renders the SI tier (the
# manuscript tier is step 13); figures whose data needs are absent are skipped
# with a message naming the missing artifact. Pure post-processing on the
# persisted per-SOW re-eval cubes and scored CSVs — no simulation.
#
# Requires: every campaign design's re-eval on the common E_test tag
# (workflow steps 08/09) with robustness_scorecard.csv (+ the
# robustness_scorecard_criteria.csv companion) and baseline/ cubes.
#
# Everything comes from the environment — no positional args, no value flags:
#   NYCOPT_REEVAL_TAG    E_test re-eval tag; defaults to the campaign tag
#                        (etest_kn_50yr_n25000_first25ch). Set it only for a
#                        non-campaign re-eval.
#   NYCOPT_RESULTS_SLUG  moea slug shared by the campaign runs (ffmp_obj8)
#   NYCOPT_FOCAL_CRITERION  focal criterion set (default: compromise)
#   FIGURES              optional comma-separated subset (names or stems)
#
# Submit (from repo root):
#   sbatch workflow/14_results_figures.sh
#
# Sizing: loads three <10 MB parquet cubes and renders matplotlib figures;
# minutes on a single shared core.
#
#SBATCH --job-name=results_figures
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/results_figures_%j.out
#SBATCH --error=logs/results_figures_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

FIGURES="${FIGURES:-}"

echo "[results_figures] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[results_figures] tag=${NYCOPT_REEVAL_TAG:-<campaign default>} slug=${NYCOPT_RESULTS_SLUG:-ffmp_obj8}"

if [[ -n "${FIGURES}" ]]; then
    ARGS=""
    for f in ${FIGURES//,/ }; do ARGS="${ARGS} --figure ${f}"; done
    # shellcheck disable=SC2086
    python3 -u -m scripts.main.figures ${ARGS}
else
    python3 -u -m scripts.main.figures --tier si
fi

echo "[results_figures] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
