#!/bin/bash
# Step 14: Render the cross-design RESULTS-figure sequence.
#
# Replaces the retired outputs/figures/comparison/*/robustness set. The
# sequence lives in scripts/main/results_figures.py, whose FIGURES registry
# defines what "the sequence" is (phase 1: satisficing diagnostics; later
# phases extend the registry). Pure post-processing on the persisted per-SOW
# re-eval cubes — no simulation, so the job is small and fast.
#
# Requires: every campaign design's re-eval on the common E_test tag
# (workflow steps 08/09) with robustness_scorecard.csv + baseline/ cubes.
#
# Everything comes from the environment — no positional args, no value flags:
#   NYCOPT_REEVAL_TAG    E_test re-eval tag; defaults to the campaign spec's
#                        tag. The interim 200-SOW subset MUST set it, e.g.
#                        NYCOPT_REEVAL_TAG=etest_kn_50yr_n25000_first10ch
#   NYCOPT_RESULTS_SLUG  moea slug shared by the campaign runs (ffmp_obj8)
#   FIGURES              optional comma-separated subset of the sequence
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_REEVAL_TAG=etest_kn_50yr_n25000_first10ch \
#       workflow/14_results_figures.sh
#
# Sizing: loads three <10 MB parquet cubes and renders matplotlib PNGs;
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
    python3 -u -m scripts.main.results_figures ${ARGS}
else
    python3 -u -m scripts.main.results_figures --all
fi

echo "[results_figures] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
