#!/bin/bash
# Hazard-axis selection on the production candidate pools
# (hazard_axis_selection_run.py then hazard_axis_selection_figures.py; design,
# decision rule and findings in docs/notes/methods/hazard_axis_selection.md).
# Selection-level only: reads statpool_10yr_n{P}_d{0,1}/hazard_image.npz, no
# simulation. Both draws run in one task; the worker count follows the CPUs.
#
# Measured on one workstation at P = 1e6, N = 300 with 8 workers: 39 min (2,455 sets).
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_SEARCH_N=300 \
#          workflow/supplemental/hazard_axis_selection.sh
#
# Output: outputs/supplemental/hazard_axis_selection/{tables,figures}/
#
#SBATCH --job-name=hazard_axsel
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hazard_axsel_%j.out
#SBATCH --error=logs/hazard_axsel_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export NYCOPT_SCENARIO_DESIGN="${NYCOPT_SCENARIO_DESIGN:-hazard_filling_stationary}"
export NYCOPT_CANDIDATE_POOL_N="${NYCOPT_CANDIDATE_POOL_N:-1000000}"
export NYCOPT_SEARCH_N="${NYCOPT_SEARCH_N:-300}"
export NYCOPT_HAXSEL_WORKERS="${NYCOPT_HAXSEL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "[hazard_axsel] P=${NYCOPT_CANDIDATE_POOL_N} N=${NYCOPT_SEARCH_N} workers=${NYCOPT_HAXSEL_WORKERS}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "${NYCOPT_HAXSEL_FIGURES_ONLY:-0}" != "1" ]]; then
  python3 -u scripts/supplemental/hazard_axis_selection_run.py
fi
python3 -u scripts/supplemental/hazard_axis_selection_figures.py
echo "[hazard_axsel] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
