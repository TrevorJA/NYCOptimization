#!/bin/bash
# Step 3: Hazard-filling selection — select the search ensemble from the active
# design's OWN candidate pool (staged by step 02).
#
# Applies ONLY to the hazard-filling designs (NYCOPT_SCENARIO_DESIGN ∈
# {hazard_filling_stationary, hazard_filling_du, hazard_filling_absolute}), via
# scripts/main/select_hazard_filling.py. Every other design GENERATES its search
# ensemble directly in step 02 and skips this step — including input_stratified,
# which is an LHS over the generator's forcing parameters with realizations
# generated at each design point, not a subsample.
#
# No array: all K draws are selected in one job, each from its own draw's pool
# image (step 02 regenerates the pool per draw, so a draw re-rolls both the pool
# and the LHS anchor plan). Serial (the selector is single-process); the daily
# pool is read only for the selected realizations. Three draws are staged (d0
# searched; d1-d2 for the SI draw-sensitivity re-evaluation); the production
# pool is selected by passing NYCOPT_CANDIDATE_POOL_N=1000000.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary,NYCOPT_CANDIDATE_POOL_N=1000000 \
#          workflow/03_subsample_ensemble.sh
#
#SBATCH --job-name=select_hazfill
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/select_hazfill_%j.out
#SBATCH --error=logs/select_hazfill_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
# Optional env file — the design can also come straight from
# --export=ALL,NYCOPT_SCENARIO_DESIGN=...
nycopt_source_env_file optional

# Selector is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[select_hazfill] design=${NYCOPT_SCENARIO_DESIGN:-<default>}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/select_hazard_filling.py
echo "[select_hazfill] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
