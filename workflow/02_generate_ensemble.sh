#!/bin/bash
# Step 2: Generate the active scenario design's realizations, one draw per
# array task.
#
# Every design generates from its own namespaced seed stream; the array index
# is the ensemble-draw index k (0..K-1, K = design.n_ensemble_draws). The
# pool-owning designs (`monte_carlo_resampled`, `hazard_filling_*`) generate
# their candidate pool (+ hazard_image.npz) here, and the pool is regenerated
# per draw (src/scenario_designs.py::pool_slug(draw)), so the array index
# applies to pools too; the two DU hazard-filling designs share the pool of a
# given draw. Three draws are staged per matched design: d0 is searched, d1-d2
# serve the SI draw-sensitivity re-evaluation (campaign_design.md).
#
# Env inputs: NYCOPT_ENV_FILE (optional; the design via NYCOPT_SCENARIO_DESIGN),
# NYCOPT_CANDIDATE_POOL_N (hazard-filling pool size; campaign 1000000, see
# workflow/supplemental/gen_pool_shards.sh for the sharded build),
# NYCOPT_ENSEMBLE_FORCE=1 to overwrite an already-staged slug.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_mc_production.env \
#          --array=0-2 workflow/02_generate_ensemble.sh
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_hazfill_stat_production.env \
#          --array=0-2 workflow/02_generate_ensemble.sh
#
# Output: outputs/synthetic_ensembles/{slug}/ per draw (search ensemble, or
# the design's candidate/resampling pool).
#
#SBATCH --job-name=gen_ensemble
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --output=logs/gen_ensemble_%A_%a.out
#SBATCH --error=logs/gen_ensemble_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

# Array index = ensemble draw k. config.py reads it back as SCENARIO_ENSEMBLE_DRAW,
# so shell and Python resolve the same staged slug.
export NYCOPT_ENSEMBLE_DRAW="${SLURM_ARRAY_TASK_ID:-${NYCOPT_ENSEMBLE_DRAW:-0}}"

# Generator is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[gen_ensemble] design=${NYCOPT_SCENARIO_DESIGN:-<default>} draw=${NYCOPT_ENSEMBLE_DRAW}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/generate_stochastic_ensemble.py --draw "${NYCOPT_ENSEMBLE_DRAW}"
echo "[gen_ensemble] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
