#!/bin/bash
# Merge staged hazard-image shards into the canonical pool artifacts, then
# verify the merged image: (a) regenerate realizations at shard boundaries
# under a third partition and assert exact row equality
# (verify_shard_boundaries.py); (b) assert the standalone P=2,000 smoke pool is
# bit-identical to the image's leading rows (verify_prefix_identity.py).
#
# Submit after the shard array (chain with --dependency=afterok:<array_jobid>):
#   sbatch --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50,\
# NYCOPT_POOL_SLUG=statpool_10yr_n1000000_d0 workflow/supplemental/staging/pool_merge.sh
#
#SBATCH --job-name=pool_merge
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pool_merge_%j.out
#SBATCH --error=logs/pool_merge_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_SCENARIO_DESIGN="${NYCOPT_SCENARIO_DESIGN:-hazard_filling_stationary}"
export NYCOPT_CANDIDATE_POOL_N="${NYCOPT_CANDIDATE_POOL_N:-1000000}"
export NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1
export NYCOPT_ENSEMBLE_SHARD_COUNT="${NYCOPT_ENSEMBLE_SHARD_COUNT:-50}"
export NYCOPT_ENSEMBLE_DRAW="${NYCOPT_ENSEMBLE_DRAW:-0}"
export NYCOPT_POOL_SLUG="${NYCOPT_POOL_SLUG:-statpool_10yr_n1000000_d0}"   # the production pool

echo "[pool_merge] merging shards for ${NYCOPT_POOL_SLUG}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
NYCOPT_ENSEMBLE_MERGE_SHARDS=1 \
python3 -u scripts/main/generate_stochastic_ensemble.py --draw "${NYCOPT_ENSEMBLE_DRAW}"

echo "[pool_merge] verifying shard boundaries (third-partition regeneration)"
python3 -u scripts/supplemental/verify_shard_boundaries.py

echo "[pool_merge] verifying smoke-pool prefix identity"
python3 -u scripts/supplemental/verify_prefix_identity.py

echo "[pool_merge] done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
