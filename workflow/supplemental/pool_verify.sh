#!/bin/bash
# Standalone post-merge verification of a sharded candidate pool.
#
# gen_pool_merge.sh runs these two checks inline, but it DELETES the shard files
# as part of the canonical merge — so once a merge has run, the checks can only be
# re-run separately (they regenerate from the recorded generation config, they do
# not read shards). This script is that separate entry point.
#
#   verify_shard_boundaries.py  regenerates realizations at shard boundaries under
#                               a THIRD partition and asserts row equality with the
#                               staged image (tolerance = 1% of each axis's robust
#                               range; a real partition bug shows O(range) errors
#                               on many axes, era-level FP drift stays far below).
#   verify_prefix_identity.py   asserts the standalone P=2,000 smoke pool is
#                               row-identical to the large pool's leading rows.
#
# Submit (per draw):
#   sbatch --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50,\
# NYCOPT_ENSEMBLE_DRAW=0,NYCOPT_NESTEDP_POOL_SLUG=statpool_10yr_n1000000_d0,\
# NYCOPT_NESTEDP_SMOKE_SLUG=statpool_10yr_n2000_d0 workflow/supplemental/pool_verify.sh
#
#SBATCH --job-name=pool_verify
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pool_verify_%j.out
#SBATCH --error=logs/pool_verify_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_SCENARIO_DESIGN="${NYCOPT_SCENARIO_DESIGN:-hazard_filling_stationary}"
export NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1
: "${NYCOPT_NESTEDP_POOL_SLUG:?set NYCOPT_NESTEDP_POOL_SLUG via --export}"

echo "[pool_verify] ${NYCOPT_NESTEDP_POOL_SLUG} (smoke=${NYCOPT_NESTEDP_SMOKE_SLUG:-<default d0>})  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[pool_verify] shard boundaries (third-partition regeneration)"
python3 -u scripts/supplemental/verify_shard_boundaries.py
echo "[pool_verify] smoke-pool prefix identity"
python3 -u scripts/supplemental/verify_prefix_identity.py
echo "[pool_verify] OK  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
