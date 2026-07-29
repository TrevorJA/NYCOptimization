#!/bin/bash
# Sharded stationary candidate-pool generation (stream-only): one array task per
# contiguous global-index slice. Rows are keyed to the GLOBAL realization index
# (methods §3.4), so the shard union is bit-identical to a serial generation;
# the merge step (gen_pool_merge.sh) verifies and writes the canonical
# artifacts. Generation is effectively single-threaded (measured TotalCPU ~
# Elapsed), so shards request 2 cores — parallelism comes from the array.
#
# Submit (defaults: P=1e6, 50 shards):
#   sbatch --array=0-49 --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50 \
#          workflow/supplemental/gen_pool_shards.sh
#
#SBATCH --job-name=gen_pool_shard
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=20:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/gen_pool_shard_%A_%a.out
#SBATCH --error=logs/gen_pool_shard_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_SCENARIO_DESIGN="${NYCOPT_SCENARIO_DESIGN:-hazard_filling_stationary}"
export NYCOPT_CANDIDATE_POOL_N="${NYCOPT_CANDIDATE_POOL_N:-1000000}"
export NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1
export NYCOPT_ENSEMBLE_SHARD_COUNT="${NYCOPT_ENSEMBLE_SHARD_COUNT:-50}"
export NYCOPT_ENSEMBLE_SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
export NYCOPT_ENSEMBLE_DRAW="${NYCOPT_ENSEMBLE_DRAW:-0}"

echo "[gen_pool_shard] P=${NYCOPT_CANDIDATE_POOL_N} shard ${NYCOPT_ENSEMBLE_SHARD_INDEX}/${NYCOPT_ENSEMBLE_SHARD_COUNT} draw=${NYCOPT_ENSEMBLE_DRAW}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/generate_stochastic_ensemble.py --draw "${NYCOPT_ENSEMBLE_DRAW}"
echo "[gen_pool_shard] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
