#!/bin/bash
# Sharded E_test generation: one array task per chunk-aligned slice of the LHS
# forcing profiles. The serial step-12 build measured ~1.65 h per 500-realization
# chunk (~84 h for all 50), so E_test parallelizes across the array exactly like
# the candidate pools: rows are keyed to the GLOBAL realization index (methods
# §3.4), so the shard union equals a serial generation. Each shard writes its
# daily chunk dir(s) plus a hazard-image shard npz (the per-shard completion
# marker: rerunning a finished shard is a no-op; delete the npz to force).
# Shard boundaries must align to chunk boundaries — with the campaign sizing
# (N_theta=1000, R=25, chunk=500 -> 50 chunks) any shard count that divides 50
# works; 50 shards (1 chunk each, default) maximizes parallelism.
#
# Submit (defaults: kn variant, 50 shards), then chain gen_etest_merge.sh:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/gen_etest_shards.sh
#
#SBATCH --job-name=gen_etest_shard
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/gen_etest_shard_%A_%a.out
#SBATCH --error=logs/gen_etest_shard_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"

# Generator is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_ENSEMBLE_SHARD_COUNT="${NYCOPT_ENSEMBLE_SHARD_COUNT:-50}"
export NYCOPT_ENSEMBLE_SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"

echo "[gen_etest_shard] variant=${VARIANT} shard ${NYCOPT_ENSEMBLE_SHARD_INDEX}/${NYCOPT_ENSEMBLE_SHARD_COUNT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u -m scripts.main.generate_test_ensemble --variant "${VARIANT}"
echo "[gen_etest_shard] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
