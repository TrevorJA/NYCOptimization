#!/bin/bash
# Merge staged E_test hazard-image shards into the canonical slug artifacts
# (hazard_image.npz, forcing_profiles.npz, chunk_index.json, _meta.json,
# manifest.json), through the same finalizer the serial step-12 path uses. The
# merge verifies every daily chunk dir is complete and tiles [0, N) before
# writing anything, refits the (deterministic) generators to persist the
# per-realization theta, and ends with the staged-E_test contract check
# (assert_staged_etest_contract) exactly like a serial run.
#
# Submit after the shard array (chain with --dependency=afterok:<array_jobid>):
#   sbatch --dependency=afterok:<jobid> \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/gen_etest_merge.sh
#
#SBATCH --job-name=gen_etest_merge
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gen_etest_merge_%j.out
#SBATCH --error=logs/gen_etest_merge_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_ENSEMBLE_MERGE_SHARDS=1

echo "[gen_etest_merge] variant=${VARIANT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u -m scripts.main.generate_test_ensemble --variant "${VARIANT}"
echo "[gen_etest_merge] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
