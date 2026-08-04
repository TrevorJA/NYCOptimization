#!/bin/bash
# Step 9b: Merge a chunked test-ensemble re-evaluation (metrics-only).
#
# Companion to step 09 run with NYCOPT_CHUNK_MERGE=off (the campaign mode):
# the simulate jobs only write per-(solution, chunk) unit files; this job
# assembles them into reeval_raw.parquet + objectives_summary.csv + the
# robustness scorecards through the same persist path, so the artifacts are
# byte-compatible with an in-job merge. Single process, resumable, refuses on
# missing units unless NYCOPT_CHUNK_MERGE_ALLOW_PARTIAL=1.
#
# Env identity MUST match the step-09 submission (same env file, same
# NYCOPT_REEVAL_ENSEMBLE_PRESET, same NYCOPT_CHUNK_POLICIES).
#
# Submit (from repo root, after the last simulate job):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/<arm>.env,\
# NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000,NYCOPT_CHUNK_POLICIES=<merged.ref> \
#          workflow/09b_merge_test_chunks.sh
#
#SBATCH --job-name=merge_test_chunks
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_test_chunks_%j.out
#SBATCH --error=logs/merge_test_chunks_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set the chunked master slug explicitly, e.g. etest_kn_50yr_n25000}"
FORMULATION="${FORMULATION:-ffmp}"
SEED="${SEED:-}"

ARGS="--formulation ${FORMULATION}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

echo "=== Merge test chunks: formulation=${FORMULATION} "\
"master=${NYCOPT_REEVAL_ENSEMBLE_PRESET} policies=${NYCOPT_CHUNK_POLICIES:-baseline} ==="
python3 -m scripts.main.merge_test_chunks ${ARGS}
echo "=== Completed: $(date) ==="
