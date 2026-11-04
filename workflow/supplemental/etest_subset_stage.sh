#!/bin/bash
# Finalize a demand-staged E_test for re-evaluation, metadata only (no simulation):
#   1. scripts/supplemental/merge_demand_utilization_realized.py --variant VARIANT —
#      concatenates every chunk's demand_utilization_realized.csv (written when
#      the chunk's scaled NYC demand series was staged) into the parent directory
#      and reduces it per SOW (realized versus nominal u; a diagnostic, never a
#      correction). Fails naming any chunk without the file.
#   2. scripts/supplemental/make_etest_subset.py --pool SLUG — stages the campaign
#      chunk-prefix subset (default: the first 25 chunks = 500 SOWs,
#      `{slug}_first25ch`) as three metadata files pointing at the parent's
#      already-staged chunk dirs, and runs assert_staged_etest_contract on it
#      (LHS x R > 1, reserved seed stream, no per-axis box override, the
#      utilization keys at the registry bounds). NYCOPT_FORCE=1 rebuilds an
#      existing subset.
#
# Runs after prep_etest_chunks.sh or stage_nyc_demand_etest_chunks.sh (chain with
# --dependency=afterok:<array_jobid>); the staging contract lists the demand series
# as a required chunk file, so the subset cannot be built before the chunks carry
# it. Seconds. Every re-evaluation step (05, 08, 09, 09b, 10, 11) then takes
# NYCOPT_REEVAL_ENSEMBLE_PRESET=<subset slug>.
#
# Submit (from repo root):
#   sbatch --dependency=afterok:<jobid> \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/etest_subset_stage.sh
#
#SBATCH --job-name=etest_subset
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/etest_subset_%j.out
#SBATCH --error=logs/etest_subset_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"
POOL_SLUG="$(python3 -c "from src.etest import get_etest_variant; print(get_etest_variant('${VARIANT}').slug)")"
FORCE_FLAG=""
[[ "${NYCOPT_FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

echo "[etest_subset] merge realized utilization variant=${VARIANT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/merge_demand_utilization_realized.py --variant "${VARIANT}"
echo "[etest_subset] prefix subset of ${POOL_SLUG}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
# shellcheck disable=SC2086
python3 -u -m scripts.supplemental.make_etest_subset --pool "${POOL_SLUG}" ${FORCE_FLAG}
echo "[etest_subset] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
