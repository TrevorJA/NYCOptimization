#!/bin/bash
# Stage E_test from a larger generated design, metadata only (no simulation):
#   1. scripts/supplemental/make_etest_subset.py --pool PARENT — cuts the registry's
#      E_test (src/etest.py: 500 SOWs x R = 25 x 50 yr = the leading 25 chunks,
#      `etest_kn_50yr_n12500`) as three metadata files plus the sliced forcing
#      profiles and nominal-utilization table, all pointing at the parent's
#      already-staged chunk dirs, and runs assert_staged_etest_contract on it
#      (LHS x R > 1, reserved seed stream, no per-axis box override, the
#      utilization keys at the registry bounds). NYCOPT_FORCE=1 re-cuts an
#      existing E_test directory.
#   2. scripts/supplemental/merge_demand_utilization_realized.py --slug E_TEST —
#      concatenates every E_test chunk's demand_utilization_realized.csv (written
#      when the chunk's scaled NYC demand series was staged) into the E_test
#      directory and reduces it per SOW (realized versus nominal u; a diagnostic,
#      never a correction). Fails naming any chunk without the file.
#
# Runs after prep_etest_chunks.sh or stage_nyc_demand_etest_chunks.sh (chain with
# --dependency=afterok:<array_jobid>); the staging contract lists the demand series
# as a required chunk file, so E_test cannot be cut before its chunks carry it.
# Seconds. Every re-evaluation step (05, 08, 09, 09b, 10, 11) then takes
# NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500. A design generated at the
# registry size needs neither step 1 nor this wrapper's --slug override.
#
# Env inputs:
#   NYCOPT_ETEST_PARENT   generated parent slug (default etest_kn_50yr_n25000, the
#                         1,000-SOW design of 2026-08-19 whose leading half is E_test)
#   NYCOPT_ETEST_VARIANT  registry variant (default kn)
#
# Submit (from repo root):
#   sbatch --dependency=afterok:<jobid> \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/etest_subset_stage.sh
#
#SBATCH --job-name=etest_stage
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/etest_stage_%j.out
#SBATCH --error=logs/etest_stage_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"
PARENT_SLUG="${NYCOPT_ETEST_PARENT:-etest_kn_50yr_n25000}"
ETEST_SLUG="$(python3 -c "from src.etest import get_etest_variant; print(get_etest_variant('${VARIANT}').slug)")"
FORCE_FLAG=""
[[ "${NYCOPT_FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

echo "[etest_stage] cut ${ETEST_SLUG} from ${PARENT_SLUG}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
# shellcheck disable=SC2086
python3 -u -m scripts.supplemental.make_etest_subset --pool "${PARENT_SLUG}" --variant "${VARIANT}" ${FORCE_FLAG}
echo "[etest_stage] merge realized utilization into ${ETEST_SLUG}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/merge_demand_utilization_realized.py --slug "${ETEST_SLUG}"
echo "[etest_stage] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
