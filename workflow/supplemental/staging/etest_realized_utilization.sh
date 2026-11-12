#!/bin/bash
# Merge E_test's realized NYC utilization, metadata only (no simulation):
# scripts/supplemental/merge_demand_utilization_realized.py concatenates every
# chunk's demand_utilization_realized.csv (written when the chunk's scaled NYC
# demand series was staged) into the E_test directory and reduces it per SOW
# (realized versus nominal u; a diagnostic, never a correction). Fails naming
# any chunk without the file.
#
# The last step of the E_test build: runs after staging/etest_prep_chunks.sh
# (chain with --dependency=afterok:<array_jobid>). Seconds. Every
# re-evaluation step (10, 11, 12 and the supplemental wrappers) then takes
# NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500.
#
# Env inputs:
#   NYCOPT_ETEST_VARIANT  registry variant (default kn)
#
# Submit (from repo root):
#   sbatch --dependency=afterok:<jobid> \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/staging/etest_realized_utilization.sh
#
#SBATCH --job-name=etest_realized_u
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/etest_realized_u_%j.out
#SBATCH --error=logs/etest_realized_u_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"

echo "[etest_realized_u] variant=${VARIANT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/merge_demand_utilization_realized.py --variant "${VARIANT}"
echo "[etest_realized_u] staged-contract check  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -c "
from src.etest import assert_staged_etest_contract, get_etest_variant
slug = get_etest_variant('${VARIANT}').slug
assert_staged_etest_contract(slug)
print(f'[etest_realized_u] {slug}: staged contract OK')
"
echo "[etest_realized_u] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
