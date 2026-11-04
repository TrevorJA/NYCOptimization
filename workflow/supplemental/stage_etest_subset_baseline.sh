#!/bin/bash
# Plant the incumbent's full-E_test cube under a design's subset re-eval tag
# (scripts/supplemental/stage_etest_subset_baseline.py): a symlink, no
# simulation. The incumbent-relative regret family needs a baseline per-SOW
# matrix at reeval/{tag}/baseline; the incumbent cube on the full E_test (step
# 09 with NYCOPT_CHUNK_POLICIES=baseline under the historic env) is a strict
# superset of the subset's SOWs and is joined by SOW label, so one source cube
# serves every design. Run once per design env BEFORE that design's step-09
# re-evaluation on the subset, with the same NYCOPT_REEVAL_ENSEMBLE_PRESET the
# step-09 submission will use.
#
# Env inputs:
#   NYCOPT_ENV_FILE                 required — the design env (target dir is design-specific)
#   NYCOPT_REEVAL_ENSEMBLE_PRESET   required — the subset slug (campaign: etest_kn_50yr_n25000_first25ch)
#   NYCOPT_BASELINE_SRC             optional — source cube dir; the chunked driver
#                                   writes outputs/historic/{slug}/reeval/{full slug}/baseline
#   SEED                            optional — iff step 09 will pass --seed
#
# Submit (from repo root), once per design env file:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_mc_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch,NYCOPT_BASELINE_SRC=outputs/historic/ffmp_obj8/reeval/etest_kn_50yr_n25000/baseline \
#          workflow/supplemental/stage_etest_subset_baseline.sh
#
#SBATCH --job-name=etest_subset_baseline
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/etest_subset_baseline_%j.out
#SBATCH --error=logs/etest_subset_baseline_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set NYCOPT_REEVAL_ENSEMBLE_PRESET to the subset slug}"
ARGS=()
[[ -n "${NYCOPT_BASELINE_SRC:-}" ]] && ARGS+=(--baseline-src "${NYCOPT_BASELINE_SRC}")
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "${SEED}")

echo "[etest_subset_baseline] env=${NYCOPT_ENV_FILE} preset=${NYCOPT_REEVAL_ENSEMBLE_PRESET}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u -m scripts.supplemental.stage_etest_subset_baseline "${ARGS[@]}"
echo "[etest_subset_baseline] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
