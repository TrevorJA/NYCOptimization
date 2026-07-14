#!/bin/bash
# Step 11: Scenario discovery on E_test failures, IN HAZARD SPACE — the
# MECHANISM TEST for the study's central claim.
#
# Runs after step 08 (re-evaluation of every design's Pareto set on the common
# held-out ensemble E_test). For each scenario design it (a) fits a
# gradient-boosted classifier of E_test failure on the realization's HAZARD
# coordinates, and (b) tests whether failure probability is positively
# associated with that design's COVERAGE DEFICIT — the hazard-space distance
# from each E_test realization to the nearest member of the design's SEARCH
# ensemble. Cheap: no re-simulation, it scores persisted artifacts.
#
# Requires:
#   * step 08 has written reeval_raw.parquet for the designs being compared
#   * E_test carries a staged hazard_image.npz (generate it with
#     compute_hazard_image=True — workflow step 02). There is no forcing-parameter
#     fallback: the coverage hypothesis is stated in hazard space.
#
# Everything comes from the env file — no positional args, no value flags:
#   NYCOPT_ENV_FILE                required — pins objectives/MOEA config (the slug)
#   NYCOPT_REEVAL_ENSEMBLE_PRESET  required — E_test, the SAME ensemble step 08 used
#   FORMULATION                    identifier, default ffmp
#   SEED                           optional, selects a per-seed re-eval subdir
#   ENSEMBLE_DRAW                  optional, default 0 — which draw's search ensembles
#   DESIGNS                        optional, comma-separated design ids
#                                  (default: the campaign designs)
#
# Analysis settings (compromise rule, classifier hyperparameters, redundancy
# threshold) are module constants in scripts/main/scenario_discovery.py, env-
# overridable via NYCOPT_SD_* — never CLI value flags.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_hazfill_du.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_10yr_n200 \
#          workflow/11_scenario_discovery.sh
#
# Local (no allocation needed — this is a single-core scoring job):
#   NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_hazfill_du.env \
#   NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_10yr_n200 \
#   bash workflow/11_scenario_discovery.sh
#
#SBATCH --job-name=scenario_discovery
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/scenario_discovery_%j.out
#SBATCH --error=logs/scenario_discovery_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required

: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set E_test explicitly — the SAME preset step 08 re-evaluated on}"
FORMULATION="${FORMULATION:-ffmp}"
SEED="${SEED:-}"
ENSEMBLE_DRAW="${ENSEMBLE_DRAW:-0}"
DESIGNS="${DESIGNS:-}"

ARGS="--formulation ${FORMULATION} --draw ${ENSEMBLE_DRAW}"
[[ -n "${SEED}" ]]    && ARGS="${ARGS} --seed ${SEED}"
[[ -n "${DESIGNS}" ]] && ARGS="${ARGS} --designs ${DESIGNS}"

echo "=== Scenario discovery (hazard space): formulation=${FORMULATION}" \
     "E_test=${NYCOPT_REEVAL_ENSEMBLE_PRESET} draw=${ENSEMBLE_DRAW}" \
     "designs=${DESIGNS:-<campaign>} ==="

python3 scripts/main/scenario_discovery.py ${ARGS}

echo "=== Completed: $(date) ==="
