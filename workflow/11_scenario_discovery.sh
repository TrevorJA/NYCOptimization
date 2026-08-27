#!/bin/bash
# Step 11: Scenario discovery on E_test failures in hazard space. For each
# scenario design it (a) fits a gradient-boosted classifier of E_test failure on
# the realization's hazard coordinates and (b) tests whether failure probability
# rises with the design's coverage deficit (hazard-space distance from each
# E_test realization to the nearest member of the design's search ensemble).
# No re-simulation; it scores persisted artifacts.
#
# Requires: the step 08/09 re-eval cubes (reeval_raw.parquet) of the designs
# compared, and E_test's staged hazard_image.npz (workflow step 12; no
# forcing-parameter fallback).
#
# Env inputs (no positional args, no value flags):
#   NYCOPT_ENV_FILE                required — pins objectives/MOEA config (the slug)
#   NYCOPT_REEVAL_ENSEMBLE_PRESET  required — E_test, the SAME preset steps 08/09 used
#   FORMULATION                    identifier, default ffmp
#   SEED                           optional, selects a per-seed re-eval subdir
#   DRAW                           optional, default 0 — which draw's search ensembles
#   DESIGNS                        optional, comma-separated design ids
#                                  (default: the campaign designs)
# Analysis settings are module constants in scripts/main/scenario_discovery.py,
# env-overridable via NYCOPT_SD_*.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_hazfill_stat_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
#          workflow/11_scenario_discovery.sh
#
# Output: outputs/comparison/{slug}/{tag}/scenario_discovery/ tables + the
# scenario_discovery figure tree.
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
# DRAW is the cross-step identifier (steps 06/08/09); ENSEMBLE_DRAW kept as
# this step's documented name.
ENSEMBLE_DRAW="${DRAW:-${ENSEMBLE_DRAW:-0}}"
DESIGNS="${DESIGNS:-}"

ARGS="--formulation ${FORMULATION} --draw ${ENSEMBLE_DRAW}"
[[ -n "${SEED}" ]]    && ARGS="${ARGS} --seed ${SEED}"
[[ -n "${DESIGNS}" ]] && ARGS="${ARGS} --designs ${DESIGNS}"

echo "=== Scenario discovery (hazard space): formulation=${FORMULATION}" \
     "E_test=${NYCOPT_REEVAL_ENSEMBLE_PRESET} draw=${ENSEMBLE_DRAW}" \
     "designs=${DESIGNS:-<campaign>} ==="

python3 scripts/main/scenario_discovery.py ${ARGS}

echo "=== Completed: $(date) ==="
