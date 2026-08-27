#!/bin/bash
# Step 10: Cross-design comparison of the re-evaluated Pareto sets
# (scripts/main/compare_designs.py): reads every campaign design's re-eval
# artifacts on ONE common E_test and writes the cross-design tables + figures
# (satisficing-threshold sweep, scorecard aggregation and ranking stability,
# variance components, raw performance distributions, attainability screen).
# Serial, no simulation; runs on a login node via `bash` as well.
#
# Env inputs (no positional args, no value flags):
#   NYCOPT_REEVAL_ENSEMBLE_PRESET  required — the SAME E_test preset steps 08/09
#                                  re-evaluated on (campaign:
#                                  etest_kn_50yr_n25000_first25ch)
#   NYCOPT_ENV_FILE                optional — objectives / MOEA config for the slug
#   FORMULATION                    identifier, default ffmp
#   SEED                           optional — restrict to one MOEA seed
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
#          workflow/10_compare_designs.sh
#
# Outputs: outputs/comparison/{slug}/{preset}/ tables + figures.
#
#SBATCH --job-name=compare_designs
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/compare_designs_%j.out
#SBATCH --error=logs/compare_designs_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set the common held-out E_test explicitly — the same preset step 08 re-evaluated on}"
FORMULATION="${FORMULATION:-ffmp}"
SEED="${SEED:-}"

ARGS="--formulation ${FORMULATION}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "=== Cross-design comparison: formulation=${FORMULATION}" \
     "E_test=${NYCOPT_REEVAL_ENSEMBLE_PRESET} seed=${SEED:-all} ==="
python3 -u scripts/main/compare_designs.py ${ARGS}
echo "=== Completed: $(date) ==="
