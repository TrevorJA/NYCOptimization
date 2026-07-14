#!/bin/bash
# Step 10: Cross-design comparison of the re-evaluated Pareto sets.
#
# The single consumer of the per-run robustness artifacts (step 08 + `python -m
# src.robustness`). Reads every campaign design's re-eval on ONE common held-out
# test ensemble E_test and emits the cross-design tables + figures:
#   - the satisficing-threshold sweep (main-text figure): design robustness vs
#     criterion stringency, plus Kendall tau_b of the DESIGN ranking at each
#     stringency against the ranking at the registry-default criterion.
#   - the cross-design scorecard aggregation, design-ranking stability across
#     metrics, and the design/draw/seed variance components.
#   - the raw re-evaluated performance distributions (with thresholds drawn on)
#     and the degeneracy screen.
#   - the pooled attainability screen.
#
# Cheap and serial: no simulation, only CSV/parquet reads. Run it on a login node
# or via `bash`; sbatch is only for convenience/logging.
#
# Everything comes from env vars — no positional args, no value flags:
#   NYCOPT_REEVAL_ENSEMBLE_PRESET  required — the SAME common E_test the designs
#                                  were re-evaluated on in step 08. Required
#                                  explicitly so cross-design comparability is a
#                                  recorded choice, never a silent default.
#   NYCOPT_ENV_FILE                optional — objectives / MOEA config, for the
#                                  fallback slug used to name output dirs.
#   FORMULATION                    identifier, default ffmp
#   SEED                           optional — restrict to one MOEA seed
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200 \
#          workflow/10_compare_designs.sh
#
# Local:
#   NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200 bash workflow/10_compare_designs.sh
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
