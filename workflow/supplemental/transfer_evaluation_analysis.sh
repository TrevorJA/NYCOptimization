#!/bin/bash
#SBATCH --job-name=tev_analysis
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/transfer_evaluation_analysis_%j.out
#SBATCH --error=logs/transfer_evaluation_analysis_%j.err

# Transfer evaluation, stage 3: merge, analyze, figures
# (docs/notes/methods/transfer_evaluation.md). Zero simulation.
#
# MERGE reassembles the per-unit artifacts into per-cell objective matrices and
# per-realization annual-unit tables, and verifies that recomposing objectives
# from the persisted tensor reproduces the per-unit composed vectors exactly.
# A cell with missing units or an inexact recomposition marks the matrix
# invalid and the analyze stage refuses to run on it.
#
# ANALYZE computes the three readouts, all existing study metrics:
#   1. fraction of each set dominating the scenario-matched FFMP baseline,
#      via src.solution_selection.dominance_mask;
#   2. the epsilon-nondominated merged reference set per target ensemble and
#      each optimization's contribution to it. MOEAFramework ResultFileMerger
#      produces the union (it merges by PLAIN Pareto dominance and ignores
#      --epsilon for archiving, measured in this repo), then the Borg
#      epsilon-box archive is applied with sensitivity_common.epsilon_nondominated.
#      Attribution is by decision vector; MOEAFramework's own Contribution
#      indicator is run as a cross-check;
#   3. hypervolume of each set under each ensemble, every call against ONE
#      shared reference set so the nine values are comparable.
# Then the exploratory-tier figures are drawn from the persisted tables.
#
# Needs Java (MOEAFramework 5.0). Java 17 lives in the project's conda env, not
# a module; the driver prepends the interpreter's own bin dir to PATH for the
# CLI subprocess, so no module load is required here.
#
# PREREQUISITE: transfer_evaluation_eval_array.sh has completed every unit (the
# merge stage reports any that are missing and marks the matrix invalid).
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/transfer_evaluation.env \
#          workflow/supplemental/transfer_evaluation_analysis.sh
#
# Switches: NYCOPT_TEV_SMOKE=1, NYCOPT_TEV_MERGE_ONLY=1 (stop after merge),
# NYCOPT_TEV_FIGURES_ONLY=1 (redraw from persisted tables),
# NYCOPT_TEV_ALLOW_PARTIAL=1 (analyze a knowingly incomplete matrix).
# Settings in supplemental_config.py (TEV_ section).
#
# Outputs: outputs/supplemental/transfer_evaluation/{tables,figures,sets}.
# Sizing: deliberately small and short so it backfills promptly - a 26-core /
# 48 GB / 2 h request was estimated seven hours out under the backlog at the
# time of writing, while this footprint schedules in minutes. Merge streams one
# cell at a time (largest is 991 units x 7,200 rows) and the readouts are
# arithmetic; exact 8-D hypervolume measured ~7 s for three sets against a
# 947-member reference, and is capped by TEV_INDICATOR_TIMEOUT_S in any case.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[tev:analysis] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (smoke=${NYCOPT_TEV_SMOKE:-0}, figures_only=${NYCOPT_TEV_FIGURES_ONLY:-0})"
if [[ "${NYCOPT_TEV_FIGURES_ONLY:-0}" != "1" ]]; then
    NYCOPT_TEV_STAGE=merge python3 -u scripts/supplemental/transfer_evaluation_run.py
    if [[ "${NYCOPT_TEV_MERGE_ONLY:-0}" != "1" ]]; then
        NYCOPT_TEV_STAGE=analyze python3 -u scripts/supplemental/transfer_evaluation_run.py
    fi
fi
if [[ "${NYCOPT_TEV_MERGE_ONLY:-0}" != "1" ]]; then
    python3 -u scripts/supplemental/transfer_evaluation_figures.py
fi
echo "[tev:analysis] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
