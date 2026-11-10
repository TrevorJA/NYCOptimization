#!/bin/bash
# Re-snapshot every persisted E_test re-eval cube onto the current satisficing
# thresholds (src/objectives_ensemble.py::_DEFAULT_THRESHOLDS) and rescore it
# (scripts/supplemental/rescore_thresholds.py). Zero simulation: the per-SOW
# objective values do not depend on the thresholds; only each cube's
# reeval_raw_meta.json snapshot and its robustness_* artifacts are rewritten.
# The replaced vector is kept in the meta's threshold_history.
#
# Run after any change of the adopted threshold vector, before any figure or
# step-12 comparison reads the scorecards. Order: the incumbent cube first,
# then every design leaf on the tag (regret is scored against the incumbent).
#
# Env inputs (no positional args, no value flags):
#   NYCOPT_ENV_FILE          required: a production env file, so the regret
#                            tolerance ladder resolves (NYCOPT_REGRET_TAU)
#   NYCOPT_REEVAL_TAG        optional: E_test tag (default: the campaign tag)
#   NYCOPT_RESULTS_SLUG      optional: moea slug (default ffmp_obj8)
#   NYCOPT_CRITERIA_VARIANT  optional: criterion-set variant (default v3_20261106)
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
#          workflow/supplemental/results_diagnostics/rescore_thresholds.sh
#
# Sizing: reads one small parquet per cube and rewrites CSVs; seconds per cube
# on one shared core.
#
#SBATCH --job-name=rescore_thr
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/rescore_thresholds_%j.out
#SBATCH --error=logs/rescore_thresholds_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_full_threads 1

echo "[rescore_thresholds] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/rescore_thresholds.py
echo "=== Completed: $(date) ==="
