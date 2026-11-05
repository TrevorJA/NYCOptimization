#!/bin/bash
# Step 6: Evaluate the default FFMP policy (no optimization) and save
# baseline objective values — the comparison anchor for optimized Pareto sets.
# Runs once per design on its searched draw (d0): the historic design on its
# single trace, an ensemble design scenario-matched on its own search ensemble
# (`--search-ensemble`); d1 is staged only for the SI draw-sensitivity
# re-evaluation (docs/notes/methods/campaign_design.md).
#
# The incumbent's per-SOW E_test cube is NOT written here: step 10 with
# NYCOPT_CHUNK_POLICIES=baseline simulates it per design on the chunked E_test.
# NYCOPT_BASELINE_REEVAL=1 opts in to the legacy unchunked `--reeval` leg,
# which needs NYCOPT_REEVAL_ENSEMBLE_PRESET set to an unchunked preset.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
#          workflow/06_run_baseline.sh
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env \
#          workflow/06_run_baseline.sh --search-ensemble
#
#SBATCH --job-name=baseline
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

python3 scripts/main/run_baseline.py "$@"

# Opt-in legacy leg: the incumbent's re-eval matrix on an UNCHUNKED preset
# (the campaign cube on the chunked E_test comes from step 10 instead).
if [[ "${NYCOPT_BASELINE_REEVAL:-0}" == "1" ]]; then
    : "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?NYCOPT_BASELINE_REEVAL=1 needs an unchunked NYCOPT_REEVAL_ENSEMBLE_PRESET}"
    echo "[06_run_baseline] persisting the baseline re-eval matrix on ${NYCOPT_REEVAL_ENSEMBLE_PRESET}"
    python3 scripts/main/run_baseline.py "$@" --reeval
fi
