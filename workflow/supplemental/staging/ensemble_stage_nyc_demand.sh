#!/bin/bash
# Add the per-realization NYC demand series to an already-staged search
# ensemble (or any staged slug via --preset), without repeating step 04's
# presim pass.
#
# NYC demand is the time-dynamic extrapolated diversion series in every
# simulation (config.NYC_DEMAND_SOURCE_ENSEMBLE); step 04 stages it with the
# other inputs for new ensembles. Ensembles staged before the series was part
# of the contract get it here. The array index is the ensemble-draw index k,
# as in step 04; `--preset NAME` targets an arbitrary staged slug instead.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=monte_carlo --array=0-1 \
#          workflow/supplemental/staging/ensemble_stage_nyc_demand.sh
#   sbatch workflow/supplemental/staging/ensemble_stage_nyc_demand.sh --preset kn_10yr_n100
#
#SBATCH --job-name=ensemble_stage_nyc_demand
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:30:00
#SBATCH --array=0
#SBATCH --output=logs/ensemble_stage_nyc_demand_%A_%a.out
#SBATCH --error=logs/ensemble_stage_nyc_demand_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

export NYCOPT_ENSEMBLE_DRAW="${SLURM_ARRAY_TASK_ID:-${NYCOPT_ENSEMBLE_DRAW:-0}}"

NTASKS="${SLURM_NTASKS:-1}"
echo "[ensemble_stage_nyc_demand] design=${NYCOPT_SCENARIO_DESIGN:-<default>} draw=${NYCOPT_ENSEMBLE_DRAW} ranks=${NTASKS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" python3 -u scripts/supplemental/stage_nyc_demand.py "$@"
echo "[ensemble_stage_nyc_demand] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
