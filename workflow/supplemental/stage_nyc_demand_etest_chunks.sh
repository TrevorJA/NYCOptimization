#!/bin/bash
# Add the per-realization NYC demand series to already-staged E_test chunks:
# one array task per chunk, no repeat of the presim pass.
#
# NYC demand is the time-dynamic extrapolated diversion series in every
# simulation (config.NYC_DEMAND_SOURCE_ENSEMBLE); step 04 stages it with the
# other inputs for new ensembles. E_test chunks staged before the series was
# part of the contract get it from this script, which runs
# scripts/supplemental/stage_nyc_demand.py --preset {slug}__chunk{JJJ} (MPI over
# the chunk's realizations) and writes diversion_nyc_extrapolated_mgd.hdf5 into
# each chunk dir. Cheap: the extrapolation is a regression plus a
# nearest-neighbour month match per realization, no system simulation.
#
# Submit (slug via env, default the campaign E_test):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
#          workflow/supplemental/stage_nyc_demand_etest_chunks.sh
#
#SBATCH --job-name=stage_nyc_demand_chunk
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:40:00
#SBATCH --array=0-49
#SBATCH --output=logs/stage_nyc_demand_chunk_%A_%a.out
#SBATCH --error=logs/stage_nyc_demand_chunk_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

ETEST_SLUG="${NYCOPT_ETEST_SLUG:-etest_kn_50yr_n25000}"
PRESET="$(printf '%s__chunk%03d' "${ETEST_SLUG}" "${SLURM_ARRAY_TASK_ID}")"

NTASKS="${SLURM_NTASKS:-1}"
echo "[stage_nyc_demand_chunk] preset=${PRESET} ranks=${NTASKS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" python3 -u scripts/supplemental/stage_nyc_demand.py --preset "${PRESET}"
echo "[stage_nyc_demand_chunk] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
