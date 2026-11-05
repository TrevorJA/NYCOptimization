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
# Chunk dirs are named after the GENERATED design ({slug}__chunkJJJ). The default
# slug is the registry's E_test; when E_test is cut from a larger parent (the
# campaign: etest_kn_50yr_n25000, whose chunks 0-24 are E_test) pass
# NYCOPT_ETEST_SLUG=<parent>. One array task per E_test chunk (25).
#
# Submit:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
#          workflow/supplemental/staging/etest_stage_nyc_demand.sh
#
#SBATCH --job-name=etest_stage_nyc_demand
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:40:00
#SBATCH --array=0-24
#SBATCH --output=logs/etest_stage_nyc_demand_%A_%a.out
#SBATCH --error=logs/etest_stage_nyc_demand_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

ETEST_SLUG="${NYCOPT_ETEST_SLUG:-$(python3 -c "from src.etest import E_TEST_VARIANT, get_etest_variant; print(get_etest_variant(E_TEST_VARIANT).slug)")}"
PRESET="$(printf '%s__chunk%03d' "${ETEST_SLUG}" "${SLURM_ARRAY_TASK_ID}")"

NTASKS="${SLURM_NTASKS:-1}"
echo "[etest_stage_nyc_demand] preset=${PRESET} ranks=${NTASKS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" python3 -u scripts/supplemental/stage_nyc_demand.py --preset "${PRESET}"
echo "[etest_stage_nyc_demand] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
