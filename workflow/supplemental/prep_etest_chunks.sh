#!/bin/bash
# Step-04 staging for a CHUNKED test ensemble: one array task per chunk.
#
# E_test's daily traces live only in its 50 sibling chunk dirs (each a
# standalone staged ensemble with local keys 0..S-1), and the chunked re-eval
# path (src/chunk_reeval.py, step 09) simulates per chunk — so the pywrdrb
# inputs (flood-node inflows, presimulated releases, predicted inflows) must be
# staged per chunk dir. Running step 04 against the parent slug fails by
# design: there is no monolithic catchment_inflow_mgd.hdf5 to read.
#
# Each task preps `--preset {slug}__chunk{JJJ}` with JJJ = the array index,
# through the ordinary scripts/main/prep_pywrdrb_inputs.py path (MPI over the
# chunk's realizations). This is the one-time full-model presim pass over
# E_test (~70 SU total across the array; methods §5.4).
#
# Submit (after the E_test merge; slug via env, default the campaign E_test):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
#          workflow/supplemental/prep_etest_chunks.sh
#
#SBATCH --job-name=prep_etest_chunk
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --time=02:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/prep_etest_chunk_%A_%a.out
#SBATCH --error=logs/prep_etest_chunk_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

ETEST_SLUG="${NYCOPT_ETEST_SLUG:-etest_kn_50yr_n25000}"
PRESET="$(printf '%s__chunk%03d' "${ETEST_SLUG}" "${SLURM_ARRAY_TASK_ID}")"

NTASKS="${SLURM_NTASKS:-1}"
echo "[prep_etest_chunk] preset=${PRESET} ranks=${NTASKS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" python3 -u scripts/main/prep_pywrdrb_inputs.py --preset "${PRESET}"
echo "[prep_etest_chunk] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
