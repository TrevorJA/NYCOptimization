#!/bin/bash
# Production-scale acceptance gate for the vectorized perfect-foresight
# predicted-inflow kernel (pywrdrb PredictedInflowEnsemblePreprocessor).
#
# Regenerates one staged E_test chunk's predicted inflows on the vectorized
# path (same 33-rank MPI layout as step 04) into outputs/tmp_bitcheck/, then
# bit-compares every dataset against the staged production artifact (produced
# by the scalar path). Exact equality expected; any nonzero diff is reported
# per column against the 1% robust-range cross-era tolerance and the job exits
# nonzero. The staged chunk is never opened for writing.
#
# Submit:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
#          workflow/supplemental/predicted_inflow_bitcheck.sh
#
#SBATCH --job-name=pf_bitcheck
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:30:00
#SBATCH --output=logs/pf_bitcheck_%j.out
#SBATCH --error=logs/pf_bitcheck_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

NTASKS="${SLURM_NTASKS:-1}"
echo "[pf_bitcheck] preset=${NYCOPT_BITCHECK_PRESET:-etest_kn_50yr_n25000__chunk000} ranks=${NTASKS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" python3 -u scripts/supplemental/predicted_inflow_bitcheck.py
echo "[pf_bitcheck] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
