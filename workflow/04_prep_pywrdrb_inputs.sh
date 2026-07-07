#!/bin/bash
# Step 3: Format a generated/subsampled ensemble into the pywrdrb-ingestible
# HDF5 inputs the trimmed optimization model reads, under
# outputs/synthetic_ensembles/{inflow_type}/:
#   catchment_inflow_with_flood_nodes_mgd.hdf5   (FlowEnsemble, flood ops on)
#   presimulated_releases_mgd.hdf5 (+ _metadata.json)   (trimmed-model releases)
#   predicted_inflows_mgd.hdf5                   (Montague/Trenton forecasts)
#
# The ensemble is the active scenario design's search ensemble
# (config.SEARCH_ENSEMBLE_SPEC); the base catchment_inflow_mgd.hdf5 must already
# exist (Step 1 generator, + Step 2 subsample for hazard-filling designs).
# Per-realization work is distributed across MPI ranks automatically.
#
# Submit:
#   sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling \
#          workflow/03_prep_pywrdrb_inputs.sh
#
#SBATCH --job-name=prep_pywrdrb_inputs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_pywrdrb_inputs_%j.out
#SBATCH --error=logs/prep_pywrdrb_inputs_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.5 || true
source venv/bin/activate

# Optional per-experiment env file (design can also come from --export).
if [[ -n "${NYCOPT_ENV_FILE:-}" && -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a; source "${NYCOPT_ENV_FILE}"; set +a
    echo "[03_prep_pywrdrb_inputs] sourced env file: ${NYCOPT_ENV_FILE}"
fi

# One BLAS thread per rank: each rank runs a pywrdrb simulation, so avoid
# oversubscribing the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

NTASKS="${SLURM_NTASKS:-1}"
echo "[03_prep_pywrdrb_inputs] design=${NYCOPT_SCENARIO_DESIGN:-<default>} ranks=${NTASKS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" python3 -u scripts/main/prep_pywrdrb_inputs.py "$@"
echo "[03_prep_pywrdrb_inputs] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
