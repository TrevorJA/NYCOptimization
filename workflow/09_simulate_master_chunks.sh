#!/bin/bash
# Step 9: Simulate + score a chunked master ensemble (metrics-only).
#
# Re-evaluates a policy set against every chunk of the chunked forcing master
# that NYCOPT_REEVAL_ENSEMBLE_PRESET resolves to, computing objectives +
# robustness from in-memory reduced metrics (no simulation-output timeseries
# persisted). MPI chunk-and-aggregate; degrades to serial.
#
# Everything comes from env vars / the env file — no positional args:
#   NYCOPT_ENV_FILE                  required
#   NYCOPT_REEVAL_ENSEMBLE_PRESET    required — the chunked master slug
#                                    (e.g. master_5yr_n128000)
#   NYCOPT_CHUNK_POLICIES            'baseline' (default) or a path to a .ref
#                                    reference set
#   NYCOPT_SEARCH_REALIZATION_BATCH  realizations per within-chunk sim batch
#                                    (bounds RAM)
#   NYCOPT_CHUNK_SIM_MODE            single | mpi (default mpi)
#   NYCOPT_CHUNK_SIM_NODES / _RANKS  MPI layout (default 4 x 16 = 64)
#   FORMULATION                      identifier, default ffmp
#   SEED                             optional
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_sal.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=master_5yr_n128000 \
#          workflow/09_simulate_master_chunks.sh
#
#SBATCH --job-name=sim_master_chunks
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/sim_master_chunks_%j.out
#SBATCH --error=logs/sim_master_chunks_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set the chunked master slug explicitly, e.g. master_5yr_n128000}"
FORMULATION="${FORMULATION:-ffmp}"
SEED="${SEED:-}"
MODE="${NYCOPT_CHUNK_SIM_MODE:-mpi}"

ARGS="--formulation ${FORMULATION}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

echo "=== Simulate master chunks: formulation=${FORMULATION} mode=${MODE} "\
"master=${NYCOPT_REEVAL_ENSEMBLE_PRESET} policies=${NYCOPT_CHUNK_POLICIES:-baseline} ==="

case "${MODE}" in
    single)
        python3 -m scripts.main.simulate_master_chunks ${ARGS}
        ;;
    mpi)
        NTASKS_MPI="$(( ${NYCOPT_CHUNK_SIM_NODES:-4} * ${NYCOPT_CHUNK_SIM_RANKS:-16} ))"
        echo "[sim-chunks] MPI mode, ${NTASKS_MPI} ranks"
        mpirun -np "${NTASKS_MPI}" ${NYCOPT_MPI_MCA_FLAGS} \
            python3 -m scripts.main.simulate_master_chunks ${ARGS}
        ;;
    *)
        echo "ERROR: unknown NYCOPT_CHUNK_SIM_MODE='${MODE}' (expected 'single' or 'mpi')" >&2
        exit 1
        ;;
esac

echo "=== Completed: $(date) ==="
