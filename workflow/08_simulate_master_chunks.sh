#!/bin/bash
# Step 8: Simulate + score a chunked master ensemble (metrics-only).
#
# Re-evaluates a policy set against every chunk of the chunked forcing master that
# NYCOPT_REEVAL_ENSEMBLE_PRESET resolves to, computing objectives + robustness from in-memory reduced
# metrics (no simulation-output timeseries persisted). MPI chunk-and-aggregate; degrades to serial.
# All values come from the env file / registries (no CLI value flags); FORMULATION/SEED are ids.
#
# Usage (SLURM with env file):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7_sal.env \
#          workflow/08_simulate_master_chunks.sh ffmp
#
# Key env:
#   NYCOPT_REEVAL_ENSEMBLE_PRESET  the chunked master slug (e.g. master_5yr_n128000)  [required]
#   NYCOPT_CHUNK_POLICIES          'baseline' (default) or a path to a .ref reference set
#   NYCOPT_SEARCH_REALIZATION_BATCH  realizations per within-chunk sim batch (bounds RAM)
#   NYCOPT_CHUNK_SIM_NODES / _RANKS  MPI layout (default 4 x 16 = 64)
#
#SBATCH --job-name=sim_master_chunks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/sim_master_chunks_%j.out
#SBATCH --error=logs/sim_master_chunks_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ffmp_obj7_sal.env}"
if [[ -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[sim-chunks] sourced env file: ${NYCOPT_ENV_FILE}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

FORMULATION="${1:-ffmp}"
SEED="${2:-}"
MODE="${NYCOPT_CHUNK_SIM_MODE:-mpi}"

ARGS="--formulation ${FORMULATION}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

echo "=== Simulate master chunks: formulation=${FORMULATION} mode=${MODE} "\
"master=${NYCOPT_REEVAL_ENSEMBLE_PRESET:-<unset>} policies=${NYCOPT_CHUNK_POLICIES:-baseline} ==="

case "${MODE}" in
    single)
        python3 -m scripts.main.simulate_master_chunks ${ARGS}
        ;;
    mpi)
        NTASKS_MPI="$(( ${NYCOPT_CHUNK_SIM_NODES:-4} * ${NYCOPT_CHUNK_SIM_RANKS:-16} ))"
        echo "[sim-chunks] MPI mode, ${NTASKS_MPI} ranks"
        mpirun -np "${NTASKS_MPI}" \
            --mca pml ob1 --mca btl self,vader,tcp \
            python3 -m scripts.main.simulate_master_chunks ${ARGS}
        ;;
    *)
        echo "ERROR: unknown NYCOPT_CHUNK_SIM_MODE='${MODE}' (expected 'single' or 'mpi')" >&2
        exit 1
        ;;
esac

echo "=== Completed: $(date) ==="
