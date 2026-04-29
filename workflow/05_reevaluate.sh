#!/bin/bash
# Step 5: Re-simulate Pareto-optimal solutions with the full Pywr-DRB model.
#
# Mode selection (single-node multiprocessing vs MPI multi-node) is driven
# by `NYCOPT_REEVAL_MODE` from the sourced env file — no CLI flag to remember.
# Slug auto-derives from active config; outputs land at
# `outputs/reevaluation/{slug}/deterministic/` (Phase 1) or
# `outputs/reevaluation/{slug}/ensemble_{id}/` (Phase 3).
#
# Usage (single-node fallback / interactive):
#   bash workflow/05_reevaluate.sh [FORMULATION] [MAX_SOLUTIONS] [SEED]
#
# Usage (SLURM with env file):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7.env \
#          workflow/05_reevaluate.sh ffmp 0
#
# Defaults:
#   FORMULATION   first positional arg, else "ffmp"
#   MAX_SOLUTIONS second positional arg, else 0 (all)
#   SEED          third positional arg (optional, for per-seed output subdir)
#
#SBATCH --job-name=reevaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/reevaluate_%j.out
#SBATCH --error=logs/reevaluate_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

# ---- Source per-experiment env file (if any) ----
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ffmp_obj7.env}"
if [[ -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[reevaluate] sourced env file: ${NYCOPT_ENV_FILE}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

FORMULATION="${1:-ffmp}"
MAX_SOLUTIONS="${2:-0}"
SEED="${3:-}"

# Mode: env-driven, with single-node default for back-compat.
MODE="${NYCOPT_REEVAL_MODE:-single}"
NJOBS="${SLURM_CPUS_ON_NODE:-1}"

ARGS="--formulation ${FORMULATION} --max ${MAX_SOLUTIONS} --njobs ${NJOBS}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

echo "=== Re-evaluation: formulation=${FORMULATION} mode=${MODE} njobs=${NJOBS} seed=${SEED:-all} ==="

case "${MODE}" in
    single)
        python3 -m src.reevaluate ${ARGS}
        ;;
    mpi)
        NTASKS_MPI="$(( ${NYCOPT_REEVAL_NODES:-4} * ${NYCOPT_REEVAL_RANKS:-16} ))"
        echo "[reevaluate] MPI mode, ${NTASKS_MPI} ranks"
        mpirun -np "${NTASKS_MPI}" \
            --mca pml ob1 --mca btl self,vader,tcp \
            python3 -m src.reevaluate_mpi ${ARGS}
        ;;
    *)
        echo "ERROR: unknown NYCOPT_REEVAL_MODE='${MODE}' (expected 'single' or 'mpi')" >&2
        exit 1
        ;;
esac

echo "=== Completed: $(date) ==="
