#!/bin/bash
# Step 7: Re-simulate Pareto-optimal solutions with the full Pywr-DRB model.
#
# Mode selection (single-node multiprocessing vs MPI multi-node) is driven
# by `NYCOPT_REEVAL_MODE` from the sourced env file — no CLI flag to remember.
# Slug auto-derives from active config; outputs land at
# `outputs/{scenario}/{slug}/reeval/{reeval_tag}[/seed_NN]/`.
#
# The SBATCH geometry below is sized for the MPI path (4 nodes x 16 ranks = 64,
# matching the env's NYCOPT_REEVAL_NODES x NYCOPT_REEVAL_RANKS). Rescale a run
# without editing this file via `sbatch --nodes=N --ntasks-per-node=M ...`;
# `mpirun -np` follows the actual allocation (SLURM_NTASKS). `single` mode uses
# one node's cores and is intended for interactive/local `bash` invocation.
#
# Usage (single-node fallback / interactive):
#   bash workflow/07_reevaluate.sh [FORMULATION] [MAX_SOLUTIONS] [SEED]
#
# Usage (SLURM with env file):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7_sal.env \
#          workflow/07_reevaluate.sh ffmp 0
#
# Defaults:
#   FORMULATION   first positional arg, else "ffmp"
#   MAX_SOLUTIONS second positional arg, else 0 (all)
#   SEED          third positional arg (optional, for per-seed output subdir)
#
#SBATCH --job-name=reevaluate
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/reevaluate_%j.out
#SBATCH --error=logs/reevaluate_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

# ---- Source per-experiment env file (if any) ----
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ffmp_obj7_sal.env}"
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

echo "=== Re-evaluation: formulation=${FORMULATION} mode=${MODE} seed=${SEED:-all} ==="

case "${MODE}" in
    single)
        # Single-node multiprocessing.Pool across this node's cores. For
        # interactive/local use, invoke with `bash` (no SLURM allocation needed).
        echo "[reevaluate] single-node mode, ${NJOBS} worker(s)"
        python3 -m src.reevaluate ${ARGS}
        ;;
    mpi)
        # Multi-node MPI: one rank per solution slice (src.reevaluate_mpi). -np is
        # the ACTUAL SLURM allocation, so the launch cannot over/under-subscribe
        # the SBATCH geometry above. Requires a SLURM allocation.
        : "${SLURM_NTASKS:?mpi mode needs a SLURM allocation — submit with sbatch, or set NYCOPT_REEVAL_MODE=single}"
        echo "[reevaluate] MPI mode, ${SLURM_NTASKS} ranks"
        mpirun -np "${SLURM_NTASKS}" \
            --mca pml ob1 --mca btl self,vader,tcp \
            python3 -m src.reevaluate_mpi ${ARGS}
        ;;
    *)
        echo "ERROR: unknown NYCOPT_REEVAL_MODE='${MODE}' (expected 'single' or 'mpi')" >&2
        exit 1
        ;;
esac

# ---- Optional offline robustness scoring (opt-in) ----
# Scores the persisted raw matrix (reeval_raw.parquet) into a multi-metric
# robustness scorecard. Cheap, no re-simulation. Regret-from-baseline also
# needs a baseline raw pass: run `python3 scripts/main/run_baseline.py
# --formulation ${FORMULATION} --reeval` first and pass --baseline-dir.
if [[ "${NYCOPT_REEVAL_SCORE:-0}" == "1" ]]; then
    echo "=== Robustness scoring ==="
    SCORE_ARGS="--formulation ${FORMULATION}"
    [[ -n "${SEED}" ]] && SCORE_ARGS="${SCORE_ARGS} --seed ${SEED}"
    [[ -n "${NYCOPT_REEVAL_BASELINE_DIR:-}" ]] && \
        SCORE_ARGS="${SCORE_ARGS} --baseline-dir ${NYCOPT_REEVAL_BASELINE_DIR}"
    python3 -m src.robustness ${SCORE_ARGS}
fi

echo "=== Completed: $(date) ==="
