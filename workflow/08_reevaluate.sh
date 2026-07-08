#!/bin/bash
# Step 8: Re-evaluate Pareto-optimal policies on the COMMON held-out ensemble
# with the full Pywr-DRB model, so scores are comparable across scenario
# designs. Merges the former per-arm and common-ensemble re-eval launchers.
#
# Everything comes from env vars / the env file — no positional args:
#   NYCOPT_ENV_FILE                required — the arm being re-evaluated
#   NYCOPT_REEVAL_ENSEMBLE_PRESET  required — the common held-out ensemble
#                                  (e.g. kn_5yr_n200). Required explicitly so
#                                  cross-arm comparability is a recorded
#                                  choice, never a silent default.
#   NYCOPT_REEVAL_MODE             single | mpi (from env file; default single)
#   FORMULATION                    identifier, default ffmp
#   SEED                           optional, per-seed output subdir
#   MAX_SOLUTIONS                  default 0 = all Pareto solutions
#   NYCOPT_REEVAL_SCORE=1          opt-in offline robustness scoring
#   NYCOPT_REEVAL_BASELINE_DIR     optional, for regret-from-baseline
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200,NYCOPT_REEVAL_SCORE=1 \
#          workflow/08_reevaluate.sh
#
# Local / single-node (no SLURM allocation needed):
#   NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env \
#   NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200 NYCOPT_REEVAL_MODE=single \
#   bash workflow/08_reevaluate.sh
#
# The SBATCH geometry below is sized for the MPI path (4 nodes x 16 ranks = 64,
# matching the env's NYCOPT_REEVAL_NODES x NYCOPT_REEVAL_RANKS). Rescale a run
# without editing this file via `sbatch --nodes=N --ntasks-per-node=M ...`;
# `mpirun -np` follows the actual allocation (SLURM_NTASKS).
#
#SBATCH --job-name=reevaluate
#SBATCH --account=x-tamestoy
#SBATCH --partition=wholenode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/reevaluate_%j.out
#SBATCH --error=logs/reevaluate_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set the common held-out re-eval ensemble explicitly, e.g. kn_5yr_n200}"
FORMULATION="${FORMULATION:-ffmp}"
MAX_SOLUTIONS="${MAX_SOLUTIONS:-0}"
SEED="${SEED:-}"
MODE="${NYCOPT_REEVAL_MODE:-single}"
NJOBS="${SLURM_CPUS_ON_NODE:-1}"

ARGS="--formulation ${FORMULATION} --max ${MAX_SOLUTIONS} --njobs ${NJOBS}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

echo "=== Re-evaluation: formulation=${FORMULATION} mode=${MODE} seed=${SEED:-all}" \
     "ensemble=${NYCOPT_REEVAL_ENSEMBLE_PRESET} ==="

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
        mpirun -np "${SLURM_NTASKS}" ${NYCOPT_MPI_MCA_FLAGS} \
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
