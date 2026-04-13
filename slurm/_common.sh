#!/bin/bash
# _common.sh — shared setup sourced by per-architecture SLURM scripts.
#
# Sourced after SBATCH directives. Does NOT call `set -e` itself; each
# caller controls its own error handling.
#
# Expected caller-set vars:
#   FORMULATION   architecture name (ffmp, rbf, tree, ann, ffmp_N)
#   SEED          optimization seed (int)
#   N_ISLANDS     MM Borg islands
#   NFE           NFE per island
#   RUNTIME_FREQ  runtime snapshot interval
#
# Optional:
#   DEBUG_SIM     "true" to use short 2018-2022 simulation window
#   CHECKPOINT    "true" to enable Borg checkpointing (currently race-prone)
#
# Also honors these environment overrides (read by config.py):
#   NYCOPT_STATE_SPEC       state vector spec (minimal|extended|full)
#   NYCOPT_OBJECTIVE_SET    objective set name (see src/objectives.py)

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs

module load python/3.11.5 || true
source venv/bin/activate

# ---- Thread pinning (prevents BLAS contention across MPI ranks) ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Debug sim dates (short window, ~13s/eval) ----
if [[ "${DEBUG_SIM:-false}" == "true" ]]; then
    export PYWRDRB_SIM_START_DATE="2018-01-01"
    export PYWRDRB_SIM_END_DATE="2022-12-31"
fi

# ---- MPI task count formula: 1 + N_ISLANDS × (workers + 1) ----
# With N_ISLANDS=2, workers_per_island=98  → 1 + 2×99 = 199 ranks
# Allocate 200 slots (5 nodes × 40) and run 199 to leave 1 slot headroom.
NTASKS_MPI=${NTASKS_MPI:-199}

# ---- Reproducibility logging ----
RUN_TAG="${FORMULATION}_seed${SEED}_${SLURM_JOB_ID:-local}"
RUN_LOG_DIR="outputs/run_manifests/${RUN_TAG}"
mkdir -p "${RUN_LOG_DIR}"

{
    echo "=== Run manifest: ${RUN_TAG} ==="
    echo "Date:            $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Host:            $(hostname)"
    echo "SLURM_JOB_ID:    ${SLURM_JOB_ID:-n/a}"
    echo "SLURM_JOB_NAME:  ${SLURM_JOB_NAME:-n/a}"
    echo "Nodes:           ${SLURM_JOB_NUM_NODES:-1}"
    echo "Alloc ntasks:    ${SLURM_NTASKS:-?}"
    echo "MPI ntasks used: ${NTASKS_MPI}"
    echo "Formulation:     ${FORMULATION}"
    echo "Seed:            ${SEED}"
    echo "N islands:       ${N_ISLANDS}"
    echo "NFE/island:      ${NFE}"
    echo "Runtime freq:    ${RUNTIME_FREQ}"
    echo "Debug sim:       ${DEBUG_SIM:-false}"
    echo "Checkpoint:      ${CHECKPOINT:-false}"
    echo "STATE_SPEC:      ${NYCOPT_STATE_SPEC:-<config default>}"
    echo "OBJECTIVE_SET:   ${NYCOPT_OBJECTIVE_SET:-<config default>}"
    echo "Python:          $(which python3)"
    echo "Python version:  $(python3 --version 2>&1)"
    echo "---- git ----"
    git rev-parse HEAD 2>/dev/null || echo "(not a git repo)"
    git status --porcelain 2>/dev/null || true
} > "${RUN_LOG_DIR}/manifest.txt"

cp config.py "${RUN_LOG_DIR}/config_snapshot.py" 2>/dev/null || true
git diff HEAD > "${RUN_LOG_DIR}/uncommitted.diff" 2>/dev/null || true

echo "=== Run manifest written to ${RUN_LOG_DIR}/manifest.txt ==="
cat "${RUN_LOG_DIR}/manifest.txt"
