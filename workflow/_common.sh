#!/bin/bash
# _common.sh — shared setup sourced by FFMP / FFMP-VR SLURM scripts.
#
# Sourced after SBATCH directives. Does NOT call `set -e` itself; each
# caller controls its own error handling.
#
# Expected caller-set vars:
#   FORMULATION   formulation name (ffmp or ffmp_N)
#   SEED          optimization seed (int)
#
# Algorithm settings (islands, NFE, runtime freq, MPI rank count) are NO LONGER
# caller-set — they come from the active MOEA config (src/moea_config.py),
# selected by NYCOPT_MOEA_CONFIG in the env file. The scenario design comes from
# NYCOPT_SCENARIO_DESIGN. This block reads them back from config so the shell and
# Python agree on a single source of truth.
#
# Optional:
#   DEBUG_SIM     "true" to use short 2018-2022 simulation window
#   CHECKPOINT    "true" to enable Borg checkpointing (currently race-prone)
#   NYCOPT_ENV_FILE  path to a `slurm/envs/*.env` file. If set, sourced
#                    before config.py is read so its NYCOPT_* knobs apply.
#                    SLURM scripts default this to slurm/envs/ffmp_obj7_sal.env
#                    when no override is provided.
#   RUN_SLUG      escape hatch — sets the output slug verbatim, bypassing
#                 derive_slug(). Most users should NOT set this; instead
#                 author or pick an env file under slurm/envs/.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs

module load python/3.11.5 || true
source venv/bin/activate

# Make `from borg import ...` work from any CWD. borg.py still loads
# libborg.so / libborgmm.so relative to CWD, so callers that start MPI
# (e.g. src/mmborg.py) chdir into lib/borg/ for that call.
export PYTHONPATH="${PWD}:${PWD}/lib/borg:${PYTHONPATH:-}"

# ---- Source per-experiment env file (if any) ----
# Default to ffmp_obj7_sal.env to preserve pre-Phase-0 behavior when no file
# is passed. The env file sets NYCOPT_* knobs that drive derive_slug().
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ffmp_obj7_sal.env}"
if [[ -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[_common] sourced env file: ${NYCOPT_ENV_FILE}"
else
    echo "[_common] env file not found: ${NYCOPT_ENV_FILE} (using config.py defaults)"
fi

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

# ---- Run identity, derived from the active config (single source of truth) ----
# One Python call reads back: scenario design name, moea slug, MOEA config name,
# MPI rank count, islands, NFE/island, runtime freq. The shell and the driver
# then agree on identical values without any value-carrying flags.
mapfile -t _CFG < <(python3 -c "
import config
mc = config.ACTIVE_MOEA_CONFIG
print(config.active_scenario_name())
print(config.derive_slug('${FORMULATION}'))
print(mc.name)
print(mc.total_ntasks_mpi if mc.total_ntasks_mpi is not None else '')
print(mc.n_islands if mc.n_islands is not None else '')
print(mc.max_evaluations if mc.max_evaluations is not None else '')
print(mc.runtime_frequency if mc.runtime_frequency is not None else '')
")
SCENARIO="${_CFG[0]}"
# RUN_SLUG escape hatch wins outright; otherwise use the derived moea slug.
if [[ -z "${RUN_SLUG:-}" ]]; then
    export RUN_SLUG="${_CFG[1]}"
fi
MOEA_CONFIG_NAME="${_CFG[2]}"
N_ISLANDS="${_CFG[4]}"
NFE="${_CFG[5]}"
RUNTIME_FREQ="${_CFG[6]}"

# MPI rank count. Precedence: caller override (NTASKS_MPI already set) wins, then
# the config's 1 + islands*(workers+1), then the SLURM allocation minus 1.
if [[ -n "${NTASKS_MPI:-}" ]]; then
    :  # caller override wins (e.g. smoke_test.sh sizing to its allocation)
elif [[ -n "${_CFG[3]}" ]]; then
    NTASKS_MPI="${_CFG[3]}"
else
    NTASKS_MPI="$(( ${SLURM_NTASKS:-200} - 1 ))"
fi

# ---- Reproducibility logging ----
RUN_TAG="${RUN_SLUG}_seed${SEED}_${SLURM_JOB_ID:-local}"
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
    echo "Env file:        ${NYCOPT_ENV_FILE}"
    echo "Formulation:     ${FORMULATION}"
    echo "Scenario design: ${SCENARIO}"
    echo "MOEA config:     ${MOEA_CONFIG_NAME}"
    echo "Slug:            ${RUN_SLUG}"
    echo "Seed:            ${SEED}"
    echo "N islands:       ${N_ISLANDS}"
    echo "NFE/island:      ${NFE}"
    echo "Runtime freq:    ${RUNTIME_FREQ}"
    echo "Debug sim:       ${DEBUG_SIM:-false}"
    echo "Checkpoint:      ${CHECKPOINT:-false}"
    echo "OBJECTIVES:      ${NYCOPT_OBJECTIVES:-<config default>}"
    echo "SCENARIO_DESIGN: ${NYCOPT_SCENARIO_DESIGN:-<config default>}"
    echo "MOEA_CONFIG:     ${NYCOPT_MOEA_CONFIG:-<config default>}"
    echo "TS_ON:           ${NYCOPT_TS_ON:-<config default>}"
    echo "CLUSTER:         ${NYCOPT_CLUSTER:-<config default>}"
    echo "Python:          $(which python3)"
    echo "Python version:  $(python3 --version 2>&1)"
    echo "---- git ----"
    git rev-parse HEAD 2>/dev/null || echo "(not a git repo)"
    git status --porcelain 2>/dev/null || true
} > "${RUN_LOG_DIR}/manifest.txt"

cp config.py "${RUN_LOG_DIR}/config_snapshot.py" 2>/dev/null || true
cp "${NYCOPT_ENV_FILE}" "${RUN_LOG_DIR}/env_snapshot.env" 2>/dev/null || true
git diff HEAD > "${RUN_LOG_DIR}/uncommitted.diff" 2>/dev/null || true

echo "=== Run manifest written to ${RUN_LOG_DIR}/manifest.txt ==="
cat "${RUN_LOG_DIR}/manifest.txt"
