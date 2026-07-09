#!/bin/bash
# workflow/_common.sh — shared setup functions sourced by every workflow script.
#
# Defines functions only (no top-level side effects); each caller composes
# exactly what it needs, in this order:
#
#   nycopt_setup_env                      every script
#   nycopt_source_env_file required|optional
#   nycopt_pin_threads                    MPI / simulation steps (NOT steps
#                                         02/03, which want full BLAS threads)
#   nycopt_read_run_identity              MM-Borg only; needs FORMULATION, SEED
#   nycopt_check_allocation               MM-Borg only; SLURM_NTASKS vs config
#   nycopt_write_manifest                 MM-Borg only
#   nycopt_preflight_mmborg               MM-Borg only
#
# Sourced after SBATCH directives. Does NOT call `set -e` itself; each caller
# controls its own error handling. Jobs must be submitted from the repo root
# (SLURM_SUBMIT_DIR is taken as the root when present).
#
# Run identity is env-file driven: there is NO default env file. Steps whose
# meaning depends on a chosen experiment (06/08/09) pass "required" and abort
# with a listing of workflow/envs/*.env when NYCOPT_ENV_FILE is unset.

# Repo root: SLURM_SUBMIT_DIR when submitted from the root; otherwise derived
# from this file's location (workflow/_common.sh -> parent dir).
NYCOPT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Centralized cluster constants (previously copy-pasted across scripts).
# The python module exists on Hopper; on Anvil there is no such module and the
# conda env's PATH (propagated by sbatch --export=ALL) supplies python3, so the
# module load below is a no-op there. Override via NYCOPT_PYTHON_MODULE.
NYCOPT_PYTHON_MODULE="${NYCOPT_PYTHON_MODULE:-python/3.11.5}"
# Anvil fallback when the submitting shell did not have the env active: the
# project env is the conda env named "venv" under the anaconda module.
# Override via NYCOPT_CONDA_MODULE / NYCOPT_CONDA_ENV.
NYCOPT_CONDA_MODULE="${NYCOPT_CONDA_MODULE:-anaconda/2024.02-py311}"
NYCOPT_CONDA_ENV="${NYCOPT_CONDA_ENV:-venv}"
# OpenMPI transport flags for the re-evaluation / chunked-simulation steps
# (NOT the MM-Borg launcher). Cluster-dependent: Hopper's default fabric
# providers hang on the preprocessors' MPI gathers, so TCP is forced there;
# Anvil should use its default (InfiniBand/UCX) transport — forcing TCP would
# route MPI over IPoIB. Resolved in _nycopt_set_mpi_flags() once
# NYCOPT_CLUSTER is known (i.e. after the env file is sourced).
NYCOPT_MPI_MCA_FLAGS="${NYCOPT_MPI_MCA_FLAGS:-}"
# MPI ranks packed per node for MM-Borg jobs. 33/node is the measured
# memory-bandwidth-safe packing for pywrdrb simulations; it is the single
# source for the suggested --nodes/--ntasks-per-node geometry printed by
# nycopt_check_allocation when an allocation doesn't fit the MOEA config.
# Override via env to test denser packings (e.g. on Anvil's 128-core nodes).
NYCOPT_RANKS_PER_NODE="${NYCOPT_RANKS_PER_NODE:-33}"

_nycopt_set_mpi_flags() {
    if [[ -z "${NYCOPT_MPI_MCA_FLAGS}" && "${NYCOPT_CLUSTER:-hopper}" == "hopper" ]]; then
        NYCOPT_MPI_MCA_FLAGS="--mca pml ob1 --mca btl self,vader,tcp"
    fi
}

# cd to repo root, load the Python module, activate ./venv, set PYTHONPATH.
nycopt_setup_env() {
    cd "${NYCOPT_ROOT}"
    mkdir -p logs
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh 2>/dev/null || true
    module load "${NYCOPT_PYTHON_MODULE}" 2>/dev/null || true
    if [[ -f venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    elif ! python3 -c 'import numpy' >/dev/null 2>&1; then
        # No ./venv and the PATH python3 can't run the project (e.g. the job
        # was submitted from a shell without the conda env active): activate
        # the Anvil conda env directly instead of relying on --export=ALL.
        module load "${NYCOPT_CONDA_MODULE}" 2>/dev/null || true
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook 2>/dev/null)" || true
            conda activate "${NYCOPT_CONDA_ENV}" 2>/dev/null || true
        fi
        if python3 -c 'import numpy' >/dev/null 2>&1; then
            echo "[_common] activated conda env '${NYCOPT_CONDA_ENV}' — using $(command -v python3)"
        else
            echo "[_common] WARNING: no usable python3 (./venv missing, conda env '${NYCOPT_CONDA_ENV}' not activatable) — using $(command -v python3)"
        fi
    else
        echo "[_common] ./venv not found — using PATH python3: $(command -v python3)"
    fi
    # Make `from borg import ...` work from any CWD. borg.py still loads
    # libborg.so / libborgmm.so relative to CWD, so callers that start MPI
    # (e.g. src/mmborg.py) chdir into lib/borg/ for that call.
    export PYTHONPATH="${PWD}:${PWD}/lib/borg:${PYTHONPATH:-}"
}

# Source the per-experiment env file (KEY=VALUE only; sets NYCOPT_* knobs that
# config.py reads at import). $1 = required | optional.
nycopt_source_env_file() {
    local mode="${1:-optional}"
    if [[ -z "${NYCOPT_ENV_FILE:-}" ]]; then
        if [[ "${mode}" == "required" ]]; then
            {
                echo "ERROR: NYCOPT_ENV_FILE is not set."
                echo "This step's run identity (scenario design, MOEA config, objectives)"
                echo "comes from an env file. Submit with:"
                echo "  sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/<file>.env ..."
                echo "Available env files:"
                ls -1 workflow/envs/*.env
            } >&2
            exit 2
        fi
        echo "[_common] no env file set (config.py defaults apply)"
        _nycopt_set_mpi_flags
        return 0
    fi
    if [[ ! -f "${NYCOPT_ENV_FILE}" ]]; then
        {
            echo "ERROR: env file not found: ${NYCOPT_ENV_FILE}"
            echo "Available env files:"
            ls -1 workflow/envs/*.env
        } >&2
        exit 2
    fi
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    export NYCOPT_ENV_FILE
    _nycopt_set_mpi_flags
    echo "[_common] sourced env file: ${NYCOPT_ENV_FILE}"
}

# Pin BLAS/OpenMP to one thread per rank (prevents contention across MPI
# ranks). Steps that want full BLAS parallelism (02/03) skip this and set
# OMP_NUM_THREADS to the allocation size themselves.
nycopt_pin_threads() {
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
}

# Read the run identity back from config.py (single source of truth) so the
# shell and the Python driver agree without value-carrying flags.
# Requires: FORMULATION (exported). Honors: RUN_SLUG, NTASKS_MPI, DEBUG_SIM.
# Sets: SCENARIO, RUN_SLUG, MOEA_CONFIG_NAME, N_ISLANDS, NFE, RUNTIME_FREQ,
# NTASKS_MPI (precedence: caller override > config's total_ntasks_mpi >
# SLURM allocation minus 1).
nycopt_read_run_identity() {
    if [[ "${DEBUG_SIM:-false}" == "true" ]]; then
        # Short window, ~13s/eval — must be exported before config is imported.
        # End pinned to config.END_DATE (2022-09-30): the trimmed model's
        # presimulated releases are generated to that water-year end, so a
        # later debug end date would run past the presim data coverage.
        export PYWRDRB_SIM_START_DATE="2018-01-01"
        export PYWRDRB_SIM_END_DATE="2022-09-30"
    fi

    local _cfg
    mapfile -t _cfg < <(python3 -c "
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
    SCENARIO="${_cfg[0]}"
    # RUN_SLUG escape hatch wins outright; otherwise use the derived moea slug.
    if [[ -z "${RUN_SLUG:-}" ]]; then
        export RUN_SLUG="${_cfg[1]}"
    fi
    MOEA_CONFIG_NAME="${_cfg[2]}"
    N_ISLANDS="${_cfg[4]}"
    NFE="${_cfg[5]}"
    RUNTIME_FREQ="${_cfg[6]}"

    if [[ -n "${NTASKS_MPI:-}" ]]; then
        :  # caller override wins (e.g. submit_smoke.sh sizing to its allocation)
    elif [[ -n "${_cfg[3]}" ]]; then
        NTASKS_MPI="${_cfg[3]}"
    else
        NTASKS_MPI="$(( ${SLURM_NTASKS:-200} - 1 ))"
    fi
}

# Verify the SLURM allocation fits the MPI launch the MOEA config demands.
# The rank count itself always comes from config (nycopt_read_run_identity);
# the static #SBATCH geometry is only a container for it. This guard makes a
# too-small allocation fail fast (before Borg starts) with the exact sbatch
# geometry to use, and flags allocations wasting a node's worth of cores.
# Requires: nycopt_read_run_identity ran first (NTASKS_MPI set).
nycopt_check_allocation() {
    [[ -z "${SLURM_NTASKS:-}" ]] && return 0   # not under SLURM (local run)
    local need="${NTASKS_MPI}"
    local have="${SLURM_NTASKS}"
    if (( have < need )); then
        local nodes=$(( (need + NYCOPT_RANKS_PER_NODE - 1) / NYCOPT_RANKS_PER_NODE ))
        {
            echo "ERROR: allocation too small for the active MOEA config."
            echo "  MOEA config '${MOEA_CONFIG_NAME}' needs ${need} MPI ranks;"
            echo "  this job allocated only ${have} tasks (SLURM_NTASKS)."
            echo "  Resubmit with a matching geometry, e.g.:"
            echo "    sbatch --nodes=${nodes} --ntasks-per-node=${NYCOPT_RANKS_PER_NODE} ..."
        } >&2
        exit 3
    fi
    if (( have - need >= NYCOPT_RANKS_PER_NODE )); then
        echo "[_common] WARNING: allocation has $(( have - need )) idle tasks" \
             "(need ${need}, allocated ${have}) — a whole node or more is unused."
    fi
    echo "[_common] allocation OK: ${need} MPI ranks in ${have} allocated tasks"
}

# Write a full reproducibility manifest (run identity, allocation, git state,
# config + env-file snapshots) to outputs/run_manifests/.
# Requires: nycopt_read_run_identity ran first; FORMULATION, SEED set.
nycopt_write_manifest() {
    local run_tag="${RUN_SLUG}_seed${SEED}_${SLURM_JOB_ID:-local}"
    local run_log_dir="outputs/run_manifests/${run_tag}"
    mkdir -p "${run_log_dir}"

    {
        echo "=== Run manifest: ${run_tag} ==="
        echo "Date:            $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Host:            $(hostname)"
        echo "SLURM_JOB_ID:    ${SLURM_JOB_ID:-n/a}"
        echo "SLURM_JOB_NAME:  ${SLURM_JOB_NAME:-n/a}"
        echo "Nodes:           ${SLURM_JOB_NUM_NODES:-1}"
        echo "Alloc ntasks:    ${SLURM_NTASKS:-?}"
        echo "MPI ntasks used: ${NTASKS_MPI}"
        echo "Env file:        ${NYCOPT_ENV_FILE:-<none>}"
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
    } > "${run_log_dir}/manifest.txt"

    cp config.py "${run_log_dir}/config_snapshot.py" 2>/dev/null || true
    [[ -n "${NYCOPT_ENV_FILE:-}" ]] && cp "${NYCOPT_ENV_FILE}" "${run_log_dir}/env_snapshot.env" 2>/dev/null
    git diff HEAD > "${run_log_dir}/uncommitted.diff" 2>/dev/null || true

    echo "=== Run manifest written to ${run_log_dir}/manifest.txt ==="
    cat "${run_log_dir}/manifest.txt"
}

# Config-derived pre-flight for the MM-Borg launcher: echo the resolved run
# identity for the job log, and fail fast only on genuine inconsistencies
# (unstaged scenario design, schema-only MOEA config, bad formulation name).
# Expectations come FROM config — nothing here hardcodes an experiment.
# Requires: FORMULATION exported.
nycopt_preflight_mmborg() {
    echo "=== Pre-flight verification (config-derived) ==="
    python3 -c "
import os
import config
from src.formulations import get_n_vars

f = os.environ['FORMULATION']
mc = config.ACTIVE_MOEA_CONFIG
spec = config.SEARCH_ENSEMBLE_SPEC
obj = config.get_objective_set()

print('Scenario design :', config.active_scenario_name())
print('Search ensemble :', None if spec is None else spec.preset_name,
      '| is_ensemble =', None if spec is None else spec.is_ensemble)
print('MOEA config     :', mc.name, '| islands =', mc.n_islands,
      '| NFE/island =', mc.max_evaluations, '| ranks =', mc.total_ntasks_mpi,
      '| seeds =', mc.n_seeds)
print('Salinity LSTM   :', config.INCLUDE_SALINITY_MODEL)
print('Temperature LSTM:', config.INCLUDE_TEMPERATURE_MODEL)
print('Formulation     :', f, '| n_vars =', get_n_vars(f))
print('n_objs          :', obj.n_objs)
print('Objectives      :', obj.names)
print('Epsilons        :', obj.epsilons)

assert spec is not None, (
    'SEARCH_ENSEMBLE_SPEC is None: the scenario design '
    f'{config.active_scenario_name()!r} could not resolve its search ensemble. '
    'Stage it first (workflow steps 02-04) or pick a staged design in the env file.')
assert None not in (mc.n_islands, mc.n_workers_per_island, mc.max_evaluations), (
    f'MOEA config {mc.name!r} is schema-only (unset numbers) and cannot launch; '
    'set NYCOPT_MOEA_CONFIG to a concrete config (see src/moea_config.py).')
print('Pre-flight OK.')
"
}
