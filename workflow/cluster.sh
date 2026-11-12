#!/bin/bash
# workflow/cluster.sh — identify the cluster and load its profile.
#
# Machine-specific settings (partition names, ranks per node, node memory, the
# Python environment, MPI transport flags) live in one profile per cluster
# under workflow/clusters/. They are NOT part of a run's identity, so no
# workflow/envs/*.env file sets them: the same env file runs on every cluster.
#
# Sourced by workflow/_common.sh (every job) and by the submit helpers. On a
# cluster whose partitions differ from the Anvil #SBATCH headers (Hopper),
# source it once in the login shell before typing sbatch lines by hand:
#
#   source workflow/cluster.sh        # Hopper Xeon nodes: NYCOPT_NODE_CLASS=xeon source workflow/cluster.sh
#
# Detection order: Slurm's cluster name (SLURM_CLUSTER_NAME in a job, scontrol
# on a login node), then a pre-set NYCOPT_CLUSTER, then the hostname; each must
# name a profile. With no match (a laptop) the anvil profile loads, which is
# what the workflow did before profiles existed.
#
# Exports NYCOPT_CLUSTER and every profile variable. A profile value already
# set in the environment wins, unless it was set by an earlier load of a
# different profile (another cluster or node class): those are reset first.
#
# When the profile sets NYCOPT_SBATCH_ARGS, `sbatch` becomes a shell function
# in the sourcing shell that prepends those arguments (and says so on stderr);
# arguments typed after them still override. `command sbatch` bypasses it.

_NYCOPT_CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clusters"
_NYCOPT_PROFILE_VARS=(
    NYCOPT_PARTITION_SERIAL NYCOPT_PARTITION_MPI NYCOPT_PARTITION_DEBUG
    NYCOPT_SLURM_ACCOUNT NYCOPT_RANKS_PER_NODE NYCOPT_CPUS_PER_RANK NYCOPT_NODE_MEMORY_GB
    NYCOPT_WALL_CAP_HOURS NYCOPT_REEVAL_UNIT_MINUTES
    NYCOPT_PYTHON_MODULE NYCOPT_CONDA_MODULE NYCOPT_CONDA_ENV
    NYCOPT_MPI_MCA_FLAGS NYCOPT_SBATCH_ARGS SBATCH_PARTITION
)

# Print the name of the profile for this machine (empty when none matches).
nycopt_detect_cluster() {
    local name="${SLURM_CLUSTER_NAME:-}"
    if [[ -z "${name}" ]] && command -v scontrol >/dev/null 2>&1; then
        # No early awk exit and `|| true`: callers may run under `set -eo pipefail`.
        name="$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" {n = $3} END {print n}' || true)"
    fi
    if [[ -n "${name}" && -f "${_NYCOPT_CLUSTER_DIR}/${name}.sh" ]]; then
        echo "${name}"; return 0
    fi
    name="${NYCOPT_CLUSTER:-}"
    if [[ -n "${name}" && -f "${_NYCOPT_CLUSTER_DIR}/${name}.sh" ]]; then
        echo "${name}"; return 0
    fi
    local profile
    for profile in "${_NYCOPT_CLUSTER_DIR}"/*.sh; do
        name="$(basename "${profile}" .sh)"
        if [[ "$(hostname -f 2>/dev/null)" == *"${name}"* ]]; then
            echo "${name}"; return 0
        fi
    done
    echo ""
}

# Called first by every profile with its tag ("<cluster>" or
# "<cluster>:<node class>"): values exported by an earlier load of a different
# profile are dropped so this profile's defaults apply.
_nycopt_profile_begin() {
    local var
    if [[ -n "${NYCOPT_CLUSTER_PROFILE:-}" && "${NYCOPT_CLUSTER_PROFILE}" != "$1" ]]; then
        for var in "${_NYCOPT_PROFILE_VARS[@]}"; do unset "${var}"; done
    fi
    export NYCOPT_CLUSTER_PROFILE="$1"
}

# Load the detected cluster's profile and export it.
nycopt_load_cluster() {
    local name var
    name="$(nycopt_detect_cluster)"
    [[ -n "${name}" ]] || name="anvil"
    # shellcheck disable=SC1090
    source "${_NYCOPT_CLUSTER_DIR}/${name}.sh" || return $?
    for var in "${_NYCOPT_PROFILE_VARS[@]}"; do
        if [[ -n "${!var+x}" ]]; then export "${var?}"; fi
    done
    export NYCOPT_CLUSTER="${name}"
    if [[ -n "${NYCOPT_SBATCH_ARGS:-}" ]]; then
        sbatch() {
            local -a _args
            read -ra _args <<< "${NYCOPT_SBATCH_ARGS}"   # word-split without globbing
            echo "[cluster] ${NYCOPT_CLUSTER_PROFILE}: sbatch ${NYCOPT_SBATCH_ARGS} ..." >&2
            command sbatch "${_args[@]}" "$@"
        }
    else
        unset -f sbatch 2>/dev/null || true
    fi
}

nycopt_load_cluster
