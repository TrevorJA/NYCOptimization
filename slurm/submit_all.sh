#!/bin/bash
# submit_all.sh — Submit an MM-Borg campaign described by an env file.
#
# Reads the formulation list from a `slurm/envs/*.env` file (sourced) and
# expands it into per-architecture sbatch submissions. Each submission
# carries the env file path forward via --export=ALL,NYCOPT_ENV_FILE so
# the per-arch SLURM scripts (and _common.sh) see the same knobs.
#
# Usage:
#   bash slurm/submit_all.sh                                  # default ffmp_obj7
#   bash slurm/submit_all.sh slurm/envs/manuscript_obj9_ts.env
#   bash slurm/submit_all.sh slurm/envs/ann_obj9_ts.env --dry-run
#   bash slurm/submit_all.sh slurm/envs/ffmp_obj9_ts.env ffmp ffmp_6
#       (override formulations after env file)
#
# Formulation names ending in `_N` (e.g. ffmp_6, ffmp_10) automatically
# dispatch to slurm/mmborg_ffmp_vr.sh with N_ZONES set.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
ENV_FILE=""
ARGS=()
for a in "$@"; do
    case "$a" in
        --dry-run)         DRY_RUN=true ;;
        slurm/envs/*.env)  ENV_FILE="$a" ;;
        */*.env|*.env)     ENV_FILE="$a" ;;
        *)                 ARGS+=("$a") ;;
    esac
done

# Default env file preserves pre-Phase-0 behavior.
ENV_FILE="${ENV_FILE:-slurm/envs/ffmp_obj7.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: env file not found: ${ENV_FILE}" >&2
    exit 1
fi
echo "[submit_all] env file: ${ENV_FILE}"
# Source it so we can read NYCOPT_FORMULATIONS without a python call.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# Resolve formulation list: explicit args > env file > default.
if [[ ${#ARGS[@]} -gt 0 ]]; then
    TARGETS=("${ARGS[@]}")
elif [[ -n "${NYCOPT_FORMULATIONS:-}" ]]; then
    IFS=',' read -ra TARGETS <<< "${NYCOPT_FORMULATIONS}"
else
    TARGETS=(ffmp ffmp_6 ffmp_8 ffmp_10 ffmp_12 ann)
fi

echo "[submit_all] targets: ${TARGETS[*]}"

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" == "false" ]]; then
        "$@"
    fi
}

for t in "${TARGETS[@]}"; do
    t="$(echo "$t" | tr -d ' ')"
    if [[ "$t" =~ ^ffmp_([0-9]+)$ ]]; then
        N="${BASH_REMATCH[1]}"
        run sbatch \
            --export=ALL,N_ZONES="${N}",NYCOPT_ENV_FILE="${ENV_FILE}" \
            --array=1-10 \
            --job-name="mmborg_ffmp_vr_N${N}" \
            slurm/mmborg_ffmp_vr.sh
    elif [[ -f "slurm/mmborg_${t}.sh" ]]; then
        run sbatch \
            --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}" \
            --array=1-10 \
            "slurm/mmborg_${t}.sh"
    else
        echo "ERROR: no SLURM script for target '${t}' (looked for slurm/mmborg_${t}.sh)" >&2
        exit 1
    fi
done

echo "=== Submission complete ==="
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run — no jobs actually submitted)"
