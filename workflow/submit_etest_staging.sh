#!/bin/bash
# submit_etest_staging.sh — complete the campaign E_test from its staged parent
# as a dependency chain (the metadata, demand and hazard-image steps under
# supplemental/staging/). Login-node helper, not an sbatch script.
#
# Usage (from the repo root):
#   bash workflow/submit_etest_staging.sh <demand|hazard|all> [--dry-run] [sbatch args...]
#
#   demand   etest_demand_utilization.sh (k(u) table + per-SOW utilization u)
#            -> etest_stage_nyc_demand.sh (one task per chunk, the scaled NYC
#            demand series) -> etest_cut.sh (cuts etest_kn_50yr_n12500 from the
#            parent and merges realized u)
#   hazard   etest_hazard_image_shards.sh (one task per chunk) ->
#            etest_hazard_image_merge.sh (the sub-window hazard image)
#   all      demand, then hazard
#   --dry-run print the sbatch lines without submitting
#   any further arguments are appended to every sbatch line verbatim
#
# Identity: the historic production env (no MOEA settings are read),
# NYCOPT_ETEST_VARIANT=kn, and NYCOPT_ETEST_PARENT / NYCOPT_ETEST_SLUG =
# etest_kn_50yr_n25000, the 1,000-SOW design whose chunks 000 to 024 are
# E_test (override both with NYCOPT_ETEST_PARENT=<slug>). The steps are
# idempotent or refuse to repeat work, so a chain can be resubmitted after a
# failure. A fresh E_test build (05 or staging/etest_shards.sh -> etest_merge.sh,
# then etest_prep_chunks.sh, which stages the scaled series itself) needs only
# the hazard chain and the cut.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

[[ $# -ge 1 ]] || usage 2
MODE="$1"; shift
case "${MODE}" in demand|hazard|all) ;; -h|--help) usage 0 ;; *) echo "ERROR: unknown mode '${MODE}'" >&2; usage 2 ;; esac

DRY_RUN=false
EXTRA_ARGS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage 0 ;;
        *) EXTRA_ARGS+=("$a") ;;
    esac
done

ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env
PARENT="${NYCOPT_ETEST_PARENT:-etest_kn_50yr_n25000}"
VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"
S=workflow/supplemental/staging
[[ -f "${ENV_FILE}" ]] || { echo "ERROR: env file not found: ${ENV_FILE}" >&2; exit 2; }
mkdir -p logs

# submit <label> <dependency-or-empty> <sbatch args...>; prints the job id.
submit() {
    local label="$1" dep="$2"; shift 2
    local cmd=(sbatch --parsable)
    [[ -n "${dep}" ]] && cmd+=("--dependency=afterok:${dep}")
    cmd+=("$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
    echo "+ ${cmd[*]}" >&2
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "<${label}>"
    else
        "${cmd[@]}"
    fi
}

LAST=""
echo "=== E_test completion (${MODE}): parent ${PARENT}, variant ${VARIANT} ==="
if [[ "${MODE}" == "demand" || "${MODE}" == "all" ]]; then
    J1="$(submit demand_u "" --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}",NYCOPT_ETEST_VARIANT="${VARIANT}" \
          "${S}/etest_demand_utilization.sh")"
    J2="$(submit demand_chunks "${J1}" \
          --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}",NYCOPT_ETEST_SLUG="${PARENT}" \
          "${S}/etest_stage_nyc_demand.sh")"
    J3="$(submit cut "${J2}" \
          --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}",NYCOPT_ETEST_VARIANT="${VARIANT}",NYCOPT_ETEST_PARENT="${PARENT}" \
          "${S}/etest_cut.sh")"
    echo "demand utilization -> ${J1}; chunk demand -> ${J2}; cut -> ${J3}"
    LAST="${J3}"
fi
if [[ "${MODE}" == "hazard" || "${MODE}" == "all" ]]; then
    J4="$(submit hazimg_shards "${LAST}" --export=ALL,NYCOPT_ETEST_VARIANT="${VARIANT}" \
          "${S}/etest_hazard_image_shards.sh")"
    J5="$(submit hazimg_merge "${J4}" --export=ALL,NYCOPT_ETEST_VARIANT="${VARIANT}" \
          "${S}/etest_hazard_image_merge.sh")"
    echo "hazard-image shards -> ${J4}; merge -> ${J5}"
fi
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run: nothing submitted)"
exit 0
