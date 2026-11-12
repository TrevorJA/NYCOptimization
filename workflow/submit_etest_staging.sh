#!/bin/bash
# submit_etest_staging.sh — build the campaign E_test (etest_kn_50yr_n12500:
# 500 LHS SOWs x 25 realizations x 50 yr, generated directly at the registry
# size of src/etest.py) as one dependency chain over the wrappers under
# supplemental/staging/. Login-node helper, not an sbatch script.
#
# Usage (from the repo root):
#   bash workflow/submit_etest_staging.sh <build|hazard|all> [--from STEP] [--after JOBID]
#                                         [--dry-run] [sbatch args...]
#
#   build    etest_shards.sh (25 array tasks, the streamflow generation)
#            -> etest_merge.sh (canonical metadata + staged-contract check)
#            -> etest_demand_utilization.sh (k(u) table + per-SOW utilization u)
#            -> etest_prep_chunks.sh (25 array tasks: step-04 pywrdrb inputs and
#               the k(u)-scaled NYC demand series per chunk)
#            -> etest_realized_utilization.sh (realized u per SOW + the full
#               staged-contract check)
#   hazard   etest_hazard_image_shards.sh (one task per chunk) ->
#            etest_hazard_image_merge.sh (the sub-window hazard image)
#   all      build, then hazard
#   --from STEP   start the build chain at STEP (shards | merge | demand | prep |
#                 realized), for resuming after a failed or already-run step
#   --after JOBID make the first submitted step wait for JOBID (afterok)
#   --dry-run     print the sbatch lines without submitting
#   any further arguments are appended to every sbatch line verbatim
#
# Identity: the historic production env (no MOEA settings are read) and
# NYCOPT_ETEST_VARIANT=kn. Every step is idempotent or refuses to repeat a
# product that exists (shards and hazard shards resume on their markers;
# NYCOPT_ENSEMBLE_FORCE=1 / NYCOPT_FORCE=1 override), so after a failure
# resubmit with --from <failed step>.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source workflow/cluster.sh

usage() {
    sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

[[ $# -ge 1 ]] || usage 2
MODE="$1"; shift
case "${MODE}" in build|hazard|all) ;; -h|--help) usage 0 ;; *) echo "ERROR: unknown mode '${MODE}'" >&2; usage 2 ;; esac

BUILD_STEPS=(shards merge demand prep realized)
DRY_RUN=false
FROM=shards
LAST=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true ;;
        --from)     shift; FROM="${1:?--from needs a step}" ;;
        --from=*)   FROM="${1#--from=}" ;;
        --after)    shift; LAST="${1:?--after needs a job id}" ;;
        --after=*)  LAST="${1#--after=}" ;;
        -h|--help)  usage 0 ;;
        *)          EXTRA_ARGS+=("$1") ;;
    esac
    shift
done
[[ " ${BUILD_STEPS[*]} " == *" ${FROM} "* ]] || { echo "ERROR: --from must be one of: ${BUILD_STEPS[*]} (got '${FROM}')" >&2; exit 2; }

ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env
VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"
S=workflow/supplemental/staging
EXPORT="ALL,NYCOPT_ENV_FILE=${ENV_FILE},NYCOPT_ETEST_VARIANT=${VARIANT}"
[[ -f "${ENV_FILE}" ]] || { echo "ERROR: env file not found: ${ENV_FILE}" >&2; exit 2; }
mkdir -p logs

# submit <label> <script>: chain behind ${LAST} (if any); sets LAST to the job id.
submit() {
    local label="$1" script="$2"
    local cmd=(sbatch --parsable)
    [[ -n "${LAST}" ]] && cmd+=("--dependency=afterok:${LAST}")
    cmd+=("--export=${EXPORT}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} "${script}")
    echo "+ ${cmd[*]}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        LAST="<${label}>"
    else
        LAST="$("${cmd[@]}")"
        LAST="${LAST%%;*}"   # --parsable prints "jobid[;cluster]"
    fi
    echo "  ${label} -> ${LAST}"
}

echo "=== E_test staging (${MODE}): variant ${VARIANT} ==="
if [[ "${MODE}" == "build" || "${MODE}" == "all" ]]; then
    started=false
    for step in "${BUILD_STEPS[@]}"; do
        [[ "${step}" == "${FROM}" ]] && started=true
        [[ "${started}" == "true" ]] || continue
        case "${step}" in
            shards)   submit shards        "${S}/etest_shards.sh" ;;
            merge)    submit merge         "${S}/etest_merge.sh" ;;
            demand)   submit demand_u      "${S}/etest_demand_utilization.sh" ;;
            prep)     submit prep_chunks   "${S}/etest_prep_chunks.sh" ;;
            realized) submit realized_u    "${S}/etest_realized_utilization.sh" ;;
        esac
    done
fi
if [[ "${MODE}" == "hazard" || "${MODE}" == "all" ]]; then
    submit hazimg_shards "${S}/etest_hazard_image_shards.sh"
    submit hazimg_merge  "${S}/etest_hazard_image_merge.sh"
fi
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run: nothing submitted)"
exit 0
