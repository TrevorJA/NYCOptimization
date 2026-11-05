#!/bin/bash
# submit_search_staging.sh — stage one campaign design's search ensemble
# (steps 02 to 04) as a dependency chain. Login-node helper, not an sbatch
# script: it issues the sbatch lines so the per-design arrays and the pool
# knob are recorded here rather than retyped at the prompt.
#
# Usage (from the repo root):
#   bash workflow/submit_search_staging.sh <design> [--dry-run] [sbatch args...]
#
#   <design>  monte_carlo | hazard_filling_stationary  (aliases: mc, hf);
#             historic has no staged ensemble and is refused
#   --dry-run print the sbatch lines without submitting
#   any further arguments are appended to every sbatch line verbatim
#
# What is chained (K = NYCOPT_STAGING_DRAWS staged draws, default 2: d0 is
# searched, d1 serves the SI draw-sensitivity re-evaluation):
#   monte_carlo                02 --array=0-(K-1)  ->  04 --array=0-(K-1)
#   hazard_filling_stationary  03 (all K draws from the P = 10^6 pool, which
#                              supplemental/staging/pool_shards.sh -> pool_merge.sh
#                              -> pool_verify.sh must have built)  ->  04 --array=0-(K-1)
# Every line carries the design's production env file; step 03 also gets
# NYCOPT_CANDIDATE_POOL_N=1000000 (the registry default is the P = 2,000 smoke
# pool). Run supplemental/checks/validate_staged_seasonality.sh afterwards.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

[[ $# -ge 1 ]] || usage 2
DESIGN_ARG="$1"; shift

DRY_RUN=false
EXTRA_ARGS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage 0 ;;
        *) EXTRA_ARGS+=("$a") ;;
    esac
done

case "${DESIGN_ARG}" in
    monte_carlo|mc)               DESIGN=monte_carlo ;;
    hazard_filling_stationary|hf) DESIGN=hazard_filling_stationary ;;
    historic|hist) echo "ERROR: historic uses the single observed trace; nothing to stage" >&2; exit 2 ;;
    *) echo "ERROR: unknown design '${DESIGN_ARG}'" >&2; usage 2 ;;
esac

ENV_FILE="workflow/envs/ffmp_obj8_${DESIGN}_production.env"
[[ -f "${ENV_FILE}" ]] || { echo "ERROR: env file not found: ${ENV_FILE}" >&2; exit 2; }
K="${NYCOPT_STAGING_DRAWS:-2}"
ARRAY="0-$(( K - 1 ))"
POOL_N=1000000
POOL_DIR="outputs/synthetic_ensembles/statpool_10yr_n${POOL_N}_d0"
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

echo "=== ${DESIGN}: staging draws ${ARRAY} with ${ENV_FILE} ==="
case "${DESIGN}" in
    monte_carlo)
        J02="$(submit j02 "" --array="${ARRAY}" --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}" \
               workflow/02_generate_ensemble.sh)"
        J04="$(submit j04 "${J02}" --array="${ARRAY}" --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}" \
               workflow/04_prep_pywrdrb_inputs.sh)"
        echo "step 02 -> ${J02}; step 04 -> ${J04}"
        ;;
    hazard_filling_stationary)
        [[ -d "${POOL_DIR}" ]] || {
            echo "ERROR: candidate pool not staged (${POOL_DIR}); build it with" \
                 "supplemental/staging/pool_shards.sh -> pool_merge.sh -> pool_verify.sh first" >&2
            exit 2
        }
        J03="$(submit j03 "" --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}",NYCOPT_CANDIDATE_POOL_N="${POOL_N}" \
               workflow/03_subsample_ensemble.sh)"
        J04="$(submit j04 "${J03}" --array="${ARRAY}" --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}" \
               workflow/04_prep_pywrdrb_inputs.sh)"
        echo "step 03 -> ${J03}; step 04 -> ${J04}"
        ;;
esac
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run: nothing submitted)"
exit 0
