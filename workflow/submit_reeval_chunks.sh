#!/bin/bash
# submit_reeval_chunks.sh — the ONE submission path for the chunked E_test
# re-evaluation (workflow steps 10 + 11). Login-node helper, not an sbatch
# script: it issues the step-10 sbatch line at the measured geometry and, for a
# design, chains step 11 behind it, so no step-10 line is typed at the prompt.
#
# Usage (from the repo root):
#   bash workflow/submit_reeval_chunks.sh baseline [--dry-run] [sbatch args...]
#   bash workflow/submit_reeval_chunks.sh <design> [--nodes N] [--no-merge | --merge-only]
#                                         [--dry-run] [sbatch args...]
#
#   baseline     the current-FFMP incumbent cube, once for all designs
#                (NYCOPT_CHUNK_POLICIES=baseline, in-job merge): `shared`, one
#                rank per chunk x 2 cpus (3.8 GB per rank at Anvil's 1,896 MB
#                per cpu), batch 25, 3 h. One wave of 25 units, ~75 SU.
#   <design>     historic | monte_carlo | hazard_filling_stationary (aliases
#                hist, mc, hf): the installed equal-NFE reference set
#                outputs/<design>/ffmp_obj8/sets/ffmp_obj8_merged.set on
#                `wholenode`, 128 ranks per node, batch 25, merge deferred to
#                step 11 (submitted with --dependency=afterok).
#   --nodes N    wholenode count for a design (default 8); sets only the wall
#   --no-merge   submit step 10 only (a resumed pass after a truncated one)
#   --merge-only submit step 11 only (after the last resumed pass)
#   --dry-run    print the sbatch lines without submitting
#   further arguments are appended to every sbatch line verbatim
#
# The E_test preset is NYCOPT_REEVAL_ENSEMBLE_PRESET (default
# etest_kn_50yr_n12500); it is deliberately not in the env files.
#
# Geometry (measured; ANVIL_NOTES.md, 2026-11-05 audit of the August jobs): one
# (policy, chunk) unit is 500 realizations x 50 yr, ~80 min per unit per rank
# and 1.27-1.33 SU per unit at 128 single-threaded ranks per node (jobs
# 19845823/19845825/19859233, 2026-08-13; node RSS 197 GB at batch 50). Batch 25
# holds ~1.2 GB per rank (~150 GB per node) for ~2 % per-model-run overhead.
# The 16 ranks x 8 cpus `shared` layout the records carried until 2026-11-05
# takes the same wall per unit and bills 8x. Node count sets only the wall: the
# request below is ceil(units / ranks) waves x 80 min x 1.1 + 15 min, capped
# at 96 h. A pass the cap truncates is finished by resubmitting with
# --no-merge (step 10 resumes per unit; NYCOPT_CHUNK_UNIT_SECONDS lets a rank
# stop cleanly before the wall), then --merge-only.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

[[ $# -ge 1 ]] || usage 2
TARGET_ARG="$1"; shift

DRY_RUN=false
NODES=8
MODE=both          # both | sim | merge
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true ;;
        --nodes)      shift; NODES="${1:?--nodes needs a count}" ;;
        --nodes=*)    NODES="${1#--nodes=}" ;;
        --no-merge)   MODE=sim ;;
        --merge-only) MODE=merge ;;
        -h|--help)    usage 0 ;;
        *)            EXTRA_ARGS+=("$1") ;;
    esac
    shift
done
[[ "${NODES}" =~ ^[0-9]+$ && "${NODES}" -ge 1 ]] || { echo "ERROR: --nodes must be a positive integer (got '${NODES}')" >&2; exit 2; }

case "${TARGET_ARG}" in
    baseline)                        TARGET=baseline ;;
    historic|hist)                   TARGET=historic ;;
    monte_carlo|mc)                  TARGET=monte_carlo ;;
    hazard_filling_stationary|hf)    TARGET=hazard_filling_stationary ;;
    *) echo "ERROR: unknown target '${TARGET_ARG}'" >&2; usage 2 ;;
esac

PRESET="${NYCOPT_REEVAL_ENSEMBLE_PRESET:-etest_kn_50yr_n12500}"
PRESET_DIR="outputs/synthetic_ensembles/${PRESET}"
[[ -f "${PRESET_DIR}/_meta.json" ]] || { echo "ERROR: E_test preset '${PRESET}' is not staged (${PRESET_DIR}/_meta.json missing)" >&2; exit 2; }
if [[ -f "${PRESET_DIR}/chunk_index.json" ]]; then
    N_CHUNKS="$(grep -o '"n_chunks": *[0-9]*' "${PRESET_DIR}/chunk_index.json" | grep -o '[0-9]*$' | head -1)"
else
    N_CHUNKS=1   # single-directory preset: one chunk
fi
[[ "${N_CHUNKS:-}" =~ ^[0-9]+$ && "${N_CHUNKS}" -ge 1 ]] || { echo "ERROR: cannot read n_chunks from ${PRESET_DIR}/chunk_index.json" >&2; exit 2; }

# Measured per-unit basis (see header); edit here if a re-measurement moves it.
UNIT_MINUTES=80
UNIT_SU=1.3
UNIT_SECONDS=4900          # NYCOPT_CHUNK_UNIT_SECONDS: 80 min + margin for the wall guard
BATCH=25                   # NYCOPT_SEARCH_REALIZATION_BATCH
RANKS_PER_NODE=128
WALL_CAP_H=96

[[ -d logs ]] || mkdir -p logs   # the #SBATCH --output paths require it

if [[ "${TARGET}" == "baseline" ]]; then
    ENV_FILE="workflow/envs/ffmp_obj8_historic_production.env"
    POLICIES="baseline"
    N_UNITS="${N_CHUNKS}"
    [[ "${MODE}" == "both" ]] || { echo "ERROR: --no-merge / --merge-only apply to a design; the baseline cube merges in-job" >&2; exit 2; }
else
    ENV_FILE="workflow/envs/ffmp_obj8_${TARGET}_production.env"
    POLICIES="outputs/${TARGET}/ffmp_obj8/sets/ffmp_obj8_merged.set"
    [[ -f "${POLICIES}" ]] || { echo "ERROR: installed reference set not found: ${POLICIES} (run step 08 --merge --install first)" >&2; exit 2; }
    N_POLICIES="$(grep -cve '^#' -e '^[[:space:]]*$' "${POLICIES}" || true)"
    [[ "${N_POLICIES}" -ge 1 ]] || { echo "ERROR: ${POLICIES} holds no policy rows" >&2; exit 2; }
    N_UNITS=$(( N_POLICIES * N_CHUNKS ))
fi
[[ -f "${ENV_FILE}" ]] || { echo "ERROR: env file not found: ${ENV_FILE}" >&2; exit 2; }

COMMON_EXPORT="ALL,NYCOPT_ENV_FILE=${ENV_FILE},DRAW=0,NYCOPT_REEVAL_ENSEMBLE_PRESET=${PRESET},NYCOPT_CHUNK_POLICIES=${POLICIES}"

run() {   # print the line; submit unless dry-run; echo the job id on success
    echo "+ $*"
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "(dry-run: not submitted)"; JOB_ID=""
    else
        local out; out="$("$@")"; echo "${out}"; JOB_ID="${out##* }"
    fi
}

SU_PER_UNIT="${UNIT_SU}"
[[ "${TARGET}" == "baseline" ]] && SU_PER_UNIT="$(awk -v s="${UNIT_SU}" 'BEGIN{print s*2}')"   # 2 cpus billed per rank on `shared`
echo "=== ${TARGET}: preset ${PRESET} (${N_CHUNKS} chunks), policies ${POLICIES}, ${N_UNITS} units, ~$(awk -v u="${N_UNITS}" -v s="${SU_PER_UNIT}" 'BEGIN{printf "%.0f", u*s}') SU ==="

JOB_ID=""
if [[ "${TARGET}" == "baseline" ]]; then
    # --ntasks-per-node, not --ntasks: the step-10 header carries
    # --ntasks-per-node=16, which --ntasks does not override, so 25 tasks on one
    # node are otherwise rejected ("Requested node configuration is not available").
    run sbatch --job-name=reeval_baseline \
        --partition=shared --nodes=1 --ntasks-per-node="${N_CHUNKS}" --cpus-per-task=2 --time=03:00:00 \
        --export="${COMMON_EXPORT},NYCOPT_SEARCH_REALIZATION_BATCH=${BATCH},NYCOPT_CHUNK_MERGE=job,NYCOPT_CHUNK_UNIT_SECONDS=${UNIT_SECONDS}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        workflow/10_simulate_test_chunks.sh
    exit 0
fi

case "${TARGET}" in
    historic)                  TAG=hist ;;
    monte_carlo)               TAG=mc ;;
    hazard_filling_stationary) TAG=hf ;;
esac

if [[ "${MODE}" != "merge" ]]; then
    RANKS=$(( NODES * RANKS_PER_NODE ))
    WAVES=$(( (N_UNITS + RANKS - 1) / RANKS ))
    WALL_MIN=$(( WAVES * UNIT_MINUTES * 11 / 10 + 15 ))
    if (( WALL_MIN > WALL_CAP_H * 60 )); then
        echo "NOTE: ${WAVES} waves of ${RANKS} ranks need ~${WALL_MIN} min; capping at ${WALL_CAP_H} h." \
             "Resubmit with --no-merge until every unit is written, then --merge-only (or raise --nodes)."
        WALL_MIN=$(( WALL_CAP_H * 60 ))
    fi
    WALL="$(printf '%02d:%02d:00' $(( WALL_MIN / 60 )) $(( WALL_MIN % 60 )))"
    echo "--- step 10: ${NODES} x ${RANKS_PER_NODE} ranks, ${WAVES} wave(s) of ~${UNIT_MINUTES} min, wall ${WALL}"
    run sbatch --job-name="reeval_${TAG}" \
        --partition=wholenode --nodes="${NODES}" --ntasks-per-node="${RANKS_PER_NODE}" --time="${WALL}" \
        --export="${COMMON_EXPORT},NYCOPT_SEARCH_REALIZATION_BATCH=${BATCH},NYCOPT_CHUNK_MERGE=off,NYCOPT_CHUNK_UNIT_SECONDS=${UNIT_SECONDS}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        workflow/10_simulate_test_chunks.sh
fi

if [[ "${MODE}" != "sim" ]]; then
    DEP=()
    [[ -n "${JOB_ID}" ]] && DEP=(--dependency="afterok:${JOB_ID}")
    echo "--- step 11: merge${JOB_ID:+ after job ${JOB_ID}}"
    run sbatch --job-name="merge_${TAG}" \
        ${DEP[@]+"${DEP[@]}"} \
        --export="${COMMON_EXPORT}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        workflow/11_merge_test_chunks.sh
fi
