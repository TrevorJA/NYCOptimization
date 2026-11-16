#!/bin/bash
# submit_reeval_chunks.sh — the ONE submission path for the chunked E_test
# re-evaluation (workflow steps 10 + 11). Login-node helper, not an sbatch
# script: it issues the step-10 sbatch line at the measured geometry and, for a
# design, chains step 11 behind it, so no step-10 line is typed at the prompt.
#
# Usage (from the repo root):
#   bash workflow/submit_reeval_chunks.sh baseline [--nodes N] [--dry-run] [sbatch args...]
#   bash workflow/submit_reeval_chunks.sh <design> [--nodes N] [--no-merge | --merge-only]
#                                         [--dry-run] [sbatch args...]
#
#   baseline     the current-FFMP incumbent cube, once for all designs
#                (NYCOPT_CHUNK_POLICIES=baseline, in-job merge). Default: the
#                serial partition, one rank per chunk x 2 cpus (3.8 GB per rank
#                at Anvil's 1,896 MB per cpu), batch 25, 3 h, ~75 SU on Anvil.
#                With --nodes N: N whole nodes at the cluster's ranks per node
#                on the MPI partition, for a short wall.
#   <design>     historic | monte_carlo | hazard_filling_stationary (aliases
#                hist, mc, hf): the installed equal-NFE reference set
#                outputs/<design>/ffmp_obj8/sets/ffmp_obj8_merged.set on the
#                MPI partition, batch 25, merge deferred to step 11 (submitted
#                with --dependency=afterok).
#   --nodes N    node count (design default 8); sets only the wall
#   --no-merge   submit step 10 only (a resumed pass after a truncated one)
#   --merge-only submit step 11 only (after the last resumed pass)
#   --dry-run    print the sbatch lines without submitting
#   further arguments are appended to every sbatch line verbatim
#
# The E_test preset is NYCOPT_REEVAL_ENSEMBLE_PRESET (default
# etest_kn_50yr_n12500); it is deliberately not in the env files.
#
# Work units: one unit is a (policy, chunk, block), a block being one
# simulation batch of the chunk (25 realizations = one SOW of the campaign
# E_test, so a 500-realization chunk is 20 units; src/chunk_reeval.py). One
# policy on 25 chunks is therefore 500 units and can use 500 ranks.
#
# Cost basis (measured on Anvil; ANVIL_NOTES.md, 2026-11-05 audit of the
# August jobs): one policy on one chunk (500 realizations x 50 yr) is ~80
# rank-minutes and 1.27-1.33 SU at 128 single-threaded ranks per node (jobs
# 19845823/19845825/19859233, 2026-08-13; node RSS 197 GB at batch 50). Batch
# 25 holds ~1.2 GB per rank (~150 GB per node) for ~2 % per-model-run overhead.
# Node count sets only the wall: the request below is ceil(units / ranks)
# waves x the per-unit wall x 1.1 + 15 min, capped at the cluster's wall cap.
# A pass the cap truncates is finished by resubmitting with --no-merge (step 10
# resumes per unit; NYCOPT_CHUNK_UNIT_SECONDS lets a rank stop cleanly before
# the wall), then --merge-only.
#
# The numbers above are Anvil's. Partition names, ranks per node, the
# rank-minutes per (policy, chunk) and the wall cap come from the cluster
# profile (workflow/clusters/), so the same command runs on Hopper at that
# cluster's geometry (one rank per physical core, no wall cap; pass --nodes for
# the node count to use there).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Cluster profile: partitions and geometry (on Hopper it also exports
# SBATCH_PARTITION and wraps sbatch, redirecting the Anvil #SBATCH headers).
# shellcheck disable=SC1091
source workflow/cluster.sh

usage() {
    sed -n '2,51p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

[[ $# -ge 1 ]] || usage 2
TARGET_ARG="$1"; shift

DRY_RUN=false
NODES=""
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
[[ -z "${NODES}" || ( "${NODES}" =~ ^[0-9]+$ && "${NODES}" -ge 1 ) ]] || { echo "ERROR: --nodes must be a positive integer (got '${NODES}')" >&2; exit 2; }

case "${TARGET_ARG}" in
    baseline)                        TARGET=baseline ;;
    historic|hist)                   TARGET=historic ;;
    monte_carlo|mc)                  TARGET=monte_carlo ;;
    hazard_filling_stationary|hf)    TARGET=hazard_filling_stationary ;;
    *) echo "ERROR: unknown target '${TARGET_ARG}'" >&2; usage 2 ;;
esac

# json_int FILE KEY: the first integer value of "KEY" in FILE (empty if absent).
json_int() { grep -o "\"$2\": *[0-9]*" "$1" 2>/dev/null | grep -o '[0-9]*$' | head -1 || true; }

PRESET="${NYCOPT_REEVAL_ENSEMBLE_PRESET:-etest_kn_50yr_n12500}"
PRESET_DIR="outputs/synthetic_ensembles/${PRESET}"
[[ -f "${PRESET_DIR}/_meta.json" ]] || { echo "ERROR: E_test preset '${PRESET}' is not staged (${PRESET_DIR}/_meta.json missing)" >&2; exit 2; }
if [[ -f "${PRESET_DIR}/chunk_index.json" ]]; then
    N_CHUNKS="$(json_int "${PRESET_DIR}/chunk_index.json" n_chunks)"
    CHUNK_SIZE="$(json_int "${PRESET_DIR}/chunk_index.json" chunk_size)"
else
    N_CHUNKS=1   # single-directory preset: one chunk
    CHUNK_SIZE="$(json_int "${PRESET_DIR}/_meta.json" n_realizations)"
fi
R_PER_SOW="$(json_int "${PRESET_DIR}/_meta.json" realizations_per_profile)"
[[ "${N_CHUNKS:-}" =~ ^[0-9]+$ && "${N_CHUNKS}" -ge 1 ]] || { echo "ERROR: cannot read n_chunks from ${PRESET_DIR}/chunk_index.json" >&2; exit 2; }
[[ "${CHUNK_SIZE:-}" =~ ^[0-9]+$ && "${CHUNK_SIZE}" -ge 1 ]] || { echo "ERROR: cannot read the chunk size of ${PRESET}" >&2; exit 2; }
[[ "${R_PER_SOW:-}" =~ ^[0-9]+$ && "${R_PER_SOW}" -ge 1 ]] || { echo "ERROR: cannot read realizations_per_profile from ${PRESET_DIR}/_meta.json" >&2; exit 2; }

# Per-(policy, chunk) basis and geometry from the cluster profile (Anvil:
# measured, see header); edit the profile if a re-measurement moves them.
UNIT_MINUTES="${NYCOPT_REEVAL_UNIT_MINUTES}"
UNIT_SU=1.3                                # Anvil billing; printed there only
BATCH=25                   # NYCOPT_SEARCH_REALIZATION_BATCH
RANKS_PER_NODE="${NYCOPT_RANKS_PER_NODE}"
WALL_CAP_H="${NYCOPT_WALL_CAP_HOURS}"      # 0 = no cap

# Unit granularity, as src.chunk_reeval.unit_layout: a block is one batch
# widened to whole SOWs (lcm), never larger than the chunk.
gcd() { local a="$1" b="$2" t; while (( b )); do t=$(( a % b )); a="$b"; b="$t"; done; echo "$a"; }
BLOCK=$(( BATCH * R_PER_SOW / $(gcd "${BATCH}" "${R_PER_SOW}") ))
(( BLOCK > CHUNK_SIZE )) && BLOCK="${CHUNK_SIZE}"
BLOCKS_PER_CHUNK=$(( (CHUNK_SIZE + BLOCK - 1) / BLOCK ))
BLOCK_SECONDS=$(( (UNIT_MINUTES * 60 * BLOCK + CHUNK_SIZE - 1) / CHUNK_SIZE ))   # wall of one unit on one rank
GUARD_SECONDS=$(( BLOCK_SECONDS + 60 ))   # NYCOPT_CHUNK_UNIT_SECONDS: unit wall + margin for the wall guard

[[ -d logs ]] || mkdir -p logs   # the #SBATCH --output paths require it

if [[ "${TARGET}" == "baseline" ]]; then
    ENV_FILE="workflow/envs/ffmp_obj8_historic_production.env"
    POLICIES="baseline"
    N_POLICIES=1
    [[ "${MODE}" == "both" ]] || { echo "ERROR: --no-merge / --merge-only apply to a design; the baseline cube merges in-job" >&2; exit 2; }
else
    ENV_FILE="workflow/envs/ffmp_obj8_${TARGET}_production.env"
    POLICIES="outputs/${TARGET}/ffmp_obj8/sets/ffmp_obj8_merged.set"
    [[ -f "${POLICIES}" ]] || { echo "ERROR: installed reference set not found: ${POLICIES} (run step 08 --merge --install first)" >&2; exit 2; }
    N_POLICIES="$(grep -cve '^#' -e '^[[:space:]]*$' "${POLICIES}" || true)"
    [[ "${N_POLICIES}" -ge 1 ]] || { echo "ERROR: ${POLICIES} holds no policy rows" >&2; exit 2; }
    [[ -n "${NODES}" ]] || NODES=8
fi
N_POLICY_CHUNKS=$(( N_POLICIES * N_CHUNKS ))
N_UNITS=$(( N_POLICY_CHUNKS * BLOCKS_PER_CHUNK ))
[[ -f "${ENV_FILE}" ]] || { echo "ERROR: env file not found: ${ENV_FILE}" >&2; exit 2; }

COMMON_EXPORT="ALL,NYCOPT_ENV_FILE=${ENV_FILE},DRAW=0,NYCOPT_REEVAL_ENSEMBLE_PRESET=${PRESET},NYCOPT_CHUNK_POLICIES=${POLICIES}"
SIM_EXPORT="${COMMON_EXPORT},NYCOPT_SEARCH_REALIZATION_BATCH=${BATCH},NYCOPT_CHUNK_UNIT_SECONDS=${GUARD_SECONDS}"

run() {   # print the line; submit unless dry-run; echo the job id on success
    echo "+ $*"
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "(dry-run: not submitted)"; JOB_ID=""
    else
        local out; out="$("$@")"; echo "${out}"; JOB_ID="${out##* }"
    fi
}

# mpi_wall RANKS: sets WAVES and WALL (HH:MM:00) for N_UNITS on RANKS ranks.
mpi_wall() {
    local ranks="$1" wall_min
    WAVES=$(( (N_UNITS + ranks - 1) / ranks ))
    wall_min=$(( (WAVES * BLOCK_SECONDS * 11 / 10 + 59) / 60 + 15 ))
    if (( WALL_CAP_H > 0 && wall_min > WALL_CAP_H * 60 )); then
        echo "NOTE: ${WAVES} waves of ${ranks} ranks need ~${wall_min} min; capping at ${WALL_CAP_H} h." \
             "Resubmit with --no-merge until every unit is written, then --merge-only (or raise --nodes)."
        wall_min=$(( WALL_CAP_H * 60 ))
    fi
    WALL="$(printf '%02d:%02d:00' $(( wall_min / 60 )) $(( wall_min % 60 )))"
}

SU_PER_UNIT="${UNIT_SU}"
[[ "${TARGET}" == "baseline" && -z "${NODES}" ]] && SU_PER_UNIT="$(awk -v s="${UNIT_SU}" 'BEGIN{print s*2}')"   # 2 cpus billed per rank on `shared`
COST=""
[[ "${NYCOPT_CLUSTER}" == "anvil" ]] && COST=", ~$(awk -v u="${N_POLICY_CHUNKS}" -v s="${SU_PER_UNIT}" 'BEGIN{printf "%.0f", u*s}') SU"
echo "=== ${TARGET} on ${NYCOPT_CLUSTER}: preset ${PRESET} (${N_CHUNKS} chunks x ${BLOCKS_PER_CHUNK} blocks of ${BLOCK}), policies ${POLICIES}, ${N_UNITS} units${COST} ==="

JOB_ID=""
if [[ "${TARGET}" == "baseline" ]]; then
    if [[ -z "${NODES}" ]]; then
        # --ntasks-per-node, not --ntasks: the step-10 header carries
        # --ntasks-per-node=16, which --ntasks does not override, so 25 tasks on one
        # node are otherwise rejected ("Requested node configuration is not available").
        echo "--- step 10: 1 x ${N_CHUNKS} ranks, ${BLOCKS_PER_CHUNK} units per rank of ~${BLOCK_SECONDS} s (pass --nodes N for more ranks)"
        run sbatch --job-name=reeval_baseline \
            --partition="${NYCOPT_PARTITION_SERIAL}" --nodes=1 --ntasks-per-node="${N_CHUNKS}" --cpus-per-task=2 --time=03:00:00 \
            --export="${SIM_EXPORT},NYCOPT_CHUNK_MERGE=job" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            workflow/10_simulate_test_chunks.sh
    else
        RANKS=$(( NODES * RANKS_PER_NODE ))
        mpi_wall "${RANKS}"
        (( RANKS > N_UNITS )) && echo "NOTE: ${RANKS} ranks for ${N_UNITS} units; $(( RANKS - N_UNITS )) rank(s) will have no work."
        echo "--- step 10: ${NODES} x ${RANKS_PER_NODE} ranks, ${WAVES} wave(s) of ~${BLOCK_SECONDS} s, wall ${WALL}"
        run sbatch --job-name=reeval_baseline \
            --partition="${NYCOPT_PARTITION_MPI}" --nodes="${NODES}" --ntasks-per-node="${RANKS_PER_NODE}" \
            --cpus-per-task="${NYCOPT_CPUS_PER_RANK}" --time="${WALL}" \
            --export="${SIM_EXPORT},NYCOPT_CHUNK_MERGE=job" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            workflow/10_simulate_test_chunks.sh
    fi
    exit 0
fi

case "${TARGET}" in
    historic)                  TAG=hist ;;
    monte_carlo)               TAG=mc ;;
    hazard_filling_stationary) TAG=hf ;;
esac

if [[ "${MODE}" != "merge" ]]; then
    RANKS=$(( NODES * RANKS_PER_NODE ))
    mpi_wall "${RANKS}"
    echo "--- step 10: ${NODES} x ${RANKS_PER_NODE} ranks, ${WAVES} wave(s) of ~${BLOCK_SECONDS} s, wall ${WALL}"
    run sbatch --job-name="reeval_${TAG}" \
        --partition="${NYCOPT_PARTITION_MPI}" --nodes="${NODES}" --ntasks-per-node="${RANKS_PER_NODE}" \
        --cpus-per-task="${NYCOPT_CPUS_PER_RANK}" --time="${WALL}" \
        --export="${SIM_EXPORT},NYCOPT_CHUNK_MERGE=off" \
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
