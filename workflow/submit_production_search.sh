#!/bin/bash
# submit_production_search.sh — the ONE submission path for a campaign search
# (workflow step 07 under the `production` MOEA config). Login-node helper, not
# an sbatch script: it issues the per-(design, seed) sbatch line that the env
# file headers, workflow/README.md and campaign_design.md §2 document, so the
# geometry and wall request are recorded here rather than retyped at the prompt.
#
# Usage (from the repo root):
#   bash workflow/submit_production_search.sh <design> <seed> [--dry-run] [sbatch args...]
#
#   <design>  historic | monte_carlo | hazard_filling_stationary
#             (aliases: hist, mc, hf)
#   <seed>    1 | 2  (the --array index = Borg seed; one seed per job because
#             the per-island NFE and therefore the wall depend on the seed:
#             seed 1 runs 750k total NFE, seed 2 500k; MOEAConfig.production)
#   --dry-run print the sbatch line without submitting
#   any further arguments are appended to sbatch verbatim
#             (e.g. --dependency=afterok:<jobid>, --time=HH:MM:SS to override)
#
# Examples:
#   bash workflow/submit_production_search.sh historic 1
#   bash workflow/submit_production_search.sh historic 2 --dependency=afterok:21951234
#   bash workflow/submit_production_search.sh monte_carlo 1 --dry-run
#
# Geometry: `production` = 1 controller + 4 islands x (382 workers + 1 master)
# = 1,533 ranks, which needs --nodes=12 --ntasks-per-node=128 on `wholenode`
# (1,536 cores, 3 idle). The rank count itself comes from the MOEA config
# inside the job (07_run_mmborg.sh -> mpirun -np); nycopt_check_allocation
# aborts the job if this container no longer fits it, so the numbers below
# are a documented request, not a second source of truth.
#
# Wall requests (campaign_design.md §2 / §6; ANVIL_NOTES.md measured costs):
#   matched designs (N = 100 ensembles): 96 h seed 1, 72 h seed 2
#     (21.3 h measured per 500k NFE on 8 nodes; the 96 h Anvil cap bounds seed 1)
#   historic (single trace, NFE-bounded): 12 h seed 1, 8 h seed 2
#     (4.07 h measured per 500k NFE; ~8.35 SU per 1,000 NFE)
# Override with a trailing --time=... (the last --time sbatch sees wins).
#
# The campaign searches draw 0 only (DRAW=0 for the ensemble designs; historic
# accepts draw 0 only). The pre-flight inside 06 refuses a (design, seed) whose
# .set file already exists unless NYCOPT_OVERWRITE=1 is exported.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

[[ $# -ge 2 ]] || usage 2
DESIGN_ARG="$1"; SEED="$2"; shift 2

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
    historic|hist)                   DESIGN=historic ;;
    monte_carlo|mc)                  DESIGN=monte_carlo ;;
    hazard_filling_stationary|hf)    DESIGN=hazard_filling_stationary ;;
    *) echo "ERROR: unknown design '${DESIGN_ARG}'" >&2; usage 2 ;;
esac

case "${SEED}" in
    1|2) ;;
    *) echo "ERROR: seed must be 1 or 2 (got '${SEED}')" >&2; usage 2 ;;
esac

# Per-design job-name tag and per-seed wall request; the env file is named
# after the design.
ENV_FILE="workflow/envs/ffmp_obj8_${DESIGN}_production.env"
case "${DESIGN}" in
    historic)                  TAG=hist; TIME=([1]=12:00:00 [2]=08:00:00) ;;
    monte_carlo)               TAG=mc;   TIME=([1]=96:00:00 [2]=72:00:00) ;;
    hazard_filling_stationary) TAG=hf;   TIME=([1]=96:00:00 [2]=72:00:00) ;;
esac

NODES=12
NTASKS_PER_NODE=128

[[ -f "${ENV_FILE}" ]] || { echo "ERROR: env file not found: ${ENV_FILE}" >&2; exit 2; }
[[ -d logs ]] || mkdir -p logs   # the #SBATCH --output path in 06 requires it

CMD=(sbatch
    --job-name="prod_${TAG}"
    --nodes="${NODES}"
    --ntasks-per-node="${NTASKS_PER_NODE}"
    --time="${TIME[${SEED}]}"
    --array="${SEED}"
    --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}",DRAW=0
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    workflow/07_run_mmborg.sh)

echo "=== ${DESIGN} seed ${SEED}: ${NODES} x ${NTASKS_PER_NODE}, ${TIME[${SEED}]}, ${ENV_FILE} ==="
echo "+ ${CMD[*]}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "(dry-run: not submitted)"
else
    "${CMD[@]}"
fi
