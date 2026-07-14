#!/bin/bash
# ensemble_cost_stage_submit.sh — stage every (N, L) ensemble the cost-surface
# benchmark measures. Run from the repo root ON A LOGIN NODE: this submits jobs,
# it is not itself a job.
#
# For each (N, L) in supplemental_config.ensemble_cost_staging_cells() that is
# not already fully staged, submits workflow step 02 (generate kn_{L}yr_n{N})
# and then step 04 (prep the pywrdrb HDF5 inputs) with --dependency=afterok.
# Both steps are idempotent, so re-running this after a partial failure submits
# only what is still missing.
#
# Usage:
#   bash workflow/supplemental/ensemble_cost_stage_submit.sh            # submit
#   bash workflow/supplemental/ensemble_cost_stage_submit.sh --dry-run  # print only
#
# Notes:
#   * "Already staged" means all five required HDF5s exist and are non-empty
#     (src.ensembles.staged_ensemble_missing) — NOT "the directory exists". A
#     metadata-only leftover directory would otherwise make step 02's own
#     already-staged check skip generation, and step 04 would then die on the
#     absent inflow file. Any cell whose directory exists but is incomplete is
#     regenerated with NYCOPT_ENSEMBLE_FORCE=1.
#   * Step 04 runs one pywrdrb preprocessor pass per realization across MPI
#     ranks, so small-N cells get fewer ranks (N=1 with 33 ranks would hand 32
#     ranks an empty slice); large-N cells get a longer walltime.
#   * The (N, L) shape is passed via --export; the generator resolves it through
#     the scaling_stationary design (NYCOPT_SCALING_KN_REALS / _YEARS).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ENV_FILE="workflow/envs/ensemble_cost.env"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Cells to stage, each tagged with what is missing. One python call: importing
# config per cell would dominate the runtime of this script.
CELLS_RAW="$(python3 -c "
import supplemental_config as scfg
scfg.configure_ensemble_cost_env()
from src.ensembles import staged_ensemble_missing

for n, ell in scfg.ensemble_cost_staging_cells():
    missing = staged_ensemble_missing(f'kn_{ell}yr_n{n}')
    if not missing:
        continue
    need02 = any(f in ('gage_flow_mgd.hdf5', 'catchment_inflow_mgd.hdf5')
                 for f in missing)
    print(n, ell, int(need02), len(missing))
")"

if [[ -z "${CELLS_RAW}" ]]; then
    echo "[stage] all ensemble-cost cells are already staged; nothing to submit."
    exit 0
fi

echo "[stage] cells needing staging:"
echo "${CELLS_RAW}" | while read -r N L NEED02 NMISS; do
    printf '  N=%-4s L=%-3s  missing=%s  step02=%s\n' "${N}" "${L}" "${NMISS}" "${NEED02}"
done

while read -r N L NEED02 _NMISS; do
    SLUG="kn_${L}yr_n${N}"
    EXPORTS="ALL,NYCOPT_ENV_FILE=${ENV_FILE},NYCOPT_SCALING_KN_REALS=${N},NYCOPT_SCALING_KN_YEARS=${L}"

    # Step 04's per-realization work is split across ranks; never ask for more
    # ranks than realizations.
    NTASKS_04=$(( N < 32 ? N : 32 ))
    (( NTASKS_04 < 1 )) && NTASKS_04=1
    # N >= 100 is ~4-7 waves of pywrdrb preprocessor passes; 1 h has no margin.
    TIME_04="01:00:00"
    (( N >= 100 )) && TIME_04="02:00:00"

    DEP=""
    if (( NEED02 )); then
        # A partially-staged directory makes the generator's own already-staged
        # check skip the cell; force it.
        CMD02=(sbatch --parsable
               --export="${EXPORTS},NYCOPT_ENSEMBLE_FORCE=1"
               --job-name="gen_${SLUG}"
               workflow/02_generate_ensemble.sh)
        if (( DRY_RUN )); then
            echo "[dry-run] ${CMD02[*]}"
            JID02="<jid02>"
        else
            JID02="$("${CMD02[@]}")"
            echo "[stage] ${SLUG}: step 02 -> job ${JID02}"
        fi
        DEP="--dependency=afterok:${JID02}"
    fi

    CMD04=(sbatch --parsable
           ${DEP}
           --export="${EXPORTS}"
           --job-name="prep_${SLUG}"
           --ntasks-per-node="${NTASKS_04}"
           --time="${TIME_04}"
           workflow/04_prep_pywrdrb_inputs.sh)
    if (( DRY_RUN )); then
        echo "[dry-run] ${CMD04[*]}"
    else
        JID04="$("${CMD04[@]}")"
        echo "[stage] ${SLUG}: step 04 -> job ${JID04} (ranks=${NTASKS_04}, time=${TIME_04})"
    fi
done <<< "${CELLS_RAW}"

echo "[stage] submitted. Watch with: squeue -u \"\$USER\" -o '%.10i %.20j %.2t %.10M %R'"
