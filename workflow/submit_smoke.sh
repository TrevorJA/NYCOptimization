#!/bin/bash
# submit_smoke.sh — DEV UTILITY: launch one small-NFE smoke test per
# formulation. Not part of the replication pipeline.
#
# Purpose: verify the full distributed MOEA pipeline (MPI, Borg, simulation,
# objectives) end-to-end for every formulation before committing to a
# multi-day production run.
#
# Each target submits the unified launcher (workflow/06_run_mmborg.sh) with
# smoke-sized sbatch overrides: Anvil `debug` queue (2 nodes x 40 tasks, 2 h —
# the debug queue's exact limits), the `smoke` MOEA config (via
# workflow/envs/smoke.env), the short 2018-2022 debug window, and 79 MPI ranks
# (1 controller + 2 islands x (38 workers + 1 master), sized to this
# allocation via the NTASKS_MPI caller override).
#
# Default targets: ffmp + variable-resolution FFMP at each N in FFMP_VR_N_SWEEP.
# ffmp_6 is structurally identical to ffmp and is omitted from the default set.
#
# Usage (from repo root):
#   bash workflow/submit_smoke.sh                    # default 4 targets
#   bash workflow/submit_smoke.sh ffmp_10            # specific N-zone variant
#   bash workflow/submit_smoke.sh --dry-run          # print sbatch cmds only
#
# After jobs finish, generate diagnostics:
#   bash workflow/07_run_diagnostics.sh smoke_ffmp smoke_ffmp_8 smoke_ffmp_10 smoke_ffmp_12

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        ARGS+=("$a")
    fi
done

DEFAULT_TARGETS=(ffmp ffmp_8 ffmp_10 ffmp_12)
if [[ ${#ARGS[@]} -eq 0 ]]; then
    TARGETS=("${DEFAULT_TARGETS[@]}")
else
    TARGETS=("${ARGS[@]}")
fi

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" == "false" ]]; then
        "$@"
    fi
}

for t in "${TARGETS[@]}"; do
    run sbatch \
        --partition=debug --nodes=2 --ntasks-per-node=40 --time=02:00:00 \
        --job-name="smoke_${t}" \
        --export=ALL,NYCOPT_ENV_FILE=workflow/envs/smoke.env,FORMULATION="${t}",RUN_SLUG="smoke_${t}",NTASKS_MPI=79,DEBUG_SIM=true \
        workflow/06_run_mmborg.sh
done

echo "=== Smoke test submission complete ==="
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run — no jobs actually submitted)"
