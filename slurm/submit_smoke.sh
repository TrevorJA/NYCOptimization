#!/bin/bash
# submit_smoke.sh — launch one small-NFE smoke test per architecture.
#
# Purpose: verify the full distributed MOEA pipeline (MPI, Borg, policy
# instantiation, simulation, objectives) end-to-end for every formulation
# before committing to a multi-seed production campaign.
#
# Default targets: ffmp rbf tree ann ffmp_6 (N-zone smoke at baseline-equivalent N).
# Each job runs 500 NFE/island × 2 islands on a 2-node debug-window allocation.
# Output slugs:  smoke_ffmp, smoke_rbf, smoke_tree, smoke_ann, smoke_ffmp_6.
#
# Usage:
#   bash slurm/submit_smoke.sh                    # all 5 targets
#   bash slurm/submit_smoke.sh ffmp rbf           # subset
#   bash slurm/submit_smoke.sh ffmp_10            # specific N-zone variant
#   bash slurm/submit_smoke.sh --dry-run          # print sbatch cmds only
#
# After jobs finish, generate diagnostics + figures:
#   bash workflow/03_run_diagnostics.sh smoke_ffmp smoke_rbf smoke_tree smoke_ann smoke_ffmp_6
#   for s in smoke_ffmp smoke_rbf smoke_tree smoke_ann smoke_ffmp_6; do
#       f=${s#smoke_}
#       bash workflow/04_plot_diagnostics.sh "$s" "$f"
#   done

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

DEFAULT_TARGETS=(ffmp rbf tree ann ffmp_6)
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
    export_args="FORMULATION=${t},RUN_SLUG=smoke_${t}"
    # If t looks like "ffmp_<N>", forward N_ZONES so mmborg_ffmp_vr.sh-style
    # dynamic formulation generation works inside the Python layer.
    if [[ "$t" =~ ^ffmp_([0-9]+)$ ]]; then
        export_args="${export_args},N_ZONES=${BASH_REMATCH[1]}"
    fi
    run sbatch --export=ALL,${export_args} \
        --job-name="smoke_${t}" \
        slurm/smoke_test.sh
done

echo "=== Smoke test submission complete ==="
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run — no jobs actually submitted)"
