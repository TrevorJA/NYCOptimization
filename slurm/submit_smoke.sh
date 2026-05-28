#!/bin/bash
# submit_smoke.sh — launch one small-NFE smoke test per architecture.
#
# Purpose: verify the full distributed MOEA pipeline (MPI, Borg, policy
# instantiation, simulation, objectives) end-to-end for every formulation
# before committing to a multi-seed production campaign.
#
# Default targets: the manuscript scope (ffmp ann ffmp_8 ffmp_10 ffmp_12).
# rbf, tree, spline, and ffmp_6 are not in the default smoke set:
#   - rbf/tree are preserved-but-not-active per the manuscript scope decision
#     (their JARs were retired 2026-04-30; rebuild via `bash slurm/build_jars.sh
#     rbf tree` if reactivated).
#   - spline never had a JAR built.
#   - ffmp_6 is structurally identical to ffmp.
# Pass any of those explicitly to smoke them.
#
# Each job runs 500 NFE/island × 2 islands on a 2-node debug-window allocation.
#
# Usage:
#   bash slurm/submit_smoke.sh                    # default 5 targets
#   bash slurm/submit_smoke.sh ffmp rbf           # subset (rebuild JAR first if needed)
#   bash slurm/submit_smoke.sh ffmp_10            # specific N-zone variant
#   bash slurm/submit_smoke.sh --dry-run          # print sbatch cmds only
#
# After jobs finish, generate diagnostics + figures:
#   bash workflow/03_run_diagnostics.sh smoke_ffmp smoke_ann smoke_ffmp_8 smoke_ffmp_10 smoke_ffmp_12
#   for s in smoke_ffmp smoke_ann smoke_ffmp_8 smoke_ffmp_10 smoke_ffmp_12; do
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

DEFAULT_TARGETS=(ffmp ann ffmp_8 ffmp_10 ffmp_12)
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
