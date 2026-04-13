#!/bin/bash
# submit_all.sh — Submit the full publication-scale MM Borg campaign.
#
# Submits:
#   - ffmp        (10 seeds, array job)
#   - rbf         (10 seeds, array job)
#   - tree        (10 seeds, array job)
#   - ann         (10 seeds, array job)
#   - ffmp_vr     (10 seeds × len(FFMP_VR_N_SWEEP) array jobs, one per N value)
#
# Usage:
#   bash slurm/submit_all.sh                  # submit everything
#   bash slurm/submit_all.sh --dry-run        # print sbatch commands only
#   bash slurm/submit_all.sh ffmp rbf         # submit only listed architectures
#
# Environment overrides (forwarded to jobs via --export=ALL):
#   NYCOPT_STATE_SPEC=extended
#   NYCOPT_OBJECTIVE_SET=default

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

# Read the N sweep list from config.py so it stays in one place.
N_SWEEP=$(python3 -c "from config import FFMP_VR_N_SWEEP; print(' '.join(str(n) for n in FFMP_VR_N_SWEEP))")

# Default targets
if [[ ${#ARGS[@]} -eq 0 ]]; then
    TARGETS=(ffmp rbf tree ann ffmp_vr)
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
    case "$t" in
        ffmp|rbf|tree|ann)
            run sbatch --export=ALL --array=1-10 "slurm/mmborg_${t}.sh"
            ;;
        ffmp_vr)
            for N in ${N_SWEEP}; do
                run sbatch --export=ALL,N_ZONES="${N}" --array=1-10 \
                    --job-name="mmborg_ffmp_vr_N${N}" \
                    slurm/mmborg_ffmp_vr.sh
            done
            ;;
        *)
            echo "Unknown target: $t" >&2
            exit 1
            ;;
    esac
done

echo "=== Submission complete ==="
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run — no jobs actually submitted)"
