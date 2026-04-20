#!/bin/bash
# Step 3: Run MOEAFramework runtime diagnostics — computes hypervolume,
# generational distance, and builds the global reference set.
#
# By default, runs all architectures in parallel as background jobs; the
# MOEAFramework CLI is I/O bound so there's no contention issue. Pass
# specific slug names to run a subset. Slugs equal formulation names for
# plain production runs; smoke-test / variant runs use custom slugs like
# smoke_ffmp or ann_reduced_state.
#
# Usage:
#   bash workflow/03_run_diagnostics.sh                         # all default slugs (parallel)
#   bash workflow/03_run_diagnostics.sh ffmp rbf                # subset
#   bash workflow/03_run_diagnostics.sh smoke_ffmp smoke_rbf    # custom slugs
#   bash workflow/03_run_diagnostics.sh --serial ffmp           # single, serial
#   sbatch workflow/03_run_diagnostics.sh
#
#SBATCH --job-name=diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/diagnostics_%j.out
#SBATCH --error=logs/diagnostics_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
module load python/3.11.5
source venv/bin/activate

SERIAL=false
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--serial" ]]; then
        SERIAL=true
    else
        ARGS+=("$a")
    fi
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
    # Default: all registered architectures + FFMP_VR sweep values.
    N_SWEEP=$(python3 -c "from config import FFMP_VR_N_SWEEP; print(' '.join(str(n) for n in FFMP_VR_N_SWEEP))")
    TARGETS=(ffmp rbf tree ann)
    for N in ${N_SWEEP}; do TARGETS+=("ffmp_${N}"); done
else
    TARGETS=("${ARGS[@]}")
fi

echo "=== Diagnostics targets: ${TARGETS[*]} (serial=${SERIAL}) ==="

run_one() {
    local slug="$1"
    echo "[${slug}] starting"
    python3 -c "
import sys; sys.path.insert(0, '.')
from src.diagnostics import run_full_diagnostics
run_full_diagnostics('${slug}')
" > "logs/diagnostics_${slug}.log" 2>&1
    echo "[${slug}] done (log: logs/diagnostics_${slug}.log)"
}

if [[ "${SERIAL}" == "true" ]]; then
    for t in "${TARGETS[@]}"; do run_one "$t"; done
else
    pids=()
    for t in "${TARGETS[@]}"; do
        run_one "$t" &
        pids+=($!)
    done
    fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=$((fail + 1))
    done
    if [[ $fail -gt 0 ]]; then
        echo "ERROR: ${fail} diagnostics job(s) failed"
        exit 1
    fi
fi

echo "=== Completed: $(date) ==="
