#!/bin/bash
# Step 7: Run MOEAFramework runtime diagnostics — builds the cross-seed
# reference set ({slug}_merged.set) and ε-BOX FILTERS it to archive
# resolution under the campaign epsilon vector (the raw plain-dominance
# union is kept as *_raw.set; see src/diagnostics.py::epsilon_box_filter_set
# — ResultFileMerger alone ignores --epsilon for archiving), plus per-seed
# merged sets and per-island runtime metrics scored against the filtered
# cross-seed set, then renders the standard post-search figure suite
# (parallel axes, hypervolume convergence, six-indicator runtime panel)
# under figures/{scenario}/{slug}/ (src/plotting/search_diagnostics.py).
# The filtered {slug}_merged.set is the FIRST-choice reference set of the
# step 08/09 re-evaluation (src/reevaluate{,_mpi}.py), so this step must run
# between search and re-evaluation for every optimization configuration.
#
# Run identity comes from the env file (optional here): NYCOPT_ENV_FILE picks
# the scenario design + MOEA config, and DRAW=k (default 0) the ensemble draw,
# exactly as in step 06 — with no positional args the target is the active
# config's own derived slug (plus the variable-resolution sweep slugs).
# Positional args are LITERAL moea slugs (an escape hatch for ad-hoc dirs) and
# are used as given. Runs targets in parallel as background jobs; the
# MOEAFramework CLI is I/O bound so there's no contention issue.
#
# Usage (from repo root):
#   NYCOPT_ENV_FILE=workflow/envs/<file>.env bash workflow/07_run_diagnostics.sh   # active slug + VR sweep
#   DRAW=1 NYCOPT_ENV_FILE=... bash workflow/07_run_diagnostics.sh                 # a d1 replicate's outputs
#   bash workflow/07_run_diagnostics.sh ffmp_obj8_mm_moderate                      # explicit slug(s)
#   bash workflow/07_run_diagnostics.sh --serial ffmp_obj8_mm_moderate             # single, serial
#   sbatch --export=ALL,NYCOPT_ENV_FILE=... workflow/07_run_diagnostics.sh
#
#SBATCH --job-name=diagnostics
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/diagnostics_%j.out
#SBATCH --error=logs/diagnostics_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
# Ensemble-draw identifier (same contract as step 06): selects which searched
# run's outputs to diagnose — the derived slug gains "_d{k}" for k>0.
export NYCOPT_ENSEMBLE_DRAW="${DRAW:-${NYCOPT_ENSEMBLE_DRAW:-0}}"

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
    # Default: the active config's derived slug for base FFMP + each
    # variable-resolution sweep formulation (draw-aware via the env above).
    mapfile -t TARGETS < <(python3 -c "
import contextlib, io, sys
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from config import FFMP_VR_N_SWEEP, derive_slug
sys.stderr.write(_buf.getvalue())
for f in ['ffmp'] + [f'ffmp_{n}' for n in FFMP_VR_N_SWEEP]:
    print(derive_slug(f))
")
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
