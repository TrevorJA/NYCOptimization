#!/bin/bash
# Step 7: MOEAFramework runtime diagnostics on the coordinator archive. Per
# seed, rebuilds the MM Borg coordinator archive at every runtime snapshot as
# the epsilon-box merge of the island runtime files
# (runtime_coordinator/seed_SS_{slug}.runtime; final entry checked against the
# seed's controller .set), builds the diagnostics reference set
# sets/{slug}_diagnostics_reference.set as the epsilon-box merge of every
# coordinator snapshot of every seed, scores one metrics/seed_SS_{slug}.metrics
# per seed with MOEAFramework's MetricsEvaluator, and renders the search figures
# under figures/{scenario}/{slug}/ (one line per seed on total NFE;
# src/diagnostics.py). All merges use the one epsilon-archive implementation in
# src/sensitivity_common.py; MOEAFramework's mergers are not used.
# The campaign reference set {slug}_merged.set of the step 08/09 re-evaluation
# is written only by scripts/main/extract_runtime_archive.py --merge --install;
# this step never touches it, so it may run before or after that script. The
# search_01 figure needs the seed-1 extraction (--seed 1) to exist.
#
# Env inputs: NYCOPT_ENV_FILE (optional; picks design + MOEA config) and DRAW=k
# (default 0), as in step 06. With no positional args the targets are the
# active config's derived slug plus the variable-resolution sweep slugs;
# positional args are literal moea slugs. Targets run in parallel unless
# --serial is passed.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/<file>.env workflow/07_run_diagnostics.sh
#   bash workflow/07_run_diagnostics.sh --serial ffmp_obj8      # explicit slug, serial
#
# Outputs: outputs/{scenario}/{slug}/{runtime_coordinator,sets,metrics}/, figures.
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
