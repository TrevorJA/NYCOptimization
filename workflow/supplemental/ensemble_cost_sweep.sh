#!/bin/bash
# ensemble_cost_sweep.sh — measure the t_eval(N, L, model) cost surface.
#
# On ONE exclusive Anvil node, run the cell list from
# supplemental_config.ENSEMBLE_COST_MODES[mode] sequentially. For each cell
# (N realizations, L years, trimmed|full model), `mpirun -np K bench_eval_worker.py`
# has every rank time 1 cold + M warm evaluations through the production
# `evaluate()` path and record wall time + peak RSS to per-rank CSV shards under
# outputs/supplemental/ensemble_cost_experiment/cells/. A fresh mpirun per cell
# gives a natural cold cache and prevents the (cached) model dict of one cell
# from surviving into the next.
#
# K is not a sweep axis here: each cell runs at the densest packing that fits
# node memory (supplemental_config.ensemble_cost_cell_k), because the packing
# sweep already showed SU/eval is minimized at full packing. Cells that cannot
# reach 128 ranks are a result, not a workaround — memory, not contention, is
# what caps the campaign's density at large N.
#
# Usage (from repo root):
#   # correctness gate on the already-staged kn_20yr_n20 (~15 min, shared):
#   sbatch --partition=shared --ntasks-per-node=4 --mem=32G --time=01:00:00 \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_cost.env,NYCOPT_BENCH_MODE=smoke \
#          workflow/supplemental/ensemble_cost_sweep.sh
#   # RSS/time calibration at the grid's corners, one rank, no contention:
#   sbatch --partition=shared --ntasks-per-node=1 --mem=64G --time=04:00:00 \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_cost.env,NYCOPT_BENCH_MODE=probe \
#          workflow/supplemental/ensemble_cost_sweep.sh
#   # the cells that price the campaign:
#   sbatch --time=06:00:00 --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_cost.env,NYCOPT_BENCH_MODE=core,NYCOPT_COST_BUDGET_S=20400 \
#          workflow/supplemental/ensemble_cost_sweep.sh
#   # the factorial remainder, one job per model variant (run concurrently):
#   sbatch --time=08:00:00 --export=ALL,...,NYCOPT_BENCH_MODE=rest_trimmed,NYCOPT_COST_BUDGET_S=27600 ...
#   sbatch --time=14:00:00 --export=ALL,...,NYCOPT_BENCH_MODE=rest_full,NYCOPT_COST_BUDGET_S=48000 ...
#
# Notes:
#   * NYCOPT_BENCH_MODE (smoke | probe | core | rest_trimmed | rest_full) selects
#     the cell list — the lists themselves live in supplemental_config.py
#     (no value flags).
#   * NYCOPT_COST_BUDGET_S guards the wall clock per cell, using each cell's own
#     estimated cost (they span ~200x): a cell starts only if its estimate fits
#     in the remaining budget, so a short allocation degrades by skipping the
#     tail rather than dying mid-cell. Default 90% of the SLURM time limit.
#   * A cell that dies (OOM at large N x L) is recorded in its step manifest and
#     the sweep continues — a missing shard at a large cell IS the memory
#     ceiling. The worker appends each eval's row as it completes, so even a
#     killed cell keeps the evals that finished.
#   * pywrdrb's per-rank /dev/shm model JSONs are never cleaned by the code
#     (src/simulation.py::_get_temp_dir); stale dirs are purged between cells so
#     RAM-backed tmpfs doesn't silently shrink usable node memory.
#   * The node must be exclusive for the wholenode modes so contention comes only
#     from our own ranks. smoke/probe run on shared with CLI overrides (they
#     measure single-rank cost and correctness, not contention).
#
#SBATCH --job-name=ens_cost
#SBATCH --account=ees260021
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/ensemble_cost_%j.out
#SBATCH --error=logs/ensemble_cost_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

MODE="${NYCOPT_BENCH_MODE:-core}"
JOB_ID="${SLURM_JOB_ID:-local}"

# Default budget = 90% of the allocated wall time, leaving room for the final
# manifest write and the exit-trap cleanup.
_time_limit_s() {
    local t
    t="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o '%l' 2>/dev/null || true)"
    [[ -z "${t}" || "${t}" == "UNLIMITED" ]] && { echo 0; return; }
    python3 - "${t}" <<'PY'
import sys
t = sys.argv[1]
days, _, rest = t.partition("-")
if not rest:
    rest, days = days, "0"
parts = [int(p) for p in rest.split(":")]
while len(parts) < 3:
    parts.insert(0, 0)
h, m, s = parts
print(int(days) * 86400 + h * 3600 + m * 60 + s)
PY
}
LIMIT_S="$(_time_limit_s)"
if [[ -n "${NYCOPT_COST_BUDGET_S:-}" ]]; then
    BUDGET_S="${NYCOPT_COST_BUDGET_S}"
elif (( LIMIT_S > 0 )); then
    BUDGET_S=$(( LIMIT_S * 9 / 10 ))
else
    BUDGET_S=18000
fi

_shm_clean() {
    find /dev/shm -maxdepth 1 -name 'pywrdrb_opt_r*' -user "${USER:-$(id -un)}" \
        -exec rm -rf {} + 2>/dev/null || true
}
trap _shm_clean EXIT

# Pre-flight: every cell's ensemble must be fully staged before we burn
# node-hours discovering it is not. Also reports each cell's chosen density and
# estimated cost, so the log records what the budget guard is working from.
CELLS_RAW="$(python3 -c "
import sys
import supplemental_config as scfg
scfg.configure_ensemble_cost_env()
from src.ensembles import staged_ensemble_missing

cells = scfg.ENSEMBLE_COST_MODES['${MODE}']
unstaged = []
for n, ell, model, m_warm, k in cells:
    slug = f'kn_{ell}yr_n{n}'
    missing = staged_ensemble_missing(slug)
    if missing:
        unstaged.append(f'{slug}: missing {\", \".join(missing)}')
if unstaged:
    sys.stderr.write(
        'ERROR: cells are not staged. Run '
        'workflow/supplemental/ensemble_cost_stage_submit.sh first:\n  '
        + '\n  '.join(unstaged) + '\n')
    raise SystemExit(2)

for n, ell, model, m_warm, k in cells:
    k = k or scfg.ensemble_cost_cell_k(n, ell, model)
    est = scfg.ensemble_cost_step_estimate_s(n, ell, model, m_warm)
    print(n, ell, model, m_warm, k, int(est))
")"
mapfile -t CELLS <<< "${CELLS_RAW}"
(( ${#CELLS[@]} > 0 )) || { echo "ERROR: no cells for mode '${MODE}'" >&2; exit 2; }

# smoke/probe run on `shared` with an --ntasks override; a cell asking for more
# ranks than the allocation has would fail at mpirun. Cap to what we hold.
NTASKS_AVAIL="${SLURM_NTASKS:-128}"

CELLS_DIR="$(python3 -c "import supplemental_config as s; print(s.ENSEMBLE_COST_CELLS_DIR)")"
MANIFEST_DIR="$(python3 -c "import supplemental_config as s; print(s.ENSEMBLE_COST_MANIFESTS_DIR)")"
mkdir -p "${CELLS_DIR}" "${MANIFEST_DIR}"
{
    echo "=== Ensemble-cost sweep manifest: ${MODE} job ${JOB_ID} ==="
    echo "Date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Host:        $(hostname)"
    echo "Partition:   ${SLURM_JOB_PARTITION:-n/a}"
    echo "Env file:    ${NYCOPT_ENV_FILE}"
    echo "Mode:        ${MODE}"
    echo "Budget (s):  ${BUDGET_S} (time limit ${LIMIT_S}s)"
    echo "Ranks avail: ${NTASKS_AVAIL}"
    echo "Cells (N L model M K est_s):"
    printf '  %s\n' "${CELLS[@]}"
    echo "Python:      $(which python3)"
    echo "Git HEAD:    $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
} > "${MANIFEST_DIR}/ensemble_cost_${MODE}_${JOB_ID}.txt"
cp "${NYCOPT_ENV_FILE}" "${MANIFEST_DIR}/ensemble_cost_${MODE}_${JOB_ID}.env"

echo "=== ensemble-cost sweep: mode=${MODE} cells=${#CELLS[@]} budget=${BUDGET_S}s ==="
for CELL in "${CELLS[@]}"; do
    read -r N L MODEL M K EST <<< "${CELL}"
    if (( K > NTASKS_AVAIL )); then
        echo "=== CAP N=${N} L=${L} ${MODEL}: K=${K} > ${NTASKS_AVAIL} allocated ranks; running at ${NTASKS_AVAIL} ==="
        K="${NTASKS_AVAIL}"
    fi
    if (( SECONDS + EST > BUDGET_S )); then
        echo "=== SKIP N=${N} L=${L} ${MODEL}: est ${EST}s, only $(( BUDGET_S - SECONDS ))s of budget left ==="
        continue
    fi
    _shm_clean
    echo "=== cell N=${N} L=${L} ${MODEL} K=${K} M=${M} (est ${EST}s) start $(date -u +%H:%M:%SZ) ==="
    T0=$(date +%s); RC=0
    # NYCOPT_USE_TRIMMED_MODEL must be EXPORTED here, after the env file is
    # sourced: config reads it at import, and if it never arrives the "full"
    # cells silently re-measure the trimmed model with no error. The analysis
    # cross-checks this by asserting the two variants' objectives differ.
    NYCOPT_BENCH_EXPERIMENT=ensemble_cost \
    NYCOPT_BENCH_WARM_EVALS="${M}" \
    NYCOPT_BENCH_MODE="${MODE}" \
    NYCOPT_SCALING_KN_REALS="${N}" \
    NYCOPT_SCALING_KN_YEARS="${L}" \
    NYCOPT_USE_TRIMMED_MODEL="$([[ "${MODEL}" == "trimmed" ]] && echo 1 || echo 0)" \
        mpirun -np "${K}" python3 -u \
        scripts/supplemental/bench_eval_worker.py \
        < /dev/null || RC=$?
    T1=$(date +%s)
    printf '{"n_realizations": %d, "realization_years": %d, "model_variant": "%s", "k": %d, "m_warm": %d, "rc": %d, "t0": %d, "t1": %d, "est_s": %d, "mode": "%s", "job_id": "%s"}\n' \
        "${N}" "${L}" "${MODEL}" "${K}" "${M}" "${RC}" "${T0}" "${T1}" "${EST}" "${MODE}" "${JOB_ID}" \
        > "${CELLS_DIR}/step_n$(printf '%03d' "${N}")_L$(printf '%02d' "${L}")_${MODEL}_k$(printf '%03d' "${K}")_${JOB_ID}.json"
    echo "=== cell N=${N} L=${L} ${MODEL} done: rc=${RC} wall=$(( T1 - T0 ))s ==="
done
echo "=== ensemble-cost sweep completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
