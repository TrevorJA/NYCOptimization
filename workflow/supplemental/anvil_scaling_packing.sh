#!/bin/bash
# anvil_scaling_packing.sh — Stage A of the Anvil scaling experiment: the
# single-node packing/concurrency sweep (manuscript supplement).
#
# On ONE exclusive 128-core Anvil node, run the K-ladder from
# supplemental_config.PACKING_MODES sequentially: for each step,
# `mpirun -np K bench_packing_worker.py` has every rank time 1 cold + M warm
# trimmed-model ensemble evaluations (the exact Borg-worker `evaluate()` path)
# and record wall time + peak RSS to per-rank CSV shards under
# outputs/supplemental/anvil_scaling_experiment/packing/. A fresh mpirun per
# step gives a natural cold cache at every K.
#
# Usage (from repo root; the allocation account is set in the header below):
#   # ~10-min smoke on the debug partition:
#   sbatch --partition=debug --time=00:30:00 \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/anvil_scaling_packing.env,NYCOPT_PACK_MODE=smoke \
#          workflow/supplemental/anvil_scaling_packing.sh
#   # full ladder (defaults below):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/anvil_scaling_packing.env \
#          workflow/supplemental/anvil_scaling_packing.sh
#   # spot re-measurement (edit PACKING_MODES["spot"] first, then):
#   sbatch --partition=debug --time=02:00:00 \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/anvil_scaling_packing.env,NYCOPT_PACK_MODE=spot,NYCOPT_PACK_BUDGET_S=6000 \
#          workflow/supplemental/anvil_scaling_packing.sh
#
# Notes:
#   * NYCOPT_PACK_MODE (smoke | ladder | spot) selects the step list — the
#     ladders themselves live in supplemental_config.py (no value flags).
#   * The node must be exclusive (wholenode, or debug which is node-exclusive)
#     so contention comes only from our own ranks; --ntasks-per-node=128
#     reserves every core for the densest step.
#   * A step that dies (e.g. OOM at K=128) is recorded and the ladder
#     continues — missing shards at high K are the memory ceiling, not a bug.
#   * NYCOPT_PACK_BUDGET_S (default 9600 s) guards the wall clock: a new step
#     starts only if at least the worst-case step time (~40 min) remains, so a
#     shorter allocation degrades by skipping tail steps instead of dying.
#   * pywrdrb's per-rank /dev/shm model JSONs are never cleaned by the code
#     (src/simulation.py::_get_temp_dir); stale dirs are purged between steps
#     so RAM-backed tmpfs doesn't silently shrink usable node memory.
#
#SBATCH --job-name=anvscale_pack
#SBATCH --account=ees260021
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=03:00:00
#SBATCH --output=logs/anvil_scaling_packing_%j.out
#SBATCH --error=logs/anvil_scaling_packing_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

MODE="${NYCOPT_PACK_MODE:-ladder}"
BUDGET_S="${NYCOPT_PACK_BUDGET_S:-9600}"
WORST_STEP_S=2400
JOB_ID="${SLURM_JOB_ID:-local}"

_shm_clean() {
    find /dev/shm -maxdepth 1 -name 'pywrdrb_opt_r*' -user "${USER:-$(id -un)}" \
        -exec rm -rf {} + 2>/dev/null || true
}
trap _shm_clean EXIT

# Pre-flight: the staged ensemble must resolve before burning node-hours.
python3 -c "
import supplemental_config as scfg
scfg.configure_anvil_scaling_env()
import config
s = config.SEARCH_ENSEMBLE_SPEC
assert s is not None and s.is_ensemble, (
    'SEARCH_ENSEMBLE_SPEC is not a staged ensemble: stage the hazard-filling '
    'design first (workflow steps 02-04 with '
    'NYCOPT_SCENARIO_DESIGN=hazard_filling).')
print(f'pre-flight OK: {s.preset_name} N={s.n_realizations} '
      f'years={s.realization_years}')
"

# Step list (K M BATCH per line) from the single source of truth. Captured
# into a variable first so a failing python3 aborts under set -e (a process
# substitution feeding mapfile would fail silently).
STEPS_RAW="$(python3 -c "
import supplemental_config as scfg
for k, m, b in scfg.PACKING_MODES['${MODE}']:
    print(k, m, b)
")"
mapfile -t STEPS <<< "${STEPS_RAW}"
(( ${#STEPS[@]} > 0 )) || { echo "ERROR: no steps for mode '${MODE}'" >&2; exit 2; }

# Output dirs + job manifest.
PACKING_DIR="$(python3 -c "import supplemental_config as s; print(s.SCALING_PACKING_DIR)")"
MANIFEST_DIR="$(python3 -c "import supplemental_config as s; print(s.SCALING_MANIFESTS_DIR)")"
mkdir -p "${PACKING_DIR}" "${MANIFEST_DIR}"
{
    echo "=== Anvil packing sweep manifest: ${MODE} job ${JOB_ID} ==="
    echo "Date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Host:       $(hostname)"
    echo "Partition:  ${SLURM_JOB_PARTITION:-n/a}"
    echo "Env file:   ${NYCOPT_ENV_FILE}"
    echo "Mode:       ${MODE}"
    echo "Budget (s): ${BUDGET_S}"
    echo "Steps (K M BATCH):"
    printf '  %s\n' "${STEPS[@]}"
    echo "Python:     $(which python3)"
    echo "Git HEAD:   $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
} > "${MANIFEST_DIR}/packing_${MODE}_${JOB_ID}.txt"
cp "${NYCOPT_ENV_FILE}" "${MANIFEST_DIR}/packing_${MODE}_${JOB_ID}.env"

echo "=== packing sweep: mode=${MODE} steps=${#STEPS[@]} budget=${BUDGET_S}s ==="
for STEP in "${STEPS[@]}"; do
    read -r K M BATCH <<< "${STEP}"
    if (( SECONDS > BUDGET_S - WORST_STEP_S )); then
        echo "=== SKIP K=${K} b=${BATCH}: ${SECONDS}s elapsed, <${WORST_STEP_S}s budget left ==="
        continue
    fi
    _shm_clean
    echo "=== step K=${K} M=${M} batch=${BATCH} start $(date -u +%H:%M:%SZ) ==="
    T0=$(date +%s); RC=0
    NYCOPT_PACK_WARM_EVALS="${M}" \
    NYCOPT_PACK_MODE="${MODE}" \
    NYCOPT_SEARCH_REALIZATION_BATCH="${BATCH}" \
        mpirun -np "${K}" python3 -u \
        scripts/supplemental/anvil_scaling/bench_packing_worker.py \
        < /dev/null || RC=$?
    T1=$(date +%s)
    printf '{"k": %d, "m_warm": %d, "batch": %d, "rc": %d, "t0": %d, "t1": %d, "mode": "%s", "job_id": "%s"}\n' \
        "${K}" "${M}" "${BATCH}" "${RC}" "${T0}" "${T1}" "${MODE}" "${JOB_ID}" \
        > "${PACKING_DIR}/step_k$(printf '%03d' "${K}")_b${BATCH}_${JOB_ID}.json"
    echo "=== step K=${K} done: rc=${RC} wall=$(( T1 - T0 ))s ==="
done
echo "=== packing sweep completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
