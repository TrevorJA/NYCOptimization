#!/bin/bash
# anvil_scaling_borg.sh — Stage B of the Anvil scaling experiment: MM Borg
# strong scaling (manuscript supplement).
#
# One job = one (scale_* geometry, seed) MM Borg run at fixed TOTAL NFE
# (1,280 across all geometries; see the scale_* entries in
# src/moea_config.py), historic single-trace design, DEBUG_SIM short window
# (~13 s/eval). Wall time to complete the fixed NFE is the strong-scaling
# measurement; a one-row timing CSV lands under
# outputs/supplemental/anvil_scaling_experiment/borg/. Borg's own .runtime
# files (NFE + elapsed seconds per snapshot) land in the standard
# outputs/historic/{slug}_scale_*/runtime/ tree and give the NFE-vs-time
# trajectories.
#
# Submit via workflow/supplemental/anvil_scaling_borg_submit.sh, which
# exports NYCOPT_MOEA_CONFIG=scale_* and sizes --ntasks/--time per geometry
# (the env file deliberately omits NYCOPT_MOEA_CONFIG so the submit-time
# export survives `set -a` sourcing). --array index = Borg seed.
#
# This is a separate script from workflow/06_run_mmborg.sh because 06's
# #SBATCH header pins --exclusive + wholenode, which cannot be unset at
# submit time; every scale_* geometry fits one node (<=69 ranks), so the
# shared partition's per-core SU charging is the right container. The shared
# node is NOT exclusive — other users' jobs may share it, so absolute wall
# times carry that (small, seed-band-visible) noise; the packing sweep
# (Stage A) is the controlled-contention measurement.
#
#SBATCH --job-name=anvscale_borg
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=01:30:00
#SBATCH --output=logs/anvil_scaling_borg_%x_%A_seed%a.out
#SBATCH --error=logs/anvil_scaling_borg_%x_%A_seed%a.err
#SBATCH --array=1-2

set -euo pipefail

# Identifiers only — geometry comes from NYCOPT_MOEA_CONFIG (submit-time
# export), everything else from the env file + registries.
export FORMULATION="${FORMULATION:-ffmp}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-1}}"
DEBUG_SIM="${DEBUG_SIM:-true}"   # short window is this stage's design point
export DEBUG_SIM

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads
nycopt_read_run_identity
nycopt_check_allocation
nycopt_write_manifest
nycopt_preflight_mmborg

_shm_clean() {
    find /dev/shm -maxdepth 1 -name 'pywrdrb_opt_r*' -user "${USER:-$(id -un)}" \
        -exec rm -rf {} + 2>/dev/null || true
}
trap _shm_clean EXIT

BORG_DIR="$(python3 -c "import supplemental_config as s; print(s.SCALING_BORG_DIR)")"
WORKERS="$(python3 -c "import config; print(config.ACTIVE_MOEA_CONFIG.n_workers_per_island)")"
mkdir -p "${BORG_DIR}"

echo "=== Launching Borg scaling run: ${MOEA_CONFIG_NAME} seed=${SEED} (${NTASKS_MPI} ranks) ==="
T0=$(date +%s); RC=0
mpirun -np "${NTASKS_MPI}" python3 -u src/mmborg_cli.py \
    --seed "${SEED}" --formulation "${FORMULATION}" || RC=$?
T1=$(date +%s)

OUT_CSV="${BORG_DIR}/timing_${MOEA_CONFIG_NAME}_seed$(printf '%02d' "${SEED}")_${SLURM_JOB_ID:-local}.csv"
{
    echo "config,seed,islands,workers_per_island,total_slots,ranks,nfe_per_island,total_nfe,t_start_epoch,t_end_epoch,wall_seconds,rc,job_id,slug,debug_sim"
    echo "${MOEA_CONFIG_NAME},${SEED},${N_ISLANDS},${WORKERS},$(( N_ISLANDS * WORKERS )),${NTASKS_MPI},${NFE},$(( N_ISLANDS * NFE )),${T0},${T1},$(( T1 - T0 )),${RC},${SLURM_JOB_ID:-local},${RUN_SLUG},${DEBUG_SIM}"
} > "${OUT_CSV}"

echo "=== Completed: rc=${RC} wall=$(( T1 - T0 ))s -> ${OUT_CSV} ==="
exit "${RC}"
