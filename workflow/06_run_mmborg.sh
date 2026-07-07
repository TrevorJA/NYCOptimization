#!/bin/bash
# Step 6: MM-Borg MOEA search. One launcher for every formulation (ffmp and
# variable-resolution ffmp_N) and every scenario design (single-trace or
# ensemble). The run is fully specified by the env file: scenario design,
# MOEA config, objectives, physics toggles. The pre-flight echoes the
# resolved identity and fails fast on unstaged designs — it derives its
# expectations from config, never hardcodes an experiment.
#
# Each submission is one independent job; there is no campaign wrapper.
# Submit one line per (env file x formulation), from the repo root:
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env \
#          --array=1-10 workflow/06_run_mmborg.sh
#
# Variable-resolution FFMP (same launcher, formulation from the identifier):
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7_sal.env,FORMULATION=ffmp_12 \
#          --array=1-10 workflow/06_run_mmborg.sh
#
# --array index = Borg seed; array tasks are independent seed replicates.
# NYCOPT_ENV_FILE is REQUIRED (no default) — the job aborts immediately with
# a listing of workflow/envs/*.env otherwise.
#
# Geometry: 5 nodes x 33 tasks = 165 ranks = 1 controller + 4 islands x
# (40 workers + 1 master) — matches MOEAConfig.total_ntasks_mpi for
# mm_pilot/mm_full. 33/node (not 40) avoids the measured memory-bandwidth
# packing penalty. _common.sh sizes `mpirun -np` from the config, so a MOEA
# config with different island/worker counts only needs matching
# --nodes/--ntasks-per-node at submission.
#
# Anvil: multi-node jobs must use the node-exclusive `wholenode` partition
# (the default `shared` partition is capped at 1 node), and 96 h is Anvil's
# hard per-job wall-time maximum — a run that needs more must restart from
# the periodic runtime snapshots. An allocation account is mandatory; set it
# once via `export SBATCH_ACCOUNT=<allocation>` (see README). Pilots may pass
# a shorter `sbatch --time=...`.
#
#SBATCH --job-name=mmborg
#SBATCH --partition=wholenode
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=33
#SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --output=logs/mmborg_%x_seed%a_%A.out
#SBATCH --error=logs/mmborg_%x_seed%a_%A.err
#SBATCH --array=1

set -euo pipefail

# Identifiers only — algorithm settings come from the env file + registries.
export FORMULATION="${FORMULATION:-ffmp}"   # ffmp | ffmp_N (registry-validated in pre-flight)
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-1}}"
DEBUG_SIM="${DEBUG_SIM:-false}"
CHECKPOINT="${CHECKPOINT:-false}"           # disabled by default: islands share a checkpoint file (race-prone)
export DEBUG_SIM

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads
nycopt_read_run_identity
nycopt_write_manifest
nycopt_preflight_mmborg

ARGS="--seed ${SEED} --formulation ${FORMULATION}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg: ${SCENARIO}/${RUN_SLUG} seed=${SEED} (${MOEA_CONFIG_NAME}, ${NTASKS_MPI} ranks) ==="
mpirun -np "${NTASKS_MPI}" python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
