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
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
#          --array=1-10 workflow/06_run_mmborg.sh
#
# Variable-resolution FFMP (same launcher, formulation from the identifier):
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj8.env,FORMULATION=ffmp_12 \
#          --array=1-10 workflow/06_run_mmborg.sh
#
# --array index = Borg seed; array tasks are independent seed replicates.
# NYCOPT_ENV_FILE is REQUIRED (no default) — the job aborts immediately with
# a listing of workflow/envs/*.env otherwise.
#
# Geometry: the MPI rank count ALWAYS comes from the MOEA config
# (MOEAConfig.total_ntasks_mpi -> mpirun -np); the #SBATCH lines below are
# only the container for it, sized for mm_moderate: 4 nodes x 128 tasks =
# 512 cores holding 511 ranks = 1 controller + 2 islands x (254 workers +
# 1 master). Dense 128/node packing (NYCOPT_RANKS_PER_NODE in _common.sh) is
# the measured Anvil choice — wholenode bills all 128 cores regardless, and
# the packing sweep (SI §S8.2) bounds the eval-time penalty at ~17-21%,
# already priced into the cost surface. A MOEA config with different
# island/worker counts only needs matching --nodes/--ntasks-per-node at
# submission (e.g. `production` at 1,021 ranks: --nodes=8) —
# nycopt_check_allocation aborts before the search starts if the allocation
# is too small and prints the geometry to use.
#
# Anvil: multi-node jobs must use the node-exclusive `wholenode` partition
# (the default `shared` partition is capped at 1 node), and 96 h is Anvil's
# hard per-job wall-time maximum — a run that needs more must restart from
# the periodic runtime snapshots. The allocation account is set in the
# #SBATCH header below. Pilots may pass a shorter `sbatch --time=...`.
#
#SBATCH --job-name=mmborg
#SBATCH --account=ees260021
#SBATCH --partition=wholenode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
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
nycopt_check_allocation
nycopt_write_manifest
nycopt_preflight_mmborg

ARGS="--seed ${SEED} --formulation ${FORMULATION}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg: ${SCENARIO}/${RUN_SLUG} seed=${SEED} (${MOEA_CONFIG_NAME}, ${NTASKS_MPI} ranks) ==="
mpirun -np "${NTASKS_MPI}" python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
