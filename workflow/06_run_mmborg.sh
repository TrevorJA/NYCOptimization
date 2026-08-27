#!/bin/bash
# Step 6: MM-Borg MOEA search — one env-file-driven launcher for every
# formulation (ffmp, ffmp_N) and scenario design. The run identity (design,
# MOEA config, objectives, physics toggles) comes from NYCOPT_ENV_FILE
# (REQUIRED, no default); the pre-flight echoes it and fails fast on unstaged
# designs.
#
# Env inputs: NYCOPT_ENV_FILE; FORMULATION (default ffmp); DRAW=k (default 0)
# selects the staged ensemble draw — the campaign searches draw 0 only;
# --array index = Borg seed. Per-island NFE may depend on the seed
# (MOEAConfig.max_evaluations_by_seed: `production` runs seed 1 to 750k and
# seed 2 to 500k), so submit ONE seed per sbatch with its own --time (see the
# env-file headers). NYCOPT_OVERWRITE=1 relaunches a (design, slug, seed)
# whose .set file already exists; otherwise the pre-flight refuses.
#
# Submit (from repo root; production needs --nodes=12 --ntasks-per-node=128):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env,DRAW=0 \
#          --array=1 --nodes=12 --ntasks-per-node=128 --time=96:00:00 workflow/06_run_mmborg.sh
#
# Geometry: the MPI rank count comes from the MOEA config
# (MOEAConfig.total_ntasks_mpi -> mpirun -np); the #SBATCH lines below are only
# the container (4 x 128 fits mm_moderate). nycopt_check_allocation aborts
# and prints the geometry to use when the allocation is too small;
# nycopt_check_memory aborts when the design's projected node RSS at
# NYCOPT_RANKS_PER_NODE exceeds the safety line (N=300 needs
# NYCOPT_SEARCH_REALIZATION_BATCH=150, set in the production env files).
# Multi-node jobs need the `wholenode` partition; 96 h is Anvil's wall cap and
# there is no resume, so every search is sized to finish in one job
# (campaign_design.md §3).
#
# Outputs: outputs/{scenario}/{slug}[_d{k}]/sets/seed_{SS}_{slug}.set plus
# runtime/ archives and a run manifest.
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
export SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-1}}"
DEBUG_SIM="${DEBUG_SIM:-false}"
CHECKPOINT="${CHECKPOINT:-false}"           # disabled by default: islands share a checkpoint file (race-prone)
export DEBUG_SIM

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
# Ensemble-draw identifier: DRAW= at submission wins, then an env-file or
# inherited NYCOPT_ENSEMBLE_DRAW, else draw 0. Exported BEFORE the first
# config import so the moea slug ("_d{k}" for k>0) and the staged-ensemble
# resolution both see the same draw.
export NYCOPT_ENSEMBLE_DRAW="${DRAW:-${NYCOPT_ENSEMBLE_DRAW:-0}}"
nycopt_pin_threads
nycopt_read_run_identity
nycopt_check_allocation
nycopt_check_memory
nycopt_write_manifest
nycopt_preflight_mmborg

ARGS="--seed ${SEED} --formulation ${FORMULATION}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg: ${SCENARIO}/${RUN_SLUG} draw=${NYCOPT_ENSEMBLE_DRAW} seed=${SEED} (${MOEA_CONFIG_NAME}, ${NTASKS_MPI} ranks) ==="
mpirun -np "${NTASKS_MPI}" python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
