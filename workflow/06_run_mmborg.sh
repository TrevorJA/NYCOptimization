#!/bin/bash
# Step 5: Launch multi-master Borg MOEA optimization via MPI.
#
# Single-formulation entry point. For larger campaigns spanning all FFMP
# layer configs + ensembles, prefer slurm/main/submit_all.sh with the appropriate
# slurm/envs/*.env file.
#
# Single-formulation entry point. Pick the experiment by editing the env file
# below (which sets NYCOPT_SCENARIO_DESIGN, NYCOPT_MOEA_CONFIG, objectives,
# physics). Algorithm settings come from the MOEA config — there are no
# value flags here. Then submit:
#   sbatch workflow/05_run_mmborg.sh
#
# For multiple seeds, submit as an array (preferred):
#   sbatch --array=1-10 slurm/main/mmborg_ffmp.sh

#SBATCH --job-name=mmborg_ffmp
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --output=logs/mmborg.out
#SBATCH --error=logs/mmborg.err
#SBATCH --time=48:00:00

set -euo pipefail

# ---- Run identifiers (edit these) ----
SEED="${SLURM_ARRAY_TASK_ID:-1}"
FORMULATION="ffmp"
CHECKPOINT=false           # disabled: both islands write same checkpoint file → race condition/segfault
DEBUG_SIM=false            # "true" -> short 2018-2022 window (~13s/eval)

# Select the experiment via an env file (scenario design + MOEA config + objs).
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ffmp_obj7_sal.env}"

# _common.sh sources the env file, pins threads, derives SCENARIO / RUN_SLUG /
# NTASKS_MPI from the active config, and writes the run manifest.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/slurm/main/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== MM-Borg Optimization (${SCENARIO}/${RUN_SLUG}, seed=${SEED}) ==="
echo "Date: $(date) | MPI ntasks: ${NTASKS_MPI} | Args: ${ARGS}"
mpirun -np "${NTASKS_MPI}" python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
