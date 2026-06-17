#!/bin/bash
# mmborg_ffmp.sh — Production MM Borg run for standard FFMP (24 DVs).
# Submit as a seed-array job:  sbatch --array=1-10 slurm/main/mmborg_ffmp.sh
#
#SBATCH --job-name=mmborg_ffmp
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/mmborg_ffmp_seed%a_%A.out
#SBATCH --error=logs/mmborg_ffmp_seed%a_%A.err
#SBATCH --array=1-10

set -euo pipefail

# Algorithm settings (islands/NFE/runtime-freq) come from the active MOEA config
# (NYCOPT_MOEA_CONFIG in the env file), not from this script. The scenario design
# comes from NYCOPT_SCENARIO_DESIGN. _common.sh reads both back from config.
FORMULATION="ffmp"
SEED="${SLURM_ARRAY_TASK_ID:-1}"
DEBUG_SIM=false
CHECKPOINT=false

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/slurm/main/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg (${SCENARIO}/${RUN_SLUG}, seed=${SEED}) ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
