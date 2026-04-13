#!/bin/bash
# mmborg_tree.sh — Production MM Borg run for Oblique Decision Tree policy.
# DV count depends on NYCOPT_STATE_SPEC (extended→78, minimal→57, full→120).
# Submit as a seed-array job:  sbatch --array=1-10 slurm/mmborg_tree.sh
#
#SBATCH --job-name=mmborg_tree
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --output=logs/mmborg_tree_seed%a_%A.out
#SBATCH --error=logs/mmborg_tree_seed%a_%A.err
#SBATCH --array=1-10

set -euo pipefail

FORMULATION="tree"
SEED="${SLURM_ARRAY_TASK_ID:-1}"
N_ISLANDS=2
NFE=1000000
RUNTIME_FREQ=500
DEBUG_SIM=false
CHECKPOINT=false

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION} --islands ${N_ISLANDS} --nfe ${NFE} --runtime-freq ${RUNTIME_FREQ}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg (${FORMULATION}, seed=${SEED}, STATE_SPEC=${NYCOPT_STATE_SPEC:-extended}) ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
