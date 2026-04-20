#!/bin/bash
# mmborg_ann.sh — Production MM Borg run for ANN external policy.
# DV count depends on NYCOPT_STATE_FEATURES (default 4 features → 137 DVs).
# Submit as a seed-array job:  sbatch --array=1-10 slurm/mmborg_ann.sh
#
#SBATCH --job-name=mmborg_ann
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/mmborg_ann_seed%a_%A.out
#SBATCH --error=logs/mmborg_ann_seed%a_%A.err
#SBATCH --array=1-10

set -euo pipefail

FORMULATION="ann"
SEED="${SLURM_ARRAY_TASK_ID:-1}"
N_ISLANDS=2
NFE=1000000
RUNTIME_FREQ=500
DEBUG_SIM=false
CHECKPOINT=false

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/slurm/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION} --slug ${RUN_SLUG} --islands ${N_ISLANDS} --nfe ${NFE} --runtime-freq ${RUNTIME_FREQ}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg (${FORMULATION}, slug=${RUN_SLUG}, seed=${SEED}) ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
