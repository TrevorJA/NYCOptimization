#!/bin/bash
# mmborg_ffmp.sh — Production MM Borg run for standard FFMP (24 DVs).
# Submit as a seed-array job:  sbatch --array=1-10 slurm/mmborg_ffmp.sh
#
#SBATCH --job-name=mmborg_ffmp
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --output=logs/mmborg_ffmp_seed%a_%A.out
#SBATCH --error=logs/mmborg_ffmp_seed%a_%A.err
#SBATCH --array=1-10

set -euo pipefail

FORMULATION="ffmp"
SEED="${SLURM_ARRAY_TASK_ID:-1}"
N_ISLANDS=2
NFE=1000000
RUNTIME_FREQ=500
DEBUG_SIM=false
CHECKPOINT=false

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION} --islands ${N_ISLANDS} --nfe ${NFE} --runtime-freq ${RUNTIME_FREQ}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg (${FORMULATION}, seed=${SEED}) ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
