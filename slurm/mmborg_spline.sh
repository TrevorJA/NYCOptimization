#!/bin/bash
# mmborg_spline.sh — Production MM Borg run for additive B-spline policy
# (single-layer KAN / GAM).
# DV count depends on NYCOPT_STATE_FEATURES
# (default 4 features -> 6 state inputs incl. temporal -> 6*8 + 1 = 49 DVs;
#  extended 7 features -> 9 state inputs -> 9*8 + 1 = 73 DVs).
# Submit as a seed-array job:  sbatch --array=1-10 slurm/mmborg_spline.sh
#
#SBATCH --job-name=mmborg_spline
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/mmborg_spline_seed%a_%A.out
#SBATCH --error=logs/mmborg_spline_seed%a_%A.err
#SBATCH --array=1-10

set -euo pipefail

FORMULATION="spline"
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
