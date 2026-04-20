#!/bin/bash
# smoke_test.sh — small-NFE end-to-end pipeline check for one architecture.
#
# Designed to be launched once per FORMULATION via slurm/submit_smoke.sh.
# Default settings: 500 NFE/island, 2 islands, DEBUG_SIM (2018-2022),
# 2 nodes × 40 tasks (79 MPI ranks used: 1 controller + 2×39 workers).
# Expected wall time: ~1-2 h on Hopper; 5-y window keeps per-eval cost low.
#
# Required env (set by the launcher via --export=ALL):
#   FORMULATION   ffmp | rbf | tree | ann | ffmp_6 | ffmp_8 | ...
#   N_ZONES       only for ffmp_N formulations (otherwise ignored)
#
# Optional env:
#   RUN_SLUG      defaults to smoke_${FORMULATION}
#   SEED          defaults to 1
#   NFE           defaults to 500
#
#SBATCH --job-name=smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%x_%j.out
#SBATCH --error=logs/smoke_%x_%j.err

set -euo pipefail

if [[ -z "${FORMULATION:-}" ]]; then
    echo "ERROR: FORMULATION env var must be set (pass via --export=ALL,FORMULATION=...)" >&2
    exit 1
fi

SEED="${SEED:-1}"
N_ISLANDS="${N_ISLANDS:-2}"
NFE="${NFE:-500}"
RUNTIME_FREQ="${RUNTIME_FREQ:-50}"
DEBUG_SIM="${DEBUG_SIM:-true}"
CHECKPOINT=false
RUN_SLUG="${RUN_SLUG:-smoke_${FORMULATION}}"
# Override default 199-rank formula: smoke allocates 2 × 40 = 80 slots;
# MM Borg needs 1 + N_ISLANDS*(workers+1) ranks. 2 islands × 39 workers + 1 = 79.
NTASKS_MPI=79

export RUN_SLUG DEBUG_SIM NTASKS_MPI

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/slurm/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION} --slug ${RUN_SLUG} --islands ${N_ISLANDS} --nfe ${NFE} --runtime-freq ${RUNTIME_FREQ}"

echo "=== SMOKE TEST: formulation=${FORMULATION} slug=${RUN_SLUG} seed=${SEED} NFE=${NFE} ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
