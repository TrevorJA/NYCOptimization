#!/bin/bash
# ===========================================================================
# 02_run_mmborg.sh - Launch Multi-Master Borg MOEA optimization.
#
# This script is designed for HPC (SLURM) submission. It runs one seed
# of MM Borg optimization using MPI across multiple nodes. For multiple
# seeds, submit this script once per seed (varying SEED).
#
# Usage (local test with 4 ranks):
#     mpirun -np 4 bash 02_run_mmborg.sh --seed 1 --islands 1
#
# Usage (SLURM):
#     sbatch 02_submit_mmborg.slurm   (see template below)
#
# Arguments (passed as environment variables or flags):
#     --seed       : Random seed number, 1-indexed (default: 1)
#     --formulation: Formulation name (default: "ffmp")
#     --islands    : Number of MM Borg islands (default: 2)
#     --nfe        : Max function evaluations (default: from config)
#     --time       : Max wall time in seconds (optional, overrides NFE)
#     --checkpoint : Enable checkpointing (flag)
#     --restore    : Restore from checkpoint file (path)
#
# Outputs:
#     outputs/optimization/{formulation}/runtime/seed_XX_{formulation}_%d.runtime
#     outputs/optimization/{formulation}/sets/seed_XX_{formulation}.set
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
SEED="${SEED:-1}"
FORMULATION="${FORMULATION:-ffmp}"
N_ISLANDS="${N_ISLANDS:-2}"
MAX_NFE=""
MAX_TIME=""
CHECKPOINT=""
RESTORE=""

# Parse command-line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)        SEED="$2"; shift 2;;
        --formulation) FORMULATION="$2"; shift 2;;
        --islands)     N_ISLANDS="$2"; shift 2;;
        --nfe)         MAX_NFE="$2"; shift 2;;
        --time)        MAX_TIME="$2"; shift 2;;
        --checkpoint)  CHECKPOINT="1"; shift;;
        --restore)     RESTORE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

echo "============================================"
echo "  02: MM Borg Optimization"
echo "  Formulation: ${FORMULATION}"
echo "  Seed:        ${SEED}"
echo "  Islands:     ${N_ISLANDS}"
echo "============================================"

# Build Python args
PY_ARGS="--seed ${SEED} --formulation ${FORMULATION} --islands ${N_ISLANDS}"
[ -n "$MAX_NFE" ] && PY_ARGS="${PY_ARGS} --nfe ${MAX_NFE}"
[ -n "$MAX_TIME" ] && PY_ARGS="${PY_ARGS} --time ${MAX_TIME}"
[ -n "$CHECKPOINT" ] && PY_ARGS="${PY_ARGS} --checkpoint"
[ -n "$RESTORE" ] && PY_ARGS="${PY_ARGS} --restore ${RESTORE}"

python3 src/mmborg_cli.py ${PY_ARGS}
