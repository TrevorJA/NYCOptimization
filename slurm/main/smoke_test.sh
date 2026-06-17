#!/bin/bash
# smoke_test.sh — small-NFE end-to-end pipeline check for one formulation.
#
# Designed to be launched once per FORMULATION via slurm/main/submit_smoke.sh.
# Default settings: 500 NFE/island, 2 islands, DEBUG_SIM (2018-2022),
# 2 nodes × 40 tasks (79 MPI ranks used: 1 controller + 2×39 workers).
# Expected wall time: ~1-2 h on Hopper; 5-y window keeps per-eval cost low.
#
# Algorithm settings (islands, NFE, runtime-freq) come from the smoke MOEA
# config (src/moea_config.py), forced here via NYCOPT_MOEA_CONFIG=smoke. The
# scenario design comes from the env file / NYCOPT_SCENARIO_DESIGN (default
# historic). The MPI rank count is sized to this script's allocation, not the
# config (smoke's worker count targets a local machine), via the NTASKS_MPI
# caller override that _common.sh honors.
#
# Required env (set by the launcher via --export=ALL):
#   FORMULATION   ffmp | ffmp_8 | ffmp_10 | ffmp_12
#                 (ffmp_6 is structurally identical to ffmp; other ffmp_N
#                 values require their MOEAFramework JAR built first)
#   N_ZONES       only for ffmp_N formulations (otherwise ignored)
#
# Optional env:
#   SEED          defaults to 1
#   NYCOPT_MOEA_CONFIG  defaults to smoke (tiny NFE)
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
DEBUG_SIM="${DEBUG_SIM:-true}"
CHECKPOINT=false
# Force the smoke algorithm config (tiny NFE) unless the caller overrides it.
export NYCOPT_MOEA_CONFIG="${NYCOPT_MOEA_CONFIG:-smoke}"
# Smoke allocates 2 × 40 = 80 slots; MM Borg needs 1 + islands*(workers+1).
# 2 islands × 39 workers + 1 = 79. Caller override of the config's MPI sizing.
NTASKS_MPI=79
export DEBUG_SIM NTASKS_MPI

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/slurm/main/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION}"

echo "=== SMOKE TEST: ${SCENARIO}/${RUN_SLUG} seed=${SEED} (${NYCOPT_MOEA_CONFIG}) ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
