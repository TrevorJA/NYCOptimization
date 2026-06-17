#!/bin/bash
# bench_ensemble.sh — wall-clock benchmark for an ensemble simulation eval.
#
# Calls scripts/supplemental/bench_ensemble_eval.py inside a SLURM allocation
# so the measurement reflects compute-node behavior (login-node bench is
# forbidden: >3 min runs on the home node violate cluster etiquette).
#
# The active search ensemble comes from the scenario design
# (NYCOPT_SCENARIO_DESIGN), sourced from an env file under slurm/envs/. Use an
# ensemble-based design (e.g. smoke_ensemble) — historic is single-trace.
#
# Usage:
#   sbatch slurm/supplemental/bench_ensemble.sh
#   NYCOPT_ENV_FILE=slurm/envs/<preset>.env sbatch slurm/supplemental/bench_ensemble.sh
#   sbatch --export=ALL,N_EVALS=3 slurm/supplemental/bench_ensemble.sh
#
#SBATCH --job-name=bench_ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_ensemble_%j.out
#SBATCH --error=logs/bench_ensemble_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.5 || true
source venv/bin/activate

# Source the per-experiment env file (override via NYCOPT_ENV_FILE before sbatch).
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-slurm/envs/ffmp_obj7_sal.env}"
if [[ -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[bench_ensemble] sourced env file: ${NYCOPT_ENV_FILE}"
else
    echo "[bench_ensemble] env file not found: ${NYCOPT_ENV_FILE}"
    exit 2
fi

# Thread pinning so BLAS doesn't oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

N_EVALS="${N_EVALS:-2}"
FORMULATION="${FORMULATION:-ffmp}"

echo "=== bench_ensemble: scenario=${NYCOPT_SCENARIO_DESIGN:-<default>} formulation=${FORMULATION} n_evals=${N_EVALS} ==="
echo "    started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/bench_ensemble_eval.py \
    --formulation "${FORMULATION}" \
    --n-evals "${N_EVALS}"
echo "=== completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
