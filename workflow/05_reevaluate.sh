#!/bin/bash
# Step 5: Re-simulate Pareto-optimal solutions with the full (untrimmed)
# Pywr-DRB model. Thin wrapper around src/reevaluate.py which supports
# parallel workers via multiprocessing.Pool.
#
# Usage:
#   bash workflow/05_reevaluate.sh [FORMULATION] [MAX_SOLUTIONS] [NJOBS] [SEED]
#     FORMULATION   architecture name (default: ffmp)
#     MAX_SOLUTIONS cap on solutions (0 = all)
#     NJOBS         parallel workers (default: SLURM_CPUS_ON_NODE or 1)
#     SEED          optional seed number for per-seed output subdir
#
# Examples:
#   bash workflow/05_reevaluate.sh ffmp 0 16
#   sbatch workflow/05_reevaluate.sh rbf 0 32 1
#
#SBATCH --job-name=reevaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/reevaluate_%j.out
#SBATCH --error=logs/reevaluate_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
module load python/3.11.5
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

FORMULATION="${1:-ffmp}"
MAX_SOLUTIONS="${2:-0}"
NJOBS="${3:-${SLURM_CPUS_ON_NODE:-1}}"
SEED="${4:-}"

ARGS="--formulation ${FORMULATION} --max ${MAX_SOLUTIONS} --njobs ${NJOBS}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

echo "=== Re-evaluation: ${FORMULATION} (njobs=${NJOBS}, seed=${SEED:-all}) ==="
python3 -m src.reevaluate ${ARGS}
echo "=== Completed: $(date) ==="
