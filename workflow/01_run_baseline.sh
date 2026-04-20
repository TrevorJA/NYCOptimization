#!/bin/bash
# Step 1: Evaluate the default FFMP policy (no optimization) and save
# baseline objective values for comparison.
#
# Usage:
#   bash 01_run_baseline.sh
#   sbatch 01_run_baseline.sh
#
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/baseline.out
#SBATCH --error=logs/baseline.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs
module load python/3.11.5
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 scripts/run_baseline.py "$@"
