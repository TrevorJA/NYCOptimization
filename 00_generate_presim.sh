#!/bin/bash
# Step 0: Run full Pywr-DRB model and save non-NYC (STARFIT) releases
# for use as boundary conditions in the trimmed optimization model.
#
# Usage:
#   bash 00_generate_presim.sh
#   sbatch 00_generate_presim.sh
#
#SBATCH --job-name=presim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/presim_%j.out
#SBATCH --error=logs/presim_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 scripts/generate_presim.py "$@"
