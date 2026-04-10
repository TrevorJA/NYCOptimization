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
#SBATCH --output=logs/presim.out
#SBATCH --error=logs/presim.err

set -euo pipefail
mkdir -p logs
module load python/3.11.5
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 scripts/generate_presim.py "$@"
