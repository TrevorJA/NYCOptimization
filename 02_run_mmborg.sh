#!/bin/bash
# Step 2: Launch multi-master Borg MOEA optimization via MPI.
# Searches for Pareto-optimal NYC reservoir operating policies.
#
# Usage:
#   sbatch 02_run_mmborg.sh --seed 3 --islands 2 --nfe 50000 --checkpoint
#   mpirun -np 129 bash 02_run_mmborg.sh --seed 1 --nfe 10000
#
#SBATCH --job-name=mmborg
#SBATCH --nodes=3
#SBATCH --ntasks=35
#SBATCH --time=24:00:00
#SBATCH --output=logs/mmborg.out
#SBATCH --error=logs/mmborg.err


set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs
module load python/3.11.5
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpirun -np ${SLURM_NTASKS} python3 src/mmborg_cli.py "$@"
