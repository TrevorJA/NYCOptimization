#!/bin/bash
# Tiny MPI test: 1 node, 8 tasks, 2 NFE, debug sim dates
#SBATCH --job-name=mmborg_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/mmborg_test_%j.out
#SBATCH --error=logs/mmborg_test_%j.err
#SBATCH --time=00:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs

module load python/3.11.5
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Debug sim dates (5-year window, ~13s/eval)
export PYWRDRB_SIM_START_DATE="2018-01-01"
export PYWRDRB_SIM_END_DATE="2022-12-31"

NTASKS=${SLURM_NTASKS:-8}

echo "=== MM-Borg Tiny Test ==="
echo "Date: $(date)"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}, Tasks: ${NTASKS}"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "==========================="

# Very small: 1 island, 2 NFE, runtime every 1
mpirun -np ${NTASKS} python3 -u src/mmborg_cli.py \
    --seed 1 \
    --formulation ffmp \
    --islands 1 \
    --nfe 2 \
    --runtime-freq 1

echo "=== Completed: $(date) ==="
