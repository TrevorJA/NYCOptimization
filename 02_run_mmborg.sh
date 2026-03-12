#!/bin/bash
# Step 2: Launch multi-master Borg MOEA optimization via MPI.
#
# Edit the settings below, then submit:
#   sbatch 02_run_mmborg.sh
#
# For multiple seeds, copy this file or loop:
#   for s in 1 2 3 4 5; do sed "s/^SEED=.*/SEED=$s/" 02_run_mmborg.sh | sbatch; done

# ---- Run settings (edit these) ----
SEED=1
FORMULATION="ffmp"
N_ISLANDS=2
NFE=1000000
CHECKPOINT=true           # writes checkpoint files for restart

# ---- SLURM settings ----
# Rank formula: ntasks = 1 + N_ISLANDS * (workers_per_island + 1)
#
# Hopper (Cornell):  40 cores/node, no --account needed
#   #SBATCH --nodes=2
#   #SBATCH --ntasks=80
#   module load python/3.11.5
#
# Anvil (ACCESS):  128 cores/node, requires --account
#   #SBATCH --nodes=2
#   #SBATCH --ntasks=129
#   #SBATCH --account=YOUR_ALLOCATION
#   module load anaconda openmpi
#
#SBATCH --job-name=mmborg
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=35
#SBATCH --time=24:00:00
#SBATCH --output=logs/mmborg_%j.out
#SBATCH --error=logs/mmborg_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs
module load python/3.11.5
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Build args and run ----
ARGS="--seed ${SEED} --formulation ${FORMULATION} --islands ${N_ISLANDS} --nfe ${NFE}"
${CHECKPOINT} && ARGS="${ARGS} --checkpoint"

mpirun -np ${SLURM_NTASKS} python3 src/mmborg_cli.py ${ARGS}
