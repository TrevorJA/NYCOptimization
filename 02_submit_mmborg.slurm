#!/bin/bash
# ===========================================================================
# 02_submit_mmborg.slurm - SLURM batch submission for MM Borg optimization.
#
# Target cluster: Anvil (NSF ACCESS)
#
# BEFORE FIRST USE:
#   1. Replace YOUR_ALLOCATION with your actual ACCESS allocation ID
#   2. Run: mkdir -p logs
#
# Submits one seed of MM Borg optimization. For multiple seeds:
#
#     for SEED in $(seq 1 10); do
#         sbatch --export=ALL,SEED=${SEED} 02_submit_mmborg.slurm
#     done
#
# Adjust nodes, ntasks, and N_ISLANDS based on your allocation.
# Rule of thumb: ntasks = N_ISLANDS * workers_per_island + N_ISLANDS + 1
# ===========================================================================
#SBATCH --job-name=mmborg_ffmp
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --output=logs/mmborg_%j.out
#SBATCH --error=logs/mmborg_%j.err

set -euo pipefail

# ---- Configuration ----
SEED="${SEED:-1}"
FORMULATION="${FORMULATION:-ffmp}"
N_ISLANDS="${N_ISLANDS:-2}"

# Create log directory (also needed before first sbatch, see header)
mkdir -p logs

echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_NNODES}"
echo "Tasks:       ${SLURM_NTASKS}"
echo "Seed:        ${SEED}"
echo "Formulation: ${FORMULATION}"
echo "Islands:     ${N_ISLANDS}"

# ---- Load modules (Anvil-specific) ----
module purge
module load anaconda
module load openmpi

# Activate conda environment
conda activate pywrdrb

# ---- Keep each MPI rank single-threaded ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONOPTIMIZE=1

# ---- Run ----
cd "${SLURM_SUBMIT_DIR}"

mpirun -np ${SLURM_NTASKS} \
    python3 src/mmborg_cli.py \
        --seed ${SEED} \
        --formulation ${FORMULATION} \
        --islands ${N_ISLANDS} \
        --checkpoint

echo "Job complete. Seed=${SEED}"
