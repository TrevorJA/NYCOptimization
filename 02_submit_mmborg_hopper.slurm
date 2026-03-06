#!/bin/bash
# ===========================================================================
# 02_submit_mmborg_hopper.slurm - SLURM batch submission for Hopper (Cornell).
#
# BEFORE FIRST USE:
#   1. Update the venv path below to your Hopper environment
#   2. Run: mkdir -p logs
#
# Submits one seed of MM Borg optimization. For multiple seeds:
#
#     for SEED in $(seq 1 10); do
#         sbatch --export=ALL,SEED=${SEED} 02_submit_mmborg_hopper.slurm
#     done
#
# Adjust nodes, ntasks-per-node, and N_ISLANDS based on your allocation.
# Rule of thumb: ntasks = N_ISLANDS * workers_per_island + N_ISLANDS + 1
# ===========================================================================
#SBATCH --job-name=mmborg_ffmp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/mmborg_%j.out
#SBATCH --error=logs/mmborg_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ms3654@cornell.edu

set -euo pipefail

# ---- Configuration ----
SEED="${SEED:-1}"
FORMULATION="${FORMULATION:-ffmp}"
N_ISLANDS="${N_ISLANDS:-2}"

# Create log directory (also needed before first sbatch, see header)
mkdir -p logs

n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_NNODES}"
echo "Tasks/node:  ${SLURM_NTASKS_PER_NODE}"
echo "Total tasks: ${n_processors}"
echo "Seed:        ${SEED}"
echo "Formulation: ${FORMULATION}"
echo "Islands:     ${N_ISLANDS}"

# ---- Load modules (Hopper-specific) ----
module load python/3.11.5

# Activate Python virtual environment
# TODO: Update this path to your Hopper environment
source /path/to/your/venv/bin/activate

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

mpirun --bind-to core --map-by ppr:${SLURM_NTASKS_PER_NODE}:node \
    -np ${n_processors} \
    python3 src/mmborg_cli.py \
        --seed ${SEED} \
        --formulation ${FORMULATION} \
        --islands ${N_ISLANDS} \
        --checkpoint

echo "Job complete. Seed=${SEED}"
