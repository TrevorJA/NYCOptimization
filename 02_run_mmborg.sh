#!/bin/bash
# Step 2: Launch multi-master Borg MOEA optimization via MPI.
#
# Edit the settings below, then submit:
#   sbatch 02_run_mmborg.sh
#
# For multiple seeds, copy this file or loop:
#   for s in 1 2 3 4 5; do sed "s/^SEED=.*/SEED=$s/" 02_run_mmborg.sh | sbatch; done

#SBATCH --job-name=mmborg_ffmp
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --output=logs/mmborg.out
#SBATCH --error=logs/mmborg.err
#SBATCH --time=48:00:00

set -euo pipefail

# ---- Run settings (edit these) ----
SEED=1
FORMULATION="ffmp"
N_ISLANDS=2
NFE=10000                 # per island; 10k × 2 islands = 20k total evals
RUNTIME_FREQ=500          # runtime snapshot every N NFE
CHECKPOINT=false           # disabled: both islands write same checkpoint file → race condition/segfault

# ---- Simulation period ----
# Full historic: 1945-2022 (~150s/eval with trimmed model, measured 2026-03-23)
# Debug:         2018-2022 (~13s/eval)
DEBUG_SIM=false

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs

module load python/3.11.5
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Export debug sim dates if enabled ----
if ${DEBUG_SIM}; then
    export PYWRDRB_SIM_START_DATE="2018-01-01"
    export PYWRDRB_SIM_END_DATE="2022-12-31"
    echo "DEBUG_SIM=true: simulation period 2018-01-01 to 2022-12-31 (~5 yrs, ~13s/eval)"
fi

# ---- Build args and run ----
ARGS="--seed ${SEED} --formulation ${FORMULATION} --islands ${N_ISLANDS} --nfe ${NFE} --runtime-freq ${RUNTIME_FREQ}"
${CHECKPOINT} && ARGS="${ARGS} --checkpoint"

NTASKS=${SLURM_NTASKS:-200}
echo "=== MM-Borg Optimization ==="
echo "Date: $(date)"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-5}, Tasks: ${NTASKS}"
echo "Settings: seed=${SEED}, formulation=${FORMULATION}, islands=${N_ISLANDS}, NFE=${NFE}"
echo "Runtime freq: ${RUNTIME_FREQ}, Checkpoint: ${CHECKPOINT}"
echo "DEBUG_SIM: ${DEBUG_SIM}"
echo "Args: ${ARGS}"
echo "Working dir: $(pwd)"
echo "==========================="

mpirun -np ${NTASKS} python3 -u src/mmborg_cli.py ${ARGS}

echo "=== Completed: $(date) ==="
