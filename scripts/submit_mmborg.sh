#!/bin/bash
###############################################################################
# submit_mmborg.sh - Submit Multi-Master Borg optimization jobs to SLURM
#
# Submits one job per random seed. Each job runs MMBorg with MPI across
# multiple nodes on Anvil CPU.
#
# Usage:
#   bash submit_mmborg.sh [formulation_name] [n_seeds]
#   bash submit_mmborg.sh ffmp 10
#
# MMBorg MPI sizing:
#   For N_ISLANDS islands, each with K workers:
#     Total MPI ranks = N_ISLANDS * (K + 1) + 1
#   The "+1" per island is the island master; the global "+1" is the controller.
#
# Anvil CPU nodes: 128 cores/node
###############################################################################

FORMULATION=${1:-ffmp}
N_SEEDS=${2:-10}

# --- HPC Configuration ---
N_NODES=4
NTASKS_PER_NODE=128
TOTAL_TASKS=$((N_NODES * NTASKS_PER_NODE))  # 512
WALL_TIME="24:00:00"
PARTITION="wholenode"
ACCOUNT="your_allocation"  # TODO: Set your ACCESS allocation

# --- Optimization Settings ---
MAX_TIME_SEC=82800  # 23 hours (leave buffer for setup/teardown)

# --- Paths ---
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/outputs/optimization/${FORMULATION}/logs"

mkdir -p "${LOG_DIR}"

echo "============================================="
echo "NYCOptimization: MMBorg Submission"
echo "============================================="
echo "Formulation: ${FORMULATION}"
echo "Seeds: 1 through ${N_SEEDS}"
echo "Nodes per job: ${N_NODES}"
echo "Tasks per job: ${TOTAL_TASKS}"
echo "Wall time: ${WALL_TIME}"
echo "Max opt time: ${MAX_TIME_SEC}s"
echo "============================================="

for SEED in $(seq 1 ${N_SEEDS}); do
    JOB_NAME="nyc_${FORMULATION}_s${SEED}"
    LOG_FILE="${LOG_DIR}/seed_${SEED}.log"
    ERR_FILE="${LOG_DIR}/seed_${SEED}.err"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${N_NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --exclusive
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${ERR_FILE}

# Load modules (Anvil-specific)
module purge
module load gcc/11.2.0
module load openmpi/4.1.6
module load python/3.11.5
module load anaconda

# Activate virtual environment
source ${PROJECT_DIR}/venv/bin/activate

# Set environment
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/borg:\${PYTHONPATH}"

echo "Job started: \$(date)"
echo "Formulation: ${FORMULATION}, Seed: ${SEED}"
echo "Nodes: ${N_NODES}, Tasks: ${TOTAL_TASKS}"

# Run MMBorg
mpirun -np ${TOTAL_TASKS} python ${SCRIPTS_DIR}/run_mmborg.py \\
    --formulation ${FORMULATION} \\
    --seed ${SEED} \\
    --max_time ${MAX_TIME_SEC}

echo "Job finished: \$(date)"
EOF

    echo "Submitted seed ${SEED}: ${JOB_NAME}"
done

echo ""
echo "All ${N_SEEDS} jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Logs in: ${LOG_DIR}/"
