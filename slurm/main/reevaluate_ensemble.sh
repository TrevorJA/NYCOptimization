#!/bin/bash
# reevaluate_ensemble.sh — MPI re-evaluation of an arm's Pareto policies on a
# COMMON (held-out) streamflow ensemble, so scores are comparable across arms.
#
# The arm (scenario design + slug + objectives) comes from the sourced env file
# (NYCOPT_ENV_FILE). The COMMON ensemble is whatever NYCOPT_REEVAL_ENSEMBLE_PRESET
# resolves to (any preset / kn_{Y}yr_n{N} slug / staged dir with _meta.json) —
# swap it by changing that one var. Per-arm output:
#   outputs/{scenario}/{slug}/reeval/{reeval_preset}/objectives_summary.csv
#
# Solutions are distributed across MPI ranks (src.reevaluate_mpi). Scale by
# changing --nodes at submission; ranks = nodes * ntasks-per-node.
#
# Submit (one arm, common ensemble = kn_5yr_n10, all solutions):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7_hazfill_pilot.env,\
# NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n10 slurm/main/reevaluate_ensemble.sh
#
# Optional positional args: FORMULATION (default ffmp), MAX_SOLUTIONS (default 0=all).
#
#SBATCH --job-name=reeval_ens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/reeval_ens_%x_%j.out
#SBATCH --error=logs/reeval_ens_%x_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.5 || true
source venv/bin/activate

# Arm config from the env file (scenario design, formulation, objectives).
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:?set NYCOPT_ENV_FILE to an arm env, e.g. slurm/envs/ffmp_obj7_hazfill_pilot.env}"
if [[ -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a; source "${NYCOPT_ENV_FILE}"; set +a
    echo "[reeval_ens] sourced env file: ${NYCOPT_ENV_FILE}"
fi

# The common re-eval ensemble. Required (the whole point is a shared ensemble).
: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set NYCOPT_REEVAL_ENSEMBLE_PRESET to the common ensemble, e.g. kn_5yr_n10}"

# One BLAS thread per rank; each rank runs a pywrdrb ensemble simulation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PWD}/lib/borg:${PYTHONPATH:-}"

FORMULATION="${1:-ffmp}"
MAX_SOLUTIONS="${2:-0}"
NTASKS="${SLURM_NTASKS:-1}"

echo "=== Re-eval (MPI): scenario=${NYCOPT_SCENARIO_DESIGN:-?} common=${NYCOPT_REEVAL_ENSEMBLE_PRESET} ranks=${NTASKS} formulation=${FORMULATION} max=${MAX_SOLUTIONS} ==="
echo "    started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" --mca pml ob1 --mca btl self,vader,tcp \
    python3 -m src.reevaluate_mpi --formulation "${FORMULATION}" --max "${MAX_SOLUTIONS}"
echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
