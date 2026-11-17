#!/bin/bash
# Constant-demand sensitivity re-evaluation (supplemental): re-simulate every
# policy of the historic run's merged set (ffmp_obj8_merged.set, 240 policies,
# plus the incumbent FFMP baseline) on the historic trace with NYC
# demand held at the 800 MGD FFMP maximum, and score the run's objectives.
# Same simulation as the search except the NYC demand source
# (scripts/supplemental/constant_demand_reeval.py has the details and the
# output layout: outputs/historic/ffmp_obj8/reeval/historic_nyc_demand_<DEMAND>/).
#
# Env vars (all optional):
#   NYCOPT_ENV_FILE  default workflow/envs/ffmp_obj8_historic_production.env
#   DEMAND           constant_max (default) | historical (control run: must
#                    reproduce the search scores in the set files)
#   FORMULATION      default ffmp
#   MAX_SOLUTIONS    default 0 = all unique policies; >0 = smoke test, written
#                    to a separate *_smoke directory
#
# Submit from the repo root. The header is sized for 4 Xeon nodes x 40 ranks
# (one per core), which are usually free on Hopper:
#   NYCOPT_NODE_CLASS=xeon source workflow/cluster.sh
#   sbatch workflow/supplemental/results_diagnostics/constant_demand_reeval.sh
# On EPYC nodes (default class of workflow/cluster.sh) use --ntasks-per-node=96.
# Smoke test on one node:
#   sbatch --nodes=1 --ntasks-per-node=8 --export=ALL,MAX_SOLUTIONS=8 \
#       workflow/supplemental/results_diagnostics/constant_demand_reeval.sh
# Anvil: add --partition=wholenode --ntasks-per-node=128 --cpus-per-task=1.
#
# 241 policies over 160 ranks is two rounds of ~110 s per evaluation on a
# loaded Xeon core (HOPPER_NOTES.md): about 5 minutes including startup.
#
#SBATCH --job-name=const_demand_reeval
#SBATCH --account=ees260021
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=logs/const_demand_reeval_%j.out
#SBATCH --error=logs/const_demand_reeval_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
export NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-workflow/envs/ffmp_obj8_historic_production.env}"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads
# /home is NFS on Hopper: HDF5 locks make every open drop the page cache
# (HOPPER_NOTES.md). Inputs are read-only here, so locking off is safe.
export HDF5_USE_FILE_LOCKING=FALSE

DEMAND="${DEMAND:-constant_max}"
FORMULATION="${FORMULATION:-ffmp}"
MAX_SOLUTIONS="${MAX_SOLUTIONS:-0}"
: "${SLURM_NTASKS:?submit with sbatch}"

echo "=== Constant-demand re-evaluation: demand=${DEMAND} formulation=${FORMULATION}" \
     "ranks=${SLURM_NTASKS} nodes=${SLURM_JOB_NODELIST} max=${MAX_SOLUTIONS} ==="
mpirun -np "${SLURM_NTASKS}" ${NYCOPT_MPI_MCA_FLAGS} \
    python3 scripts/supplemental/constant_demand_reeval.py \
        --formulation "${FORMULATION}" --demand "${DEMAND}" --max "${MAX_SOLUTIONS}"
echo "=== Completed: $(date) ==="
