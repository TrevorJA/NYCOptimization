#!/bin/bash
# mmborg_ffmp_vr.sh — Variable-Resolution FFMP N-sweep.
#
# The `ffmp_N` formulations are resolved dynamically: N is the number of
# storage zone boundary curves (N=6 ≈ baseline 7-level FFMP). The sweep
# measures hypervolume potential as a function of N, isolating policy-class
# resolution from architecture family.
#
# Submit via slurm/main/submit_all.sh, or manually:
#     sbatch --array=1-10 --export=ALL,N_ZONES=10 slurm/main/mmborg_ffmp_vr.sh
#
# The N value is passed via the N_ZONES environment variable so that a
# single SLURM script can submit one array job per N value in the sweep.
#
#SBATCH --job-name=mmborg_ffmp_vr
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/mmborg_ffmp_vr_N%x_seed%a_%A.out
#SBATCH --error=logs/mmborg_ffmp_vr_N%x_seed%a_%A.err
#SBATCH --array=1-10

set -euo pipefail

if [[ -z "${N_ZONES:-}" ]]; then
    echo "ERROR: N_ZONES environment variable not set."
    echo "Submit via: sbatch --export=ALL,N_ZONES=<N> slurm/main/mmborg_ffmp_vr.sh"
    exit 1
fi

# Algorithm settings (islands/NFE/runtime-freq) come from the active MOEA config
# (NYCOPT_MOEA_CONFIG in the env file), not from this script. The scenario design
# comes from NYCOPT_SCENARIO_DESIGN. _common.sh reads both back from config.
FORMULATION="ffmp_${N_ZONES}"
SEED="${SLURM_ARRAY_TASK_ID:-1}"
DEBUG_SIM=false
CHECKPOINT=false

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/slurm/main/_common.sh"

ARGS="--seed ${SEED} --formulation ${FORMULATION}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg (${SCENARIO}/${RUN_SLUG}, seed=${SEED}, N=${N_ZONES}) ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
