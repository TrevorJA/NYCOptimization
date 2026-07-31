#!/bin/bash
# satisfaction_factor.sh — MPI-parallel weekly satisfaction-factor sweep for
# ONE scenario design, then the serial cross-design analysis. Re-evaluates the
# epsilon-calibration feasible-policy population (identical seed/count, so
# cube rows align across the two experiments) on the design's search ensemble,
# storing per-unit failing-week counts + §1 weekly reliability for the NYC/NJ
# delivery objectives at each factor in SF_FACTOR_GRID
# (docs/notes/methods/framing_convention_diagnostics.md diagnostic 2).
#
# REQUIRES NYCOPT_ENV_FILE (the design is the run identity — same contract as
# the epsilon calibration). Submit once per ensemble campaign design:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_fixed_probabilistic.env \
#       workflow/supplemental/satisfaction_factor.sh
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_hazard_filling_stationary.env \
#       workflow/supplemental/satisfaction_factor.sh
#
# PREREQUISITES: steps 01-04 staged for the design (same as epsilon
# calibration). All settings live in `supplemental_config.py` (SF_* section) —
# this script carries NO value flags.
#
# Sizing: identical evaluation cost to the epsilon calibration (513 policies x
# ~174 s/eval on 128 ranks ~= 15-20 min per design); the factor axis is
# computed from each realization's weekly sums at no extra simulation cost.
#
#SBATCH --job-name=sf_sweep
#SBATCH --account=ees260021
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --output=logs/satisfaction_factor_%j.out
#SBATCH --error=logs/satisfaction_factor_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

# TCP libfabric provider: filesystem-barrier shard combine avoids comm.gather,
# but keep this set so any incidental MPI collective stays on a stable provider.
export FI_PROVIDER=tcp

NTASKS_MPI="${SLURM_NTASKS:-8}"

echo "=== Satisfaction-factor sweep (design env: ${NYCOPT_ENV_FILE}, ranks=${NTASKS_MPI}) ==="
mpirun -np "${NTASKS_MPI}" python3 -u \
    scripts/supplemental/satisfaction_factor_run.py
echo "=== Evaluation complete; analyzing all design cubes present ==="
python3 -u scripts/supplemental/satisfaction_factor_figures.py
echo "=== Completed: $(date) ==="
