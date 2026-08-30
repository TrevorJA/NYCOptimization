#!/bin/bash
# epsilon_calibration.sh — MPI evaluation of the feasible-policy population on
# ONE design's search ensemble (scripts/supplemental/epsilon_calibration_run.py),
# then the serial cross-design analysis (epsilon_calibration_figures.py).
# Requires NYCOPT_ENV_FILE (the design is the run identity); one job per
# design. Settings: supplemental_config.py (EPS_* section; EPS_SMOKE=False for
# the full run), no CLI value flags.
#
# Prerequisites: step 01 and, for the ensemble designs, steps 02-04 under the
# same design.
#
# Submit (from repo root), once per design:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_monte_carlo.env \
#       workflow/supplemental/epsilon_calibration.sh
#   (also eps_calib_historic.env, eps_calib_hazard_filling_stationary.env)
#
# Outputs: outputs/supplemental/epsilon_calibration/{cubes,tables,figures}.
# Sizing: 513 policies x ~540 s/eval on 128 ranks ~= 1 h on one wholenode node
# for an N=300 design; historic is far cheaper (single trace).
#
#SBATCH --job-name=eps_calib
#SBATCH --account=ees260021
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --output=logs/epsilon_calibration_%j.out
#SBATCH --error=logs/epsilon_calibration_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

# TCP libfabric provider: filesystem-barrier shard combine avoids comm.gather,
# but keep this set so any incidental MPI collective stays on a stable provider.
export FI_PROVIDER=tcp

NTASKS_MPI="${SLURM_NTASKS:-8}"

echo "=== Epsilon calibration (design env: ${NYCOPT_ENV_FILE}, ranks=${NTASKS_MPI}) ==="
mpirun -np "${NTASKS_MPI}" python3 -u \
    scripts/supplemental/epsilon_calibration_run.py
echo "=== Evaluation complete; analyzing all design cubes present ==="
python3 -u scripts/supplemental/epsilon_calibration_figures.py
echo "=== Completed: $(date) ==="
