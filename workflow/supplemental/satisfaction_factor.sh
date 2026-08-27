#!/bin/bash
# satisfaction_factor.sh — MPI weekly satisfaction-factor sweep for ONE design
# (scripts/supplemental/satisfaction_factor_run.py; framing_convention_diagnostics.md
# diagnostic 2), then the serial cross-design analysis. Requires NYCOPT_ENV_FILE
# (the design is the run identity); prerequisites and settings as for the
# epsilon calibration (supplemental_config.py SF_* section, no CLI value flags).
#
# Submit (from repo root), once per ensemble campaign design:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_fixed_probabilistic.env \
#       workflow/supplemental/satisfaction_factor.sh
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_hazard_filling_stationary.env \
#       workflow/supplemental/satisfaction_factor.sh
#
# Outputs: outputs/supplemental/satisfaction_factor/{cubes,tables,figures}.
# Sizing: the same evaluation cost as the epsilon calibration.
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
