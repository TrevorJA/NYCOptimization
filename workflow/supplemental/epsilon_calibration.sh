#!/bin/bash
# epsilon_calibration.sh — MPI-parallel epsilon-calibration evaluation for ONE
# scenario design, then the serial cross-design analysis. Evaluates (baseline +
# EPS_N_POLICIES constraint-feasible random DV vectors) on the design's search
# ensemble through the same batched path Borg workers run, persisting the
# per-unit annual-metric cube; the figures script then derives the per-objective
# signal/noise/granularity floors, the archive-size sweep, and the campaign
# epsilon recommendation — the max over the ENSEMBLE campaign designs
# (supplemental_config.EPS_CAMPAIGN_DESIGNS); the historic cube is analyzed
# and reported as a reference arm but excluded from the max.
#
# REQUIRES NYCOPT_ENV_FILE (the design is the run identity — same contract as
# workflow/06_run_mmborg.sh). Submit once per campaign design:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_historic.env \
#       workflow/supplemental/epsilon_calibration.sh
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_fixed_probabilistic.env \
#       workflow/supplemental/epsilon_calibration.sh
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_hazard_filling_stationary.env \
#       workflow/supplemental/epsilon_calibration.sh
#
# PREREQUISITES: steps 01 (presim) and, for the ensemble designs, 02-04
# (generate / subsample / prep the search ensemble) under the same design.
#
# All other settings (sample size, seed, bootstrap, output paths) live in
# `supplemental_config.py` (EPS_* section) — this script carries NO value
# flags. Set EPS_SMOKE=False there for the full run.
#
# Sizing: 513 policies x ~540 s/eval (N=300 x 10-yr trimmed, batched at 150
# realizations per model run via supplemental_config.EPS_REALIZATION_BATCH) on
# 128 ranks = ~5 eval waves ~= 1 h on one wholenode node; the feasible-DV
# rejection sample and the analysis stage add minutes. historic evaluations are
# cheaper (single trace).
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
