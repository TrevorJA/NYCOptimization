#!/bin/bash
# Step 3: Subsample the staged master ensemble into the reduced search
# ensemble for the active scenario design.
#
# Currently implements the hazard-filling designs (NYCOPT_SCENARIO_DESIGN ∈
# {hazard_filling, hazard_filling_absolute}) via
# scripts/main/subsample_hazard_filling.py, slicing the staged master pool
# into the reduced N=64 ensemble. Designs that need no subsampling (historic,
# fixed/resampled probabilistic) skip this step. Serial (the selector is
# single-process); loads the pool HDF5 into memory.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling \
#          workflow/03_subsample_ensemble.sh
#
#SBATCH --job-name=subsample_ens
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/subsample_ens_%j.out
#SBATCH --error=logs/subsample_ens_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
# Optional env file — the design can also come straight from
# --export=ALL,NYCOPT_SCENARIO_DESIGN=...
nycopt_source_env_file optional

# Selector is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[subsample_ens] design=${NYCOPT_SCENARIO_DESIGN:-<default>}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/subsample_hazard_filling.py
echo "[subsample_ens] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
