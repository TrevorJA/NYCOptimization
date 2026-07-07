#!/bin/bash
# Step 2: Generate the stochastic streamflow ensemble / forcing master.
#
# Configuration is entirely env-driven. Point NYCOPT_ENV_FILE at one of the
# ensemble-generation env files (or set NYCOPT_ENSEMBLE_* vars directly):
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_kn_long.env \
#          workflow/02_generate_ensemble.sh
#
# Output: outputs/synthetic_ensembles/kn_{Y}yr_n{N}/ (or the forcing master
# for CMIP6-forced scenario designs).
#
#SBATCH --job-name=gen_ensemble
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_ensemble_%j.out
#SBATCH --error=logs/gen_ensemble_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

# Generator is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

python3 -u scripts/main/generate_stochastic_ensemble.py
