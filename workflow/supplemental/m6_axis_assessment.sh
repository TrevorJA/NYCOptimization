#!/bin/bash
# Score candidate m=6 axis sets against the adequacy gate on prefix rungs of
# the staged P=1e6 pool image (selection-level only; minutes).
#
#SBATCH --job-name=m6_assess
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/m6_assess_%j.out
#SBATCH --error=logs/m6_assess_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

python3 -u scripts/supplemental/assess_m6_axis_sets.py
