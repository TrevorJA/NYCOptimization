#!/bin/bash
# Step 1: Run the full Pywr-DRB model once and save non-NYC (STARFIT) releases
# for use as boundary conditions in the trimmed optimization model.
#
# Usage (from repo root):
#   sbatch workflow/01_generate_presim.sh
#   bash   workflow/01_generate_presim.sh
#
#SBATCH --job-name=presim
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/presim.out
#SBATCH --error=logs/presim.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

python3 scripts/main/generate_presim.py
