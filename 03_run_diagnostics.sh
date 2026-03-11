#!/bin/bash
# Step 3: Run MOEAFramework runtime diagnostics — computes hypervolume,
# generational distance, and builds the global reference set.
#
# Usage:
#   bash 03_run_diagnostics.sh [FORMULATION]
#   bash 03_run_diagnostics.sh ffmp
#   sbatch 03_run_diagnostics.sh
#
#SBATCH --job-name=diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/diagnostics_%j.out
#SBATCH --error=logs/diagnostics_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs
source venv/bin/activate

FORMULATION="${1:-ffmp}"

python3 -c "
import sys; sys.path.insert(0, '.')
from src.diagnostics import run_full_diagnostics
run_full_diagnostics('${FORMULATION}')
"
