#!/bin/bash
# Step 2: Subsample the large stochastic ensemble using LHS-style sampling
# over hydrologic-metric space to produce the space-filling ensembles
# (and matched probabilistic ensembles for comparison).
#
# Placeholder — implementation deferred until the ensemble-design
# discussion finalizes the hydrologic metric vector and LHS construction.
#
# Usage:
#   bash workflow/02_subsample_lhs_ensemble.sh
#   sbatch workflow/02_subsample_lhs_ensemble.sh
#
#SBATCH --job-name=subsample_lhs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/subsample_lhs_%j.out
#SBATCH --error=logs/subsample_lhs_%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}/..}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

echo "[02_subsample_lhs_ensemble] TODO: implement LHS subsampling over hydrologic-metric space."
python3 scripts/ensemble/subsample_lhs_ensemble.py "$@"
