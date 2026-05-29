#!/bin/bash
# Step 1: Generate the large stochastic streamflow ensemble that the
# space-filling and probabilistic sub-samples will draw from.
#
# Placeholder — implementation deferred until the ensemble-design
# discussion finalizes generator method, realization count, and storage.
#
# Usage:
#   bash workflow/01_generate_stochastic_ensemble.sh
#   sbatch workflow/01_generate_stochastic_ensemble.sh
#
#SBATCH --job-name=gen_stoch_ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_stoch_ensemble_%j.out
#SBATCH --error=logs/gen_stoch_ensemble_%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}/..}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

echo "[01_generate_stochastic_ensemble] TODO: implement large stochastic streamflow ensemble generation."
python3 scripts/ensemble/generate_stochastic_ensemble.py "$@"
