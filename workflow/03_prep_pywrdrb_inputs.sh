#!/bin/bash
# Step 3: Format subsampled ensembles into pywrdrb-ingestible HDF5 inputs
# under outputs/synthetic_ensembles/{inflow_type}/.
#
# Placeholder — implementation deferred. Expected to produce, for each
# scenario design's search ensemble (see src/scenario_designs.py):
#   catchment_inflow_mgd.hdf5    per-realization inflows
#   predicted_inflows_mgd.hdf5   per-realization Montague/Trenton lag forecasts
#
# Usage:
#   bash workflow/03_prep_pywrdrb_inputs.sh
#   sbatch workflow/03_prep_pywrdrb_inputs.sh
#
#SBATCH --job-name=prep_pywrdrb_inputs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_pywrdrb_inputs_%j.out
#SBATCH --error=logs/prep_pywrdrb_inputs_%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}/..}"
mkdir -p logs
module load python/3.11.5 || true
source venv/bin/activate

echo "[03_prep_pywrdrb_inputs] TODO: format subsampled ensembles into pywrdrb HDF5 inputs."
python3 scripts/main/prep_pywrdrb_inputs.py "$@"
