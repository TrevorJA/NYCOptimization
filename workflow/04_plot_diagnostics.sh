#!/bin/bash
# Step 4: Generate diagnostic figures (hypervolume convergence, seed
# reliability, parallel coordinates) from optimization results.
#
# Usage:
#   bash 04_plot_diagnostics.sh [FORMULATION]
#   bash 04_plot_diagnostics.sh ffmp
#   sbatch 04_plot_diagnostics.sh
#
#SBATCH --job-name=plot_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/plot_diag_%j.out
#SBATCH --error=logs/plot_diag_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs
source venv/bin/activate

FORMULATION="${1:-ffmp}"

python3 -c "
import sys; sys.path.insert(0, '.')
from config import OUTPUTS_DIR
from src.plotting.hypervolume_convergence import plot_hypervolume_convergence
from src.plotting.seed_reliability import plot_seed_reliability
from src.plotting.parallel_coordinates import plot_parallel_coordinates

formulation = '${FORMULATION}'
fig_dir = OUTPUTS_DIR / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
metrics_dir = OUTPUTS_DIR / 'diagnostics' / formulation
ref_file = OUTPUTS_DIR / 'reference_sets' / f'{formulation}.ref'

plot_hypervolume_convergence(metrics_dir, formulation, output_file=fig_dir / f'{formulation}_hypervolume.png')
plot_seed_reliability(metrics_dir, formulation, output_file=fig_dir / f'{formulation}_seed_reliability.png')
plot_parallel_coordinates(ref_file, formulation, output_file=fig_dir / f'{formulation}_parallel_coords.png')
print(f'Figures saved to: {fig_dir}')
"
