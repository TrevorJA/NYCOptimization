#!/bin/bash
# ===========================================================================
# 04_plot_diagnostics.sh - Generate diagnostic plots from MOEA results.
#
# Creates:
#   - Hypervolume convergence across NFE (per seed)
#   - Seed reliability (box/strip plot of final hypervolumes)
#   - Parallel coordinate plot of reference set objectives
#
# Usage:
#     bash 04_plot_diagnostics.sh [formulation]
#
# Outputs:
#     outputs/figures/{formulation}_hypervolume.png
#     outputs/figures/{formulation}_seed_reliability.png
#     outputs/figures/{formulation}_parallel_coords.png
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORMULATION="${1:-ffmp}"

echo "============================================"
echo "  04: Diagnostic Plots"
echo "  Formulation: ${FORMULATION}"
echo "============================================"

python3 - <<PYEOF
import sys
from pathlib import Path

sys.path.insert(0, ".")

from config import OUTPUTS_DIR, get_n_vars
from src.plotting.hypervolume_convergence import plot_hypervolume_convergence
from src.plotting.seed_reliability import plot_seed_reliability
from src.plotting.parallel_coordinates import plot_parallel_coordinates

formulation = "${FORMULATION}"
fig_dir = OUTPUTS_DIR / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

metrics_dir = OUTPUTS_DIR / "diagnostics" / formulation
ref_file = OUTPUTS_DIR / "reference_sets" / f"{formulation}.ref"

# 1. Hypervolume convergence
print("Plotting hypervolume convergence...")
plot_hypervolume_convergence(
    metrics_dir, formulation,
    output_file=fig_dir / f"{formulation}_hypervolume.png",
)

# 2. Seed reliability
print("Plotting seed reliability...")
plot_seed_reliability(
    metrics_dir, formulation,
    output_file=fig_dir / f"{formulation}_seed_reliability.png",
)

# 3. Parallel coordinates
print("Plotting parallel coordinates...")
plot_parallel_coordinates(
    ref_file, formulation,
    output_file=fig_dir / f"{formulation}_parallel_coords.png",
)

print(f"\nFigures saved to: {fig_dir}")
PYEOF
