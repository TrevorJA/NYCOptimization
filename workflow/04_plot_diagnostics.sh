#!/bin/bash
# Step 4: Generate diagnostic figures (hypervolume convergence, seed
# reliability, parallel coordinates) from optimization results.
#
# Slugs == formulation names for plain runs. For smoke or variant runs
# (e.g. smoke_ffmp, ann_reduced_state) pass the slug as first arg and
# (optionally) the formulation as second arg for DV/objective parsing.
#
# Usage:
#   bash workflow/04_plot_diagnostics.sh                    # default: ffmp slug+formulation
#   bash workflow/04_plot_diagnostics.sh ffmp               # single slug
#   bash workflow/04_plot_diagnostics.sh smoke_rbf rbf      # slug + formulation override
#   sbatch workflow/04_plot_diagnostics.sh
#
#SBATCH --job-name=plot_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/plot_diag_%j.out
#SBATCH --error=logs/plot_diag_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
module load python/3.11.5
source venv/bin/activate

SLUG="${1:-ffmp}"
FORMULATION="${2:-${SLUG}}"

python3 - <<PYEOF
import sys; sys.path.insert(0, '.')
from pathlib import Path
from config import OUTPUTS_DIR
from src.plotting.hypervolume_convergence import plot_hypervolume_convergence
from src.plotting.seed_reliability import plot_seed_reliability
from src.plotting.parallel_coordinates import plot_parallel_coordinates

slug = "${SLUG}"
formulation = "${FORMULATION}"
opt_dir = OUTPUTS_DIR / "optimization" / slug
metrics_dir = opt_dir / "metrics"
sets_dir = opt_dir / "sets"
fig_dir = OUTPUTS_DIR / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Pick a merged reference set (prefer single-seed, fall back to any)
candidates = sorted(sets_dir.glob(f"{slug}_seed*_merged.set")) + \
             sorted(sets_dir.glob(f"{slug}_merged.set"))
ref_file = candidates[0] if candidates else None

plot_hypervolume_convergence(metrics_dir, slug,
    output_file=fig_dir / f"hv_convergence_{slug}.png")
plot_seed_reliability(metrics_dir, slug,
    output_file=fig_dir / f"seed_reliability_{slug}.png")
if ref_file is not None:
    plot_parallel_coordinates(ref_file, formulation,
        output_file=fig_dir / f"parallel_coords_{slug}.png")
else:
    print(f"[{slug}] no merged .set file in {sets_dir}; skipped parallel coordinates")

print(f"Figures saved to: {fig_dir}")
PYEOF
