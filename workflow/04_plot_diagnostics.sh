#!/bin/bash
# Step 4: Generate diagnostic figures (hypervolume convergence, seed
# reliability, parallel coordinates) from optimization results.
#
# Slug auto-derives from active config (formulation + ACTIVE_OBJECTIVES + T/S
# state). For smoke or variant runs, pass an explicit slug as first arg and
# (optionally) the formulation as second arg for DV/objective parsing.
#
# Figures land in figures/{convergence,parallel_coords,...}/{slug}/, NOT in
# outputs/figures/.
#
# Usage:
#   bash workflow/04_plot_diagnostics.sh                       # default ffmp + auto-slug
#   bash workflow/04_plot_diagnostics.sh ffmp_obj9_ts          # explicit slug
#   bash workflow/04_plot_diagnostics.sh smoke_rbf rbf         # slug + formulation override
#   NYCOPT_ENV_FILE=slurm/envs/ffmp_obj9_ts.env \
#       bash workflow/04_plot_diagnostics.sh                   # via env file
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

# Source per-experiment env file if provided (sets NYCOPT_TS_ON, OBJECTIVES, etc.)
NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:-}"
if [[ -n "${NYCOPT_ENV_FILE}" && -f "${NYCOPT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${NYCOPT_ENV_FILE}"
    set +a
    echo "[plot_diag] sourced env file: ${NYCOPT_ENV_FILE}"
fi

# First arg: explicit slug (overrides auto-derive). Second arg: formulation.
EXPLICIT_SLUG="${1:-}"
FORMULATION="${2:-ffmp}"

python3 - <<PYEOF
import sys; sys.path.insert(0, '.')
from config import (
    OUTPUTS_DIR, OUTPUT_OPTIMIZATION_DIR,
    derive_slug, figure_dir_for,
)
from src.plotting.hypervolume_convergence import plot_hypervolume_convergence
from src.plotting.seed_reliability import plot_seed_reliability
from src.plotting.parallel_coordinates import plot_parallel_coordinates

explicit_slug = "${EXPLICIT_SLUG}"
formulation = "${FORMULATION}"
slug = explicit_slug if explicit_slug else derive_slug(formulation)
print(f"[plot_diag] slug={slug} formulation={formulation}")

opt_dir = OUTPUT_OPTIMIZATION_DIR / slug
metrics_dir = opt_dir / "metrics"
sets_dir = opt_dir / "sets"

# Pick a merged reference set (prefer single-seed, fall back to any)
candidates = sorted(sets_dir.glob(f"{slug}_seed*_merged.set")) + \
             sorted(sets_dir.glob(f"{slug}_merged.set"))
ref_file = candidates[0] if candidates else None

conv_dir = figure_dir_for("convergence", slug)
pc_dir = figure_dir_for("parallel_coords", slug)

plot_hypervolume_convergence(metrics_dir, slug,
    output_file=conv_dir / f"hv_convergence_{slug}.png")
plot_seed_reliability(metrics_dir, slug,
    output_file=conv_dir / f"seed_reliability_{slug}.png")
if ref_file is not None:
    plot_parallel_coordinates(ref_file, formulation,
        output_file=pc_dir / f"parallel_coords_{slug}.png")
else:
    print(f"[{slug}] no merged .set file in {sets_dir}; skipped parallel coordinates")

print(f"Figures saved under: figures/<category>/{slug}/")
PYEOF
