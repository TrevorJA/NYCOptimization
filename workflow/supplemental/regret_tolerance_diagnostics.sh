#!/bin/bash
#SBATCH --job-name=rtol_diag
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/regret_tolerance_diagnostics_%j.out
#SBATCH --error=logs/regret_tolerance_diagnostics_%j.err

# Regret-tolerance diagnostics (docs/notes/methods/regret_tolerance_diagnostics.md;
# scripts/supplemental/regret_tolerance_diagnostics.py). Zero simulation.
# Pass A needs only the step-05 incumbent cube (RTOL_REEVAL_BASELINE_DIR);
# pass B needs the re-evaluated policy sets with their baseline/ subdirs on the
# same preset and is skipped with a printed reason when they are absent.
# Settings: supplemental_config.py (RTOL_ section), no CLI value flags.
#
# Submit: sbatch workflow/supplemental/regret_tolerance_diagnostics.sh
#
# Outputs: outputs/supplemental/regret_tolerance_diagnostics/{tables,figures}.
# Sizing: pandas/numpy over the incumbent cube (~1-2 min); the paired bootstrap
# (RTOL_BOOTSTRAP_N resamples) dominates pass B.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[rtol] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/regret_tolerance_diagnostics.py
echo "[rtol] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
