#!/bin/bash
# Post-shakeout epsilon-revision diagnostics: re-filter a completed run's
# Pareto sets under candidate epsilon vectors (no simulation; numpy + one
# MOEAFramework ResultFileMerger cross-check). See
# scripts/supplemental/epsilon_refilter_sweep.py.
#
# Submit from the repo root:
#   sbatch workflow/supplemental/epsilon_refilter_sweep.sh [--slug ... --scenario ...]
#
#SBATCH --job-name=eps_refilter
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/eps_refilter_%j.out
#SBATCH --error=logs/eps_refilter_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

python3 scripts/supplemental/epsilon_refilter_sweep.py "$@"
echo "=== Completed: $(date) ==="
