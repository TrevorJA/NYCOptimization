#!/bin/bash
# Grouped-epsilon re-assessment on the production ensemble fronts: re-filter
# the 500k-NFE Pareto sets under grouped candidate epsilon vectors (no
# simulation; pure numpy on the archived sets). See
# scripts/supplemental/epsilon_ensemble_refilter.py.
#
# Submit from the repo root:
#   sbatch workflow/supplemental/epsilon_ensemble_refilter.sh \
#       [--slug ffmp_obj8 --scenarios historic fixed_probabilistic ...]
#
#SBATCH --job-name=eps_ens_refilter
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/eps_ens_refilter_%j.out
#SBATCH --error=logs/eps_ens_refilter_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

PYTHONUNBUFFERED=1 python3 scripts/supplemental/epsilon_ensemble_refilter.py "$@"
echo "=== Completed: $(date) ==="
