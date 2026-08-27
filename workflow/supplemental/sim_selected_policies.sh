#!/bin/bash
# Supplemental: simulate the selected policies (scripts/main/explore_results.py
# picks them) plus the FFMP baseline on the historic trace in one job and one
# model mode, and cache the traces under outputs/{scenario}/{slug}/timeseries/
# so explore_results can re-render the timeseries figures on a login node.
#
# Env inputs: NYCOPT_ENV_FILE (optional); extra args pass through to
# explore_results (e.g. --slug, --n-diverse).
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
#          workflow/supplemental/sim_selected_policies.sh
#
#SBATCH --job-name=sim_selected
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/sim_selected_%j.out
#SBATCH --error=logs/sim_selected_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

python3 -m scripts.main.explore_results --simulate-timeseries "$@"

echo "=== Completed: $(date) ==="
