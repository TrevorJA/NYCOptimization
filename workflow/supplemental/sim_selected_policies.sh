#!/bin/bash
# Supplemental: simulate the selected policies + the FFMP baseline on the
# historic trace and render the timeseries figures.
#
# scripts/main/explore_results.py picks the representatives, so this job
# simulates whatever the current selection rules choose (baseline first, then
# each highlighted policy) in ONE process and ONE model mode — the persisted
# outputs/baseline/ffmp_baseline.hdf5 was written with the FULL model while
# search uses the TRIMMED model, so a mixed-mode comparison would attribute a
# model difference to policy. Traces are cached under
# outputs/{scenario}/{slug}/timeseries/ so the figures can be re-rendered on a
# login node afterwards with a plain `python3 -m scripts.main.explore_results`.
#
# ~31 s per historic simulation on Anvil; a handful of policies fits easily in
# the wall time below.
#
# Usage (from repo root):
#   sbatch workflow/supplemental/sim_selected_policies.sh
#   sbatch workflow/supplemental/sim_selected_policies.sh --slug ffmp_obj8_mm_full --n-diverse 4
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
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
