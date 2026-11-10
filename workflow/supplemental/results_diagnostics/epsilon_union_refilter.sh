#!/bin/bash
# Archive size under candidate epsilon vectors, estimated from every solution one
# completed seed ever held in an island archive (all runtime snapshots), plus the
# per-axis value structure of that union. No simulation. Candidates are defined in
# scripts/supplemental/epsilon_union_refilter.py (CANDIDATES).
#
# Submit from the repo root:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
#       workflow/supplemental/results_diagnostics/epsilon_union_refilter.sh \
#       --scenario historic --slug ffmp_obj8 --seed 1
#
# Outputs: outputs/supplemental/epsilon_union_refilter/{scenario}/{slug}/.
# Sizing: parses the island runtime files once and box-filters a few thousand rows
# per candidate; about a minute on 2 cores.
#
#SBATCH --job-name=epsilon_union_refilter
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/epsilon_union_refilter_%j.out
#SBATCH --error=logs/epsilon_union_refilter_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

echo "=== Started: $(date) on $(hostname) ==="
python3 -u scripts/supplemental/epsilon_union_refilter.py "$@"
echo "=== Completed: $(date) ==="
