#!/bin/bash
# Full-model HDF5 simulations of the policies highlighted by explore_results
# (figures/{scenario}/{slug}/selected_solutions.csv) plus the incumbent FFMP
# policy, on the historic trace. Driver: scripts/supplemental/
# simulate_selected_policies.py (src.policy_simulation). Outputs under
# outputs/{scenario}/{slug}/timeseries/selected_policies/ (symlink that
# directory into projects space first; runs are ~140 MB each). Re-run
# explore_results afterwards to render figures 05/06 from these traces.
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
#       workflow/supplemental/sim_selected_policies.sh --slug ffmp_obj8 [--rows ...] [--all-rules]
#
#SBATCH --job-name=sim_selected
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/sim_selected_%j.out
#SBATCH --error=logs/sim_selected_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

echo "=== Started: $(date) on $(hostname) ==="
python3 scripts/supplemental/simulate_selected_policies.py "$@"
echo "=== Completed: $(date) ==="
