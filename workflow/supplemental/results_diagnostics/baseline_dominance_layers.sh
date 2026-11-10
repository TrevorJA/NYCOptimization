#!/bin/bash
# Baseline dominance at each archive layer of one search seed (final .set, island
# archives at their last snapshot, every island snapshot, every coordinator
# snapshot): were baseline-dominating policies found and then removed by the
# epsilon-box archive, or never found? See scripts/supplemental/baseline_dominance_layers.py.
# Needs step 09 (runtime_coordinator/) and the scenario-matched step-06 baseline.
#
# Submit from the repo root:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
#       workflow/supplemental/results_diagnostics/baseline_dominance_layers.sh \
#       --scenario historic --slug ffmp_obj8 --seed 1
#
# Outputs: outputs/supplemental/baseline_dominance_layers/{scenario}/{slug}/.
# Sizing: parses the island runtime files once; about a minute on 2 cores.
#
#SBATCH --job-name=baseline_dominance_layers
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/baseline_dominance_layers_%j.out
#SBATCH --error=logs/baseline_dominance_layers_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

echo "=== Started: $(date) on $(hostname) ==="
python3 -u scripts/supplemental/baseline_dominance_layers.py "$@"
echo "=== Completed: $(date) ==="
