#!/bin/bash
# Exploratory figure suite of one completed search (scripts/main/explore_results.py):
# the front against the scenario-matched FFMP baseline on parallel axes, the
# selected representatives, Spearman trade-off structure, pairwise trade-off
# scatters, the operating rules of each representative, and the DV ranges of
# criterion-satisfying subsets. Pure post-processing of a .set file; no simulation.
# Figures 05/06 (historic time series) additionally need the HDF5 runs of
# sim_selected_policies.sh; pass --skip-timeseries until those exist.
#
# Outputs: outputs/figures/_exploratory/{scenario}/{slug}/explore/ (figures,
# selected_solutions.csv, selected_dv_distances.csv).
#
# All arguments pass through to explore_results.py. The default set is
# {slug}_merged.set; pass --set-file for a single seed before the merge.
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
#       workflow/supplemental/figures/explore_results.sh --scenario historic --slug ffmp_obj8 \
#       --set-file outputs/historic/ffmp_obj8/sets/seed_01_ffmp_obj8.set --skip-timeseries
#
# Sizing: reads one set of at most a few thousand rows; about a minute on 2 cores.
#
#SBATCH --job-name=explore_results
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/explore_results_%j.out
#SBATCH --error=logs/explore_results_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

echo "=== Started: $(date) on $(hostname) ==="
python3 -u -m scripts.main.explore_results "$@"
echo "=== Completed: $(date) ==="
