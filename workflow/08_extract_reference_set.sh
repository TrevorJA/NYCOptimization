#!/bin/bash
# Step 8: Equal-NFE reference set from the island runtime files
# (scripts/main/extract_runtime_archive.py). Two modes, passed through as the
# script's own arguments:
#   --seed N            one seed's coordinator-archive snapshot at the reporting
#                       NFE -> sets/seed_NN_{slug}_nfe{NFE}.set (run after the seed
#                       finishes; seed 1 before the epsilon re-filter)
#   --merge --install   the epsilon-box union of every seed at equal NFE ->
#                       sets/{slug}_merged_nfe{NFE}.set, installed as
#                       sets/{slug}_merged.set, the step-10 input (run after the
#                       last seed)
# Step 09 writes its own diagnostics reference set and never this one, so the
# two steps run in either order after `--seed 1`.
#
# Env inputs: NYCOPT_ENV_FILE (required); DRAW=k (default 0) as in step 07;
# FORMULATION (default ffmp).
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/<design>_production.env \
#          workflow/08_extract_reference_set.sh --seed 1
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/<design>_production.env \
#          workflow/08_extract_reference_set.sh --merge --install
#
# Sizing: reads the island runtime files once (hundreds of MB at 750k NFE) and
# epsilon-merges in memory; minutes on a few shared cores.
#
#SBATCH --job-name=extract_reference_set
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/extract_reference_set_%j.out
#SBATCH --error=logs/extract_reference_set_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
export NYCOPT_ENSEMBLE_DRAW="${DRAW:-${NYCOPT_ENSEMBLE_DRAW:-0}}"
nycopt_pin_threads

[[ $# -gt 0 ]] || { echo "ERROR: pass --seed N or --merge [--install] after the script path" >&2; exit 2; }
FORMULATION="${FORMULATION:-ffmp}"

echo "=== Extract reference set: formulation=${FORMULATION} draw=${NYCOPT_ENSEMBLE_DRAW} args: $* ==="
python3 -u scripts/main/extract_runtime_archive.py --formulation "${FORMULATION}" "$@"
echo "=== Completed: $(date) ==="
