#!/bin/bash
#SBATCH --job-name=esd_hazard
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ensemble_size_hazard_%j.out
#SBATCH --error=logs/ensemble_size_hazard_%j.err

# Ensemble-size diagnostics, stage 1: Layer A (hazard-space representativeness
# vs N, selection level, no simulation) + the fixed policy set + the library
# plan (docs/notes/methods/ensemble_size_diagnostics.md §§3-4).
#
# Reads the staged P=1e6 pool hazard images (d0-d2), the two matched designs'
# epsilon-filtered reference sets, and their campaign re-eval cubes
# (supplemental_config.ESD_POLICY_REEVAL_TAG). Writes
# tables under outputs/supplemental/ensemble_size_diagnostics/tables/,
# including library_plan.json — the array size of the next stage:
#   python3 -c 'import json;print(len(json.load(open("outputs/supplemental/ensemble_size_diagnostics/tables/library_plan.json"))["chunks"]))'
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_size_diagnostics.env \
#          workflow/supplemental/ensemble_size_hazard.sh
# Smoke: add NYCOPT_ESD_SMOKE=1 to the --export list (P=2,000 pool, minutes).
#
# Sizing: one 1e6 x 8 image (64 MB) per pool draw; the LHS-NN selector builds
# a cKDTree on 1e6 x 6 per (prefix, N) and queries N anchors — seconds each;
# ~500 selections in total plus 200 x 8 random subsets. Tens of minutes.
# All settings live in supplemental_config.py (ESD_ section) — no CLI flags.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[esd:hazard] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (smoke=${NYCOPT_ESD_SMOKE:-0})"
python3 -u scripts/supplemental/ensemble_size_hazard.py
echo "[esd:hazard] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
