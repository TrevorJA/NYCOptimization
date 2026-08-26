#!/bin/bash
#SBATCH --job-name=esd_analysis
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ensemble_size_analysis_%j.out
#SBATCH --error=logs/ensemble_size_analysis_%j.err

# Ensemble-size diagnostics, stage 4: statistics, the pre-registered decision
# table (N_min per design, N_common), n_eff, the epsilon-cube cross-check, the
# NFE-asymptote reading from the existing runtime archives, cost pricing, and
# every figure (docs/notes/methods/ensemble_size_diagnostics.md §§4.3, 6).
# Pure post-processing of the Layer-A tables and the Layer-B library; zero
# simulation. Layer-B outputs are skipped with a message when the library is
# absent, so the Layer-A figures can be regenerated at any time.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_size_diagnostics.env \
#          workflow/supplemental/ensemble_size_analysis.sh
# Smoke: add NYCOPT_ESD_SMOKE=1.
#
# Sizing: the library is ~150 MB; the bootstraps (1,000 draws x ~10 policies x
# 8 objectives x 5 replicates x 2 designs x 8 rungs) are the long pole,
# minutes with 4 threads.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[esd:analysis] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (smoke=${NYCOPT_ESD_SMOKE:-0})"
python3 -u scripts/supplemental/ensemble_size_figures.py
echo "[esd:analysis] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
