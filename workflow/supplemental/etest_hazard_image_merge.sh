#!/bin/bash
# Merge the E_test sub-window hazard-image shards into the canonical
# hazard_image_subwindows.npz. Verifies every shard file exists (fails listing
# the missing chunk indices), merges through the same lexsort-by-(rid, window)
# block the serial path uses, and unlinks the shards — no recomputation.
#
# Submit after the shard array (chain with --dependency=afterok:<array_jobid>):
#   sbatch --dependency=afterok:<jobid> \
#          --export=ALL,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/etest_hazard_image_merge.sh
#
#SBATCH --job-name=etest_hazimg_merge
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/etest_hazimg_merge_%j.out
#SBATCH --error=logs/etest_hazimg_merge_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_ETEST_HAZARD_MERGE=1

echo "[etest_hazimg_merge] variant=${VARIANT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/compute_etest_hazard_image.py
echo "[etest_hazimg_merge] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
