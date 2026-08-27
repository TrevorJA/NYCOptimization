#!/bin/bash
# Sharded E_test sub-window hazard image: one array task per staged chunk
# (scripts/main/compute_etest_hazard_image.py). Each task writes the chunk's
# hazard_image_subwindows_shard_{i:03d}.npz, which is also the resume marker
# (rerunning a finished task is a no-op). The merge is a pure function of the
# shard contents (rows lexsorted by (rid, window), unique keys), so
# shard-then-merge is byte-identical to the serial loop.
#
# Submit, then chain etest_hazard_image_merge.sh:
#   sbatch --export=ALL,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/etest_hazard_image_shards.sh
#
#SBATCH --job-name=etest_hazimg_shard
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-49
#SBATCH --output=logs/etest_hazimg_shard_%A_%a.out
#SBATCH --error=logs/etest_hazimg_shard_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"

# Single-process numpy/pandas workload; give BLAS the allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

export NYCOPT_ETEST_HAZARD_SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"

echo "[etest_hazimg_shard] variant=${VARIANT} chunk ${NYCOPT_ETEST_HAZARD_SHARD_INDEX}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/compute_etest_hazard_image.py
echo "[etest_hazimg_shard] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
