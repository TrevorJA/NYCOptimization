#!/bin/bash
# Diagnosis of the merge-job boundary-verification mismatches: regenerate the
# affected + control indices under OMP=2 (the shard jobs' env) and OMP=4 (the
# merge job's env), test in-process batch-composition invariance, compare the
# OMP=8 smoke pool against the merged image's prefix, and run the (slow-marked,
# previously deselected) end-to-end determinism regression test. Report-only.
#
#SBATCH --job-name=nestedp_diag
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nestedp_diag_%j.out
#SBATCH --error=logs/nestedp_diag_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

DIAG="scripts/supplemental/diagnose_partition_mismatch.py"

echo "[nestedp_diag] pass 1: OMP=2 (shard-job env)  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 -u "${DIAG}" || echo "[nestedp_diag] pass 1 rc=$?"

echo "[nestedp_diag] pass 2: OMP=4 (merge-job env)  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python3 -u "${DIAG}" || echo "[nestedp_diag] pass 2 rc=$?"

echo "[nestedp_diag] done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
