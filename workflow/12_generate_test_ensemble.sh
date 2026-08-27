#!/bin/bash
# Step 12: Generate the held-out test ensemble E_test — LHS over the widened
# DU box, N_theta=1000 SOWs x R=25 realizations x L=50 yr in 50 chunks of 500
# realizations, hazard image streamed during generation (step 11 needs it).
# The campaign re-evaluates on the leading 500 SOWs (25 chunks;
# scripts/supplemental/make_etest_subset.py). Sizing lives in src/etest.py
# (env-overridable via NYCOPT_ETEST_*) and nowhere else. Independent of steps
# 02-07; the sharded build is workflow/supplemental/gen_etest_shards.sh.
#
# Env inputs: NYCOPT_ETEST_VARIANT (kn = the campaign variant, default; hmm =
# opt-in generator-structure sensitivity), NYCOPT_ENV_FILE (optional),
# NYCOPT_ENSEMBLE_FORCE=1 to overwrite an already-staged slug.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/12_generate_test_ensemble.sh
#
# Output: outputs/synthetic_ensembles/etest_{gen}_{L}yr_n{N}[__chunkJJJ]/;
# NYCOPT_REEVAL_ENSEMBLE_PRESET then names that slug (or its prefix subset)
# for steps 05/08/09/10/11.
#
#SBATCH --job-name=gen_etest
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_etest_%j.out
#SBATCH --error=logs/gen_etest_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"

# Generator is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[gen_etest] variant=${VARIANT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u -m scripts.main.generate_test_ensemble --variant "${VARIANT}"
echo "[gen_etest] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
