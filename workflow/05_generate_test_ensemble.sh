#!/bin/bash
# Step 5: Generate the held-out test ensemble E_test — LHS over the widened
# DU box, N_theta=500 SOWs x R=25 realizations x L=50 yr in 25 chunks of 500
# realizations, hazard image streamed during generation (scenario discovery needs it).
# Sizing lives in src/etest.py (env-overridable via NYCOPT_ETEST_*) and nowhere
# else. Independent of the staging, baseline and search steps (02 to 04, 06 to 09); the sharded build is
# workflow/supplemental/staging/etest_shards.sh. (The campaign E_test was not built
# by this step: it is cut from the larger design generated 2026-08-19 by
# scripts/supplemental/make_etest_subset.py, metadata only.)
#
# Env inputs: NYCOPT_ETEST_VARIANT (kn = the campaign variant, default; hmm =
# opt-in generator-structure sensitivity), NYCOPT_ENV_FILE (optional),
# NYCOPT_ENSEMBLE_FORCE=1 to overwrite an already-staged slug.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/05_generate_test_ensemble.sh
#
# Output: outputs/synthetic_ensembles/etest_{gen}_{L}yr_n{N}[__chunkJJJ]/;
# NYCOPT_REEVAL_ENSEMBLE_PRESET then names that slug for steps 10 to 12.
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
nycopt_full_threads 8

echo "[gen_etest] variant=${VARIANT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u -m scripts.main.generate_test_ensemble --variant "${VARIANT}"
echo "[gen_etest] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
