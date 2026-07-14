#!/bin/bash
# Step 12: Generate the held-out test ensemble E_test.
#
# E_test is the MEASURING STICK — every design's Pareto policies are re-simulated
# on it (step 08/09) and the cross-design comparison is made there and nowhere
# else. It is NOT a scenario design: it never enters search, is never subsampled,
# and is never a control, so (unlike the search-side candidate pools) it is
# sampled by LHS, over a DELIBERATELY WIDER DU box than any design searched in.
#
#   LHS over the full (widened) harmonic DU box  ->  N_theta_test SOWs
#     x R_test realizations per SOW              ->  natural variability WITHIN each SOW
#     x L_test years                             ->  N_test = N_theta x R, chunked
#
# R_test > 1 is load-bearing: it is what makes the SOW-unit robustness metric
# (Herman 2014 / Trindade 2017 / Gold 2022 lineage) computable offline from the
# persisted re-eval cube. The hazard image is streamed during generation because
# step 11 (scenario discovery) hard-fails without it.
#
# Variants:
#   --variant kn    THE test ensemble: Kirsch-Nowak over the wide DU box. The
#                   DEFAULT and the only variant the campaign requires; this is what
#                   NYCOPT_REEVAL_ENSEMBLE_PRESET points at.  (seed domain etest:kn)
#   --variant hmm   OPT-IN, NOT part of the campaign, built by nothing automatically:
#                   a multi-site Gaussian-mixture HMM on annual flows, as a
#                   generator-structure sensitivity.           (seed domain etest:hmm)
#
# Sizing (N_theta_test, R_test, L_test, chunk) is PROVISIONAL and lives in
# src/etest.py — env-overridable (NYCOPT_ETEST_*), hardcoded nowhere else. No
# value flags here; --variant is an identifier.
#
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/12_generate_test_ensemble.sh
#
# Output: outputs/synthetic_ensembles/etest_{gen}_{L}yr_n{N}[__chunkJJJ]/
# Then point NYCOPT_REEVAL_ENSEMBLE_PRESET at that slug for steps 05/08/09/11.
# Set NYCOPT_ENSEMBLE_FORCE=1 to overwrite an already-staged slug.
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
