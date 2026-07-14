#!/bin/bash
# Step 2: Generate the active scenario design's realizations.
#
# Every design GENERATES its own realizations from its own namespaced seed
# stream — no design is subsampled from a shared master. The array index is the
# independent ensemble-draw index k (0..K-1, K = design.n_ensemble_draws), which
# is now the natural parallel axis: draws are independent GENERATIONS.
#
# COST DISCLOSURE: per-design construction is not free. For `fixed_probabilistic`
# and `input_stratified` each draw is a fresh N x L generation (not a re-index of
# shared data), so step-02 cost scales with K — K x the single-draw generation
# cost, per design. The pool-owning designs (`resampled_probabilistic`,
# `hazard_filling_*`) instead build ONE draw-invariant pool: array tasks k>0 are
# no-ops there, and the two DU hazard designs share a single candidate pool.
#
# Configuration is entirely env-driven; the design comes from the run's env file
# (or --export). No value flags.
#
#   # 10 independent draws of the fixed-probabilistic design:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_fixprob.env \
#          --array=0-9 workflow/02_generate_ensemble.sh
#
#   # A pool-owning design: one draw is enough.
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_hazfill_pilot.env \
#          workflow/02_generate_ensemble.sh
#
# Output: outputs/synthetic_ensembles/{slug}/ — the per-draw search ensemble, or
# the design's candidate/resampling pool (+ hazard_image.npz for hazard-filling).
# Set NYCOPT_ENSEMBLE_FORCE=1 to overwrite an already-staged slug.
#
#SBATCH --job-name=gen_ensemble
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --output=logs/gen_ensemble_%A_%a.out
#SBATCH --error=logs/gen_ensemble_%A_%a.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

# Array index = ensemble draw k. config.py reads it back as SCENARIO_ENSEMBLE_DRAW,
# so shell and Python resolve the same staged slug.
export NYCOPT_ENSEMBLE_DRAW="${SLURM_ARRAY_TASK_ID:-${NYCOPT_ENSEMBLE_DRAW:-0}}"

# Generator is single-process; let BLAS use the full allocation (no pinning).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[gen_ensemble] design=${NYCOPT_SCENARIO_DESIGN:-<default>} draw=${NYCOPT_ENSEMBLE_DRAW}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/main/generate_stochastic_ensemble.py --draw "${NYCOPT_ENSEMBLE_DRAW}"
echo "[gen_ensemble] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
