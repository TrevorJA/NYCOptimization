#!/bin/bash
#SBATCH --job-name=esd_eval
#SBATCH --account=ees260021
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --output=logs/ensemble_size_eval_%j.out
#SBATCH --error=logs/ensemble_size_eval_%j.err

# Ensemble-size diagnostics, stage 3: the per-realization annual-unit LIBRARY
# (docs/notes/methods/ensemble_size_diagnostics.md §2, §4.2). MPI task farm
# over (policy, staged ensemble, block of <= ESD_EVAL_BLOCK realizations)
# through src.simulation.evaluate_annual_units — the re-eval path, the same
# simulation and stage-(i) reduction as search — with shard-and-merge and the
# composition / regeneration QC on rank 0.
#
# PREREQUISITES: every chunk of the library plan staged + prepped
# (ensemble_size_library_stage.sh) and the production search ensembles
# (fixprob_/hazfill_ d0-d2) prepped by step 04.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_size_diagnostics.env \
#          workflow/supplemental/ensemble_size_library_eval.sh
# Smoke (shared partition, a few ranks):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_size_diagnostics.env,NYCOPT_ESD_SMOKE=1 \
#          --partition=shared --ntasks-per-node=8 --mem=32G --time=01:00:00 \
#          workflow/supplemental/ensemble_size_library_eval.sh
#
# Sizing: ~10 policies x ~10,000 realizations = ~1,000 blocks of 100 at
# 173.8 s each on 128 ranks = ~25 min of evaluation (+ model-build overhead);
# ~1 GB per rank. ~50 SU.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

export FI_PROVIDER=tcp
export NYCOPT_ESD_STAGE=evaluate
NTASKS_MPI="${SLURM_NTASKS:-8}"

echo "[esd:eval] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ranks=${NTASKS_MPI} (smoke=${NYCOPT_ESD_SMOKE:-0})"
mpirun -np "${NTASKS_MPI}" python3 -u scripts/supplemental/ensemble_size_library_run.py
echo "[esd:eval] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
