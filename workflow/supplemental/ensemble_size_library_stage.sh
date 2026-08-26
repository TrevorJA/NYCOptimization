#!/bin/bash
#SBATCH --job-name=esd_stage
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0
#SBATCH --output=logs/ensemble_size_stage_%A_%a.out
#SBATCH --error=logs/ensemble_size_stage_%A_%a.err

# Ensemble-size diagnostics, stage 2: regenerate + prep ONE library chunk per
# array task (docs/notes/methods/ensemble_size_diagnostics.md §4.2).
#
# Task j regenerates chunk j of tables/library_plan.json from the stream-only
# candidate pool (serial; ~2.1 s per realization, so ~35 min per 1,000-member
# chunk) into the staged chunk dir (kept on projects space via a symlink), then
# stages the step-04 pywrdrb inputs for that chunk over the allocation's ranks
# (scripts/main/prep_pywrdrb_inputs.py --preset; the one-time full-model
# presim pass, ~200 s per 100 realizations per rank).
#
# Submit with the array sized to the plan (see ensemble_size_hazard.sh):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_size_diagnostics.env \
#          --array=0-9 workflow/supplemental/ensemble_size_library_stage.sh
# Smoke: add NYCOPT_ESD_SMOKE=1 (chunks of <= 300; --array=0-1).
# NYCOPT_ESD_FORCE=1 re-regenerates an already-staged chunk.
#
# Home-quota note: chunks land under supplemental_config.ESD_STAGING_ROOT
# (projects space) and are symlinked into outputs/synthetic_ensembles/.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

export NYCOPT_ESD_STAGE=materialize
export NYCOPT_ESD_CHUNK="${SLURM_ARRAY_TASK_ID:-0}"

# The chunk slug and size come from the persisted plan (smoke-aware via
# supplemental_config). The prep rank count is capped at the chunk size: the
# pywrdrb ensemble preprocessors hang when a rank owns zero realizations
# (observed 2026-08-25 on a 6-member chunk under 8 ranks).
read -r CHUNK_SLUG CHUNK_N < <(python3 -c "import json, sys, supplemental_config as s; c = json.load(open(s.esd_json_path('library_plan')))['chunks'][int(sys.argv[1])]; print(c['slug'], c['n_realizations'])" "${NYCOPT_ESD_CHUNK}")
PREP_NP="${SLURM_NTASKS:-1}"
(( CHUNK_N < PREP_NP )) && PREP_NP="${CHUNK_N}"

echo "[esd:stage] chunk ${NYCOPT_ESD_CHUNK} (${CHUNK_SLUG}) start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (smoke=${NYCOPT_ESD_SMOKE:-0})"
python3 -u scripts/supplemental/ensemble_size_library_run.py
echo "[esd:stage] chunk regenerated; staging pywrdrb inputs on ${PREP_NP} ranks (${CHUNK_N} realizations)"
mpirun -np "${PREP_NP}" python3 -u scripts/main/prep_pywrdrb_inputs.py --preset "${CHUNK_SLUG}"
echo "[esd:stage] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
