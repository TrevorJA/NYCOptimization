#!/bin/bash
#SBATCH --job-name=tev_eval
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=26G
#SBATCH --time=00:30:00
#SBATCH --array=0-39
#SBATCH --output=logs/transfer_evaluation_eval_%A_%a.out
#SBATCH --error=logs/transfer_evaluation_eval_%A_%a.err

# Transfer evaluation, stage 2: the full nine-cell matrix as a shared work pool
# (docs/notes/methods/transfer_evaluation.md). MPI task farm over every
# (source solution, target ensemble) unit, through
# src.simulation.evaluate_annual_units with the target passed as an explicit
# ensemble_spec, then pooled through the search's own unit operators by
# src.reeval_core.sow_objective_matrix.
#
# The diagonal is simulated rather than read from the stored .set objective
# columns. Measured 2026-09-11 by the check stage: those columns come from
# searches at commit dc7e70b (2026-08-11/12), before commit a1e88bd
# (2026-08-18, "align metrics on June 1") moved START_DATE from 1945-10-01 to
# 1945-12-01 and introduced ENSEMBLE_START_DATE. The historic columns are
# multiples of 1/76 where the current code gives 1/77, and the ensemble designs
# differ by 1-4 epsilon. Mixing the two inside one nondominated sort would bias
# which solutions enter a merged reference set, so every cell uses one path.
#
# GEOMETRY, and why it is many small tasks rather than one big job. The work is
# ~200 core-hours over 6,330 independent units, so SU is flat in rank count and
# the only question is what the scheduler will start. Under the backlog at the
# time of writing (22,264 jobs pending on shared, 1,192 on wholenode) a 4-node
# wholenode request was estimated nine hours out and a 96-core shared job
# nineteen hours out, while a 60-core 45-minute shared job started in ninety
# seconds. Forty 16-core half-hour tasks fit backfill windows and together
# supply 320 core-hours of capacity against the 200 needed.
#
# All tasks of one array share a single claim space (keyed by SLURM_ARRAY_JOB_ID)
# and stride the global work list by global rank, so they pull from one pool
# without ever evaluating the same unit twice, and a task that starts late
# simply finds less left to do. Per-unit writes are atomic and completion is
# read back from the unit files, so RESUBMITTING THIS SAME ARRAY IS THE RESUME.
#
# The wall guard stops a rank cleanly rather than letting SLURM kill it
# mid-unit: no unit starts within 1.25 x NYCOPT_TEV_UNIT_SECONDS of the job's
# end, computed below from the actual SLURM end time.
#
# PREREQUISITES: the adopted sets and the staged d0 search ensembles (step 04),
# plus the historic presim CSV. Run the check stage first.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/transfer_evaluation.env \
#          workflow/supplemental/transfer_evaluation_eval_array.sh
# Resume: resubmit unchanged. Fewer/more workers: --array=0-N.
#
# Widening the farm: the per-user cap on `shared` is cpu=2048, but one array
# ramps up only as backfill windows appear. Submit ADDITIONAL arrays with
# NYCOPT_TEV_CLAIM_TAG set to the first array's job id and they join the same
# work pool instead of duplicating it:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=...,NYCOPT_TEV_CLAIM_TAG=<first_array_jobid> \
#          --array=0-59 workflow/supplemental/transfer_evaluation_eval_array.sh
#
# Switches: NYCOPT_TEV_SMOKE=1, NYCOPT_TEV_RETRY_FAILED=1,
# NYCOPT_TEV_CLAIM_TAG=<tag> (share a work pool across submissions),
# NYCOPT_TEV_INCLUDE_DRAWS=1 (own-draw d1 cells, the SI draw-sensitivity
# item), NYCOPT_TEV_CELL=<i> (restrict a task to one cell).
# Settings in supplemental_config.py (TEV_ section).
#
# Outputs: per-unit parquet under TEV_UNITS_ROOT/evaluate/<cell>/ with a
# provenance.json per cell. Merge is a SEPARATE stage
# (transfer_evaluation_analysis.sh).
# Sizing: 16 ranks x ~1.1 GB = ~18 GB against the 26 GB request.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

export FI_PROVIDER=tcp
export NYCOPT_TEV_STAGE=evaluate

# Longest unit is the N=100 x L=10 evaluation at ~160 s measured; pad to 175 s.
export NYCOPT_TEV_UNIT_SECONDS="${NYCOPT_TEV_UNIT_SECONDS:-175}"
END_TS="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)"
if [[ -n "${END_TS}" ]]; then
    export NYCOPT_TEV_STOP_EPOCH="$(date -d "${END_TS}" +%s 2>/dev/null || echo 0)"
else
    export NYCOPT_TEV_STOP_EPOCH=0
fi

NTASKS_MPI="${SLURM_NTASKS:-8}"
echo "[tev:eval] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) array_task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1} ranks=${NTASKS_MPI} stop_epoch=${NYCOPT_TEV_STOP_EPOCH}"
mpirun -np "${NTASKS_MPI}" python3 -u scripts/supplemental/transfer_evaluation_run.py
echo "[tev:eval] done: $(date -u +%Y-%m-%dT%H:%M:%SZ) array_task=${SLURM_ARRAY_TASK_ID:-0}"
