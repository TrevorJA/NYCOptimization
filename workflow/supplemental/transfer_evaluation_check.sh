#!/bin/bash
#SBATCH --job-name=tev_check
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/transfer_evaluation_check_%j.out
#SBATCH --error=logs/transfer_evaluation_check_%j.err

# Transfer evaluation, stage 1: the path-consistency check
# (docs/notes/methods/transfer_evaluation.md). Evaluates TEV_CHECK_N_SOLUTIONS
# policies per design through the NEW driver against that design's OWN d0
# ensemble, and differences the result against the objective columns stored in
# the same design's .set file. Reports |delta| / epsilon per objective.
#
# This is not only a smoke test. The matrix reads its diagonal cells from the
# stored .set columns and its off-diagonal cells from the driver, and the
# merged-reference-set readout puts both into one nondominated sort. The
# measured ratio is what licenses that: if |delta| is far below epsilon, no
# solution changes epsilon box and the merge is unaffected. It also validates
# the explicit ensemble_spec override, the trimmed-model and presimulated-
# release path, the pinned flow prediction mode, the objective set and unit
# operators, and the identity of each .set file. It additionally measures the
# per-unit wall time and peak per-rank RSS used to size the evaluate job.
#
# PREREQUISITES: the adopted sets at
#   outputs/{historic,monte_carlo,hazard_filling_stationary}/ffmp_obj8/sets/ffmp_obj8_merged_eps20260812.set
# and the staged d0 search ensembles fixprob_10yr_n100_d0 and
# hazfill_stat_abs_10yr_n100_d0 (step 04), plus the historic presim CSV.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/transfer_evaluation.env \
#          workflow/supplemental/transfer_evaluation_check.sh
#
# Switches: NYCOPT_TEV_SMOKE=1 (smoke_ prefixed artifacts),
# NYCOPT_TEV_RETRY_FAILED=1 (re-attempt previously failed units).
# Settings in supplemental_config.py (TEV_ section).
#
# Outputs: outputs/supplemental/transfer_evaluation/tables/tev_path_consistency.csv;
# per-unit artifacts under TEV_UNITS_ROOT/check/.
# Sizing: 30 units (10 per design) at ~155 s (ensembles) / ~31 s (historic)
# = ~1 core-hour total; ~1.1 GB per rank. Well under 10 SU.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

export FI_PROVIDER=tcp
export NYCOPT_TEV_STAGE=check
NTASKS_MPI="${SLURM_NTASKS:-8}"

echo "[tev:check] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ranks=${NTASKS_MPI} (smoke=${NYCOPT_TEV_SMOKE:-0})"
mpirun -np "${NTASKS_MPI}" python3 -u scripts/supplemental/transfer_evaluation_run.py
echo "[tev:check] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
