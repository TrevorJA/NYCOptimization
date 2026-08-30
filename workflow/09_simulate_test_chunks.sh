#!/bin/bash
# Step 9: Simulate + score a policy set on the chunked test ensemble
# (metrics-only): every chunk of the E_test preset NYCOPT_REEVAL_ENSEMBLE_PRESET
# resolves to, objectives + robustness from in-memory reduced metrics (no
# timeseries persisted). MPI chunk-and-aggregate; degrades to serial.
#
# Env inputs (no positional args):
#   NYCOPT_ENV_FILE                  required
#   NYCOPT_REEVAL_ENSEMBLE_PRESET    required — the chunked E_test slug
#                                    (campaign: etest_kn_50yr_n25000_first25ch)
#   NYCOPT_CHUNK_POLICIES            'baseline' (default) or a .ref reference set
#   NYCOPT_SEARCH_REALIZATION_BATCH  realizations per within-chunk sim batch
#                                    (bounds RAM; required above 64 ranks/node)
#   NYCOPT_CHUNK_SIM_MODE            single | mpi (default mpi)
#   NYCOPT_CHUNK_SIM_NODES / _RANKS  local (non-SLURM) mpi fallback only
#                                    (default 4 x 16); under SLURM the rank
#                                    count is the allocation
#   FORMULATION                      identifier, default ffmp
#   DRAW                             optional, default 0 — draw of the searched
#                                    run (selects the "_d{k}" slug for k>0)
#   SEED                             optional
#   NYCOPT_CHUNK_INCREMENTAL         1 (default): per-unit atomic flush + resume
#   NYCOPT_CHUNK_SCHEDULE            claim (default) | interleave | contiguous
#   NYCOPT_CHUNK_MERGE               job (default) | off — campaign runs use
#                                    off + workflow/09b_merge_test_chunks.sh
#   NYCOPT_CHUNK_UNIT_SECONDS        measured per-unit wall (enables the
#                                    stop-before-the-wall guard)
#
# Campaign submission (chain identical resubmissions, each resumes, then 09b):
#   sbatch --nodes=8 --ntasks-per-node=128 --time=24:00:00 \
#          --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_mc_production.env,\
# NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch,NYCOPT_CHUNK_POLICIES=<merged.ref>,\
# NYCOPT_CHUNK_MERGE=off,NYCOPT_SEARCH_REALIZATION_BATCH=150 \
#          workflow/09_simulate_test_chunks.sh
#
# Output: outputs/{scenario}/{slug}/reeval/{preset}[/seed_NN]/ (reeval_raw.parquet + meta).
#
#SBATCH --job-name=sim_test_chunks
#SBATCH --account=ees260021
#SBATCH --partition=wholenode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/sim_test_chunks_%j.out
#SBATCH --error=logs/sim_test_chunks_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
# Ensemble-draw identifier (same contract as step 06): selects which searched
# run's outputs to score — the slug gains "_d{k}" for k>0.
export NYCOPT_ENSEMBLE_DRAW="${DRAW:-${NYCOPT_ENSEMBLE_DRAW:-0}}"
nycopt_pin_threads

: "${NYCOPT_REEVAL_ENSEMBLE_PRESET:?set the chunked E_test slug explicitly, e.g. etest_kn_50yr_n25000_first25ch}"
FORMULATION="${FORMULATION:-ffmp}"
SEED="${SEED:-}"
MODE="${NYCOPT_CHUNK_SIM_MODE:-mpi}"

# Wall guard: export the job's end time so ranks stop cleanly (finishing and
# flushing their current unit) instead of being killed mid-unit. Takes effect
# only when NYCOPT_CHUNK_UNIT_SECONDS is also set (the measured unit wall).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    _END_TIME="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*EndTime=\([^ ]*\).*/\1/p' | head -1)"
    if [[ -n "${_END_TIME}" && "${_END_TIME}" != "Unknown" ]]; then
        _END_EPOCH="$(date -d "${_END_TIME}" +%s 2>/dev/null || echo "")"
        if [[ -n "${_END_EPOCH}" ]]; then
            # 15-minute margin for the final flush + teardown.
            export NYCOPT_CHUNK_STOP_EPOCH="$(( _END_EPOCH - 900 ))"
        fi
    fi
fi

# High rank-per-node densities need an explicit realization batch: batch=0
# holds a whole 500-scenario model's results per rank and will OOM 128
# ranks/node. Fail at submit-shell speed, not 8 nodes deep into an OOM.
_RANKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-0}"
if [[ "${_RANKS_PER_NODE}" -gt 64 && -z "${NYCOPT_SEARCH_REALIZATION_BATCH:-}" ]]; then
    echo "ERROR: ${_RANKS_PER_NODE} ranks/node requires an explicit" \
         "NYCOPT_SEARCH_REALIZATION_BATCH (memory bound per rank)." >&2
    exit 2
fi

ARGS="--formulation ${FORMULATION}"
[[ -n "${SEED}" ]] && ARGS="${ARGS} --seed ${SEED}"

echo "=== Simulate test-ensemble chunks: formulation=${FORMULATION} mode=${MODE} "\
"ensemble=${NYCOPT_REEVAL_ENSEMBLE_PRESET} policies=${NYCOPT_CHUNK_POLICIES:-baseline} ==="

case "${MODE}" in
    single)
        python3 -m scripts.main.simulate_test_chunks ${ARGS}
        ;;
    mpi)
        # Under SLURM the rank count IS the allocation (cannot mismatch the
        # #SBATCH geometry); the NODES x RANKS product is a local fallback.
        NTASKS_MPI="${SLURM_NTASKS:-$(( ${NYCOPT_CHUNK_SIM_NODES:-4} * ${NYCOPT_CHUNK_SIM_RANKS:-16} ))}"
        echo "[sim-chunks] MPI mode, ${NTASKS_MPI} ranks"
        mpirun -np "${NTASKS_MPI}" ${NYCOPT_MPI_MCA_FLAGS} \
            python3 -m scripts.main.simulate_test_chunks ${ARGS}
        ;;
    *)
        echo "ERROR: unknown NYCOPT_CHUNK_SIM_MODE='${MODE}' (expected 'single' or 'mpi')" >&2
        exit 1
        ;;
esac

echo "=== Completed: $(date) ==="
