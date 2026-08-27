#!/bin/bash
# submit_search_memory_smoke.sh — DEV UTILITY: one-node memory + timing smoke
# of the campaign search path (N=300, realization batch 150, 128 ranks/node)
# through the unified launcher. Not part of the replication pipeline.
#
# Why: N=300 x L=10 does not fit 128 ranks/node unbatched (projected ~259
# GB/node), so the production env files set NYCOPT_SEARCH_REALIZATION_BATCH=150
# — a path that has run in step-09 re-evaluations but never inside a Borg
# search. This job runs the `smoke` MOEA config (200 NFE/island, 2 islands)
# on ONE wholenode node at the production packing density with the batch set,
# via workflow/envs/smoke_search_batch.env, and samples node memory every
# NYCOPT_MEM_SAMPLE_S seconds (default 30) into logs/mem_<jobid>_<host>.log.
#
# Read after completion:
#   * peak "used" MB in logs/mem_<jobid>_*.log        -> must be <= ~217,000 MB
#   * warm per-evaluation time from the runtime files  -> expected ~540 s
#     (173.8 s x (300/100)^0.951 x 1.09); > 650 s means the N-extrapolation or
#     the batch penalty is worse than budgeted (campaign_design.md §6)
#   * sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,AveRSS,MaxRSSNode
#
# Geometry: 127 MPI ranks = 1 controller + 2 islands x (62 workers + 1 master)
# in 128 allocated tasks (NTASKS_MPI caller override; the `smoke` config's
# island count is what Borg uses, the worker count follows the MPI world).
# ~3 evaluation waves at ~9-10 min each plus model build -> ~1 h; 2 h wall.
# Cost: ~256 SU (2 wholenode node-hours).
#
# Usage (from repo root, after steps 02 -> 04 at N=300 for fixed_probabilistic):
#   bash workflow/submit_search_memory_smoke.sh            # submit
#   bash workflow/submit_search_memory_smoke.sh --dry-run  # print the sbatch line

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" == "false" ]]; then
        "$@"
    fi
}

run sbatch \
    --partition=wholenode --nodes=1 --ntasks-per-node=128 --exclusive --time=02:00:00 \
    --job-name="smoke_search_batch" \
    --export=ALL,NYCOPT_ENV_FILE=workflow/envs/smoke_search_batch.env,FORMULATION=ffmp,RUN_SLUG=smoke_search_batch_ffmp,NTASKS_MPI=127,NYCOPT_MEM_SAMPLE_S="${NYCOPT_MEM_SAMPLE_S:-30}" \
    workflow/06_run_mmborg.sh

echo "=== Search memory smoke submitted ==="
[[ "${DRY_RUN}" == "true" ]] && echo "(dry-run — no job actually submitted)"
