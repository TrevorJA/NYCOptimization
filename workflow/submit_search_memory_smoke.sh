#!/bin/bash
# submit_search_memory_smoke.sh — DEV UTILITY: one-node memory + timing smoke
# of the batched N=300 search path (realization batch 150, 128 ranks/node,
# `smoke` MOEA config) via workflow/envs/smoke_search_batch.env, sampling node
# memory every NYCOPT_MEM_SAMPLE_S seconds (default 30) into
# logs/mem_<jobid>_<host>.log. Not part of the replication pipeline.
#
# Read after completion: peak "used" MB in logs/mem_<jobid>_*.log (must be
# <= ~217,000 MB) and the warm per-evaluation time from the runtime files
# (expected ~540 s; campaign_design.md §6).
#
# Usage (from repo root, after steps 02 -> 04 at N=300 for monte_carlo):
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
