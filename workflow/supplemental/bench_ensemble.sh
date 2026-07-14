#!/bin/bash
# bench_ensemble.sh — wall-clock benchmark for an ensemble simulation eval.
#
# Calls scripts/supplemental/bench_ensemble_eval.py inside a SLURM allocation
# so the measurement reflects compute-node behavior (login-node bench is
# forbidden: >3 min runs on the home node violate cluster etiquette).
#
# The active search ensemble comes from the scenario design
# (NYCOPT_SCENARIO_DESIGN), sourced from an env file under workflow/envs/.
# Use an ensemble-based design (e.g. fixed_probabilistic) — historic is
# single-trace.
#
# Usage (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/<preset>.env \
#          workflow/supplemental/bench_ensemble.sh
#   sbatch --export=ALL,NYCOPT_ENV_FILE=...,N_EVALS=3 workflow/supplemental/bench_ensemble.sh
#
#SBATCH --job-name=bench_ensemble
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_ensemble_%j.out
#SBATCH --error=logs/bench_ensemble_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

N_EVALS="${N_EVALS:-2}"
FORMULATION="${FORMULATION:-ffmp}"

echo "=== bench_ensemble: scenario=${NYCOPT_SCENARIO_DESIGN:-<default>} formulation=${FORMULATION} n_evals=${N_EVALS} ==="
echo "    started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/bench_ensemble_eval.py \
    --formulation "${FORMULATION}" \
    --n-evals "${N_EVALS}"
echo "=== completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
