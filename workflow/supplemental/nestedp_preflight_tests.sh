#!/bin/bash
# Test-suite preflight for the sharded-generation chain: both suites must be
# green before the 50-task shard array burns allocation on edited code.
#
#SBATCH --job-name=nestedp_pytest
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/nestedp_pytest_%j.out
#SBATCH --error=logs/nestedp_pytest_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[nestedp_pytest] scengen  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
(cd ../NYCOptimization_scenario_generation && python3 -m pytest -q)

echo "[nestedp_pytest] NYCOptimization (-m 'not slow')  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -m pytest -q -m "not slow"

echo "[nestedp_pytest] both suites green  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
