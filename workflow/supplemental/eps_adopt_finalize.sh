#!/bin/bash
# Finalize the 2026-08-12 epsilon adoption: write the preserved re-filtered
# reference-set copies ({slug}_merged_eps20260812.set — originals untouched)
# and run the epsilon/tau-touching test files against the edited registry.
# See scripts/supplemental/write_refiltered_sets.py and the adoption record
# in docs/notes/methods/epsilon_calibration_experiment.md.
#
# Submit from the repo root:
#   sbatch workflow/supplemental/eps_adopt_finalize.sh
#
#SBATCH --job-name=eps_adopt_finalize
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/eps_adopt_finalize_%j.out
#SBATCH --error=logs/eps_adopt_finalize_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

PYTHONUNBUFFERED=1 python3 scripts/supplemental/write_refiltered_sets.py \
    --slug ffmp_obj8 --suffix eps20260812

mpirun -np 1 python -m pytest \
    tests/test_epsilon_calibration.py \
    tests/test_objectives_ensemble.py \
    tests/test_robustness.py \
    tests/test_regret_tolerance_diagnostics.py \
    tests/test_compare_designs.py \
    tests/test_ensemble_simulation.py \
    tests/test_terminology.py -q

echo "=== Completed: $(date) ==="
