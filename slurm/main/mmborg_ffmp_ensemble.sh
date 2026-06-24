#!/bin/bash
# mmborg_ffmp_ensemble.sh — MM Borg run for standard FFMP (24 DVs) over a
# MULTI-REALIZATION search ENSEMBLE (e.g. hazard_filling, hazard_filling_absolute,
# fixed_probabilistic_short). Identical to mmborg_ffmp.sh except the pre-flight
# requires an ENSEMBLE search spec (is_ensemble=True) instead of the historic
# single trace. The scenario design comes from NYCOPT_SCENARIO_DESIGN in the env
# file; algorithm settings (islands/NFE/runtime-freq) come from the active MOEA
# config (NYCOPT_MOEA_CONFIG). 165 ranks: 5 nodes x 33 tasks.
#
# Submit (pilot, 5k NFE):
#   sbatch --time=03:00:00 \
#     --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7_hazfill_pilot.env \
#     slurm/main/mmborg_ffmp_ensemble.sh
# Multiple seeds:  add --array=1-N at submission.
#
#SBATCH --job-name=mmborg_ffmp_ens
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=33
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/mmborg_ffmp_ens_seed%a_%A.out
#SBATCH --error=logs/mmborg_ffmp_ens_seed%a_%A.err
#SBATCH --array=1

set -euo pipefail

# Algorithm settings (islands/NFE/runtime-freq) come from the active MOEA config
# (NYCOPT_MOEA_CONFIG in the env file), not from this script. The scenario design
# comes from NYCOPT_SCENARIO_DESIGN. _common.sh reads both back from config.
FORMULATION="ffmp"
SEED="${SLURM_ARRAY_TASK_ID:-1}"
DEBUG_SIM=false
CHECKPOINT=false

# No default env file: an ensemble design MUST be selected explicitly.
export NYCOPT_ENV_FILE="${NYCOPT_ENV_FILE:?set NYCOPT_ENV_FILE to an ensemble env, e.g. slurm/envs/ffmp_obj7_hazfill_pilot.env}"

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/slurm/main/_common.sh"

# ---- Pre-flight: verify the problem definition before burning an allocation. ----
# Runs on the compute node (module/venv already loaded by _common.sh), never on
# the login node. Aborts the job if the spec is not an ensemble or objs != 7.
echo "=== Pre-flight verification ==="
python3 -c "
import config
from src.formulations import get_objective_set, get_n_vars
obj = get_objective_set()
spec = config.SEARCH_ENSEMBLE_SPEC
print('Scenario design :', config.active_scenario_name())
print('Search ensemble :', None if spec is None else spec.preset_name,
      '| is_ensemble =', None if spec is None else spec.is_ensemble)
print('Salinity LSTM   :', config.INCLUDE_SALINITY_MODEL)
print('Temp LSTM       :', config.INCLUDE_TEMPERATURE_MODEL)
print('Formulation     : ffmp |  n_vars =', get_n_vars('ffmp'))
print('n_objs          :', obj.n_objs)
print('Objectives      :', obj.names)
print('Epsilons        :', obj.epsilons)
assert spec is not None and spec.is_ensemble is True, 'expected an ENSEMBLE search spec; is it staged?'
assert obj.n_objs == 7, f'expected 7 objectives, got {obj.n_objs}'
assert config.INCLUDE_SALINITY_MODEL is False, 'salinity LSTM should be OFF for the 7 baseline objectives'
print('Pre-flight OK.')
"

ARGS="--seed ${SEED} --formulation ${FORMULATION}"
[[ "${CHECKPOINT}" == "true" ]] && ARGS="${ARGS} --checkpoint"

echo "=== Launching MM-Borg ensemble (${SCENARIO}/${RUN_SLUG}, seed=${SEED}) ==="
mpirun -np ${NTASKS_MPI} python3 -u src/mmborg_cli.py ${ARGS}
echo "=== Completed: $(date) ==="
