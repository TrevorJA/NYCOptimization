#!/bin/bash
# E_test demand-utilization axis, metadata only (no streamflow, no simulation):
#   1. scripts/supplemental/demand_utilization_calibration.py — tabulates u(k) for
#      the NYC utilization axis once from Pywr-DRB's extrapolation preprocessor
#      (outputs/diagnostics/demand_utilization/, SI figure S12); deterministic,
#      so re-running it is a no-op in substance.
#   2. scripts/supplemental/append_demand_utilization.py --variant VARIANT — draws
#      the per-SOW utilization column u from the reserved `{seed_domain}:demand`
#      stream, maps it to the demand scale k(u) through the calibration table and
#      writes it into the staged E_test's forcing_profiles.npz, every chunk's
#      _meta.json and demand_utilization_nominal.csv. Idempotent: a slug that
#      already carries the keys is refused unless NYCOPT_FORCE=1.
#
# Runs after staging/etest_merge.sh (or on an E_test staged before the axis existed)
# and before staging/etest_prep_chunks.sh / staging/etest_stage_nyc_demand.sh, which read
# the nominal table when they stage each chunk's scaled demand series. Seconds.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
#          workflow/supplemental/staging/etest_demand_utilization.sh
#
#SBATCH --job-name=etest_demand_u
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/etest_demand_u_%j.out
#SBATCH --error=logs/etest_demand_u_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

VARIANT="${NYCOPT_ETEST_VARIANT:-kn}"

echo "[etest_demand_u] calibration  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/demand_utilization_calibration.py
echo "[etest_demand_u] append variant=${VARIANT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u scripts/supplemental/append_demand_utilization.py --variant "${VARIANT}"
echo "[etest_demand_u] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
