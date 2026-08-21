#!/bin/bash
#SBATCH --job-name=seasval_hazfill
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/seasval_hazfill_%j.out
#SBATCH --error=logs/seasval_hazfill_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/workflow/_common.sh"
nycopt_setup_env
for k in 0 1 2; do
  echo "### hazfill_stat_abs_10yr_n100_d${k}"
  NYCOPT_SEASVAL_SLUG=hazfill_stat_abs_10yr_n100_d${k} \
    python3 scripts/supplemental/validate_staged_seasonality.py
done
echo "ALL_HAZFILL_SEASVAL_OK"
