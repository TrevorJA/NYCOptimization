#!/bin/bash
#SBATCH --job-name=fig4_composition
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fig4_composition_%j.out
#SBATCH --error=logs/fig4_composition_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/workflow/_common.sh"
nycopt_setup_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

# fixed_probabilistic stages flows only (its generator skips the hazard image,
# since no selection happens), so score it post hoc for the composition figure.
for k in 0 1 2; do
  echo "### post-hoc hazard image: fixprob_10yr_n100_d${k}"
  NYCOPT_SCENARIO_DESIGN=fixed_probabilistic NYCOPT_ENSEMBLE_DRAW=${k} \
    NYCOPT_HAZIMG_SLUG=fixprob_10yr_n100_d${k} \
    python3 -u scripts/supplemental/compute_staged_hazard_image.py
done

echo "### rendering manuscript figure 4"
python3 -u -m scripts.main.figures --figure ensemble_composition
echo "FIG4_DONE"
