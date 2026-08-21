#!/bin/bash
# Step 13: Render the MAIN-MANUSCRIPT figure sequence.
#
# The single entry point for every polished figure that appears in the
# manuscript body. Two stages:
#
#   1. Compute caches that are too expensive to rebuild per render.
#      Currently just the flow-duration-curve cache behind Figure 3 panel (d),
#      which reads ~11 GB of E_test daily flows across 50 staged chunks.
#      Skipped when the cache already exists (NYCOPT_FIG_REFRESH=1 forces it).
#   2. Render the manuscript tier from the unified registry
#      (src/figures/registry.py) through scripts/main/figures.py; figures
#      whose data needs are absent are skipped with a message.
#
# SI-tier figures render in step 14 (same driver, --tier si); off-pipeline
# diagnostic figures keep their own drivers:
#   workflow/supplemental/si_figures_design.sh   (pre-campaign, design support)
#   workflow/supplemental/si_figures_results.sh  (needs campaign outputs)
#
# Requires:
#   * the staged campaign E_test (workflow step 12) with forcing_profiles.npz
#     and its daily chunk directories
#   * the sibling CMIP6_multimodel_streamflow repo (change-factor table +
#     per-run daily pywrdrb inputs)
#
# Everything comes from the environment — no positional args, no value flags:
#   NYCOPT_FIG_REFRESH        1 rebuilds the FDC cache even if present
#   NYCOPT_FIG_ENVELOPE_PCTL  envelope definition for Fig 3 (c)/(d), default "0,100"
#                             e.g. "5,95" for a trimmed band
#   NYCOPT_FIG_STRIDE         keep every Nth E_test realization in the cache
#                             (default 1 = all; >1 is for smoke runs only)
#   FIGURES                   optional comma-separated subset of the sequence
#
# Submit (from repo root):
#   sbatch workflow/13_main_figures.sh
#   sbatch --export=ALL,NYCOPT_FIG_REFRESH=1 workflow/13_main_figures.sh
#
# Sizing: the cache build is single-core HDF5 I/O over ~11 GB (tens of minutes
# on Lustre); rendering is seconds.
#
#SBATCH --job-name=main_figures
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/main_figures_%j.out
#SBATCH --error=logs/main_figures_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

FIG_REFRESH="${NYCOPT_FIG_REFRESH:-0}"
FIG_STRIDE="${NYCOPT_FIG_STRIDE:-1}"
FIGURES="${FIGURES:-}"

CACHE="$(python3 -c 'from scripts.main.forcing_fdc_cache import DEFAULT_CACHE; print(DEFAULT_CACHE)')"

echo "[main_figures] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[main_figures] envelope=${NYCOPT_FIG_ENVELOPE_PCTL:-0,100} refresh=${FIG_REFRESH} stride=${FIG_STRIDE}"

if [[ "${FIG_REFRESH}" == "1" || ! -f "${CACHE}" ]]; then
    echo "[main_figures] building FDC cache -> ${CACHE}"
    python3 -u -m scripts.main.forcing_fdc_cache --stride "${FIG_STRIDE}"
else
    echo "[main_figures] FDC cache present, reusing: ${CACHE}"
fi

if [[ -n "${FIGURES}" ]]; then
    ARGS=""
    for f in ${FIGURES//,/ }; do ARGS="${ARGS} --figure ${f}"; done
    # shellcheck disable=SC2086
    python3 -u -m scripts.main.figures ${ARGS}
else
    python3 -u -m scripts.main.figures --tier manuscript
fi

echo "[main_figures] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
