#!/bin/bash
#SBATCH --job-name=hfm_diag
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/hf_design_metrics_%j.out
#SBATCH --error=logs/hf_design_metrics_%j.err

# Design metrics of the Hazard Filling (HF) search ensemble
# (docs/notes/methods/hf_design_metrics.md; SI Text S4). Zero simulation: on
# hazard images alone, replays the HF selection from its recorded seed, solves
# the exact minimum-total-displacement assignment (certified) for the gap of
# the sequential rule, and scores coverage (minimax distance, per-axis KS to
# uniform), diversity (MST edge lengths), range (per-axis span), and the
# nearest-member redistribution with its effective sample size for the HF
# ensemble, its Latin hypercube targets, the MC ensemble, random N-subsets of
# the candidate ensemble, and the historical 10-year windows. Figures
# regenerate from tables.
#
# Prerequisites:
#   outputs/synthetic_ensembles/statpool_10yr_n1000000_d{0,1,2}/hazard_image.npz
#     (or the HF images, which embed the candidate H)
#   outputs/synthetic_ensembles/hazfill_stat_abs_10yr_n300_d{0,1,2}/{hazard_image.npz,_meta.json}
#   outputs/synthetic_ensembles/fixprob_10yr_n300_d{0,1,2}/hazard_image.npz (skipped if absent)
#   outputs/supplemental/historic_hazard_windows/hazard_windows_10yr.npz (skipped if absent)
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_size_diagnostics.env \
#          workflow/supplemental/hf_design_metrics.sh
# Switches: NYCOPT_HFM_SMOKE=1 (local P=300 / N=40 images, smoke_ prefix),
# NYCOPT_HFM_FIGURES_ONLY=1 (redraw from persisted tables). Settings in
# supplemental_config.py (HFM_ section); no CLI flags.
#
# Outputs: outputs/supplemental/hf_design_metrics/{tables,figures}.
# Sizing: one 1e6 x 8 image plus its cKDTree resident (< 1 GB); each of the
# ~100 random references is one 1e6-point query (about 0.4 s); a draw runs in
# about a minute on 8 threads.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[hfm] start: $(date -u +%Y-%m-%dT%H:%M:%SZ) (smoke=${NYCOPT_HFM_SMOKE:-0}, figures_only=${NYCOPT_HFM_FIGURES_ONLY:-0})"
if [[ "${NYCOPT_HFM_FIGURES_ONLY:-0}" != "1" ]]; then
    python3 -u scripts/supplemental/hf_design_metrics_run.py
fi
python3 -u scripts/supplemental/hf_design_metrics_figures.py
echo "[hfm] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
