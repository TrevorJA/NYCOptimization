#!/bin/bash
# Step 4: Format a generated/subsampled ensemble into the pywrdrb-ingestible
# HDF5 inputs the trimmed optimization model reads, under
# outputs/synthetic_ensembles/{inflow_type}/:
#   catchment_inflow_with_flood_nodes_mgd.hdf5   (FlowEnsemble, flood ops on)
#   presimulated_releases_mgd.hdf5 (+ _metadata.json)   (trimmed-model releases)
#   predicted_inflows_mgd.hdf5                   (Montague/Trenton forecasts)
#
# The ensemble is the active scenario design's search ensemble
# (config.SEARCH_ENSEMBLE_SPEC); the base catchment_inflow_mgd.hdf5 must
# already exist (step 02 generator, + step 03 subsample for hazard-filling
# designs). Pass `--preset NAME` to stage an arbitrary ensemble instead (e.g.
# the held-out re-eval ensemble). Per-realization work is distributed across
# MPI ranks automatically.
#
# Submit (from repo root):
#   sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling \
#          workflow/04_prep_pywrdrb_inputs.sh
#
#SBATCH --job-name=prep_pywrdrb_inputs
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_pywrdrb_inputs_%j.out
#SBATCH --error=logs/prep_pywrdrb_inputs_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
# Optional env file — the design can also come from --export.
nycopt_source_env_file optional
# One BLAS thread per rank: each rank runs a pywrdrb simulation, so avoid
# oversubscribing the node.
nycopt_pin_threads

NTASKS="${SLURM_NTASKS:-1}"
echo "[prep_pywrdrb_inputs] design=${NYCOPT_SCENARIO_DESIGN:-<default>} ranks=${NTASKS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mpirun -np "${NTASKS}" python3 -u scripts/main/prep_pywrdrb_inputs.py "$@"
echo "[prep_pywrdrb_inputs] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
