#!/bin/bash
# Real-simulation acceptance gate for the chunked re-eval rewrite: stage a
# mini E_test (N_theta=2 x R=2 x 10 yr -> slug etest_kn_10yr_n4, 2 chunks of 2)
# end-to-end, then re-evaluate the baseline policy twice —
#   (a) legacy path: NYCOPT_CHUNK_INCREMENTAL=0, serial, s-major contiguous;
#   (b) new path:    incremental + claim scheduling, 4 MPI ranks
# — and require the persisted reeval_raw tables to be BIT-IDENTICAL (same
# evaluate_raw calls, same batch boundaries => no FP tolerance). Complements
# the analytic-stub tests in tests/test_chunk_reeval.py with the real
# pywrdrb/evaluate_raw integration.
#
# Idempotent: the mini E_test stages once and is reused (delete
# outputs/synthetic_ensembles/etest_kn_10yr_n4* to force).
#
# Submit:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
#          workflow/supplemental/mini_etest_reeval_bitcompare.sh
#
#SBATCH --job-name=mini_reeval_bitcmp
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mini_reeval_bitcmp_%j.out
#SBATCH --error=logs/mini_reeval_bitcmp_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file optional
nycopt_pin_threads

# Mini E_test sizing (env knobs from src/etest.py). Chunk size must be a
# multiple of R so SOWs are never split.
export NYCOPT_ETEST_N_THETA=2
export NYCOPT_ETEST_R=2
export NYCOPT_ETEST_YEARS=10
export NYCOPT_ETEST_CHUNK_SIZE=2
export NYCOPT_SALINITY_ON=0
export NYCOPT_TEMPERATURE_ON=0

SLUG="etest_kn_10yr_n4"

echo "[mini-bitcmp] staging ${SLUG} (idempotent) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "outputs/synthetic_ensembles/${SLUG}/hazard_image.npz" ]]; then
    python3 -u -m scripts.main.generate_test_ensemble --variant kn
fi
for J in 000 001; do
    if [[ ! -f "outputs/synthetic_ensembles/${SLUG}__chunk${J}/predicted_inflows_mgd.hdf5" ]]; then
        python3 -u scripts/main/prep_pywrdrb_inputs.py --preset "${SLUG}__chunk${J}"
    fi
done

export NYCOPT_REEVAL_ENSEMBLE_PRESET="${SLUG}"
export NYCOPT_CHUNK_POLICIES=baseline

echo "[mini-bitcmp] (a) legacy path (incremental=0, serial, seed 91)"
NYCOPT_CHUNK_INCREMENTAL=0 NYCOPT_CHUNK_SCHEDULE=contiguous \
    python3 -u -m scripts.main.simulate_test_chunks --formulation ffmp --seed 91

echo "[mini-bitcmp] (b) new path (incremental + claim, 4 ranks, seed 92)"
NYCOPT_CHUNK_INCREMENTAL=1 NYCOPT_CHUNK_SCHEDULE=claim \
    mpirun -np 4 ${NYCOPT_MPI_MCA_FLAGS:-} \
    python3 -u -m scripts.main.simulate_test_chunks --formulation ffmp --seed 92

echo "[mini-bitcmp] comparing reeval_raw (seed_91 vs seed_92)"
python3 - <<'PY'
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, ".")
import config
from src.ensembles import get_ensemble_spec
from src.reeval_core import reeval_output_dir
from config import active_scenario_name, derive_slug

spec = get_ensemble_spec("etest_kn_10yr_n4")
dirs = [reeval_output_dir(active_scenario_name(), derive_slug("ffmp"), spec, s)
        for s in (91, 92)]

def load(d: Path) -> pd.DataFrame:
    p = d / "reeval_raw.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(d / "reeval_raw.csv.gz")
    return df.sort_values(["solution_id", "realization_id", "objective"]
                          ).reset_index(drop=True)

a, b = load(dirs[0]), load(dirs[1])
assert a.shape == b.shape and a.shape[0] > 0, (a.shape, b.shape)
assert (a["solution_id"].values == b["solution_id"].values).all()
assert (a["realization_id"].values == b["realization_id"].values).all()
assert (a["objective"].values == b["objective"].values).all()
va, vb = a["value"].to_numpy(), b["value"].to_numpy()
exact = np.array_equal(va, vb) or (
    np.array_equal(np.isnan(va), np.isnan(vb))
    and np.array_equal(va[~np.isnan(va)], vb[~np.isnan(vb)])
)
if not exact:
    diff = np.nanmax(np.abs(va - vb))
    print(f"[mini-bitcmp] FAIL: values differ (max |diff| = {diff:.3e})")
    sys.exit(1)
print(f"[mini-bitcmp] PASS: {a.shape[0]} cells bit-identical across paths.")
PY
echo "[mini-bitcmp] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
