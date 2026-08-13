#!/bin/bash
# Acceptance gate for the metadata-only E_test chunk-prefix subset
# (scripts/supplemental/make_etest_subset.py): on the mini E_test fixture
# (etest_kn_10yr_n4: N_theta=2 x R=2 x 10 yr, 2 chunks of 2), re-evaluate the
# baseline policy against
#   (a) the FULL mini pool (2 chunks, SOWs {0,1}), and
#   (b) the first1ch SUBSET (1 chunk, SOW {0}),
# IN THE SAME JOB, and require the subset's reeval_raw rows to be BIT-IDENTICAL
# to the full run's rows restricted to SOW 0. Same job/node/env => bit-identity
# is a valid gate (the cross-job FP nondeterminism caveat does not apply; cf.
# workflow/supplemental/mini_etest_reeval_bitcompare.sh, the same-job precedent).
# Identity holds by construction: both paths evaluate the identical
# (solution, chunk000) unit via the same chunk spec and batch boundaries.
#
# Also asserts the subset run's meta records exactly 1 SOW (no phantom NaN SOWs
# — the failure mode that ruled out ALLOW_PARTIAL merging on the full slug).
#
# Idempotent: reuses the staged mini fixture if present; regenerates it only if
# missing (tiny; the REAL E_test is never touched by any of this).
#
# Submit:
#   sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
#          workflow/supplemental/etest_subset_reeval_check.sh
#
#SBATCH --job-name=etest_subset_check
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/etest_subset_check_%j.out
#SBATCH --error=logs/etest_subset_check_%j.err
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env
nycopt_source_env_file required
nycopt_pin_threads

# Mini E_test sizing (only consulted if the fixture must be (re)staged).
export NYCOPT_ETEST_N_THETA=2
export NYCOPT_ETEST_R=2
export NYCOPT_ETEST_YEARS=10
export NYCOPT_ETEST_CHUNK_SIZE=2
export NYCOPT_SALINITY_ON=0
export NYCOPT_TEMPERATURE_ON=0

POOL="etest_kn_10yr_n4"
SUB="${POOL}_first1ch"

echo "[subset-check] ensuring mini fixture ${POOL} is staged (idempotent)"
if [[ ! -f "outputs/synthetic_ensembles/${POOL}/hazard_image.npz" && \
      ! -f "outputs/synthetic_ensembles/${POOL}/chunk_index.json" ]]; then
    python3 -u -m scripts.main.generate_test_ensemble --variant kn
fi
for J in 000 001; do
    if [[ ! -f "outputs/synthetic_ensembles/${POOL}__chunk${J}/predicted_inflows_mgd.hdf5" ]]; then
        python3 -u scripts/main/prep_pywrdrb_inputs.py --preset "${POOL}__chunk${J}"
    fi
done

echo "[subset-check] staging metadata-only subset ${SUB}"
python3 -u -m scripts.supplemental.make_etest_subset \
    --pool "${POOL}" --n-chunks 1 --force

export NYCOPT_CHUNK_POLICIES=baseline

echo "[subset-check] (a) FULL mini pool, serial (seed 93)"
NYCOPT_REEVAL_ENSEMBLE_PRESET="${POOL}" \
    python3 -u -m scripts.main.simulate_test_chunks --formulation ffmp --seed 93

echo "[subset-check] (b) first1ch SUBSET, serial (seed 94)"
NYCOPT_REEVAL_ENSEMBLE_PRESET="${SUB}" \
    python3 -u -m scripts.main.simulate_test_chunks --formulation ffmp --seed 94

echo "[subset-check] comparing subset rows vs full rows @ SOW 0"
POOL="${POOL}" SUB="${SUB}" python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import active_scenario_name, derive_slug
from src.ensembles import get_ensemble_spec
from src.reeval_core import reeval_output_dir

pool, sub = os.environ["POOL"], os.environ["SUB"]
scen, slug = active_scenario_name(), derive_slug("ffmp")
d_full = reeval_output_dir(scen, slug, get_ensemble_spec(pool), 93)
d_sub = reeval_output_dir(scen, slug, get_ensemble_spec(sub), 94)

def load(d: Path) -> pd.DataFrame:
    p = d / "reeval_raw.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(d / "reeval_raw.csv.gz")
    return df.sort_values(["solution_id", "sow_id", "objective"]).reset_index(drop=True)

full, subset = load(d_full), load(d_sub)

meta = json.loads((d_sub / "reeval_raw_meta.json").read_text())
assert meta["n_sow"] == 1 and meta["sow_labels"] == [0], (
    f"subset meta records n_sow={meta['n_sow']}, sow_labels={meta['sow_labels']}; "
    f"expected exactly SOW [0] — phantom SOWs would bias satisficing fractions.")
assert meta["n_realizations"] == 2 and meta["realizations_per_sow"] == 2, meta

sow0 = full[full["sow_id"] == 0].reset_index(drop=True)
assert subset.shape == sow0.shape and subset.shape[0] > 0, (subset.shape, sow0.shape)
for col in ("solution_id", "sow_id", "objective"):
    assert (subset[col].values == sow0[col].values).all(), f"key column {col} differs"
va, vb = subset["value"].to_numpy(), sow0["value"].to_numpy()
exact = np.array_equal(va, vb) or (
    np.array_equal(np.isnan(va), np.isnan(vb))
    and np.array_equal(va[~np.isnan(va)], vb[~np.isnan(vb)])
)
if not exact:
    print(f"[subset-check] FAIL: values differ (max |diff| = "
          f"{np.nanmax(np.abs(va - vb)):.3e})")
    sys.exit(1)
print(f"[subset-check] PASS: {subset.shape[0]} subset cells bit-identical to the "
      f"full run's SOW-0 cells; subset meta n_sow=1 as expected.")
PY
echo "[subset-check] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
