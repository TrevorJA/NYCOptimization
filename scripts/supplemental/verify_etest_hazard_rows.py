"""verify_etest_hazard_rows.py - Row-level acceptance gate for the E_test
sub-window hazard image path (reference-fit cache + shard/merge dispatch).

Recomputes a few staged chunks' (realization x window) hazard rows through the
CURRENT _score_chunk kernel (i.e. with the scengen reference-fit cache) and
compares them against the corresponding rows of the production
``hazard_image_subwindows.npz`` — read-only; nothing under the staged dir is
written. Same-era comparisons must be exact; cross-era regeneration on Anvil
formally tolerates <= 1% of each axis's robust range (the
verify_shard_boundaries convention). Exits nonzero on any beyond-tolerance row.

Run under srun/sbatch (compute nodes). Env config (no CLI value flags):

    NYCOPT_ETEST_VARIANT        E_test variant (default "kn")
    NYCOPT_HAZROW_CHUNKS        comma-separated chunk indices (default "0,49")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

from src.ensembles import pool_chunk_specs, staged_ensemble_dir  # noqa: E402
from src.etest import E_TEST_VARIANT, get_etest_variant  # noqa: E402

from scripts.main.compute_etest_hazard_image import (  # noqa: E402
    SCENARIO_YEARS,
    _reference_series,
    _score_chunk,
)


def main() -> None:
    variant = get_etest_variant(E_TEST_VARIANT)
    slug = variant.slug
    out_dir = staged_ensemble_dir(slug)
    art_path = out_dir / "hazard_image_subwindows.npz"
    if not art_path.exists():
        raise FileNotFoundError(
            f"Production artifact not found: {art_path} — run "
            f"compute_etest_hazard_image (serial or sharded) first."
        )
    art = np.load(art_path, allow_pickle=True)
    H_ref, rid_ref, win_ref = art["H"], art["realization_ids"], art["window_index"]

    chunk_ids = [int(x) for x in
                 os.environ.get("NYCOPT_HAZROW_CHUNKS", "0,49").split(",")]
    chunks = pool_chunk_specs(slug)
    from src.etest import assert_staged_etest_contract

    meta = assert_staged_etest_contract(slug)
    L = int(meta.get("realization_years") or meta.get("n_years"))
    n_windows = L // SCENARIO_YEARS
    flowtype = meta.get("flowtype", "pub_nhmv10_BC_withObsScaled")
    reference_monthly, reference_daily = _reference_series(flowtype)

    n_bad = 0
    for ci in chunk_ids:
        spec, gids = chunks[ci]
        chunk_dir = staged_ensemble_dir(spec.inflow_type)
        H, rid, win, _axes = _score_chunk(
            chunk_dir, list(range(len(gids))), [int(g) for g in gids],
            n_windows, reference_monthly, reference_daily,
        )
        # Align to the production rows for this chunk's rid range.
        sel = np.isin(rid_ref, rid)
        order_ref = np.lexsort((win_ref[sel], rid_ref[sel]))
        order_new = np.lexsort((win, rid))
        ref_rows = H_ref[sel][order_ref]
        new_rows = H[order_new]
        if not np.array_equal(rid_ref[sel][order_ref], rid[order_new]):
            print(f"[hazrow] chunk {ci}: rid alignment mismatch"); n_bad += 1
            continue
        if np.array_equal(new_rows, ref_rows):
            print(f"[hazrow] chunk {ci}: EXACT ({new_rows.shape[0]} rows).")
            continue
        rng = (np.percentile(ref_rows, 99, axis=0)
               - np.percentile(ref_rows, 1, axis=0))
        rng[rng <= 0] = 1.0
        rel = np.max(np.abs(new_rows - ref_rows) / rng, axis=0)
        worst = float(rel.max())
        verdict = "within" if worst <= 0.01 else "BEYOND"
        print(f"[hazrow] chunk {ci}: not exact; worst axis diff "
              f"{worst:.3e} of robust range ({verdict} the 1% cross-era "
              f"tolerance).")
        if worst > 0.01:
            n_bad += 1
    if n_bad:
        print(f"[hazrow] FAIL: {n_bad} chunk(s) beyond tolerance.")
        sys.exit(1)
    print("[hazrow] PASS.")


if __name__ == "__main__":
    main()
