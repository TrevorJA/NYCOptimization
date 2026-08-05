"""verify_hazfill_d0_restoration.py - Verify the hazfill d0 selection restoration.

The 2026-08-04 15:01 mis-configured step-03 run overwrote
``hazfill_stat_abs_10yr_n100_d0``'s selection files (gage/catchment HDF5s,
hazard_image.npz, _meta.json) with a P=2,000 smoke-pool selection; the Aug-3
staged step-04 artifacts (flood / presim / predicted HDF5s) were untouched and
encode the original P=1e6 selection. After step 03 re-stages the selection from
the P=1e6 pool images (deterministic seeds -> the original selection), this
script proves the restoration end-to-end:

  1. regenerate ``presimulated_releases_mgd.hdf5`` in place from the RESTORED
     catchment inflows (STARFITReleaseEnsemblePreprocessor, force=True); then
  2. bit-compare every dataset against a pre-restoration snapshot of the Aug-3
     staged presim artifact (same-input reference -> exact equality applies).

A match proves the restored selection is the one the Aug-3 inputs were built
from (the presim map inflows -> releases is deterministic and injective for
our purposes; the forensic check, job 19666041, showed the bad selection FAILS
this comparison). On mismatch the snapshot is copied back into place so the
staged dir keeps the authoritative Aug-3 artifact, and the script exits 1.

Configuration (env only, no CLI value flags):

    NYCOPT_D0_VERIFY_SLUG        staged ensemble slug
                                 (default hazfill_stat_abs_10yr_n100_d0)
    NYCOPT_D0_PRESIM_SNAPSHOT    path to the pre-restoration copy of
                                 presimulated_releases_mgd.hdf5 (required)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from src.ensembles import (  # noqa: E402
    get_ensemble_spec,
    register_ensemble_path,
    staged_ensemble_dir,
)

SLUG = os.environ.get("NYCOPT_D0_VERIFY_SLUG", "hazfill_stat_abs_10yr_n100_d0")
SNAPSHOT = os.environ.get("NYCOPT_D0_PRESIM_SNAPSHOT", "")


def _collect(h5: h5py.File) -> dict:
    """Map every dataset path in the file to its ndarray."""
    out: dict = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj[()]

    h5.visititems(visit)
    return out


def _compare(a_path: Path, b_path: Path) -> list:
    """Return a list of human-readable differences (empty = bit-identical)."""
    diffs = []
    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
        da, db = _collect(fa), _collect(fb)
        only_a = sorted(set(da) - set(db))
        only_b = sorted(set(db) - set(da))
        if only_a:
            diffs.append(f"datasets only in regenerated: {only_a}")
        if only_b:
            diffs.append(f"datasets only in snapshot: {only_b}")
        for name in sorted(set(da) & set(db)):
            xa, xb = da[name], db[name]
            if xa.dtype != xb.dtype or xa.shape != xb.shape:
                diffs.append(
                    f"{name}: dtype/shape mismatch "
                    f"({xa.dtype}{xa.shape} vs {xb.dtype}{xb.shape})"
                )
            elif not np.array_equal(xa, xb):
                if np.issubdtype(xa.dtype, np.number):
                    worst = float(np.max(np.abs(xa.astype(np.float64)
                                                - xb.astype(np.float64))))
                    diffs.append(f"{name}: values differ (worst |diff| {worst:.3e})")
                else:
                    diffs.append(f"{name}: values differ (non-numeric)")
    return diffs


def main() -> None:
    if not SNAPSHOT:
        sys.exit("NYCOPT_D0_PRESIM_SNAPSHOT must point at the Aug-3 presim copy.")
    snapshot = Path(SNAPSHOT)
    if not snapshot.exists():
        sys.exit(f"snapshot not found: {snapshot}")

    from pywrdrb.pre.generate_presimulated_releases import (
        STARFITReleaseEnsemblePreprocessor,
    )

    spec = get_ensemble_spec(SLUG)
    register_ensemble_path(spec.inflow_type)
    flows_dir = staged_ensemble_dir(spec.inflow_type)
    presim = flows_dir / "presimulated_releases_mgd.hdf5"

    print(f"[verify-d0] slug={SLUG} flows_dir={flows_dir}", flush=True)
    print(f"[verify-d0] snapshot={snapshot}", flush=True)

    print("[verify-d0] regenerating presim from restored inflows (force=True)...",
          flush=True)
    pp = STARFITReleaseEnsemblePreprocessor(
        inflow_type=spec.inflow_type,
        realization_ids=list(spec.realization_indices),
        use_mpi=False,
        force=True,
        initial_volume_frac=config.INITIAL_VOLUME_FRAC,
    )
    pp.run()

    diffs = _compare(presim, snapshot)
    if diffs:
        for d in diffs:
            print(f"[verify-d0] DIFF {d}", flush=True)
        print("[verify-d0] restoring the Aug-3 snapshot into place.", flush=True)
        shutil.copy2(snapshot, presim)
        print(f"[verify-d0] FAIL: regenerated presim does not match the Aug-3 "
              f"artifact ({len(diffs)} difference(s)); d0 selection is NOT "
              "restored.", flush=True)
        sys.exit(1)

    print("[verify-d0] PASS: regenerated presim is bit-identical to the Aug-3 "
          "staged artifact across all datasets - d0 selection restored "
          "bit-exactly.", flush=True)


if __name__ == "__main__":
    main()
