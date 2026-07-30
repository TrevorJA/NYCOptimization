"""compare_serial_vs_shard.py - Serial-vs-sharded generation cross-check (report-only).

Compares the serially generated P = 100,000 pool image (`statpool_10yr_n100000_d0`)
against the first 100,000 rows of the sharded P = 1e6 image
(`statpool_10yr_n1000000_d0`). Both were generated in the same environment era
from the same seed domain, so rows spanning four shard boundaries should be
bit-identical; any differences are reported with per-axis counts and magnitudes
(never a hard failure — this is the final documentation datum for the sharded
generation path in `nested_P_saturation.md`).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

import config  # noqa: E402
from scengen.diagnostics import load_hazard_image  # noqa: E402

SERIAL_SLUG = "statpool_10yr_n100000_d0"
SHARD_SLUG = "statpool_10yr_n1000000_d0"


def main() -> None:
    serial = load_hazard_image(config.STAGED_ENSEMBLE_DIR / SERIAL_SLUG / "hazard_image.npz")
    shard = load_hazard_image(config.STAGED_ENSEMBLE_DIR / SHARD_SLUG / "hazard_image.npz")
    axes = list(serial["hazard_axes"])
    n = len(serial["H"])
    A, B = serial["H"], shard["H"][:n]
    if list(shard["hazard_axes"]) != axes:
        shard_axes = list(shard["hazard_axes"])
        print(f"[serial_vs_shard] AXIS MISMATCH: {axes} vs {shard_axes}")
        return
    diff = np.abs(A - B)
    rows_differ = int((diff > 0).any(axis=1).sum())
    print(f"[serial_vs_shard] {n} rows compared (spanning 4 shard boundaries): "
          f"{n - rows_differ} bit-identical, {rows_differ} differ "
          f"({rows_differ / n:.3%}).")
    if rows_differ:
        rng = np.percentile(A, 99, axis=0) - np.percentile(A, 1, axis=0)
        for j, a in enumerate(axes):
            c = int((diff[:, j] > 0).sum())
            if c:
                print(f"[serial_vs_shard]   {a}: {c} rows, max diff "
                      f"{diff[:, j].max():.4g} ({diff[:, j].max() / rng[j]:.3%} "
                      f"of robust range)")


if __name__ == "__main__":
    main()
