"""robustness_threshold_anchor.py - Historic-trace anchor for the threshold diagnostics.

Recomputes the 8 whole-trace BASE objective values (weekly reliabilities,
deficit CVaR %, flood exceedance ft·days/yr, storage p5 %) for the default
FFMP policy on the persisted historic baseline simulation
(outputs/baseline/ffmp_baseline.hdf5) and caches them as JSON. Zero
simulation: the HDF5 is loaded through the same ``pywrdrb.Data()`` path that
``run_baseline.py`` scores from, and each metric is ``objective.base.compute``
— byte-identical to the per-realization metrics in the E_test re-eval cube.

The persisted ``outputs/baseline/ffmp_baseline_objectives.csv`` is in the
ANNUAL-UNIT search-objective space and cannot anchor the base-metric
satisficing thresholds; this cache is the apples-to-apples anchor. It is the
seam that keeps ``robustness_threshold_figures.py`` pywrdrb-free.

Configuration lives in supplemental_config.py (RTD_* section) — no CLI value
flags. Env:

    NYCOPT_RTD_REFRESH=1    force recompute even if the cache exists

Usage (never on a login node):
    srun --partition=shared --account=ees260021 --ntasks=1 --time=00:15:00 \\
        python3 scripts/supplemental/robustness_threshold_anchor.py
or via the wrapper:
    sbatch workflow/supplemental/robustness_threshold_diagnostics.sh
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_rtd_env()


def load_meta() -> dict:
    """Parse the baseline cube's reeval_raw_meta.json (no cube load)."""
    meta_path = scfg.RTD_REEVAL_BASELINE_DIR / "reeval_raw_meta.json"
    if not meta_path.exists():
        sys.exit(f"[rtd-anchor] missing {meta_path} — run step 05 --reeval first.")
    with open(meta_path) as f:
        return json.load(f)


def compute_historic_anchor(base_names: list) -> dict:
    """Whole-trace base metrics of the default policy on the historic trace.

    Builds the ensemble objective set from the cube's own ``base_names`` (the
    moving-measuring-stick guard: the anchor is scored for exactly the metric
    columns the cube carries, not whatever the live registry currently lists).
    """
    from src.objectives_ensemble import build_ensemble_objective_set  # noqa: E402
    from src.simulation import _load_results_from_hdf5  # noqa: E402

    if not scfg.RTD_BASELINE_HDF5.exists():
        sys.exit(f"[rtd-anchor] missing {scfg.RTD_BASELINE_HDF5} — "
                 "run scripts/main/run_baseline.py first.")

    print(f"[rtd-anchor] loading {scfg.RTD_BASELINE_HDF5}", flush=True)
    data = _load_results_from_hdf5(scfg.RTD_BASELINE_HDF5)

    objs = list(build_ensemble_objective_set(base_names))
    got = [o.base.name for o in objs]
    if got != list(base_names):
        sys.exit(f"[rtd-anchor] base-name order mismatch: {got} != {base_names}")

    anchor = {}
    for o in objs:
        val = float(o.base.compute(data))
        anchor[o.base.name] = val
        print(f"[rtd-anchor]   {o.base.name} = {val:.6f}", flush=True)
    return anchor


def main() -> None:
    refresh = os.environ.get("NYCOPT_RTD_REFRESH", "0") == "1"
    if scfg.RTD_ANCHOR_CACHE.exists() and not refresh:
        print(f"[rtd-anchor] cache exists, skipping: {scfg.RTD_ANCHOR_CACHE} "
              "(NYCOPT_RTD_REFRESH=1 to force)", flush=True)
        return

    meta = load_meta()
    base_names = meta["base_names"]
    anchor = compute_historic_anchor(base_names)

    bad = [k for k, v in anchor.items() if not (v == v)]  # NaN check
    if bad:
        sys.exit(f"[rtd-anchor] non-finite anchor values for {bad}")

    scfg.RTD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor": anchor,
        "base_names": list(base_names),
        "source_hdf5": str(scfg.RTD_BASELINE_HDF5),
        "source_mtime": scfg.RTD_BASELINE_HDF5.stat().st_mtime,
        "computed": date.today().isoformat(),
    }
    with open(scfg.RTD_ANCHOR_CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[rtd-anchor] wrote {scfg.RTD_ANCHOR_CACHE}", flush=True)


if __name__ == "__main__":
    main()
