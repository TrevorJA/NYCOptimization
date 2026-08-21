"""One-off diagnosis of the merge-job boundary-verification mismatches.

Regenerates a set of realizations of the merged P=1e6 pool image under the
current BLAS thread environment and reports, per index, whether the row matches
the staged image exactly (with per-axis diffs when not). Also regenerates the
last shard's final partial block as ONE batch and compares against the
single-realization regeneration of the same indices — an in-process test of
batch-composition (partition) invariance that isolates it from cross-job
environment effects.

Report-only: never exits nonzero. Run once per OMP setting (the BLAS thread
count is fixed at process start).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path("/home/x-tamestoy/Research/DRB/Pywr-DRB/NYCOptimization")
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

import config
from scengen.diagnostics import load_hazard_image
from scengen.forcing_ensemble import ForcingEnsembleConfig
from scengen.hazard_filling import daily_to_monthly
from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
from src.ensemble_generation import (
    Ensemble,
    _disaggregate_fill_inflow,
    _generate_profile_monthly,
    _hazard_block,
    _prepare_generators,
)
from src.load.historical_flows import load_historical_flows

POOL_SLUG = "statpool_10yr_n1000000_d0"
OMP = os.environ.get("OMP_NUM_THREADS", "?")

# The three merge-job mismatches, the five merge-job exact matches, and a
# spread of mid-shard controls.
SINGLES = [0, 19999, 20000, 499999, 500000, 979999, 980000, 999999,
           123456, 250007, 333333, 411111, 555555, 640001, 707070, 808080,
           870001, 930303, 985432, 990001]
# The last shard's final partial block, regenerated as one batch (matches the
# shard job's block composition [999968, 1000000)).
BATCH = list(range(999968, 1000000))


def main() -> None:
    staged = config.STAGED_ENSEMBLE_DIR / POOL_SLUG
    meta = json.loads((staged / "_meta.json").read_text())
    img = load_hazard_image(staged / "hazard_image.npz")
    H, axes = img["H"], list(img["hazard_axes"])

    cfg = ForcingEnsembleConfig(
        root_seed=int(meta["root_seed"]),
        seed_domain=meta.get("seed_domain"),
        n_forcing_profiles=int(meta["n_realizations"]),
        realizations_per_profile=1,
        realization_years=int(meta["realization_years"]),
        population="stationary",
        theta_sampler="iid",
        compute_hazard_image=True,
        flowtype=meta["flowtype"],
        output_dir=staged,
        store_daily=False,
        start_date=meta["start_date"],
    )
    print(f"[diag] OMP={OMP}: fitting generators...")
    setup = _prepare_generators(cfg)
    ref = load_historical_flows(gage=False, period="full", flowtype=cfg.flowtype)
    ref_daily = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    reference_monthly = daily_to_monthly(ref_daily, agg="mean")
    reference_daily = ref_daily.to_numpy(dtype=float)

    def hazard_rows(ks: list[int]) -> np.ndarray:
        monthly, md = {}, None
        for k in ks:
            m, md_k = _generate_profile_monthly(setup, cfg, k)
            monthly.update(m)
            md = md or md_k
        _, inflow, _ = _disaggregate_fill_inflow(
            Ensemble(monthly, metadata=md), nowak=setup.nowak, kdes=setup.kdes,
            root_seed=cfg.root_seed, start_date=cfg.start_date,
        )
        rows, row_axes = _hazard_block(
            inflow, sorted(inflow), DEFAULT_NYC_INFLOW_NODES,
            reference_monthly, reference_daily,
            n_years=cfg.realization_years,
        )
        assert row_axes == axes
        return rows

    print(f"[diag] --- singles (block-of-1) vs image, OMP={OMP} ---")
    n_bad = 0
    single_rows = {}
    for k in SINGLES:
        row = hazard_rows([k])[0]
        single_rows[k] = row
        if np.array_equal(row, H[k]):
            print(f"[diag] k={k}: EXACT")
        else:
            n_bad += 1
            d = row - H[k]
            worst = int(np.argmax(np.abs(d)))
            print(f"[diag] k={k}: MISMATCH max|d|={np.max(np.abs(d)):.4g} "
                  f"on {axes[worst]}; per-axis "
                  + " ".join(f"{a}={v:+.3g}" for a, v in zip(axes, d) if v != 0))
    print(f"[diag] singles: {n_bad}/{len(SINGLES)} mismatch image at OMP={OMP}")

    print(f"[diag] --- batch [999968,1e6) as one 32-block, OMP={OMP} ---")
    batch = hazard_rows(BATCH)
    img_eq = sum(bool(np.array_equal(batch[i], H[k])) for i, k in enumerate(BATCH))
    print(f"[diag] batch vs image: {img_eq}/{len(BATCH)} rows exact")
    for i, k in enumerate(BATCH):
        if k in single_rows:
            same = np.array_equal(batch[i], single_rows[k])
            print(f"[diag] batch-vs-single k={k}: "
                  f"{'IDENTICAL' if same else 'DIFFER (partition-dependence!)'}")
    print("[diag] done.")


if __name__ == "__main__":
    main()
