"""verify_shard_boundaries.py - Bit-identity check of sharded pool generation.

Sharded candidate-pool generation rests on the §3.4 determinism contract:
realization k is fully determined by a child stream keyed to its GLOBAL index,
invariant to how the index range is partitioned. This verifies the contract on
the merged image at its riskiest points — the shard boundaries — by regenerating
a handful of realizations under a THIRD partition (single-realization blocks,
which also exercises hazard-block-size invariance) through the exact same code
path the generator used, and asserting exact row equality with the staged image.

All configuration is via environment variables (no CLI value flags):

    NYCOPT_NESTEDP_POOL_SLUG      the merged pool's staged slug (required)
    NYCOPT_ENSEMBLE_SHARD_COUNT   shard count used at generation (default 50)

Exits nonzero on any mismatch — the ladder must not run on a broken image.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

import config  # noqa: E402
from scengen.diagnostics import load_hazard_image  # noqa: E402
from scengen.forcing_ensemble import ForcingEnsembleConfig  # noqa: E402
from scengen.hazard_filling import daily_to_monthly  # noqa: E402
from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES  # noqa: E402
from src.ensemble_generation import (  # noqa: E402
    Ensemble,
    _disaggregate_fill_inflow,
    _generate_profile_monthly,
    _hazard_block,
    _prepare_generators,
    shard_profile_range,
)
from src.load.historical_flows import load_historical_flows  # noqa: E402

POOL_SLUG = os.environ["NYCOPT_NESTEDP_POOL_SLUG"]
SHARD_COUNT = int(os.environ.get("NYCOPT_ENSEMBLE_SHARD_COUNT", "50"))


def _check_indices(n: int) -> list[int]:
    """Global indices to regenerate: both sides of a few shard boundaries + ends."""
    bounds = [shard_profile_range(n, SHARD_COUNT, i)[0] for i in (1, SHARD_COUNT // 2,
                                                                  SHARD_COUNT - 1)]
    ks = {0, n - 1}
    for b in bounds:
        ks.update((b - 1, b))
    return sorted(ks)


def main() -> None:
    """Regenerate boundary realizations and assert exact image-row equality."""
    staged = config.STAGED_ENSEMBLE_DIR / POOL_SLUG
    meta = json.loads((staged / "_meta.json").read_text())
    if meta["population"] != "stationary":
        raise SystemExit(f"[verify_shards] pool '{POOL_SLUG}' is not stationary.")
    img = load_hazard_image(staged / "hazard_image.npz")
    H, axes = img["H"], list(img["hazard_axes"])
    n = int(meta["n_realizations"])

    cfg = ForcingEnsembleConfig(
        root_seed=int(meta["root_seed"]),
        seed_domain=meta.get("seed_domain"),
        n_forcing_profiles=n,
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
    print(f"[verify_shards] fitting generators for '{POOL_SLUG}' "
          f"(N={n}, {SHARD_COUNT} shards)...")
    setup = _prepare_generators(cfg)
    ref = load_historical_flows(gage=False, period="full", flowtype=cfg.flowtype)
    ref_daily = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    reference_monthly = daily_to_monthly(ref_daily, agg="mean")
    reference_daily = ref_daily.to_numpy(dtype=float)

    bad = []
    for k in _check_indices(n):
        monthly, md = _generate_profile_monthly(setup, cfg, k)
        _, inflow, _ = _disaggregate_fill_inflow(
            Ensemble(monthly, metadata=md), nowak=setup.nowak, kdes=setup.kdes,
            root_seed=cfg.root_seed, start_date=cfg.start_date,
        )
        row, row_axes = _hazard_block(
            inflow, [k], DEFAULT_NYC_INFLOW_NODES, reference_monthly, reference_daily,
        )
        if row_axes != axes or not np.array_equal(row[0], H[k]):
            bad.append((k, np.max(np.abs(row[0] - H[k]))))
            print(f"[verify_shards] MISMATCH at k={k}: max abs diff {bad[-1][1]:.3e}")
        else:
            print(f"[verify_shards] k={k}: exact match.")
    if bad:
        raise SystemExit(
            f"[verify_shards] FAILED at {len(bad)} of {len(_check_indices(n))} "
            f"indices — the sharded image is not partition-invariant; do not use it."
        )
    print(f"[verify_shards] OK: all {len(_check_indices(n))} regenerated rows "
          f"(shard boundaries + ends) match the staged image exactly.")


if __name__ == "__main__":
    main()
