"""validate_staged_seasonality.py - Staged-ensemble seasonal-alignment check (Text S2).

Verifies that a STAGED ensemble's stamped months carry the right statistical
season: the monthly climatology of the staged flows, grouped by TRUE index
month, must align with the historic record's monthly climatology at circular
shift zero. This is the check the generator-side validations cannot perform —
they score the generator's own frames, while a stamping defect lives only in
the staged artifact (the +3-month rotation this check exists to catch was
invisible to every generator-side diagnostic).

For each of the twelve circular shifts s, the staged cycle rotated by s is
correlated with the historic cycle; the best-aligned shift must be 0. The
per-month synthetic/historic mean ratio at shift 0 is reported so level
distortions (e.g. a forced population's intended wetting/drying) stay
distinguishable from rotations.

Run as a build QC step after any ensemble is staged. Exits nonzero on failure.

Configuration is via environment variables (no CLI value flags):

    NYCOPT_SEASVAL_SLUG      staged slug to validate. Default: the active
                             design's ``search_ensemble_slug(SCENARIO_ENSEMBLE_DRAW)``.
    NYCOPT_SEASVAL_MAX_REAL  realizations pooled for the cycle (default 50;
                             chunked ensembles read from the leading chunks).

Run standalone::

    NYCOPT_SEASVAL_SLUG=hazfill_stat_abs_10yr_n100_d0 \
        python scripts/supplemental/validate_staged_seasonality.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.ensembles import pool_chunk_specs, staged_ensemble_dir  # noqa: E402
from src.load.historical_flows import load_historical_flows  # noqa: E402


def staged_monthly_cycle(slug: str, max_realizations: int) -> tuple[np.ndarray, pd.Timestamp]:
    """Pooled per-true-month mean of the staged aggregate NYC inflow.

    Args:
        slug: Staged-ensemble slug (single-dir or chunked).
        max_realizations: Cap on realizations pooled across the leading chunks.

    Returns:
        ``(cycle, start)`` — the length-12 monthly-mean array (Jan..Dec) and
        the staged index's first timestamp.
    """
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
    from synhydro import Ensemble

    parts: list[pd.Series] = []
    start = None
    remaining = max_realizations
    for spec, _gids in pool_chunk_specs(slug):
        h5 = staged_ensemble_dir(spec.inflow_type) / "catchment_inflow_mgd.hdf5"
        if not h5.exists():
            raise FileNotFoundError(
                f"'{slug}' has no staged daily flows ({h5} missing); a stream-only "
                f"pool cannot be validated directly — validate a materialized "
                f"ensemble drawn from it instead."
            )
        n = min(remaining, spec.n_realizations)
        ens = Ensemble.from_hdf5(str(h5), realization_subset=list(range(n)))
        for df in ens.data_by_realization.values():
            agg = df.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
            parts.append(agg)
            if start is None:
                start = pd.Timestamp(pd.DatetimeIndex(agg.index)[0])
        remaining -= n
        if remaining <= 0:
            break
    pooled = pd.concat(parts)
    cycle = pooled.groupby(pooled.index.month).mean().reindex(range(1, 13)).to_numpy()
    return cycle, start


def historic_monthly_cycle(flowtype: str) -> np.ndarray:
    """Per-month mean of the historic aggregate NYC inflow (true dates)."""
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES

    ref = load_historical_flows(gage=False, period="full", flowtype=flowtype)
    agg = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    return agg.groupby(agg.index.month).mean().reindex(range(1, 13)).to_numpy()


def validate(slug: str, *, max_realizations: int = 50) -> bool:
    """Run the alignment check for one staged slug; print the report.

    Returns:
        True when the best-aligned circular shift is zero.
    """
    import json

    meta = json.loads((staged_ensemble_dir(slug) / "_meta.json").read_text())
    flowtype = meta.get("flowtype", "pub_nhmv10_BC_withObsScaled")

    syn, start = staged_monthly_cycle(slug, max_realizations)
    hist = historic_monthly_cycle(flowtype)

    corr = np.array([np.corrcoef(np.roll(syn, -s), hist)[0, 1] for s in range(12)])
    best = int(np.argmax(corr))

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print(f"[seasval] '{slug}': staged start {start.date()}, "
          f"stamp convention {config.ENSEMBLE_START_DATE}")
    print(f"[seasval] correlation by circular shift (months): "
          f"{np.round(corr, 3).tolist()}")
    print("[seasval] per-month synthetic/historic mean ratio at shift 0:")
    for m, s, h in zip(months, syn, hist):
        print(f"[seasval]   {m}: {s / h:6.3f}")
    if best != 0:
        print(f"[seasval] FAIL: staged cycle best aligns with the historic cycle "
              f"at a {best}-month rotation — the stamped months carry the wrong "
              f"statistical season.")
        return False
    print("[seasval] OK: best-aligned shift is 0 (stamped months carry the "
          "right season).")
    return True


def main() -> None:
    slug = os.environ.get("NYCOPT_SEASVAL_SLUG", "").strip() or (
        config.ACTIVE_SCENARIO_DESIGN.search_ensemble_slug(config.SCENARIO_ENSEMBLE_DRAW)
    )
    max_real = int(os.environ.get("NYCOPT_SEASVAL_MAX_REAL", "50"))
    if not validate(slug, max_realizations=max_real):
        sys.exit(1)


if __name__ == "__main__":
    main()
