"""compute_historic_hazard_windows.py - Hazard coordinates of the historical record's windows.

Scores every disjoint ``SCENARIO_YEARS``-year window of the reconstructed historical
record on the 8-axis candidate hazard image under the exact pool convention: SSI-6
controlling-event run-theory dry axes on the window's monthly aggregate NYC inflow,
POT wet axes on its daily series with the leading ``METRIC_EXCLUSION_MONTHS`` cut by
date, and the SSI fit / POT threshold / reference mean fitted once on the FULL
historical record. Window coordinates are therefore commensurable with the
candidate-pool images, the realized search ensembles, and the E_test sub-window
image. Each window is scored independently, exactly as a pool scenario would be, so a
drought event straddling a window boundary is truncated by it (the same convention
every ensemble scenario lives under).

Windows anchor at the month scenarios start in (``config.ENSEMBLE_START_DATE``,
December): the record is cut into windows exactly the way every synthetic scenario
window is cut — including the trailing-partial trim, so each window's scored span is
[Jun 1 year 1, May 31 year L] — and the historic marker layer and the ensemble
layers occupy one commensurable hazard space.

Feeds the historical-record marker layer of the ensemble-composition figure
(manuscript figure 4; ``src/plotting/ensemble_composition.py``).

Writes a cached ``hazard_windows_{L}yr.npz`` under
``outputs/supplemental/historic_hazard_windows/`` carrying its anchor month and
reference start as provenance; the loader recomputes when they do not match the
current convention. Pass ``force=True`` to recompute unconditionally.

Run standalone::

    python scripts/main/compute_historic_hazard_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from config import METRIC_EXCLUSION_MONTHS, SCENARIO_YEARS  # noqa: E402
from scripts.main.compute_etest_hazard_image import _reference_series  # noqa: E402
from src.load.historical_flows import load_historical_flows  # noqa: E402

#: pywrdrb inflow-dataset key of the reconstructed record (matches the staged pools).
FLOWTYPE = "pub_nhmv10_BC_withObsScaled"

#: Calendar month anchoring the disjoint windows: the month every scenario
#: window starts in (see the module docstring).
WINDOW_ANCHOR_MONTH = pd.Timestamp(config.ENSEMBLE_START_DATE).month

CACHE_PATH = (
    config.OUTPUTS_DIR / "supplemental" / "historic_hazard_windows"
    / f"hazard_windows_{SCENARIO_YEARS}yr.npz"
)


def _first_anchor(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """The first ``WINDOW_ANCHOR_MONTH`` 1 on or after the record start."""
    start = idx[0]
    if start.month == WINDOW_ANCHOR_MONTH and start.day == 1:
        return start
    year = start.year + (1 if (start.month, start.day) > (WINDOW_ANCHOR_MONTH, 1) else 0)
    return pd.Timestamp(year=year, month=WINDOW_ANCHOR_MONTH, day=1)


def historic_hazard_windows(
    flowtype: str = FLOWTYPE, *, force: bool = False
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Hazard coordinates of the record's disjoint 10-yr windows, cached.

    Args:
        flowtype: pywrdrb inflow-dataset key of the historical reconstruction.
        force: Recompute even when the cache file exists.

    Returns:
        ``(H, hazard_axes, window_starts)`` with one row of ``H`` per disjoint
        ``SCENARIO_YEARS``-year window, in chronological order.
    """
    from scengen.hazard_filling import daily_to_monthly
    from scengen.hazard_metrics import (
        _REFERENCE_START,
        _SCENARIO_STAMP_START,
        DEFAULT_NYC_INFLOW_NODES,
        compute_candidate_hazard_image,
    )

    if CACHE_PATH.exists() and not force:
        with np.load(CACHE_PATH, allow_pickle=True) as z:
            # Convention provenance: a cache from another anchor/reference/
            # stamp convention (or predating provenance) silently recomputes.
            stale = (
                "anchor_month" not in z
                or int(z["anchor_month"]) != WINDOW_ANCHOR_MONTH
                or str(z["reference_start"]) != _REFERENCE_START
                or "scenario_stamp_start" not in z
                or str(z["scenario_stamp_start"]) != _SCENARIO_STAMP_START
            )
            if not stale:
                return (
                    z["H"],
                    [str(a) for a in z["hazard_axes"]],
                    pd.DatetimeIndex([str(s) for s in z["window_starts"]]),
                )
        print(f"[hist-hazard] cache {CACHE_PATH.name} predates the current date "
              f"convention; recomputing.")

    reference_monthly, reference_daily = _reference_series(flowtype)
    flows = load_historical_flows(gage=False, period="full", flowtype=flowtype)
    agg = flows.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    idx = pd.DatetimeIndex(agg.index)

    anchor = _first_anchor(idx)
    end = idx[-1]
    starts: list[pd.Timestamp] = []
    k = 0
    while anchor + pd.DateOffset(years=SCENARIO_YEARS * (k + 1)) - pd.Timedelta(days=1) <= end:
        starts.append(anchor + pd.DateOffset(years=SCENARIO_YEARS * k))
        k += 1
    if not starts:
        raise ValueError(
            f"Historical record {idx[0].date()}..{end.date()} holds no complete "
            f"{SCENARIO_YEARS}-yr anchor-aligned window."
        )

    rows, axes = [], []
    for w0 in starts:
        cutoff = w0 + pd.DateOffset(months=METRIC_EXCLUSION_MONTHS)
        # Pool convention: the scored window ends with the last complete FFMP
        # year, [cutoff, cutoff + (SCENARIO_YEARS - 1) yr) — the trailing
        # Jun-Nov partial is cut, exactly as _hazard_block cuts it.
        metric_end = cutoff + pd.DateOffset(years=SCENARIO_YEARS - 1)
        in_win = (idx >= w0) & (idx < metric_end)
        wet_cut = int(((idx >= w0) & (idx < cutoff)).sum())
        w_daily = agg.loc[in_win]
        H_win, axes = compute_candidate_hazard_image(
            np.asarray(daily_to_monthly(w_daily, agg="mean"), dtype=float)[None, :],
            w_daily.to_numpy(dtype=float)[None, :],
            reference_monthly, reference_daily, wet_exclusion_days=wet_cut,
        )
        rows.append(H_win[0])

    H = np.vstack(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH,
        H=H,
        hazard_axes=np.asarray(axes, dtype=object),
        window_starts=np.asarray([str(s.date()) for s in starts], dtype=object),
        window_years=np.asarray(SCENARIO_YEARS),
        exclusion_months=np.asarray(METRIC_EXCLUSION_MONTHS),
        flowtype=np.asarray(flowtype, dtype=object),
        anchor_month=np.asarray(WINDOW_ANCHOR_MONTH),
        reference_start=np.asarray(_REFERENCE_START, dtype=object),
        scenario_stamp_start=np.asarray(_SCENARIO_STAMP_START, dtype=object),
    )
    print(f"[hist-hazard] wrote {CACHE_PATH} ({H.shape[0]} windows x {H.shape[1]} axes; "
          f"{starts[0].date()} .. {starts[-1].date()} starts).")
    return H, list(axes), pd.DatetimeIndex(starts)


if __name__ == "__main__":
    historic_hazard_windows(force=True)
