"""historical_flows.py - Load pywrdrb-shipped historical streamflow datasets.

Thin adapter around pywrdrb's path navigator. Used by Step 1 (Kirsch-Nowak
ensemble generation) to load the ``pub_nhmv10_BC_withObsScaled`` reconstruction
into pandas DataFrames suitable for SynHydro's ``KirschGenerator.preprocessing()``.

Mirrors the loader in
``../StochasticExploratoryExperiment/methods/load.py::load_baseline_historical_flow``,
trimmed to just what NYCOpt needs.
"""

from __future__ import annotations

import pandas as pd


_FLOWTYPE_DEFAULT = "pub_nhmv10_BC_withObsScaled"
_BASELINE_START = "1980-01-01"
_BASELINE_END = "2019-12-31"


def load_historical_flows(
    *,
    gage: bool = True,
    period: str = "full",
    flowtype: str = _FLOWTYPE_DEFAULT,
) -> pd.DataFrame:
    """Load a pywrdrb-shipped daily-streamflow CSV as a DataFrame.

    Args:
        gage: ``True`` loads ``gage_flow_mgd.csv`` (cumulative-upstream gage
            flows). ``False`` loads ``catchment_inflow_mgd.csv`` (marginal
            per-catchment inflows).
        period: ``"full"`` returns the full record; ``"baseline"`` clips to
            1980-01-01 to 2019-12-31 (the Kirsch fit window per the reference
            implementation).
        flowtype: pywrdrb inflow-dataset key. Defaults to
            ``pub_nhmv10_BC_withObsScaled`` (Amestoy et al. 2026 BC-reconstructed
            DRB record, 1945-2023).

    Returns:
        DataFrame with a daily ``DatetimeIndex`` and pywrdrb node names as
        columns.
    """
    if period not in ("baseline", "full"):
        raise ValueError(f"period must be 'baseline' or 'full', got {period!r}")

    from pywrdrb.path_manager import get_pn_object

    pn = get_pn_object()
    filename = "gage_flow_mgd.csv" if gage else "catchment_inflow_mgd.csv"
    path = str(pn.sc.get(f"flows/{flowtype}") / filename)

    Q = pd.read_csv(path, index_col=0, parse_dates=True)
    Q.index = pd.to_datetime(Q.index)

    if period == "baseline":
        Q = Q.loc[_BASELINE_START:_BASELINE_END, :]
    return Q
