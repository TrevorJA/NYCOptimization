"""
tests/test_demand_covariate_diagnostics.py - Pure reductions of the demand-covariate run.

The simulation pass itself is slow and local; these tests pin the reductions
``scripts/supplemental/demand_covariate/run.py`` applies to its series, on hand
values: event counting on a level index, the storage-difference window screen
(threshold, merge gap, padding, signed peak), the drought-zone day/event table,
and the demand summary (capped months, cap shares, FFMP-year utilization).

Run:
    NYCOPT_SALINITY_ON=0 NYCOPT_TEMPERATURE_ON=0 \\
        venv/Scripts/python.exe -m pytest tests/test_demand_covariate_diagnostics.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("NYCOPT_SALINITY_ON", "0")
os.environ.setdefault("NYCOPT_TEMPERATURE_ON", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


@pytest.fixture(scope="module")
def run():
    path = PROJECT_DIR / "scripts" / "supplemental" / "demand_covariate" / "run.py"
    spec = importlib.util.spec_from_file_location("dcv_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_count_events_counts_maximal_runs(run):
    assert run.count_events([]) == 0
    assert run.count_events([False, False]) == 0
    assert run.count_events([True]) == 1
    assert run.count_events([True, True, False, True, False, False, True, True]) == 3
    assert run.count_events([False, True, True, True]) == 1


def test_screen_windows_merges_close_runs_and_reports_signed_peaks(run):
    idx = pd.date_range("2000-01-01", periods=1000, freq="D")
    diff = pd.Series(0.0, index=idx)
    diff.iloc[100:110] = 3.0        # run A
    diff.iloc[150:155] = -5.0       # run B, 40 days after A: merges with A (gap <= 60)
    diff.iloc[600:603] = 2.5        # run C, far away: its own window
    diff.iloc[800] = 1.9            # below threshold: ignored
    win = run.screen_windows(diff, threshold=2.0, merge_gap_days=60, pad_days=10)
    assert len(win) == 2
    a = win.iloc[0]
    assert a["start"] == str(idx[100].date()) and a["end"] == str(idx[154].date())
    assert a["n_days_exceeding"] == 15
    assert a["peak_diff"] == -5.0 and a["peak_date"] == str(idx[150].date())
    assert a["panel_start"] == str((idx[100] - pd.Timedelta(days=10)).date())
    assert a["panel_end"] == str((idx[154] + pd.Timedelta(days=10)).date())
    c = win.iloc[1]
    assert c["start"] == str(idx[600].date()) and c["peak_diff"] == 2.5
    empty = run.screen_windows(diff, threshold=10.0, merge_gap_days=60, pad_days=10)
    assert len(empty) == 0 and list(empty.columns) == list(win.columns)


def test_zone_table_days_and_events(run):
    idx = pd.date_range("2000-01-01", periods=10, freq="D")
    level = pd.Series([0, 0, 4, 4, 6, 6, 4, 0, 6, 0], index=idx)
    table = run.zone_table(level).set_index("zone")
    assert table.loc["level1a", "days"] == 4 and table.loc["level1a", "events"] == 3
    assert table.loc["level3", "days"] == 3 and table.loc["level3", "events"] == 2
    assert table.loc["level5", "days"] == 3 and table.loc["level5", "events"] == 2
    assert table.loc["level4", "days"] == 0 and table.loc["level4", "events"] == 0
    # at-or-below level3 (index >= 4): days 2..6 and 8 -> 6 days in 2 runs
    assert table.loc["level3", "days_at_or_below"] == 6
    assert table.loc["level3", "events_at_or_below"] == 2
    assert table["days"].sum() == len(level)


def test_demand_statistics_caps_and_utilization(run):
    from src import demand_utilization as du

    idx = pd.date_range("1945-12-01", "1949-11-30", freq="D")
    series = pd.Series(600.0, index=idx)
    series[idx.year == 1947] = 820.0      # 12 capped months (> 800)
    series[(idx.year == 1948) & (idx.month == 7)] = 700.0
    caps = {"level3": 680.0, "level4": 560.0, "level5": 520.0}
    stats = run.demand_statistics(series, caps)
    assert stats["n_months"] == 48
    assert stats["n_months_capped_800"] == 12
    assert stats["share_months_capped_800"] == pytest.approx(0.25)
    n_above_l3 = int(((idx.year == 1947) | ((idx.year == 1948) & (idx.month == 7))).sum())
    assert stats["share_days_above_level3_cap"] == pytest.approx(n_above_l3 / len(idx))
    assert stats["share_days_above_level5_cap"] == pytest.approx(1.0)
    assert stats["share_days_above_level4_cap"] == pytest.approx(1.0)
    assert stats["utilization_ffmp_years"] == pytest.approx(du.realized_utilization(series))
    assert stats["monthly_mean_max"] == pytest.approx(820.0)
    assert stats["n_zero_days"] == 0


def test_ffmp_year_reduce_labels_complete_june_to_may_years(run):
    idx = pd.date_range("1945-12-01", "1949-11-30", freq="D")
    series = pd.Series(idx.year.astype(float) + idx.dayofyear / 1000.0, index=idx)
    means = run.ffmp_year_reduce(series)
    # spin-up to 1946-06-01, then complete years 1946-06..1947-05, 1947-06..1948-05, 1948-06..1949-05
    assert list(means.index) == [1946, 1947, 1948]
    assert means.loc[1946] == pytest.approx(series["1946-06-01":"1947-05-31"].mean())
    minima = run.ffmp_year_reduce(series, np.min)
    assert minima.loc[1947] == pytest.approx(series["1947-06-01":"1948-05-31"].min())
