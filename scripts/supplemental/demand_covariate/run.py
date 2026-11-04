"""run.py - Simulation pass for the NYC demand-covariate diagnostics.

Measures what putting the NYC diversion regressions on one flow covariate (the
sum of the inflows at the three NYC reservoir nodes of the model's inflow
reconstruction, for training AND the historic extrapolation; Pywr-DRB
``extrapolate_nyc_nj_diversions.py``) changed, old (the seven-gauge USGS sum)
versus new, with everything else identical:

  1. The demand series on the historic record (config START_DATE to END_DATE):
     refit statistics per quarter, annual (June-May FFMP year) means, monthly
     climatology, utilization over complete FFMP years, the distribution of
     monthly means, months capped at 800 MGD, the share of days above the FFMP
     drought caps, per era of the old covariate's gauge availability; and the
     k = 1 utilization of the ``DCV_FIXTURE_SLUG`` fixture before (saved HDF5)
     and after (re-staged live), so the HIST-versus-ensemble consistency is
     measured, not asserted.
  2. NYC system outcomes of the current FFMP policy from the step-05 baseline
     configuration (full model, single trace, default decision variables): one
     cached model dict, deep-copied and patched twice with only the
     ``demand_nyc`` dataframe source swapped. Combined storage, FFMP-year
     minimum storage, days and distinct events per FFMP drought zone, NYC
     delivery and shortfall, Montague and Trenton target compliance, and the
     screened windows where the two variants' storage differs materially.
  3. The eight active annual-unit objectives (``config.get_objective_set()``,
     the step-05 computation) under both demands, the difference in units of
     each objective's epsilon.

Outputs -> outputs/supplemental/demand_covariate/tables (figures.py renders).
Configuration lives in supplemental_config.py (DCV_* section) -- no CLI value
flags. Old-variant inputs (the git HEAD CSV, the old fit table, the old fixture
HDF5) live in outputs/supplemental/demand_covariate/inputs; the CSV is
recovered from ``DCV_OLD_PYWRDRB_REF`` when absent.

Usage:
    python scripts/supplemental/demand_covariate/run.py
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_dcv_env()

from config import (  # noqa: E402
    END_DATE,
    NYC_DECREE_DIVERSION_CAP_MGD,
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    START_DATE,
    get_objective_set,
)
from src import demand_utilization as du  # noqa: E402
from src.formulations import get_baseline_values  # noqa: E402
from src.objectives import _metric_window  # noqa: E402
from src.objectives_ensemble import ffmp_year_unit_slices  # noqa: E402
from src.simulation import (  # noqa: E402
    _DROUGHT_LEVELS,
    _ZONE_LEVELS,
    InMemoryRecorder,
    _extract_results_from_recorder,
    _get_cached_model_dict,
    _get_temp_dir,
    _patch_model_dict,
    dvs_to_config,
)

VARIANTS = ("old", "new")
NYC_DEMAND_CSV = "diversion_nyc_extrapolated_mgd.csv"
#: Pywr-DRB sibling checkout the old CSV is recovered from.
PYWRDRB_DIR = PROJECT_DIR.parent / "Pywr-DRB"


###############################################################################
# Inputs
###############################################################################

def old_csv_path() -> Path:
    """The USGS-covariate historic CSV, recovered from ``DCV_OLD_PYWRDRB_REF`` if absent."""
    path = scfg.DCV_OLD_CSV
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = f"{scfg.DCV_OLD_PYWRDRB_REF}:{scfg.DCV_PYWRDRB_CSV_RELPATH}"
        text = subprocess.run(["git", "-C", str(PYWRDRB_DIR), "show", spec],
                              check=True, capture_output=True, text=True).stdout
        path.write_text(text, encoding="utf-8")
        print(f"[dcv_run] recovered old CSV from Pywr-DRB {spec} -> {path}")
    return path


def new_csv_path() -> Path:
    """The regenerated historic CSV pywrdrb ships (the ``historical`` NYC source)."""
    from pywrdrb.path_manager import get_pn_object

    return Path(get_pn_object().diversions.get_str(NYC_DEMAND_CSV))


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found: {path}. It is produced by the Pywr-DRB code at "
            f"{scfg.DCV_OLD_PYWRDRB_REF} (the USGS-covariate variant) and saved under "
            f"{scfg.DCV_INPUTS_DIR} before the covariate change; check out that ref to rebuild it.")
    return path


def load_demand(path: Path) -> pd.Series:
    """Daily ``aggregate`` NYC demand (MGD) over the historic simulation window."""
    s = pd.read_csv(path, index_col="datetime", parse_dates=True)["aggregate"].astype(float)
    s = s.loc[START_DATE:END_DATE]
    s.index = pd.DatetimeIndex(s.index)
    return s.rename("demand_mgd")


###############################################################################
# Demand-series reductions (pure functions of the series)
###############################################################################

def monthly_means(series: pd.Series) -> pd.Series:
    """Calendar-month means, indexed by month start."""
    return series.resample("MS").mean()


def ffmp_year_reduce(series: pd.Series, reducer=np.mean) -> pd.Series:
    """``reducer`` over each complete June-May FFMP year after the spin-up, by start year."""
    idx = pd.DatetimeIndex(series.index)
    out = {int(idx[sl.start].year): float(reducer(series.iloc[sl].to_numpy()))
           for sl in ffmp_year_unit_slices(idx)}
    return pd.Series(out, name=series.name, dtype=float).rename_axis("ffmp_year")


def demand_statistics(series: pd.Series, caps: dict[str, float],
                      allocation: float = NYC_DECREE_DIVERSION_CAP_MGD) -> dict:
    """Summary statistics of one daily demand series.

    Args:
        series: Daily demand (MGD), full simulation window.
        caps: FFMP drought-stage delivery caps (MGD) keyed by level name.
        allocation: The 800 MGD Decree allocation.

    Returns:
        Flat dict: mean, utilization over complete FFMP years, capped-month
        count and share, share of days above each cap, monthly-mean quantiles,
        zero-demand days.
    """
    monthly = monthly_means(series)
    capped = monthly >= allocation - 1e-6
    out = {
        "mean_mgd": float(series.mean()),
        "utilization_ffmp_years": du.realized_utilization(series, allocation),
        "n_months": int(len(monthly)),
        "n_months_capped_800": int(capped.sum()),
        "share_months_capped_800": float(capped.mean()),
        "n_zero_days": int((series <= 0).sum()),
        "monthly_mean_min": float(monthly.min()),
        "monthly_mean_p05": float(monthly.quantile(0.05)),
        "monthly_mean_p25": float(monthly.quantile(0.25)),
        "monthly_mean_p50": float(monthly.quantile(0.50)),
        "monthly_mean_p75": float(monthly.quantile(0.75)),
        "monthly_mean_p95": float(monthly.quantile(0.95)),
        "monthly_mean_max": float(monthly.max()),
    }
    for level, cap in caps.items():
        out[f"share_days_above_{level}_cap"] = float((series > cap).mean())
        out[f"cap_{level}_mgd"] = float(cap)
    return out


def era_table(old: pd.Series, new: pd.Series, eras=scfg.DCV_ERAS,
              allocation: float = NYC_DECREE_DIVERSION_CAP_MGD) -> pd.DataFrame:
    """Per-era means of both variants and their difference."""
    rows = []
    for name, start, end in eras:
        o, n = old.loc[start:end], new.loc[start:end]
        rows.append({
            "era": name, "start": start, "end": end, "n_days": int(len(o)),
            "old_mean_mgd": float(o.mean()), "new_mean_mgd": float(n.mean()),
            "diff_mgd": float(n.mean() - o.mean()),
            "old_mean_over_allocation": float(o.mean() / allocation),
            "new_mean_over_allocation": float(n.mean() / allocation),
            "max_abs_daily_diff_mgd": float((n - o).abs().max()),
        })
    return pd.DataFrame(rows)


def climatology_table(old: pd.Series, new: pd.Series) -> pd.DataFrame:
    """Monthly climatology of the monthly means: mean and interquartile range per month."""
    rows = []
    mo, mn = monthly_means(old), monthly_means(new)
    for m in range(1, 13):
        a, b = mo[mo.index.month == m], mn[mn.index.month == m]
        rows.append({
            "month": m,
            "old_mean": float(a.mean()), "old_p25": float(a.quantile(0.25)), "old_p75": float(a.quantile(0.75)),
            "new_mean": float(b.mean()), "new_p25": float(b.quantile(0.25)), "new_p75": float(b.quantile(0.75)),
            "diff_mean": float(b.mean() - a.mean()),
        })
    return pd.DataFrame(rows)


###############################################################################
# Regression refit
###############################################################################

def regression_fit_table() -> pd.DataFrame:
    """Quarterly OLS statistics, old (saved table) versus new (fitted live)."""
    from pywrdrb.pre.extrapolate_nyc_nj_diversions import ExtrapolatedDiversionPreprocessor

    old = pd.read_csv(_require(scfg.DCV_OLD_FIT_CSV, "old regression fit table")).set_index("quarter")
    pp = ExtrapolatedDiversionPreprocessor(loc="nyc")
    pp.process()
    rows = []
    for q in pp.quarters:
        r = pp.lrrs[q]
        o = old.loc[q]
        rows.append({
            "quarter": q, "n": int(r.nobs),
            "slope_old": float(o["slope"]), "slope_new": float(r.params[1]),
            "intercept_old": float(o["intercept"]), "intercept_new": float(r.params[0]),
            "r2_old": float(o["r2"]), "r2_new": float(r.rsquared),
            "resid_sd_old": float(o["resid_sd"]), "resid_sd_new": float(np.std(r.resid, ddof=0)),
            "mean_log_inflow_old": float(o["mean_log_inflow"]),
            "mean_log_inflow_new": float(pp.df_m.loc[pp.df_m["quarter"] == q, "flow_log"].mean()),
        })
    df = pd.DataFrame(rows)
    df["covariate_log_shift"] = df["mean_log_inflow_new"] - df["mean_log_inflow_old"]
    df["training_covariate_mean_mgd_new"] = float(pp.training_flow["NYC_inflow"].mean())
    return df


def old_covariate_gauge_availability() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Availability of the seven USGS gauges the old covariate summed (skipna).

    Returns:
        ``(per_gauge, per_year)``: first/last valid date and missing days per
        gauge over the simulation window, and the number of gauges with any
        valid day per calendar year.
    """
    from pywrdrb.path_manager import get_pn_object
    from pywrdrb.pywr_drb_node_data import nyc_reservoirs, obs_site_matches

    usgs = Path(get_pn_object().observations.get_str("_raw", "streamflow_daily_usgs_mgd.csv"))
    flow = pd.read_csv(usgs, index_col=0, parse_dates=True).loc[START_DATE:END_DATE]
    gauges = [(res, g) for res in nyc_reservoirs for g in obs_site_matches[res]]
    rows, by_year = [], {}
    for res, g in gauges:
        s = flow[g]
        rows.append({"reservoir": res, "gauge": g,
                     "first_valid": str(s.first_valid_index().date()),
                     "last_valid": str(s.last_valid_index().date()),
                     "missing_days": int(s.isna().sum()), "mean_mgd": float(s.mean())})
        by_year[g] = s.notna().groupby(s.index.year).sum()
    per_year = pd.DataFrame(by_year).rename_axis("year")
    per_year["n_gauges_with_data"] = (per_year > 0).sum(axis=1)
    per_year["old_covariate_mean_mgd"] = flow[[g for _r, g in gauges]].sum(axis=1).groupby(flow.index.year).mean()
    return pd.DataFrame(rows), per_year.reset_index()


###############################################################################
# Fixture (k = 1) utilization before and after
###############################################################################

def _h5_series(path: Path) -> dict[str, pd.Series]:
    out = {}
    with h5py.File(path, "r") as f:
        for rid in sorted(f.keys(), key=lambda k: int(k)):
            g = f[rid]
            idx = pd.to_datetime(g["datetime"][:].astype(str))
            if "demand_scale" in g and float(g["demand_scale"][()]) != 1.0:
                raise ValueError(f"{path}: realization {rid} is scaled (k != 1).")
            out[rid] = pd.Series(g["aggregate"][:], index=idx, dtype=float)
    return out


def fixture_utilization_table() -> pd.DataFrame:
    """Realized k = 1 utilization per fixture realization, old (saved) versus new (re-staged)."""
    from src.ensemble_prep import stage_nyc_demand
    from src.ensembles import get_ensemble_spec

    old = _h5_series(_require(scfg.DCV_OLD_FIXTURE_H5, "old fixture demand HDF5"))
    spec = get_ensemble_spec(scfg.DCV_FIXTURE_SLUG)
    new = _h5_series(Path(stage_nyc_demand(spec, force=True)))
    assert list(old) == list(new), (list(old), list(new))
    rows = []
    for rid in old:
        o, n = old[rid], new[rid]
        rows.append({
            "realization": int(rid),
            "old_mean_mgd": float(o.mean()), "new_mean_mgd": float(n.mean()),
            "old_utilization": du.realized_utilization(o),
            "new_utilization": du.realized_utilization(n),
        })
    df = pd.DataFrame(rows)
    df["diff_utilization"] = df["new_utilization"] - df["old_utilization"]
    mean = df.drop(columns="realization").mean().to_dict()
    mean["realization"] = "mean"
    return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)


###############################################################################
# Simulation: the step-05 baseline configuration with the demand source swapped
###############################################################################

def current_policy_config():
    """The current FFMP policy as an NYCOperationsConfig."""
    return dvs_to_config(np.array(get_baseline_values(scfg.DCV_FORMULATION), dtype=float),
                         formulation_name=scfg.DCV_FORMULATION)


def simulate(nyc_config, demand_csv: Path) -> dict:
    """Full-model single-trace run of ``nyc_config`` with ``demand_nyc`` read from ``demand_csv``.

    Mirrors ``src.simulation.run_simulation_inmemory`` (the step-05 path): the
    cached full-model dict is deep-copied and patched, then only the
    ``demand_nyc`` dataframe parameter's ``url`` is replaced. Every named node
    and parameter is recorded so the drought-level index is available.

    Returns:
        The ``_extract_results_from_recorder`` dict plus ``drought_level`` (the
        daily ``drought_level_agg_nyc`` index, 0 = level1a .. 6 = level5).
    """
    import pywrdrb

    model_dict = copy.deepcopy(_get_cached_model_dict(use_trimmed=False, nyc_config=nyc_config))
    _patch_model_dict(model_dict, nyc_config)
    param = model_dict["parameters"]["demand_nyc"]
    if param.get("type") != "dataframe" or "url" not in param:
        raise RuntimeError(f"demand_nyc is not the historic dataframe source: {param}")
    param["url"] = str(demand_csv)
    json_path = Path(_get_temp_dir()) / "dcv_model.json"
    json_path.write_text(json.dumps(model_dict))
    model = pywrdrb.Model.load(str(json_path))
    recorder = InMemoryRecorder(model, minimal=False)
    model.run()
    idx = pd.DatetimeIndex(model.timestepper.datetime_index.to_timestamp())
    data = _extract_results_from_recorder(recorder.recorder_dict, idx)
    data["drought_level"] = pd.Series(
        recorder.recorder_dict["drought_level_agg_nyc"].data[:, 0].astype(int), index=idx)
    del model, recorder
    return data


def storage_pct(data: dict) -> pd.Series:
    """Daily combined NYC storage, % of total capacity."""
    return 100.0 * data["res_storage"][NYC_RESERVOIRS].sum(axis=1) / NYC_TOTAL_CAPACITY


def count_events(mask) -> int:
    """Number of maximal runs of consecutive True values."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0
    return int(m[0]) + int(np.count_nonzero(m[1:] & ~m[:-1]))


def zone_table(level: pd.Series, levels=_DROUGHT_LEVELS) -> pd.DataFrame:
    """Days and distinct events in each FFMP zone, plus 'at or below' cumulative counts."""
    lv = level.to_numpy(dtype=int)
    rows = []
    for i, name in enumerate(levels):
        rows.append({"zone": name, "zone_index": i,
                     "days": int((lv == i).sum()), "events": count_events(lv == i),
                     "days_at_or_below": int((lv >= i).sum()),
                     "events_at_or_below": count_events(lv >= i)})
    return pd.DataFrame(rows)


def outcomes(data: dict) -> dict:
    """NYC system outcomes of one run over the metric window (after the spin-up)."""
    st = _metric_window(storage_pct(data))
    lv = _metric_window(data["drought_level"]).to_numpy(dtype=int)
    dem = _metric_window(data["ibt_demands"]["demand_nyc"])
    dlv = _metric_window(data["ibt_diversions"]["delivery_nyc"])
    short = (dem - dlv).clip(lower=0.0)
    out = {
        "storage_mean_pct": float(st.mean()),
        "storage_min_pct": float(st.min()),
        "storage_p01_pct": float(st.quantile(0.01)),
        "storage_p05_pct": float(st.quantile(0.05)),
        "days_in_drought_watch_or_worse": int((lv >= _DROUGHT_LEVELS.index("level3")).sum()),
        "days_in_drought_warning_or_worse": int((lv >= _DROUGHT_LEVELS.index("level4")).sum()),
        "days_in_drought_emergency": int((lv >= _DROUGHT_LEVELS.index("level5")).sum()),
        "nyc_demand_mean_mgd": float(dem.mean()),
        "nyc_delivery_mean_mgd": float(dlv.mean()),
        "nyc_shortfall_total_mg": float(short.sum()),
        "nyc_shortfall_pct_of_demand": float(100.0 * short.sum() / dem.sum()),
        "nyc_shortfall_days": int((short > 1e-6).sum()),
        "nyc_shortfall_max_daily_mgd": float(short.max()),
    }
    for loc, label in (("delMontague", "montague"), ("delTrenton", "trenton")):
        flow = _metric_window(data["major_flow"][loc])
        target = _metric_window(data["mrf_target"][loc])
        deficit = (target - flow).clip(lower=0.0)
        below = deficit > scfg.DCV_TARGET_TOL_MGD
        out[f"{label}_days_below_target"] = int(below.sum())
        out[f"{label}_share_days_below_target"] = float(below.mean())
        out[f"{label}_target_deficit_total_mg"] = float(deficit.sum())
        out[f"{label}_flow_mean_mgd"] = float(flow.mean())
    return out


def screen_windows(diff: pd.Series, threshold: float, merge_gap_days: int,
                   pad_days: int) -> pd.DataFrame:
    """Windows where ``|diff|`` exceeds ``threshold``, merged across short gaps.

    Args:
        diff: Daily difference series (new minus old) with a DatetimeIndex.
        threshold: Exceedance threshold in the series' units.
        merge_gap_days: Runs whose gap is at most this many days merge.
        pad_days: Padding added to each side for the figure panel bounds.

    Returns:
        One row per window: start, end, padded bounds, exceedance days, peak
        (signed, with its date) and mean difference inside the window.
    """
    idx = pd.DatetimeIndex(diff.index)
    exceed = (diff.abs() > threshold).to_numpy()
    runs = []
    i = 0
    while i < exceed.size:
        if exceed[i]:
            j = i
            while j + 1 < exceed.size and exceed[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for run in runs:
        if merged and (idx[run[0]] - idx[merged[-1][1]]).days <= merge_gap_days:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    rows = []
    for a, b in merged:
        win = diff.iloc[a:b + 1]
        peak_pos = int(win.abs().to_numpy().argmax())
        rows.append({
            "start": str(idx[a].date()), "end": str(idx[b].date()),
            "panel_start": str((idx[a] - pd.Timedelta(days=pad_days)).date()),
            "panel_end": str((idx[b] + pd.Timedelta(days=pad_days)).date()),
            "n_days_exceeding": int((win.abs() > threshold).sum()),
            "peak_diff": float(win.iloc[peak_pos]),
            "peak_date": str(win.index[peak_pos].date()),
            "mean_diff": float(win.mean()),
        })
    return pd.DataFrame(rows, columns=["start", "end", "panel_start", "panel_end",
                                       "n_days_exceeding", "peak_diff", "peak_date", "mean_diff"])


def window_outcomes(windows: pd.DataFrame, series: dict[str, pd.Series],
                    levels: dict[str, pd.Series]) -> pd.DataFrame:
    """Append each variant's minimum storage and worst drought zone inside every window."""
    rows = []
    for _, w in windows.iterrows():
        row = w.to_dict()
        for v in VARIANTS:
            st = series[v].loc[w["start"]:w["end"]]
            lv = levels[v].loc[w["start"]:w["end"]]
            row[f"{v}_min_storage_pct"] = float(st.min())
            row[f"{v}_worst_zone"] = _DROUGHT_LEVELS[int(lv.max())]
        rows.append(row)
    return pd.DataFrame(rows)


def zone_boundaries(nyc_config) -> pd.DataFrame:
    """FFMP storage-zone boundaries by day of year, % of combined capacity."""
    out = pd.DataFrame({"doy": np.arange(1, 367)})
    for level in _ZONE_LEVELS:
        out[level] = 100.0 * nyc_config.get_storage_zone_profile(level)
    return out


###############################################################################
# Objectives
###############################################################################

def objective_table(results: dict[str, dict]) -> pd.DataFrame:
    """The active objectives under both demands; difference in epsilons."""
    objs = get_objective_set()
    vals = {v: objs.compute([results[v]]) for v in VARIANTS}
    rows = []
    for i, name in enumerate(objs.names):
        o, n = float(vals["old"][i]), float(vals["new"][i])
        eps = float(objs.epsilons[i])
        maximize = int(objs.directions[i]) == 1
        diff = n - o
        rows.append({"objective": name,
                     "direction": "maximize" if maximize else "minimize",
                     "epsilon": eps, "old": o, "new": n, "diff_new_minus_old": diff,
                     "diff_in_epsilons": diff / eps,
                     "new_better": bool(diff > 0 if maximize else diff < 0)})
    return pd.DataFrame(rows)


###############################################################################
# Main
###############################################################################

def main() -> None:
    t_start = time.perf_counter()
    scfg.DCV_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    nyc_config = current_policy_config()
    caps = {level: float(nyc_config.get_constant(f"{level}_factor_delivery_nyc")) * NYC_DECREE_DIVERSION_CAP_MGD
            for level in scfg.DCV_DROUGHT_CAP_LEVELS}

    csv = {"old": old_csv_path(), "new": new_csv_path()}
    demand = {v: load_demand(csv[v]) for v in VARIANTS}
    if not demand["old"].index.equals(demand["new"].index):
        raise RuntimeError("old and new demand series do not share the simulation window index.")
    obs_lo, obs_hi = [e for e in scfg.DCV_ERAS if e[0] == "observed_2000_2021"][0][1:]
    if not np.allclose(demand["old"].loc[obs_lo:obs_hi], demand["new"].loc[obs_lo:obs_hi]):
        raise RuntimeError("the observed River Master window differs between the two CSVs.")
    print(f"[dcv_run] demand series {START_DATE} to {END_DATE}: "
          + ", ".join(f"{v} mean {demand[v].mean():.1f} MGD" for v in VARIANTS), flush=True)

    # 1. Demand tables
    summary = pd.DataFrame({v: demand_statistics(demand[v], caps) for v in VARIANTS}).T
    summary.index.name = "variant"
    summary.to_csv(scfg.dcv_table_path("demand_summary"))
    era_table(demand["old"], demand["new"]).to_csv(scfg.dcv_table_path("demand_summary_by_era"), index=False)
    annual = pd.DataFrame({v: ffmp_year_reduce(demand[v]) for v in VARIANTS})
    annual["diff"] = annual["new"] - annual["old"]
    annual.to_csv(scfg.dcv_table_path("demand_annual_mean"))
    monthly = pd.DataFrame({v: monthly_means(demand[v]) for v in VARIANTS})
    monthly["diff"] = monthly["new"] - monthly["old"]
    monthly.rename_axis("month_start").to_csv(scfg.dcv_table_path("demand_monthly_means"))
    climatology_table(demand["old"], demand["new"]).to_csv(
        scfg.dcv_table_path("demand_monthly_climatology"), index=False)
    fit = regression_fit_table()
    fit.to_csv(scfg.dcv_table_path("regression_fit"), index=False)
    per_gauge, per_year = old_covariate_gauge_availability()
    per_gauge.to_csv(scfg.dcv_table_path("old_covariate_gauge_availability"), index=False)
    per_year.to_csv(scfg.dcv_table_path("old_covariate_gauges_by_year"), index=False)
    fixture = fixture_utilization_table()
    fixture.to_csv(scfg.dcv_table_path("fixture_utilization"), index=False)
    print("[dcv_run] utilization (complete FFMP years): historic "
          + ", ".join(f"{v} {summary.loc[v, 'utilization_ffmp_years']:.3f}" for v in VARIANTS)
          + "; fixture mean "
          + ", ".join(f"{v} {float(fixture.iloc[-1][f'{v}_utilization']):.3f}" for v in VARIANTS),
          flush=True)

    # 2. Simulations
    results, secs = {}, {}
    for v in VARIANTS:
        t0 = time.perf_counter()
        results[v] = simulate(nyc_config, csv[v])
        secs[v] = time.perf_counter() - t0
        print(f"[dcv_run] simulated {v} demand in {secs[v]:.0f} s", flush=True)
        sim_dem = results[v]["ibt_demands"]["demand_nyc"].loc[START_DATE:END_DATE]
        if not np.allclose(sim_dem.to_numpy(), demand[v].reindex(sim_dem.index).to_numpy()):
            raise RuntimeError(f"{v}: simulated demand_nyc is not the {v} CSV series.")

    out = pd.DataFrame({v: outcomes(results[v]) for v in VARIANTS}).T
    out.index.name = "variant"
    out.to_csv(scfg.dcv_table_path("outcomes_summary"))
    zones = pd.concat([zone_table(_metric_window(results[v]["drought_level"])).assign(variant=v)
                       for v in VARIANTS], ignore_index=True)
    zones.to_csv(scfg.dcv_table_path("drought_zone_days"), index=False)

    st = {v: storage_pct(results[v]) for v in VARIANTS}
    lv = {v: results[v]["drought_level"] for v in VARIANTS}
    annual_min = pd.DataFrame({v: ffmp_year_reduce(st[v], np.min) for v in VARIANTS})
    annual_min["diff"] = annual_min["new"] - annual_min["old"]
    annual_min.rename_axis("ffmp_year").to_csv(scfg.dcv_table_path("storage_annual_minimum"))

    diff = _metric_window(st["new"] - st["old"])
    windows = window_outcomes(
        screen_windows(diff, scfg.DCV_STORAGE_DIFF_THRESHOLD_PCT,
                       scfg.DCV_WINDOW_MERGE_GAP_DAYS, scfg.DCV_WINDOW_PAD_DAYS), st, lv)
    windows.to_csv(scfg.dcv_table_path("storage_difference_windows"), index=False)
    print(f"[dcv_run] {len(windows)} storage-difference windows above "
          f"{scfg.DCV_STORAGE_DIFF_THRESHOLD_PCT} % of capacity: "
          + "; ".join(f"{w.start[:7]} to {w.end[:7]} (peak {w.peak_diff:+.1f})" for w in windows.itertuples()),
          flush=True)

    daily = pd.DataFrame(index=st["old"].index)
    for v in VARIANTS:
        daily[f"demand_{v}"] = results[v]["ibt_demands"]["demand_nyc"]
        daily[f"storage_pct_{v}"] = st[v]
        daily[f"level_{v}"] = lv[v]
        daily[f"delivery_{v}"] = results[v]["ibt_diversions"]["delivery_nyc"]
        for loc, label in (("delMontague", "montague"), ("delTrenton", "trenton")):
            daily[f"{label}_flow_{v}"] = results[v]["major_flow"][loc]
            daily[f"{label}_target_{v}"] = results[v]["mrf_target"][loc]
    daily.rename_axis("date").to_csv(scfg.dcv_table_path("daily_series"), float_format="%.4f")
    zone_boundaries(nyc_config).to_csv(scfg.dcv_table_path("zone_boundaries"), index=False)

    # 3. Objectives
    objective_table(results).to_csv(scfg.dcv_table_path("baseline_objectives"), index=False)

    pywrdrb_head = subprocess.run(["git", "-C", str(PYWRDRB_DIR), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": [START_DATE, END_DATE],
        "formulation": scfg.DCV_FORMULATION,
        "old_csv": str(csv["old"]), "new_csv": str(csv["new"]),
        "old_pywrdrb_ref": scfg.DCV_OLD_PYWRDRB_REF, "pywrdrb_head": pywrdrb_head,
        "fixture": scfg.DCV_FIXTURE_SLUG,
        "drought_caps_mgd": caps,
        "storage_diff_threshold_pct": scfg.DCV_STORAGE_DIFF_THRESHOLD_PCT,
        "sim_seconds": {v: round(s, 1) for v, s in secs.items()},
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    with open(scfg.DCV_TABLES_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[dcv_run] tables -> {scfg.DCV_TABLES_DIR} ({manifest['total_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
