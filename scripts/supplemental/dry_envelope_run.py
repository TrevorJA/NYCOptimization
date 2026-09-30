"""dry_envelope_run.py - Drought hazard at the dry end of the E_test forcing box.

Supplemental diagnostic (SI Text S6) behind the lower bound of the
re-evaluation ensemble's annual-volume axis. Stream-only: no Pywr-DRB
simulation, no large ensemble.

Local leg (default). Ten-year windows are generated at fixed annual-volume
multipliers e^m. Every level is crossed with ONE shared Latin hypercube plan
over the seasonal amplitudes (r1, r2) of the E_test box and ONE shared set of
bootstrap streams, so the level effect is paired realization by realization.
The levels are the unchanged volume (m = 0), the driest CMIP6 run (the
full-range lower bound), and the E_test lower bound at the current widening
margin and at candidate wider margins. Each window is scored on the eight
candidate hazard axes and the supplement under the current scoring rules,
beside (a) a stationary sample from the same generator, the local stand-in for
the P = 10^6 candidate ensemble's dry tail, (b) the staged N = 300 stationary
image, and (c) the historical record's disjoint 10-year windows plus one window
centred on the 1960s drought of record.

Production leg (``NYCOPT_DRYENV_PRODUCTION=1``, Anvil, after the June 1 hazard
recompute). Reads the staged E_test sub-window hazard image, its forcing
profiles, and the P = 10^6 pool image, bins the E_test SOWs on e^m, and reports
the same tail statistics against the pool's own quantiles.

Configuration in ``supplemental_config.py`` (``DRYENV_*``); no CLI value flags.
Persisted windows are reused unless ``NYCOPT_DRYENV_REFRESH=1``. Outputs under
``outputs/supplemental/dry_envelope/tables/``; figures are drawn by
``dry_envelope_figures.py``. Run::

    python scripts/supplemental/dry_envelope_run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_dryenv_env()

import config  # noqa: E402
from scengen import forcing_space as fs  # noqa: E402
from scengen.diagnostics import check_hazard_image_provenance  # noqa: E402
from scengen.forcing_ensemble import ForcingEnsembleConfig  # noqa: E402
from scengen.hazard_filling import daily_to_monthly  # noqa: E402
from scengen.hazard_metrics import (  # noqa: E402
    DEFAULT_NYC_INFLOW_NODES,
    compute_candidate_hazard_image,
)
from scengen.subsample import generate_lhs_samples  # noqa: E402
from synhydro import Ensemble  # noqa: E402

from scripts.main.compute_etest_hazard_image import _reference_series  # noqa: E402
from scripts.main.compute_historic_hazard_windows import (  # noqa: E402
    CACHE_PATH as HIST_CACHE,
    FLOWTYPE,
    historic_hazard_windows,
)
from src.ensemble_generation import (  # noqa: E402
    _disaggregate_fill_inflow,
    _generate_profile_monthly,
    _hazard_block,
    _prepare_generators,
)
from src.ensembles import hazard_image_provenance, staged_ensemble_dir  # noqa: E402
from src.etest import E_TEST_BOUND_PCT, E_TEST_MARGIN, E_TEST_VOLUME_MULTIPLIER_MIN  # noqa: E402
from src.load.historical_flows import load_historical_flows  # noqa: E402

TABLES = scfg.DRYENV_TABLES_DIR
AXES_OF_INTEREST = tuple(scfg.DRYENV_DROUGHT_AXES) + tuple(scfg.DRYENV_DROUGHT_SUPPLEMENT)


###############################################################################
# Forcing levels
###############################################################################

def volume_levels() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Annual-volume multiplier levels derived from the CMIP6 harmonic box.

    Returns:
        ``(levels, envelope, fit)``: one row per level with ``level``, ``margin``
        (NaN for the unchanged volume), ``m`` and ``em = exp(m)``; the CMIP6
        envelope; and the harmonic fit the box is built from.
    """
    env = fs.load_cmip6_envelope(config.ENSEMBLE_FORCING_MEAN_FRAC_CSV)
    fit = fs.fit_harmonic_params(env, order=2)
    lo, hi, names = fs.harmonic_param_box(fit, bound_pct=E_TEST_BOUND_PCT, margin=0.0)
    k = names.index("m")
    m_lo, m_hi = float(lo[k]), float(hi[k])
    rows = [{"level": "volume_1.00", "margin": np.nan, "m": 0.0},
            {"level": "cmip6_min", "margin": 0.0, "m": m_lo}]
    for mg in scfg.DRYENV_CANDIDATE_MARGINS:
        rows.append({"level": f"margin_{mg:.2f}", "margin": float(mg),
                     "m": m_lo - mg * (m_hi - m_lo)})
    levels = pd.DataFrame(rows)
    levels["em"] = np.exp(levels["m"])
    levels["is_margin_bound"] = np.isclose(levels["margin"].fillna(-1.0), E_TEST_MARGIN)
    levels["m_hi_full"] = m_hi
    levels["adopted_em"] = E_TEST_VOLUME_MULTIPLIER_MIN
    if not levels["is_margin_bound"].any():
        raise ValueError(
            f"E_TEST_MARGIN={E_TEST_MARGIN} is not among DRYENV_CANDIDATE_MARGINS="
            f"{scfg.DRYENV_CANDIDATE_MARGINS}; the margin box's own bound must be one level."
        )
    return levels, env, fit


def seasonal_plan(fit: dict, n: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """Shared LHS plan over (r1, r2) inside the E_test box."""
    lo, hi, names = fs.harmonic_param_box(fit, bound_pct=E_TEST_BOUND_PCT, margin=E_TEST_MARGIN)
    idx = [names.index("r1"), names.index("r2")]
    return generate_lhs_samples(n, 2, lo[idx], hi[idx], seed=seed), ["r1", "r2"]


def level_profiles(m: float, plan: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Water-year change-factor profiles at a fixed ``m`` with canonical phases."""
    n = plan.shape[0]
    full = np.column_stack([
        np.full(n, m), plan[:, 0], np.full(n, psi[0]), plan[:, 1], np.full(n, psi[1]),
    ])
    return fs.reconstruct_harmonic(full, order=2)


def etest_share_table(levels: pd.DataFrame) -> pd.DataFrame:
    """Share of a uniform-in-m design below the margin box's bound and the CMIP6 minimum.

    Every widened box keeps the upper bound; the LHS is uniform in ``m``, so the
    share of SOWs below a level is a ratio of ``m`` intervals. The adopted bound
    (``E_TEST_VOLUME_MULTIPLIER_MIN``) is the last row.
    """
    m_hi = float(levels["m_hi_full"].iloc[0])
    m_margin = float(levels.loc[levels["is_margin_bound"], "m"].iloc[0])
    m_cmip = float(levels.loc[levels["level"] == "cmip6_min", "m"].iloc[0])
    rows = []
    candidates = [(r["level"], r["margin"], r["m"]) for _, r in levels[levels["margin"].notna()].iterrows()]
    candidates.append(("adopted", np.nan, float(np.log(E_TEST_VOLUME_MULTIPLIER_MIN))))
    for label, margin, m_lo in candidates:
        width = m_hi - m_lo
        rows.append({
            "lower_bound_level": label, "margin": margin, "em_lower": float(np.exp(m_lo)),
            "share_below_margin_bound": max(0.0, (m_margin - m_lo) / width),
            "share_below_cmip6_min": max(0.0, (m_cmip - m_lo) / width),
        })
    return pd.DataFrame(rows)


###############################################################################
# Generation and scoring
###############################################################################

def _score_profiles(setup, cfg, profiles: list[int], ref_m: np.ndarray, ref_d: np.ndarray,
                    tag: str) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Generate ``profiles`` (R = 1 each) in blocks and score their hazard rows in order."""
    H_parts, S_parts, axes, names = [], [], [], []
    t0 = time.time()
    for b0 in range(0, len(profiles), scfg.DRYENV_BLOCK):
        block = profiles[b0:b0 + scfg.DRYENV_BLOCK]
        monthly, md = {}, None
        for p in block:
            frames, md_p = _generate_profile_monthly(setup, cfg, p)
            monthly.update(frames)
            md = md if md is not None else md_p
        _, inflow, _ = _disaggregate_fill_inflow(
            Ensemble(monthly, metadata=md), nowak=setup.nowak, kdes=setup.kdes,
            root_seed=cfg.root_seed, start_date=cfg.start_date,
        )
        H, axes, S, names = _hazard_block(
            inflow, sorted(inflow), DEFAULT_NYC_INFLOW_NODES, ref_m, ref_d,
            n_years=cfg.realization_years,
        )
        H_parts.append(H)
        S_parts.append(S)
        print(f"[dryenv] {tag}: {b0 + len(block)}/{len(profiles)} windows "
              f"({time.time() - t0:.0f} s)", flush=True)
    return np.vstack(H_parts), axes, np.vstack(S_parts), names


def _rows(H, axes, S, names, **labels) -> pd.DataFrame:
    df = pd.DataFrame(H, columns=axes)
    for j, n in enumerate(names):
        df[n] = S[:, j]
    for k, v in labels.items():
        df.insert(0, k, v)
    return df


def generate_local_windows(levels: pd.DataFrame, env: pd.DataFrame, fit: dict,
                           ref_m: np.ndarray, ref_d: np.ndarray) -> pd.DataFrame:
    """Forced levels (paired) plus the stationary sample, one row per window."""
    years = scfg.DRYENV_YEARS
    if years != config.SCENARIO_YEARS:
        raise ValueError(f"DRYENV_YEARS={years} must equal SCENARIO_YEARS={config.SCENARIO_YEARS}")
    n = scfg.DRYENV_N_PROFILES_PER_LEVEL
    plan, plan_names = seasonal_plan(fit, n, scfg.DRYENV_SEED_FORCED)
    psi = fs.canonical_phases(env, order=2)

    cfg = ForcingEnsembleConfig(
        root_seed=scfg.DRYENV_SEED_FORCED, n_forcing_profiles=n, realizations_per_profile=1,
        realization_years=years, population="du_forced", theta_sampler="lhs",
        mean_frac_csv=config.ENSEMBLE_FORCING_MEAN_FRAC_CSV,
        bound_pct=E_TEST_BOUND_PCT, margin=E_TEST_MARGIN,
    )
    setup = _prepare_generators(cfg)
    frames = []
    for _, lv in levels.iterrows():
        setup.a_wy = level_profiles(float(lv["m"]), plan, psi)
        setup.v_wy = None
        H, axes, S, names = _score_profiles(setup, cfg, list(range(n)), ref_m, ref_d, lv["level"])
        df = _rows(H, axes, S, names)
        df.insert(0, "r2", plan[:, 1])
        df.insert(0, "r1", plan[:, 0])
        df.insert(0, "em", float(lv["em"]))
        df.insert(0, "m", float(lv["m"]))
        df.insert(0, "window", np.arange(n))
        df.insert(0, "source", lv["level"])
        frames.append(df)

    cfg_s = ForcingEnsembleConfig(
        root_seed=scfg.DRYENV_SEED_STATIONARY, n_forcing_profiles=scfg.DRYENV_STATIONARY_N,
        realizations_per_profile=1, realization_years=years, population="stationary",
    )
    setup_s = _prepare_generators(cfg_s)
    H, axes, S, names = _score_profiles(
        setup_s, cfg_s, list(range(scfg.DRYENV_STATIONARY_N)), ref_m, ref_d, "stationary")
    df = _rows(H, axes, S, names)
    df.insert(0, "r2", np.nan)
    df.insert(0, "r1", np.nan)
    df.insert(0, "em", np.nan)
    df.insert(0, "m", np.nan)
    df.insert(0, "window", np.arange(scfg.DRYENV_STATIONARY_N))
    df.insert(0, "source", "stationary")
    frames.append(df)
    return pd.concat(frames, ignore_index=True)


###############################################################################
# References: staged stationary image, historical windows
###############################################################################

def staged_stationary_rows() -> pd.DataFrame | None:
    """Hazard rows of the staged N = 300 stationary image (current provenance), or None."""
    path = Path(staged_ensemble_dir(scfg.DRYENV_STAGED_STATIONARY_SLUG)) / "hazard_image.npz"
    if not path.exists():
        print(f"[dryenv] staged image {path} absent; skipped")
        return None
    with np.load(path, allow_pickle=True) as z:
        check_hazard_image_provenance(z, path)
        H, axes = z["H"], [str(a) for a in z["hazard_axes"]]
        S, names = z["supplement"], [str(a) for a in z["supplement_names"]]
    df = _rows(H, axes, S, names)
    df.insert(0, "window", np.arange(len(df)))
    df.insert(0, "source", scfg.DRYENV_STAGED_STATIONARY_SLUG)
    return df


def historic_rows(ref_m: np.ndarray, ref_d: np.ndarray) -> pd.DataFrame:
    """The record's disjoint 10-year windows plus the drought-of-record window."""
    H, axes, starts = historic_hazard_windows()
    with np.load(HIST_CACHE, allow_pickle=True) as z:
        S, names = z["supplement"], [str(a) for a in z["supplement_names"]]
    df = _rows(H, axes, S, names)
    df.insert(0, "window_start", [str(s.date()) for s in starts])
    df.insert(0, "source", "historic_disjoint")

    flows = load_historical_flows(gage=False, period="full", flowtype=FLOWTYPE)
    agg = flows.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    idx = pd.DatetimeIndex(agg.index)
    w0 = pd.Timestamp(scfg.DRYENV_DROUGHT_OF_RECORD_START)
    cutoff = w0 + pd.DateOffset(months=config.METRIC_EXCLUSION_MONTHS)
    metric_end = cutoff + pd.DateOffset(years=scfg.DRYENV_YEARS - 1)
    in_win = (idx >= w0) & (idx < metric_end)
    wet_cut = int(((idx >= w0) & (idx < cutoff)).sum())
    w_daily = agg.loc[in_win]
    H1, axes1, S1, names1 = compute_candidate_hazard_image(
        np.asarray(daily_to_monthly(w_daily, agg="mean"), dtype=float)[None, :],
        w_daily.to_numpy(dtype=float)[None, :], ref_m, ref_d,
        wet_exclusion_days=wet_cut, return_supplement=True, scenario_start=w0,
    )
    df1 = _rows(H1, axes1, S1, names1)
    df1.insert(0, "window_start", str(w0.date()))
    df1.insert(0, "source", "historic_drought_of_record")
    return pd.concat([df, df1], ignore_index=True)


###############################################################################
# Summaries
###############################################################################

def _drier_is_higher(axis: str) -> bool:
    """Whether larger values of ``axis`` mean a drier window (False for the low-flow descriptors)."""
    return not axis.startswith("lowflow")


def _dry_tail_quantile(values: pd.Series, q: float, axis: str) -> float:
    """The ``q`` tail quantile in the dry direction (the ``1 - q`` quantile for low-flow descriptors)."""
    return float(values.quantile(q if _drier_is_higher(axis) else 1.0 - q))


def _beyond(values: pd.Series, threshold: float, axis: str) -> float:
    """Share of ``values`` drier than ``threshold``."""
    if _drier_is_higher(axis):
        return float((values > threshold).mean())
    return float((values < threshold).mean())


def reference_table(windows: pd.DataFrame, staged: pd.DataFrame | None,
                    hist: pd.DataFrame) -> pd.DataFrame:
    """Per drought axis: the stationary and staged dry-tail quantiles and the historic values.

    ``*_tail90`` and ``*_tail99`` are the 90th and 99th percentiles in the dry
    direction (the 10th and 1st percentiles for the low-flow descriptors);
    ``*_extreme`` is the driest value.
    """
    stat = windows[windows["source"] == "stationary"]
    rows = []
    for ax in AXES_OF_INTEREST:
        row = {"axis": ax, "drier_is_higher": _drier_is_higher(ax), "stationary_n": len(stat),
               "stationary_q50": stat[ax].quantile(0.50),
               "stationary_extreme": stat[ax].max() if _drier_is_higher(ax) else stat[ax].min()}
        for q in scfg.DRYENV_TAIL_QUANTILES:
            row[f"stationary_tail{int(round(q * 100))}"] = _dry_tail_quantile(stat[ax], q, ax)
        if staged is not None:
            row["staged_n"] = len(staged)
            row["staged_q50"] = staged[ax].quantile(0.50)
            for q in scfg.DRYENV_TAIL_QUANTILES:
                row[f"staged_tail{int(round(q * 100))}"] = _dry_tail_quantile(staged[ax], q, ax)
            row["staged_extreme"] = staged[ax].max() if _drier_is_higher(ax) else staged[ax].min()
        dis = hist[hist["source"] == "historic_disjoint"]
        drier_idx = dis[ax].idxmax() if _drier_is_higher(ax) else dis[ax].idxmin()
        row["historic_disjoint_driest"] = float(dis.loc[drier_idx, ax])
        row["historic_disjoint_driest_start"] = dis.loc[drier_idx, "window_start"]
        row["historic_drought_of_record"] = float(
            hist.loc[hist["source"] == "historic_drought_of_record", ax].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def level_table(windows: pd.DataFrame, levels: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Per level and drought axis: distribution, tail exceedance and paired shift."""
    base = windows[windows["source"] == "volume_1.00"].set_index("window")
    rows = []
    for _, lv in levels.iterrows():
        sub = windows[windows["source"] == lv["level"]].set_index("window")
        for ax in AXES_OF_INTEREST:
            r = ref.set_index("axis").loc[ax]
            v = sub[ax]
            row = {"level": lv["level"], "margin": lv["margin"], "em": lv["em"], "m": lv["m"],
                   "axis": ax, "n": len(v), "median": v.median(),
                   "tail90": _dry_tail_quantile(v, 0.90, ax),
                   "extreme": v.max() if _drier_is_higher(ax) else v.min(),
                   "paired_median_shift_vs_volume_1.00": float((v - base[ax]).median())}
            for q in scfg.DRYENV_TAIL_QUANTILES:
                tag = int(round(q * 100))
                row[f"frac_beyond_stationary_tail{tag}"] = _beyond(v, r[f"stationary_tail{tag}"], ax)
            row["frac_beyond_drought_of_record"] = _beyond(v, r["historic_drought_of_record"], ax)
            rows.append(row)
    return pd.DataFrame(rows)


###############################################################################
# Production leg
###############################################################################

def production_leg() -> None:
    """Bin the staged E_test SOWs on e^m and score their sub-window drought tails vs the pool."""
    edir = Path(staged_ensemble_dir(scfg.DRYENV_ETEST_SLUG))
    sub_path = edir / "hazard_image_subwindows.npz"
    with np.load(sub_path, allow_pickle=True) as z:
        check_hazard_image_provenance(z, sub_path)
        H = z["H"]
        axes = [str(a) for a in z["hazard_axes"]]
        S = z["supplement"]
        names = [str(a) for a in z["supplement_names"]]
        theta_index = z["theta_index"].astype(int)
    with np.load(edir / "forcing_profiles.npz", allow_pickle=True) as z:
        theta = z["theta_params"]
        tnames = [str(a) for a in z["theta_param_names"]]
        R = int(z["realizations_per_profile"])
    m_by_sow = theta[::R, tnames.index("m")]
    em_sow = np.exp(m_by_sow[theta_index])

    pool_path = Path(staged_ensemble_dir(scfg.DRYENV_POOL_SLUG)) / "hazard_image.npz"
    with np.load(pool_path, allow_pickle=True) as z:
        check_hazard_image_provenance(z, pool_path)
        pool = _rows(z["H"], [str(a) for a in z["hazard_axes"]], z["supplement"],
                     [str(a) for a in z["supplement_names"]])

    sub = _rows(H, axes, S, names)
    sub.insert(0, "em", em_sow)
    sub.insert(0, "sow", theta_index)

    levels, _, _ = volume_levels()
    cmip6_min = float(levels.loc[levels["level"] == "cmip6_min", "em"].iloc[0])
    sow_em = pd.Series(np.exp(m_by_sow))
    edges = sow_em.quantile(list(scfg.DRYENV_PRODUCTION_BINS)).to_numpy()
    bins = [("below_cmip6_min", sub["em"] < cmip6_min)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        bins.append((f"em_{lo:.3f}_{hi:.3f}", (sub["em"] >= lo) & (sub["em"] <= hi)))

    ref_rows, rows = [], []
    for ax in AXES_OF_INTEREST:
        r = {"axis": ax, "drier_is_higher": _drier_is_higher(ax), "pool_n": len(pool),
             "pool_q50": pool[ax].quantile(0.5),
             "pool_extreme": pool[ax].max() if _drier_is_higher(ax) else pool[ax].min()}
        for q in scfg.DRYENV_TAIL_QUANTILES:
            r[f"pool_tail{int(round(q * 100))}"] = _dry_tail_quantile(pool[ax], q, ax)
        ref_rows.append(r)
        for label, mask in bins:
            v = sub.loc[mask, ax]
            row = {"bin": label, "n_subwindows": int(mask.sum()),
                   "n_sow": int(sub.loc[mask, "sow"].nunique()),
                   "em_min": sub.loc[mask, "em"].min(), "em_max": sub.loc[mask, "em"].max(),
                   "axis": ax, "median": v.median(),
                   "tail90": _dry_tail_quantile(v, 0.90, ax),
                   "tail99": _dry_tail_quantile(v, 0.99, ax),
                   "extreme": v.max() if _drier_is_higher(ax) else v.min()}
            for q in scfg.DRYENV_TAIL_QUANTILES:
                tag = int(round(q * 100))
                row[f"frac_beyond_pool_tail{tag}"] = _beyond(v, r[f"pool_tail{tag}"], ax)
            rows.append(row)
    pd.DataFrame(ref_rows).to_csv(TABLES / "dryenv_production_reference.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES / "dryenv_production_bins.csv", index=False)
    print(f"[dryenv] production tables written to {TABLES}")


###############################################################################
# Main
###############################################################################

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    if scfg.DRYENV_PRODUCTION:
        production_leg()
        return

    t0 = time.time()
    levels, env, fit = volume_levels()
    levels.to_csv(TABLES / "dryenv_level_definitions.csv", index=False)
    etest_share_table(levels).to_csv(TABLES / "dryenv_etest_share.csv", index=False)
    print(levels.to_string(index=False))

    ref_m, ref_d = _reference_series(FLOWTYPE)
    hist = historic_rows(ref_m, ref_d)
    hist.to_csv(TABLES / "dryenv_historic_windows.csv", index=False)
    staged = staged_stationary_rows()

    windows_path = TABLES / "dryenv_windows.csv"
    if windows_path.exists() and not scfg.DRYENV_REFRESH:
        print(f"[dryenv] reusing persisted windows {windows_path} (NYCOPT_DRYENV_REFRESH=1 regenerates)")
        windows = pd.read_csv(windows_path)
    else:
        windows = generate_local_windows(levels, env, fit, ref_m, ref_d)
        windows.to_csv(windows_path, index=False)

    ref = reference_table(windows, staged, hist)
    ref.to_csv(TABLES / "dryenv_reference.csv", index=False)
    lvl = level_table(windows, levels, ref)
    lvl.to_csv(TABLES / "dryenv_levels.csv", index=False)

    prov = {k: (v.item() if hasattr(v, "item") else str(v))
            for k, v in hazard_image_provenance().items()}
    meta = {
        "years": scfg.DRYENV_YEARS, "n_profiles_per_level": scfg.DRYENV_N_PROFILES_PER_LEVEL,
        "stationary_n": scfg.DRYENV_STATIONARY_N,
        "seed_forced": scfg.DRYENV_SEED_FORCED, "seed_stationary": scfg.DRYENV_SEED_STATIONARY,
        "bound_pct": list(E_TEST_BOUND_PCT), "margin": E_TEST_MARGIN,
        "candidate_margins": list(scfg.DRYENV_CANDIDATE_MARGINS),
        "drought_of_record_window_start": scfg.DRYENV_DROUGHT_OF_RECORD_START,
        "levels": levels[["level", "margin", "m", "em"]].to_dict(orient="records"),
        "provenance": prov, "elapsed_s": round(time.time() - t0, 1),
    }
    (TABLES / "dryenv_meta.json").write_text(json.dumps(meta, indent=2))
    cols = ["level", "em", "axis", "median", "tail90", "frac_beyond_stationary_tail90",
            "frac_beyond_stationary_tail99", "frac_beyond_drought_of_record"]
    print(lvl[lvl["axis"] == "drought_magnitude"][cols].round(3).to_string(index=False))
    print(f"[dryenv] done in {time.time() - t0:.0f} s; tables in {TABLES}")


if __name__ == "__main__":
    main()
