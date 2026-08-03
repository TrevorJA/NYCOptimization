"""flood_exceedance_baseline_figures.py - SI illustrations of the flood-exceedance metric.

Manuscript-SI figures conveying the interpretation of the recommended flood
objective — **downstream flood exceedance**: the sum over days of the
max-across-gauges exceedance above NWS flood stage, (stage − minor)⁺, in
ft·days per year (`flood_objective_diagnostics.md` §0b). All figures show the
DEFAULT FFMP policy (the baseline decision vector) under the two reference
simulations:

  * the historic trace (`pub_nhmv10_BC_withObsScaled`, WY1946-2022), and
  * the stationary Kirsch-Nowak baseline ensemble (`kn_50yr_n5`, N=5 x 50 yr).

Figures (under outputs/supplemental/flood_objective/figures/):

  S_flood_exceedance_event_anatomy   how ft·days accumulate through a flood
                                   event: per-gauge stage relative to its own
                                   NWS flood stage, the worst-gauge envelope,
                                   and the daily exceedance increments, for the
                                   two largest simulated historic events.
  S_flood_exceedance_annual_series   water-year exceedance across the full
                                   historic trace and the five KN
                                   realizations, with the objective value
                                   (mean-annual exceedance) marked.
  S_flood_exceedance_return_period   unit-year exceedance vs return period
                                   (Weibull plotting positions), historic vs
                                   KN pooled, with zero-year fractions.

The baseline simulations (~1 min total) are cached to
outputs/supplemental/flood_objective/cube/baseline_daily_exceedance.npz and
reused on re-plot; delete the cache to force re-simulation. Configuration
lives in supplemental_config.py (FLOODOBJ_* section) — no CLI value flags.

Usage:
    python scripts/supplemental/flood_exceedance_baseline_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_floodobj_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.plotting.style import apply_style, save_figure  # noqa: E402

METRIC_NAME = "downstream flood exceedance"
METRIC_DEF = ("ft·days above NWS flood stage at the worst-affected gauge, "
              "$\\Sigma_{days}\\,\\max_{gauges}\\,(stage - flood\\ stage)^+$")

GAUGES = ["01426500", "01421000", "01436690"]
GAUGE_LABELS = {
    "01426500": "Hale Eddy (below Cannonsville)",
    "01421000": "Fishs Eddy (below Pepacton)",
    "01436690": "Bridgeville (below Neversink)",
}
# Okabe-Ito, CVD-validated triple; identity is also carried by direct labels.
GAUGE_COLORS = {"01426500": "#0072B2", "01421000": "#009E73",
                "01436690": "#CC79A7"}
# Domain pair (historic vs stationary ensemble), CVD-validated.
HIST_COLOR, ENS_COLOR = "#0072B2", "#D55E00"
SEVERITY_COLOR = "#333333"

#: Positive-exceedance days closer than this many days are one event.
EVENT_GAP_DAYS = 5
#: Days of context drawn either side of an event window.
EVENT_PAD_DAYS = 12

CACHE = scfg.FLOODOBJ_CUBE_DIR / "baseline_daily_exceedance.npz"


def _minor_series() -> pd.Series:
    from pywrdrb.flood_thresholds import flood_stage_thresholds

    return pd.Series({g: flood_stage_thresholds[g]["minor"] for g in GAUGES})


def load_baseline_daily() -> dict:
    """Daily per-gauge stage for the default FFMP policy, both domains.

    Simulates once (trimmed model, metric window applied) and caches; the
    cache is keyed to the staged inputs only through its existence — delete
    it after any re-staging.
    """
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}

    from src.ensembles import get_ensemble_spec
    from src.formulations import get_baseline_values
    from src.objectives import _metric_window
    from src.simulation import (
        dvs_to_config,
        run_simulation_ensemble_inmemory,
        run_simulation_inmemory,
    )

    cfg = dvs_to_config(get_baseline_values(scfg.FLOODOBJ_FORMULATION),
                        scfg.FLOODOBJ_FORMULATION)
    print("[exceedance_fig] simulating default FFMP baseline (historic) ...",
          flush=True)
    hist = _metric_window(run_simulation_inmemory(cfg)["flood_stage"][GAUGES])

    print("[exceedance_fig] simulating default FFMP baseline (KN ensemble) ...",
          flush=True)
    spec = get_ensemble_spec(scfg.FLOODOBJ_ENSEMBLE_SLUG)
    per_real = run_simulation_ensemble_inmemory(cfg, spec)
    ens_stage = []
    ens_dates = None
    for data_r in per_real:
        w = _metric_window(data_r["flood_stage"][GAUGES])
        ens_stage.append(w.to_numpy())
        ens_dates = w.index
    out = {
        "hist_dates": hist.index.to_numpy("datetime64[D]"),
        "hist_stage": hist.to_numpy(),
        "ens_dates": ens_dates.to_numpy("datetime64[D]"),
        "ens_stage": np.stack(ens_stage),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def daily_exceedance(stage: np.ndarray, minor: np.ndarray) -> np.ndarray:
    """Max-across-gauges positive exceedance above flood stage (ft), daily."""
    return np.clip(stage - minor[None, :], 0.0, None).max(axis=1)


def wy_totals(dates: np.ndarray, sev: np.ndarray) -> pd.Series:
    """Complete-water-year exceedance totals (ft·days) of a daily series."""
    from src.objectives_ensemble import water_year_unit_slices

    idx = pd.DatetimeIndex(dates)
    ser = pd.Series(sev, index=idx)
    out = {}
    for sl in water_year_unit_slices(idx):
        sub = ser.iloc[sl]
        wy = sub.index[-1].year
        out[wy] = float(sub.sum())
    return pd.Series(out)


def find_events(dates: np.ndarray, sev: np.ndarray) -> pd.DataFrame:
    """Contiguous positive-exceedance events (gap-merged), largest first."""
    pos = np.flatnonzero(sev > 0)
    if pos.size == 0:
        return pd.DataFrame(columns=["start", "end", "total_ftd"])
    breaks = np.flatnonzero(np.diff(pos) > EVENT_GAP_DAYS) + 1
    rows = []
    for grp in np.split(pos, breaks):
        rows.append({
            "start": pd.Timestamp(dates[grp[0]]),
            "end": pd.Timestamp(dates[grp[-1]]),
            "total_ftd": float(sev[grp].sum()),
        })
    return (pd.DataFrame(rows)
            .sort_values("total_ftd", ascending=False)
            .reset_index(drop=True))


###############################################################################
# Figure 1 — event anatomy
###############################################################################

def fig_event_anatomy(daily: dict, minor: pd.Series) -> None:
    dates = pd.DatetimeIndex(daily["hist_dates"])
    stage = daily["hist_stage"]
    sev = daily_exceedance(stage, minor.to_numpy())
    events = find_events(daily["hist_dates"], sev).head(2)

    fig, axes = plt.subplots(
        2, 2, figsize=(12.6, 6.8), sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for col, (_, ev) in enumerate(events.iterrows()):
        lo = ev["start"] - pd.Timedelta(days=EVENT_PAD_DAYS)
        hi = ev["end"] + pd.Timedelta(days=EVENT_PAD_DAYS)
        m = (dates >= lo) & (dates <= hi)
        d = dates[m]
        rel = stage[m] - minor.to_numpy()[None, :]
        env = np.clip(rel, 0.0, None).max(axis=1)

        ax = axes[0, col]
        ax.axhline(0.0, color="#555555", lw=1.2)
        ax.fill_between(d, 0.0, env, color=SEVERITY_COLOR, alpha=0.18,
                        lw=0, label="worst-gauge exceedance\n(the daily exceedance increment)")
        for gi, g in enumerate(GAUGES):
            ax.plot(d, rel[:, gi], color=GAUGE_COLORS[g], lw=1.7,
                    label=GAUGE_LABELS[g])
        ax.text(d[2], 0.12, "NWS flood stage", color="#555555", fontsize=8,
                va="bottom")
        ax.set_title(f"{ev['start']:%b %Y} event — total exceedance "
                     f"{ev['total_ftd']:.1f} ft·days", fontsize=10)
        if col == 0:
            ax.set_ylabel("stage relative to each gauge's\n"
                          "NWS flood stage (ft)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        axb = axes[1, col]
        axb.bar(d, env, width=0.85, color=SEVERITY_COLOR)
        axb.set_xlabel("date")
        axb.annotate(f"$\\Sigma$ = {env.sum():.1f} ft·days",
                     xy=(d[int(np.argmax(env))], env.max()),
                     xytext=(8, -2), textcoords="offset points", fontsize=9)
        if col == 0:
            axb.set_ylabel("daily exceedance\nincrement (ft)")
        for a in (ax, axb):
            a.tick_params(axis="x", labelrotation=25)
    fig.suptitle(
        f"How {METRIC_NAME} accumulates — default FFMP policy, historic "
        "trace\n(each day contributes the largest exceedance above NWS "
        "flood stage across the three reservoir-tail gauges)",
        y=1.04)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("S_flood_exceedance_event_anatomy"))
    plt.close(fig)


###############################################################################
# Figure 2 — annual series, both domains
###############################################################################

def fig_annual_series(daily: dict, minor: pd.Series) -> None:
    mn = minor.to_numpy()
    hist_wy = wy_totals(daily["hist_dates"],
                        daily_exceedance(daily["hist_stage"], mn))
    hist_years = len(hist_wy)
    hist_mean = hist_wy.sum() / hist_years

    ens_units = []
    for r in range(daily["ens_stage"].shape[0]):
        ens_units.append(wy_totals(daily["ens_dates"],
                                   daily_exceedance(daily["ens_stage"][r], mn)))
    pooled_mean = float(np.mean(np.concatenate(
        [u.to_numpy() for u in ens_units])))

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 6.6))

    ax = axes[0]
    ax.bar(hist_wy.index, hist_wy.values, width=0.8, color=HIST_COLOR)
    ax.axhline(hist_mean, color=SEVERITY_COLOR, ls="--", lw=1.3)
    ax.text(hist_wy.index[1], hist_mean + 0.25, "objective value: mean-annual "
            f"exceedance = {hist_mean:.2f} ft·days/yr",
            va="bottom", fontsize=8.5, color=SEVERITY_COLOR)
    for wy, v in hist_wy.nlargest(3).items():
        ax.annotate(f"WY{wy}", (wy, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_title(f"historic trace (WY{hist_wy.index[0]}–{hist_wy.index[-1]})",
                 fontsize=10)
    ax.set_ylabel("water-year exceedance\n(ft·days)")
    ax.set_xlabel("water year")

    ax = axes[1]
    offset = 0
    n_per = len(ens_units[0])
    for r, u in enumerate(ens_units):
        x = np.arange(offset, offset + len(u))
        ax.bar(x, u.values, width=0.8, color=ENS_COLOR)
        if r > 0:
            ax.axvline(offset - 0.5, color="#AAAAAA", lw=0.8)
        offset += len(u)
    tick_pos = [n_per / 2 + i * n_per for i in range(len(ens_units))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"realization {i + 1}" for i in range(len(ens_units))],
                       fontsize=9)
    ax.axhline(pooled_mean, color=SEVERITY_COLOR, ls="--", lw=1.3)
    ax.text(1, pooled_mean + 0.55, "search objective: pooled-annual mean = "
            f"{pooled_mean:.2f} ft·days/yr", va="bottom", fontsize=8.5,
            color=SEVERITY_COLOR)
    ax.set_title("stationary Kirsch–Nowak baseline ensemble "
                 f"({len(ens_units)} × {n_per} unit-years)", fontsize=10)
    ax.set_ylabel("unit-year exceedance\n(ft·days)")

    fig.suptitle(
        f"{METRIC_NAME.capitalize()} of the default FFMP policy\n"
        f"({METRIC_DEF})", y=1.03)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("S_flood_exceedance_annual_series"))
    plt.close(fig)


###############################################################################
# Figure 3 — return-period view
###############################################################################

def fig_return_period(daily: dict, minor: pd.Series) -> None:
    mn = minor.to_numpy()
    hist_wy = wy_totals(daily["hist_dates"],
                        daily_exceedance(daily["hist_stage"], mn)).to_numpy()
    ens_wy = np.concatenate([
        wy_totals(daily["ens_dates"],
                  daily_exceedance(daily["ens_stage"][r], mn)).to_numpy()
        for r in range(daily["ens_stage"].shape[0])
    ])

    def weibull(v: np.ndarray):
        v = np.sort(v)[::-1]
        n = len(v)
        t = (n + 1) / np.arange(1, n + 1)
        return t, v

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    zero_note = []
    for vals, color, label in (
        (hist_wy, HIST_COLOR,
         f"historic trace ({len(hist_wy)} water years)"),
        (ens_wy, ENS_COLOR,
         f"stationary KN ensemble ({len(ens_wy)} unit-years, pooled)"),
    ):
        t, v = weibull(vals)
        ax.plot(t, v, "o", ms=5, mec="white", mew=0.4, color=color,
                label=label)
        zero_frac = float((vals == 0).mean())
        zero_note.append(f"{zero_frac:.0%} of "
                         f"{label.split('(')[0].strip()} years are zero")
    ax.text(0.03, 0.74, "zero-exceedance years:\n  " + "\n  ".join(zero_note),
            transform=ax.transAxes, fontsize=8.5, va="top",
            color=SEVERITY_COLOR)
    ax.set_xscale("log")
    ax.set_xlabel("return period of the water-year exceedance (years)")
    ax.set_ylabel("water-year exceedance (ft·days)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(
        f"{METRIC_NAME.capitalize()} is episodic: most years contribute "
        "zero,\nthe objective integrates the exceedance of the rare flood "
        "years", fontsize=10)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("S_flood_exceedance_return_period"))
    plt.close(fig)


def main() -> None:
    apply_style()
    scfg.FLOODOBJ_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    minor = _minor_series()
    daily = load_baseline_daily()
    fig_event_anatomy(daily, minor)
    fig_annual_series(daily, minor)
    fig_return_period(daily, minor)
    print(f"[exceedance_fig] figures -> {scfg.FLOODOBJ_FIGURES_DIR}")


if __name__ == "__main__":
    main()
