"""src/plotting/forcing_space.py - Panel library for the DU forcing-space figure.

The deeply-uncertain forcing space of the re-evaluation ensemble (E_test) is a
low-order harmonic model of the monthly streamflow change factor, sampled by
Latin hypercube over a box read from the 27 mid-century (2020-2059) runs of a
CMIP6 multimodel ensemble (``config.ENSEMBLE_FORCING_PERIOD``; methods: ``docs/notes/methods/forcing_parameterization.md``; manuscript
Section 3.4.1; Quinn et al. 2018 for the fixed-phase amplitude-sampling design)::

    ln a(t) = m + r1*cos(w*(t - tau1*)) + r2*cos(2*w*(t - tau2*)),   w = 2*pi/12

with the phases ``(tau1*, tau2*)`` FIXED at the canonical CMIP6 shape, so the
three sampled coordinates are the amplitudes ``theta = (m, r1, r2)``:

    m   -> log annual-mean change; ``e^m`` is the water-year volume multiplier
    r1  -> annual-harmonic amplitude (winter-wettening / summer-drying)
    r2  -> semiannual amplitude (snowmelt shoulder / bimodal shape)

This module holds the four panels of the main-manuscript forcing-space figure
and the loaders they share with the SI figures
(``scripts/supplemental/figures_forcing_parameterization.py``). Every function
here is pure: panels draw onto a caller-supplied ``ax`` and never write files.

Panels
------
(a) :func:`panel_harmonic_decomposition` - the model, built up one term at a time
(b) :func:`panel_parameter_space`        - fitted CMIP6 params, LHS draws, sampling box
(c) :func:`panel_monthly_change`         - monthly change factors: LHS envelope vs CMIP6
(d) :func:`panel_flow_duration`          - flow duration curves: E_test, CMIP6-shifted record, record

Provenance rule
---------------
Panels (b) and (c) read the **as-built** sample from the staged ensemble's
``forcing_profiles.npz`` rather than re-running the sampler. Re-sampling would
draw a *different* Latin hypercube and quietly describe an ensemble that was
never simulated. Panel (d) picks its rows out of the FDC cache by global
realization id, so a cache built on the parent of a prefix-subset staging
still describes the subset (:func:`etest_cache_rows`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ---------------------------------------------------------------------------
# Shared visual vocabulary (identical across every panel)
# ---------------------------------------------------------------------------

#: Water-year month initials (October through September).
WY_LABELS: list[str] = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]

#: Water-year month index 0..11.
WY_T: np.ndarray = np.arange(12)

#: Fundamental angular frequency of the harmonic model (radians per month).
OMEGA: float = 2 * np.pi / 12

# Okabe-Ito, colour-vision-deficiency safe. One colour per SOURCE of
# information, held fixed across all four panels: CMIP6 evidence, the sampled
# E_test envelope, and the observed record.
CMIP6_COLOR: str = "#D55E00"      # vermillion - CMIP6 futures (points in b, lines in c/d)
ETEST_COLOR: str = "#0072B2"      # blue       - the LHS sample / E_test envelope
HISTORIC_COLOR: str = "#000000"   # black      - the observed historical record

#: Greyscale ramp for panel (a): each successive term of the harmonic model is
#: darker, so the decomposition reads without colour.
DECOMP_GREYS: dict[str, str] = {
    "level":      "0.72",   # e^m alone (annual-mean level)
    "annual":     "0.45",   # + the annual harmonic
    "semiannual": "0.10",   # + the semiannual harmonic (the fitted profile)
}

#: Alpha for the shaded sample envelope in panels (c) and (d).
ENVELOPE_ALPHA: float = 0.30

#: Exceedance-probability range plotted in panel (d), in percent. The cached
#: FDCs run the full 0-100%, but the two endpoints are each a SINGLE day of a
#: single realization, so across 25,000 realizations the min-max envelope at
#: those endpoints is set by one extreme value and spans several extra decades
#: of the log axis, compressing everything informative. Trimming the endpoints
#: is the usual FDC plotting convention; the cache keeps the full range so this
#: stays a display choice.
FDC_PLOT_DOMAIN: tuple[float, float] = (1.0, 99.0)


def envelope_pctl() -> tuple[float, float]:
    """Percentile pair defining the shaded sample envelope in panels (c) and (d).

    Default ``(0, 100)`` is the full min-max range: a deeply-uncertain envelope
    makes no probability claim, so its full extent is the honest depiction, and
    it is the extent panel (b)'s sampling box advertises. Override with
    ``NYCOPT_FIG_ENVELOPE_PCTL="5,95"`` to re-render both panels as a trimmed
    band. No central tendency is ever drawn.
    """
    raw = os.environ.get("NYCOPT_FIG_ENVELOPE_PCTL", "0,100")
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"NYCOPT_FIG_ENVELOPE_PCTL must be 'lo,hi' percentiles; got {raw!r}"
        )
    lo, hi = float(parts[0]), float(parts[1])
    if not 0.0 <= lo < hi <= 100.0:
        raise ValueError(f"NYCOPT_FIG_ENVELOPE_PCTL needs 0 <= lo < hi <= 100; got {raw!r}")
    return lo, hi


def envelope_label(pctl: tuple[float, float]) -> str:
    """Legend text naming the envelope definition actually plotted."""
    lo, hi = pctl
    if (lo, hi) == (0.0, 100.0):
        return "E$_{test}$ envelope (full range)"
    return f"E$_{{test}}$ envelope ({lo:g}–{hi:g}th pctile)"


# ---------------------------------------------------------------------------
# CMIP6 anchors: load and fit
# ---------------------------------------------------------------------------

def _run_metadata(column: str) -> tuple[str, str, str]:
    """Parse ``(ssp, period, gcm)`` out of a CMIP6 run's column name."""
    return (
        re.search(r"ssp(\d+)", column).group(1),
        re.search(r"(\d{4}_\d{4})$", column).group(1),
        re.search(r"RAPID_(.+?)_ssp", column).group(1),
    )


def cmip6_period_label(period: str) -> str:
    """Legend text for the CMIP6 runs of ``period``, e.g. ``CMIP6 2020–2059``."""
    return f"CMIP6 {period.replace('_', '–')}"


def load_cmip6_fits(csv_path=None, *, order: int = 2) -> dict:
    """Fit the harmonic model to every CMIP6 run of the forcing period.

    ``scengen.forcing_space.load_cmip6_envelope`` keeps the runs of
    ``config.ENSEMBLE_FORCING_PERIOD``: the 27 mid-century runs (2 hydrologic
    models x 7 GCMs x 2 SSPs, less the missing PRMS/BCC-CSM2-MR ssp370 run),
    each expressed as a change against its own ``1980_2019`` baseline.

    Args:
        csv_path: CMIP6 multiplicative monthly change-factor table. Defaults to
            ``config.ENSEMBLE_FORCING_MEAN_FRAC_CSV`` - the same table the
            ensemble generator reads, so the figure cannot drift from the build.
        order: Number of harmonics (2 = annual + semiannual, the campaign model).

    Returns:
        Dict with ``period`` (the projection-period token), ``env`` (the
        ``(12, K)`` water-year envelope), ``A`` the ``(K, 12)`` raw change
        factors, ``fit`` (see ``scengen.forcing_space.fit_harmonic_params``),
        ``recon`` the ``(K, 12)`` fitted profiles, and ``df`` a ``K``-row frame
        of per-run parameters and fit quality.
    """
    import config
    from scengen import forcing_space as fs

    if csv_path is None:
        csv_path = config.ENSEMBLE_FORCING_MEAN_FRAC_CSV
    forcing_period = config.ENSEMBLE_FORCING_PERIOD

    env = fs.load_cmip6_envelope(csv_path, period=forcing_period)
    A = env.values.T                                   # (K, 12) raw change factors
    fit = fs.fit_harmonic_params(env, order=order)

    m, r1, psi1 = fit["m"], fit["amp"][:, 0], fit["phase"][:, 0]
    r2, psi2 = fit["amp"][:, 1], fit["phase"][:, 1]
    # reconstruct_harmonic expects rows [m, r1, psi1, r2, psi2, ...]. Use a
    # negligible floor so the fit is not clipped where it is being scored.
    recon = fs.reconstruct_harmonic(
        np.column_stack([m, r1, psi1, r2, psi2]), order=order, floor=1e-6
    )

    # Shape R^2 in log space: how much of each profile's SEASONAL structure the
    # retained harmonics explain (the annual-mean level is fitted exactly).
    log_obs, log_fit = np.log(A), np.log(recon)
    ss_res = ((log_obs - log_fit) ** 2).sum(axis=1)
    ss_tot = ((log_obs - log_obs.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    shape_r2 = 1 - ss_res / ss_tot

    ssp, period, gcm = zip(*[_run_metadata(c) for c in env.columns])
    df = pd.DataFrame({
        "scenario": list(env.columns), "gcm": gcm, "ssp": ssp, "period": period,
        "m_log": m, "vol_mult_expm": np.exp(m),
        "r1_annual_amp": r1, "phase1_rad": psi1, "peak1_wy_month": (psi1 / OMEGA) % 12,
        "r2_semiann_amp": r2, "phase2_rad": psi2,
        "shape_R2": shape_r2,
    })
    return {"period": forcing_period, "env": env, "A": A, "fit": fit, "recon": recon,
            "df": df}


def etest_param_box(fit: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sampling box of the re-evaluation ensemble, in fitted-parameter space.

    Reads the bounds from :mod:`src.etest` rather than hard-coding them, so the
    drawn box always matches the ensemble contract. E_test uses the FULL
    empirical CMIP6 range widened by ``E_TEST_MARGIN`` on every axis (annual-volume
    multiplier 0.899 to 1.28, no per-axis override); the SEARCH-side pool uses a trimmed 5th-95th
    percentile box with no widening (:func:`search_param_box`). Confusing the
    two is what made the earlier version of this figure show a box far
    narrower than the ensemble it drew.

    Returns:
        ``(lo, hi, names)`` with names ``[m, r1, psi1, r2, psi2]``.
    """
    from scengen import forcing_space as fs

    from src import etest

    return fs.harmonic_param_box(
        fit, bound_pct=etest.E_TEST_BOUND_PCT, margin=etest.E_TEST_MARGIN
    )


def search_param_box(fit: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sampling box of the SEARCH-side candidate pool (SI figures only)."""
    from scengen import forcing_space as fs

    import config

    return fs.harmonic_param_box(
        fit,
        bound_pct=config.ENSEMBLE_FORCING_BOUND_PCT,
        margin=config.ENSEMBLE_FORCING_MARGIN,
    )


# ---------------------------------------------------------------------------
# The as-built E_test sample
# ---------------------------------------------------------------------------

def load_etest_sample(ensemble_dir) -> dict:
    """SOW-level forcing coordinates and profiles actually used to build E_test.

    Realizations are stored SOW-major (``realizations_per_profile`` consecutive
    rows share one forcing point), so striding recovers one row per SOW.

    Args:
        ensemble_dir: Staged ensemble directory holding ``forcing_profiles.npz``.

    Returns:
        Dict with ``theta`` ``(n_sow, 3)``, ``theta_names``, ``a`` ``(n_sow, 12)``
        water-year change factors (``None`` for a metadata-only staging that
        carries no ``mean_factor_a``), ``n_sow``, and ``realization_ids``
        ``(n_sow * R,)``, the global id of every staged realization (unstrided;
        :func:`panel_flow_duration` selects its FDC-cache rows by these). When the staged E_test
        carries the NYC entitlement-utilization factor (the ``demand_utilization``
        key written by ``scripts/supplemental/append_demand_utilization.py``,
        ``src.demand_utilization``), the dict also holds ``u`` ``(n_sow,)``, the
        nominal utilization, ``k`` ``(n_sow,)``, the demand scale applied to the
        SOW's realizations, and ``k_clipped`` ``(n_sow,)`` bool, all strided
        like ``theta``. ``u`` is a separate key and never a column of ``theta``:
        :func:`assert_box_contains` and the forcing-space panels read ``theta``
        as the three climate axes only.
    """
    npz = Path(ensemble_dir) / "forcing_profiles.npz"
    with np.load(npz, allow_pickle=True) as z:
        rpp = int(z["realizations_per_profile"])
        theta = np.asarray(z["theta_params"], dtype=float)[::rpp]
        a = (np.asarray(z["mean_factor_a"], dtype=float)[::rpp]
             if "mean_factor_a" in z.files else None)
        names = [str(x) for x in z["theta_param_names"]]
        out = {"theta": theta, "theta_names": names, "a": a, "n_sow": theta.shape[0],
               "realization_ids": np.asarray(z["realization_ids"], dtype=np.int64)}
        if "demand_utilization" in z.files:
            out["u"] = np.asarray(z["demand_utilization"], dtype=float)[::rpp]
            out["k"] = np.asarray(z["demand_scale_k"], dtype=float)[::rpp]
            out["k_clipped"] = np.asarray(z["demand_scale_clipped"]).astype(bool)[::rpp]
    return out


def assert_box_contains(theta: np.ndarray, names: list[str],
                        lo: np.ndarray, hi: np.ndarray, box_names: list[str],
                        *, tol: float = 1e-6) -> None:
    """Check the drawn box brackets the draws it is supposed to describe.

    The guard that would have caught the original wrong-box bug: a box plotted
    from the wrong percentile/margin settings does not contain the sample.

    Raises:
        AssertionError: If any draw falls outside its axis bounds.
    """
    for j, name in enumerate(names):
        k = box_names.index(name)
        col = theta[:, j]
        if col.min() < lo[k] - tol or col.max() > hi[k] + tol:
            raise AssertionError(
                f"forcing box does not contain the as-built draws on axis {name!r}: "
                f"draws [{col.min():.4f}, {col.max():.4f}] vs box [{lo[k]:.4f}, {hi[k]:.4f}]. "
                "The box percentile/margin settings do not match the staged ensemble."
            )


# ---------------------------------------------------------------------------
# Small drawing helpers, shared so the panels stay visually identical
# ---------------------------------------------------------------------------

def _draw_envelope(ax, x, Y, pctl, *, color, label=None):
    """Shade the ``pctl`` band of sample ``Y`` ``(n_sample, len(x))`` over ``x``."""
    lo, hi = np.percentile(np.asarray(Y, dtype=float), pctl, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=ENVELOPE_ALPHA, lw=0,
                    label=label, zorder=1)
    # A hairline edge keeps the extent legible where the band is thin.
    ax.plot(x, lo, color=color, lw=0.8, zorder=2)
    ax.plot(x, hi, color=color, lw=0.8, zorder=2)


def _cmip6_scatter(ax, x, y, *, size: float = 42):
    """Scatter the CMIP6 runs in one colour and marker."""
    ax.scatter(np.asarray(x), np.asarray(y), c=CMIP6_COLOR, marker="o", s=size,
               edgecolor="white", linewidth=0.5, zorder=3)


def _water_year_axis(ax):
    """Label the x axis with water-year months."""
    ax.set_xticks(WY_T)
    ax.set_xticklabels(WY_LABELS)
    ax.set_xlim(-0.4, 11.4)
    ax.set_xlabel("Month of water year")


#: Every panel carries its own legend, drawn alike.
LEGEND_KW: dict = {"frameon": False, "handlelength": 1.8, "borderpad": 0.2,
                   "labelspacing": 0.35}


def _headroom(ax, factor: float) -> None:
    """Extend the y range upward so an in-axes legend clears the data."""
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y0 + (y1 - y0) * factor)


def _marker_handle(marker: str, color: str, *, ms: float, label: str,
                   alpha: float = 1.0, edge: str = "white") -> Line2D:
    """Legend proxy for a scatter series, at a legible size."""
    return Line2D([0], [0], ls="none", marker=marker, ms=ms, markerfacecolor=color,
                  markeredgecolor=edge, markeredgewidth=0.5, alpha=alpha,
                  color="none", label=label)


def _line_handle(label: str, *, color: str = CMIP6_COLOR, lw: float = 1.2,
                 alpha: float = 0.75) -> Line2D:
    """Legend proxy for a bundle of thin per-run lines."""
    return Line2D([0], [0], lw=lw, color=color, alpha=alpha, label=label)


def _envelope_handle(label: str) -> Patch:
    """Legend proxy for the shaded E_test envelope."""
    return Patch(facecolor=ETEST_COLOR, edgecolor=ETEST_COLOR, alpha=0.55, label=label)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def best_fit_run(df: pd.DataFrame) -> int:
    """Row of the CMIP6 run the two-harmonic model fits best (highest shape R2).

    The decomposition panels illustrate the model on this run, so each term's
    contribution is read against points the full model reproduces. Fit quality
    across all runs, the worst fits included, is the subject of the SI figures.
    """
    return int(np.argmax(df["shape_R2"].to_numpy()))


def panel_harmonic_decomposition(ax, fits: dict) -> dict:
    """(a) The harmonic model built up one term at a time, on one CMIP6 run.

    The illustrated run is the best-fitting one (:func:`best_fit_run`); its
    identity is carried in the title.

    Returns:
        Dict describing the chosen run (``index``, ``label``).
    """
    df, A, recon = fits["df"], fits["A"], fits["recon"]

    k = best_fit_run(df)
    m, r1, psi1 = df.m_log[k], df.r1_annual_amp[k], df.phase1_rad[k]

    level = np.exp(np.full(12, m))
    plus_annual = np.exp(m + r1 * np.cos(OMEGA * WY_T - psi1))

    ax.axhline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)
    ax.plot(WY_T, level, color=DECOMP_GREYS["level"], lw=1.8, ls="--",
            label="Annual mean level, $e^{m}$", zorder=2)
    ax.plot(WY_T, plus_annual, color=DECOMP_GREYS["annual"], lw=1.8, ls="-",
            label="+ annual harmonic, $r_1$", zorder=3)
    ax.plot(WY_T, recon[k], color=DECOMP_GREYS["semiannual"], lw=2.2, ls="-",
            label="+ semiannual harmonic, $r_2$", zorder=4)

    run_label = (f"{df.gcm[k]}, SSP{df.ssp[k][0]}-{df.ssp[k][1]}.{df.ssp[k][2]}, "
                 f"{df.period[k].replace('_', '–')}")
    ax.plot(WY_T, A[k], ls="none", marker="o", ms=6, color=CMIP6_COLOR,
            markeredgecolor="white", markeredgewidth=0.5, zorder=5,
            label="CMIP6 change factor")

    _water_year_axis(ax)
    ax.set_ylabel("Change factor, $a_j$")
    # The illustrated run is named in the title rather than the legend: it is
    # provenance for the panel, not a fifth series to distinguish.
    ax.set_title(f"(a) Harmonic decomposition of $a_j$\n{run_label}", loc="left")
    # Headroom so the legend clears the winter peak without shrinking the data.
    _headroom(ax, 1.52)
    ax.legend(loc="upper left", **LEGEND_KW)
    return {"index": k, "label": run_label}


def panel_parameter_space(ax, fits: dict, sample: dict) -> None:
    """(b) Fitted CMIP6 parameters, the E_test LHS draws, and the sampling box.

    Axes are the two coordinates with a direct hydrologic reading: the water-year
    volume multiplier ``e^m`` and the seasonal amplitude ``r1``. The
    ``r1``-``r2`` slice is an SI figure.
    """
    df = fits["df"]
    lo, hi, box_names = etest_param_box(fits["fit"])
    assert_box_contains(sample["theta"], sample["theta_names"], lo, hi, box_names)

    theta = sample["theta"]
    i_m = sample["theta_names"].index("m")
    i_r1 = sample["theta_names"].index("r1")
    k_m, k_r1 = box_names.index("m"), box_names.index("r1")

    ax.scatter(np.exp(theta[:, i_m]), theta[:, i_r1], s=9, c=ETEST_COLOR,
               alpha=0.28, linewidths=0, zorder=1)
    ax.add_patch(Rectangle(
        (np.exp(lo[k_m]), lo[k_r1]),
        np.exp(hi[k_m]) - np.exp(lo[k_m]), hi[k_r1] - lo[k_r1],
        fill=False, edgecolor=ETEST_COLOR, ls="--", lw=1.6, zorder=4,
    ))
    ax.axvline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)
    _cmip6_scatter(ax, df.vol_mult_expm, df.r1_annual_amp)

    ax.set_xlabel("Water-year volume multiplier, $e^{m}$")
    ax.set_ylabel("Seasonal amplitude, $r_1$")
    ax.set_title("(b) Sampled forcing space", loc="left")

    # The E_test blues, then the CMIP6 vermillion; one column, since two do
    # not fit the square panel at the figure's type size.
    handles = [
        _marker_handle("o", ETEST_COLOR, ms=7, alpha=0.55, edge="none",
                       label="E$_{test}$ Latin hypercube draws"),
        Line2D([0], [0], ls="--", lw=1.6, color=ETEST_COLOR,
               label="E$_{test}$ sampling box"),
        _marker_handle("o", CMIP6_COLOR, ms=8, label=cmip6_period_label(fits["period"])),
    ]
    _headroom(ax, 1.40)
    ax.legend(handles=handles, loc="upper left", **LEGEND_KW)


def panel_monthly_change(ax, fits: dict, sample: dict,
                         pctl: tuple[float, float]) -> None:
    """(c) Monthly change factors: the sampled envelope against the CMIP6 fits."""
    _draw_envelope(ax, WY_T, sample["a"], pctl, color=ETEST_COLOR)
    for profile in fits["recon"]:
        ax.plot(WY_T, profile, color=CMIP6_COLOR, lw=0.7, alpha=0.55, zorder=3)
    ax.axhline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)

    _water_year_axis(ax)
    ax.set_ylabel("Change factor, $a_j$")
    ax.set_title("(c) Monthly streamflow change", loc="left")

    handles = [_envelope_handle(envelope_label(pctl)),
               _line_handle("CMIP6 harmonic fits")]
    _headroom(ax, 1.22)
    ax.legend(handles=handles, loc="upper right", **LEGEND_KW)


def cmip6_shifted_historic_fdcs(fits: dict, exceedance: np.ndarray,
                                nodes: tuple[str, ...]) -> np.ndarray:
    """FDCs of the historical record after each CMIP6 run's change factors.

    Each run's raw monthly change factor ``a_j`` (the CMIP6 table that panels
    (a)-(c) are fitted on) multiplies every day of the matching water-year
    month of the reconstructed aggregate NYC inflow, and the scaled record is
    reduced to a flow duration curve on ``exceedance``. This is the record-based
    counterpart of the E_test realizations, which the generator drives with the
    sampled harmonic change factors on the same record: the two sources share
    one baseline, and the raw CMIP6 flows' hydrologic-model bias never enters
    the panel. The raw factors are used rather than the harmonic fits so the
    panel checks the parameterization instead of restating it.

    Args:
        fits: Output of :func:`load_cmip6_fits`.
        exceedance: Exceedance grid in percent (the cache's ``exceedance``).
        nodes: Catchments summed into the aggregate inflow (the cache's
            ``nyc_nodes``, so the record matches the cached historical FDC).

    Returns:
        ``(K, len(exceedance))`` flows in MGD, rows in the order of ``fits["df"]``.
    """
    from scengen import forcing_space as fs
    from scripts.main.forcing_fdc_cache import fdc_on_grid
    from src.load.historical_flows import load_historical_flows

    flows = load_historical_flows(gage=False, period="full")
    record = flows.loc[:, list(nodes)].sum(axis=1)
    slot_of_month = np.empty(13, dtype=int)
    for j, month in enumerate(fs.WATER_YEAR_MONTHS):
        slot_of_month[month] = j
    slot = slot_of_month[record.index.month]
    base = record.to_numpy(dtype=float)
    exceedance = np.asarray(exceedance, dtype=float)
    return np.vstack([fdc_on_grid(base * a[slot], exceedance) for a in fits["A"]])


def etest_cache_rows(cached_ids: np.ndarray, realization_ids: np.ndarray) -> np.ndarray:
    """Row indices of the staged E_test's realizations in the FDC cache.

    The cache stores one row per realization of the ensemble it was built on,
    keyed by global realization id. The campaign E_test is a chunk prefix of a
    larger generated parent, so a cache built on the parent is a superset of
    the cut's rows; selecting by id keeps panel (d) on the same ensemble as
    panels (b) and (c).

    Args:
        cached_ids: The cache's ``etest_realization_ids`` column.
        realization_ids: Global ids of every staged realization
            (``load_etest_sample(...)["realization_ids"]``).

    Returns:
        Integer row indices into the cache's ``etest_fdc``.

    Raises:
        KeyError: If any staged realization has no cached row.
    """
    cached_ids = np.asarray(cached_ids, dtype=np.int64)
    wanted = np.unique(np.asarray(realization_ids, dtype=np.int64))
    missing = np.setdiff1d(wanted, cached_ids)
    if missing.size:
        raise KeyError(
            f"FDC cache has no row for {missing.size} of the staged E_test's "
            f"{wanted.size} realizations (first missing id {missing[0]}). Rebuild it: "
            "`sbatch --export=ALL,NYCOPT_FIG_REFRESH=1 workflow/13_main_figures.sh`."
        )
    return np.flatnonzero(np.isin(cached_ids, wanted))


def panel_flow_duration(ax, cache, cmip6_fdc: np.ndarray, pctl: tuple[float, float],
                        *, realization_ids: np.ndarray | None = None) -> None:
    """(d) Flow duration curves of aggregate NYC reservoir inflow.

    Three sources on one log flow axis, all built on the reconstructed
    historical record so they differ only by the forcing applied to it:

        E_test   the staged realizations, the Kirsch-Nowak generator driven by
                 the sampled harmonic change factors (the envelope also carries
                 the generator's internal variability, which is part of the
                 ensemble's spread and is not removed)
        CMIP6    the record scaled by each run's raw monthly change factors
                 (:func:`cmip6_shifted_historic_fdcs`), one line per run
        record   the historical record itself

    Args:
        ax: Target axes.
        cache: The FDC cache written by ``scripts/main/forcing_fdc_cache.py``
            (E_test rows, the historical FDC and the exceedance grid).
        cmip6_fdc: ``(K, n_grid)`` CMIP6-shifted record FDCs on the cache's
            exceedance grid.
        pctl: Envelope percentiles (:func:`envelope_pctl`).
        realization_ids: Global ids of the staged E_test's realizations. When
            given, only their cache rows are drawn (:func:`etest_cache_rows`);
            ``None`` draws every cached row.
    """
    exceedance = np.asarray(cache["exceedance"], dtype=float)
    keep = (exceedance >= FDC_PLOT_DOMAIN[0]) & (exceedance <= FDC_PLOT_DOMAIN[1])
    exceedance = exceedance[keep]

    etest = np.asarray(cache["etest_fdc"])
    if realization_ids is not None:
        etest = etest[etest_cache_rows(cache["etest_realization_ids"], realization_ids)]

    _draw_envelope(ax, exceedance, etest[:, keep], pctl, color=ETEST_COLOR)
    for curve in np.asarray(cmip6_fdc, dtype=float)[:, keep]:
        ax.plot(exceedance, curve, color=CMIP6_COLOR, lw=0.7, alpha=0.55, zorder=3)
    ax.plot(exceedance, np.asarray(cache["historic_fdc"], dtype=float)[keep],
            color=HISTORIC_COLOR, lw=2.0, zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(*FDC_PLOT_DOMAIN)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Aggregate NYC reservoir inflow (MGD)")
    ax.set_title("(d) Flow duration curves", loc="left")

    # Two-line label: on one line it would run under the envelope's dry tail.
    handles = [_envelope_handle(envelope_label(pctl)),
               _line_handle("Historical record ×\nCMIP6 change factors"),
               Line2D([0], [0], lw=2.0, color=HISTORIC_COLOR, label="Historical record")]
    ax.legend(handles=handles, loc="lower left", **LEGEND_KW)

