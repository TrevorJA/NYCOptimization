"""src/plotting/forcing_space.py - Panel library for the DU forcing-space figure.

The deeply-uncertain forcing space of the re-evaluation ensemble (E_test) is a
low-order harmonic model of the monthly streamflow change factor, sampled by
Latin hypercube over a box read from a 54-member CMIP6 multimodel ensemble
(methods: ``docs/notes/methods/forcing_parameterization.md``; manuscript
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
(d) :func:`panel_flow_duration`          - change in flow duration curve: E_test vs CMIP6

Provenance rule
---------------
Panels (b) and (c) read the **as-built** sample from the staged ensemble's
``forcing_profiles.npz`` rather than re-running the sampler. Re-sampling would
draw a *different* Latin hypercube and quietly describe an ensemble that was
never simulated.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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

#: CMIP6 projection period -> marker. Period is the only sub-encoding kept in
#: the main figure; the SSP breakdown lives in the SI.
PERIOD_MARKERS: dict[str, str] = {"2020_2059": "o", "2060_2099": "^"}

#: Human-readable period labels for legends.
PERIOD_LABELS: dict[str, str] = {
    "2020_2059": "CMIP6 2020–2059",
    "2060_2099": "CMIP6 2060–2099",
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


def load_cmip6_fits(csv_path=None, *, order: int = 2) -> dict:
    """Fit the harmonic model to every CMIP6 future run.

    ``scengen.forcing_space.load_cmip6_envelope`` drops the ``1980_2019``
    historical siblings, leaving the 54 future runs (2 hydrologic models x
    7 GCMs x 2 SSPs x 2 periods, less 2 missing PRMS/BCC-CSM2-MR ssp370 runs).

    Args:
        csv_path: CMIP6 multiplicative monthly change-factor table. Defaults to
            ``config.ENSEMBLE_FORCING_MEAN_FRAC_CSV`` - the same table the
            ensemble generator reads, so the figure cannot drift from the build.
        order: Number of harmonics (2 = annual + semiannual, the campaign model).

    Returns:
        Dict with ``env`` (the ``(12, K)`` water-year envelope), ``A`` the
        ``(K, 12)`` raw change factors, ``fit`` (see
        ``scengen.forcing_space.fit_harmonic_params``), ``recon`` the ``(K, 12)``
        fitted profiles, and ``df`` a ``K``-row frame of per-run parameters and
        fit quality.
    """
    import config
    from scengen import forcing_space as fs

    if csv_path is None:
        csv_path = config.ENSEMBLE_FORCING_MEAN_FRAC_CSV

    env = fs.load_cmip6_envelope(csv_path)
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
    return {"env": env, "A": A, "fit": fit, "recon": recon, "df": df}


def etest_param_box(fit: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sampling box of the re-evaluation ensemble, in fitted-parameter space.

    Reads the bounds from :mod:`src.etest` rather than hard-coding them, so the
    drawn box always matches the ensemble contract. E_test uses the FULL
    empirical CMIP6 range widened by ``E_TEST_MARGIN``; the SEARCH-side pool
    uses a trimmed 5th-95th percentile box with no widening
    (:func:`search_param_box`). Confusing the two is what made the earlier
    version of this figure show a box far narrower than the ensemble it drew.

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
        water-year change factors, and ``n_sow``.
    """
    npz = Path(ensemble_dir) / "forcing_profiles.npz"
    with np.load(npz, allow_pickle=True) as z:
        rpp = int(z["realizations_per_profile"])
        theta = np.asarray(z["theta_params"], dtype=float)[::rpp]
        a = np.asarray(z["mean_factor_a"], dtype=float)[::rpp]
        names = [str(x) for x in z["theta_param_names"]]
    return {"theta": theta, "theta_names": names, "a": a, "n_sow": theta.shape[0]}


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


def _cmip6_scatter(ax, df, x, y, *, size: float = 42):
    """Scatter CMIP6 runs, one colour, marker by projection period."""
    x, y = np.asarray(x), np.asarray(y)
    for period, marker in PERIOD_MARKERS.items():
        sel = (df["period"] == period).to_numpy()
        if not sel.any():
            continue
        ax.scatter(x[sel], y[sel], c=CMIP6_COLOR, marker=marker, s=size,
                   edgecolor="white", linewidth=0.5, zorder=3)


def _water_year_axis(ax):
    """Label the x axis with water-year months."""
    ax.set_xticks(WY_T)
    ax.set_xticklabels(WY_LABELS)
    ax.set_xlim(-0.4, 11.4)
    ax.set_xlabel("Month of water year")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_harmonic_decomposition(ax, fits: dict) -> dict:
    """(a) The harmonic model built up one term at a time, on one CMIP6 run.

    The illustrative run is chosen deterministically as the run closest to the
    ensemble median in both sampled shape coordinates, so it is representative
    rather than cherry-picked. Its identity is carried in the legend.

    Returns:
        Dict describing the chosen run (``index``, ``label``).
    """
    df, A, recon = fits["df"], fits["A"], fits["recon"]

    k = int(np.argmin(np.abs(df.r1_annual_amp - df.r1_annual_amp.median())
                      + np.abs(df.m_log - df.m_log.median())))
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
    ax.set_title(f"(a) Harmonic decomposition of the change factor\n{run_label}",
                 loc="left")
    # Headroom so the legend clears the winter peak without shrinking the data.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y0 + (y1 - y0) * 1.42)
    ax.legend(loc="upper left", frameon=False, handlelength=1.8,
              borderpad=0.2, labelspacing=0.35)
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
    _cmip6_scatter(ax, df, df.vol_mult_expm, df.r1_annual_amp)

    ax.set_xlabel("Water-year volume multiplier, $e^{m}$")
    ax.set_ylabel("Seasonal amplitude, $r_1$")
    ax.set_title("(b) Sampled forcing space", loc="left")


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


def _pct_change(future: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Percent change of ``future`` against ``baseline``, elementwise.

    Broadcasting-friendly, so one baseline row serves a whole ensemble. Any
    non-positive baseline flow is returned as NaN rather than a spike: on the
    trimmed exceedance domain the aggregate NYC inflow is strictly positive, so
    this is a guard, not a routine path.
    """
    baseline = np.asarray(baseline, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (np.asarray(future, dtype=float) / baseline - 1.0)
    return np.where(baseline > 0, out, np.nan)


def panel_flow_duration(ax, cache, pctl: tuple[float, float]) -> None:
    """(d) Change in the flow duration curve of aggregate NYC reservoir inflow.

    Plotted as the percent change in flow at each exceedance probability rather
    than the absolute curves: on a log flow axis the three sources overlie each
    other almost exactly, and it is the *shift* that the forcing space is
    supposed to span.

    Each source is differenced against its OWN baseline, which is what makes the
    two comparable:

        E_test   against the reconstructed historical record it was generated
                 from, so the change is the sampled forcing signal (plus the
                 Kirsch-Nowak generator's internal variability, which is part of
                 the ensemble's spread and is not removed)
        CMIP6    against that run's own 1980-2019 historical sibling, so each
                 hydrologic model's bias relative to the reconstruction cancels
                 and the line is the model's projected change

    This is the same baseline convention the monthly change factors of panels
    (b)/(c) were fitted on, so panel (d) is an independent daily-flow check on
    the parameterization rather than a restatement of it. The historical record
    is the reference and therefore plots as the 0% line.
    """
    exceedance = np.asarray(cache["exceedance"], dtype=float)
    keep = (exceedance >= FDC_PLOT_DOMAIN[0]) & (exceedance <= FDC_PLOT_DOMAIN[1])
    exceedance = exceedance[keep]

    if "cmip6_baseline_fdc" not in cache.files:
        raise KeyError(
            "FDC cache predates the change-based panel (d): it has no "
            "'cmip6_baseline_fdc'. Rebuild it with "
            "`sbatch --export=ALL,NYCOPT_FIG_REFRESH=1 workflow/13_main_figures.sh`."
        )

    historic = np.asarray(cache["historic_fdc"], dtype=float)[keep]
    etest_pct = _pct_change(np.asarray(cache["etest_fdc"])[:, keep], historic)
    cmip6_pct = _pct_change(np.asarray(cache["cmip6_fdc"])[:, keep],
                            np.asarray(cache["cmip6_baseline_fdc"])[:, keep])

    _draw_envelope(ax, exceedance, etest_pct, pctl, color=ETEST_COLOR)
    for curve in cmip6_pct:
        ax.plot(exceedance, curve, color=CMIP6_COLOR, lw=0.7, alpha=0.55, zorder=3)
    ax.axhline(0.0, color=HISTORIC_COLOR, lw=2.0, zorder=5)

    ax.set_xlim(*FDC_PLOT_DOMAIN)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Change in streamflow (%)")
    ax.set_title("(d) Change in the flow duration curve", loc="left")


# ---------------------------------------------------------------------------
# Shared legend
# ---------------------------------------------------------------------------

#: Column count of the shared figure legend: one column per information
#: SOURCE, matching the colour vocabulary (E_test blue, CMIP6 vermillion,
#: observed black).
LEGEND_NCOL: int = 3


def _legend_spacer() -> Line2D:
    """Invisible, unlabelled entry used to pad a short legend column."""
    return Line2D([], [], ls="none", marker="none", color="none", label=" ")


def shared_legend_handles(pctl: tuple[float, float]) -> list:
    """Handles for the figure-level legend covering panels (b)-(d).

    Panel (a) carries its own legend because the decomposition stages appear
    nowhere else; everything shared by the other three panels is named once here.

    Ordered for a COLUMN-MAJOR ``ncol=LEGEND_NCOL`` legend so each column is one
    colour, i.e. one source of information: the E_test blues, then the CMIP6
    vermillions, then the observed record. Matplotlib fills columns top-to-bottom
    and balances the row count itself, so the short black column is padded with
    invisible spacers to hold the grouping.
    """
    etest = [
        Line2D([0], [0], ls="none", marker="s", ms=10, markerfacecolor=ETEST_COLOR,
               markeredgecolor="none", alpha=0.55, label=envelope_label(pctl)),
        Line2D([0], [0], ls="none", marker="o", ms=7, markerfacecolor=ETEST_COLOR,
               markeredgecolor="none", alpha=0.55,
               label="E$_{test}$ Latin hypercube draws"),
        Line2D([0], [0], ls="--", lw=1.6, color=ETEST_COLOR,
               label="E$_{test}$ sampling box"),
    ]
    cmip6 = [
        Line2D([0], [0], ls="none", marker=PERIOD_MARKERS[p], ms=8,
               markerfacecolor=CMIP6_COLOR, markeredgecolor="white",
               markeredgewidth=0.5, color="none", label=PERIOD_LABELS[p])
        for p in ("2020_2059", "2060_2099")
    ]
    # The vermillion LINES of panels (c) and (d): one per CMIP6 run, each its
    # own fitted change factor / own change against its own historical sibling.
    cmip6.append(
        Line2D([0], [0], lw=1.2, color=CMIP6_COLOR, alpha=0.75,
               label="CMIP6 model-specific change")
    )
    historic = [
        Line2D([0], [0], lw=2.0, color=HISTORIC_COLOR,
               label="Historical record (panel d reference)")
    ]

    rows = max(len(etest), len(cmip6), len(historic))
    columns = [etest, cmip6, historic]
    return [h
            for col in columns
            for h in col + [_legend_spacer()] * (rows - len(col))]
