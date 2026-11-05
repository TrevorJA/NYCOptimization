"""
demand_utilization.py - SI figures for the NYC entitlement-utilization axis of E_test.

The fourth deeply uncertain factor of the re-evaluation ensemble is the NYC
entitlement utilization ``u`` (``src.demand_utilization``; methods note
``docs/notes/methods/forcing_parameterization.md``): a per-SOW multiplicative
scale ``k(u)`` on the regression-structured demand series, capped afterwards at
the 800 MGD monthly mean. Four builders, all registered in the SI tier of
``src/figures/registry.py`` and rendered by ``scripts/main/figures.py``:

- :func:`fig_demand_utilization_calibration` -- the calibration ``u(k)`` and
  the monthly demand climatology it implies at four utilization levels
  (reads the tables of ``scripts/supplemental/demand_utilization_calibration.py``).
- :func:`fig_demand_utilization_sampling` -- the sampled ``(e^m, u)`` and
  ``(r1, u)`` planes of the staged E_test with the sampling box.
- :func:`fig_demand_utilization_realized` -- nominal versus realized ``u`` per
  SOW (realized: the staged demand series' mean over complete FFMP years).
- :func:`fig_regret_vs_utilization` -- per-objective regret and gain against
  the FFMP incumbent across utilization bins
  (``design_regret_by_utilization.csv`` of ``scripts/main/compare_designs.py``).

Every builder has the registry signature ``(ctx, out_stub, table_dir)``, saves
through :func:`src.plotting.style.save_figure`, and writes its companion CSV
under ``table_dir``; exact numbers go there, never into panel annotations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import config
from src.plotting import forcing_space as fspace
from src.plotting import style

#: Utilization levels of the climatology panel (the sampled floor, the
#: unscaled series' nominal level, a high level, and the flat-cap limit).
CLIMATOLOGY_LEVELS: tuple[float, ...] = (0.45, 0.70, 0.90, 1.0)

#: Reference scales marked on the calibration curve.
REFERENCE_SCALES: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 2.0, 3.0)

#: Greyscale ramp for the climatology levels, lightest = lowest utilization.
_LEVEL_GREYS: tuple[str, ...] = ("0.75", "0.55", "0.30", "0.0")

#: Calendar-month initials, January through December.
_MONTH_LABELS: tuple[str, ...] = ("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")

#: Realized-utilization colour (the E_test blue of the forcing-space figure).
_REALIZED_COLOR: str = fspace.ETEST_COLOR


def _band_lines(ax, *, axis: str) -> None:
    """Dotted lines at the utilization-band edges on one axis."""
    from src.demand_utilization import UTILIZATION_BAND_EDGES

    for edge in UTILIZATION_BAND_EDGES:
        if axis == "x":
            ax.axvline(edge, color="0.55", lw=0.8, ls=":", zorder=0)
        else:
            ax.axhline(edge, color="0.55", lw=0.8, ls=":", zorder=0)


# ---------------------------------------------------------------------------
# Calibration u(k) + implied monthly climatology
# ---------------------------------------------------------------------------

def fig_demand_utilization_calibration(ctx, out_stub: Path, table_dir: Path) -> dict:
    """The calibration ``u(k)`` and the monthly demand climatology it implies.

    Left: the nominal utilization of the capped, scaled reference extrapolation
    as a function of the scale ``k`` (the persisted calibration table), the
    reference scales marked, and the clip at ``DEMAND_SCALE_MAX``. Right: the
    mean demand by calendar month of ``cap_800(k(u) * D_ref)`` at the levels of
    :data:`CLIMATOLOGY_LEVELS`; ``u = 1`` is the flat 800 MGD limit.

    Raises:
        FileNotFoundError: If the calibration tables are absent.
    """
    from src.demand_utilization import (CALIBRATION_CSV, DEMAND_SCALE_MAX,
                                        REFERENCE_SERIES_CSV, cap_monthly_mean,
                                        k_from_u, u_from_k)

    missing = [p for p in (CALIBRATION_CSV, REFERENCE_SERIES_CSV) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"demand-utilization calibration tables missing: {missing}. Run "
            f"scripts/supplemental/demand_utilization_calibration.py first."
        )
    cap = float(config.NYC_DECREE_DIVERSION_CAP_MGD)
    table = pd.read_csv(CALIBRATION_CSV)
    ref = pd.read_csv(REFERENCE_SERIES_CSV, parse_dates=["datetime"])
    series = pd.Series(ref["aggregate"].to_numpy(dtype=float),
                       index=pd.DatetimeIndex(ref["datetime"]))

    fig, axes = plt.subplots(1, 2, figsize=style.FIGSIZE_WIDE)

    ax = axes[0]
    ax.plot(table["k"], table["u"], color="0.15", lw=2.0, zorder=3,
            label="calibration $u(k)$ (reference record)")
    ref_k = np.array([k for k in REFERENCE_SCALES if k <= float(table["k"].max())])
    ref_u = np.asarray(u_from_k(ref_k), dtype=float)
    ax.plot(ref_k, ref_u, ls="none", marker="o", ms=6, color=fspace.CMIP6_COLOR,
            markeredgecolor="white", zorder=4, label="reference scales")
    ax.axvline(DEMAND_SCALE_MAX, color="0.55", lw=1.0, ls="--", zorder=1,
               label=f"clip, $k_\\max$ = {DEMAND_SCALE_MAX:g}")
    ax.axhline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)
    _band_lines(ax, axis="y")
    ax.set_xlabel("Demand scale, $k$")
    ax.set_ylabel("Nominal entitlement utilization, $u$")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("(a) Calibration of the demand scale", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    clim_rows = []
    for level, grey in zip(CLIMATOLOGY_LEVELS, _LEVEL_GREYS):
        if level >= 1.0:
            clim = pd.Series(cap, index=range(1, 13))
            k_level, clipped = float("nan"), True
        else:
            k_arr, clip_arr = k_from_u(np.array([level]))
            k_level, clipped = float(np.asarray(k_arr)[0]), bool(np.asarray(clip_arr)[0])
            capped = cap_monthly_mean(series * k_level, cap)
            clim = capped.groupby(capped.index.month).mean().reindex(range(1, 13))
        ax.plot(np.arange(12), clim.to_numpy(dtype=float), color=grey, lw=2.0,
                marker="o", ms=4, zorder=3, label=f"$u$ = {level:.2f}")
        clim_rows += [{"u": level, "k": k_level, "k_clipped": clipped,
                       "month": int(m), "demand_mgd": float(v)}
                      for m, v in clim.items()]
    ax.axhline(cap, color=fspace.CMIP6_COLOR, lw=1.2, ls="--", zorder=1,
               label=f"{cap:.0f} MGD monthly-mean cap")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Mean NYC demand (MGD)")
    ax.set_title("(b) Implied monthly climatology", loc="left")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    style.save_figure(fig, out_stub)
    plt.close(fig)

    table.to_csv(table_dir / "demand_utilization_calibration_table.csv", index=False)
    pd.DataFrame(clim_rows).to_csv(table_dir / "demand_utilization_climatology.csv",
                                   index=False)
    return {"n_grid": int(len(table)), "levels": list(CLIMATOLOGY_LEVELS)}


# ---------------------------------------------------------------------------
# Sampled (e^m, u) and (r1, u) planes
# ---------------------------------------------------------------------------

def fig_demand_utilization_sampling(ctx, out_stub: Path, table_dir: Path) -> dict:
    """The sampled ``(e^m, u)`` and ``(r1, u)`` planes of the staged E_test.

    The as-built sample (``forcing_profiles.npz`` of ``ctx.tag``) with the
    sampling box: the climate bounds from
    :func:`src.plotting.forcing_space.etest_param_box` and the utilization
    bounds from ``src.demand_utilization``. The utilization column is an
    independent stratified LHS column appended to the climate design, so each
    plane is a joint Latin hypercube.

    Raises:
        FileNotFoundError: If the staged E_test is absent.
        ValueError: If it carries no utilization factor.
    """
    from src.demand_utilization import DEMAND_UTILIZATION_BOUNDS

    ensemble_dir = config.STAGED_ENSEMBLE_DIR / ctx.tag
    if not ensemble_dir.is_dir():
        raise FileNotFoundError(f"staged E_test not found: {ensemble_dir}")
    sample = fspace.load_etest_sample(ensemble_dir)
    if "u" not in sample:
        raise ValueError(
            f"staged E_test '{ctx.tag}' carries no demand-utilization factor; run "
            f"scripts/supplemental/append_demand_utilization.py first."
        )
    fits = fspace.load_cmip6_fits()
    lo, hi, box_names = fspace.etest_param_box(fits["fit"])
    theta, names = sample["theta"], sample["theta_names"]
    fspace.assert_box_contains(theta, names, lo, hi, box_names)
    u = np.asarray(sample["u"], dtype=float)
    u_lo, u_hi = DEMAND_UTILIZATION_BOUNDS
    em = np.exp(theta[:, names.index("m")])
    r1 = theta[:, names.index("r1")]
    k_m, k_r1 = box_names.index("m"), box_names.index("r1")

    fig, axes = plt.subplots(1, 2, figsize=style.FIGSIZE_WIDE)
    panels = (
        (axes[0], em, "Water-year volume multiplier, $e^{m}$",
         (np.exp(lo[k_m]), np.exp(hi[k_m])), "(a) Volume and utilization"),
        (axes[1], r1, "Seasonal amplitude, $r_1$",
         (lo[k_r1], hi[k_r1]), "(b) Seasonal amplitude and utilization"),
    )
    for ax, x, xlabel, (x0, x1), title in panels:
        ax.scatter(x, u, s=9, c=fspace.ETEST_COLOR, alpha=0.28, linewidths=0, zorder=1)
        ax.add_patch(Rectangle((x0, u_lo), x1 - x0, u_hi - u_lo, fill=False,
                               edgecolor=fspace.ETEST_COLOR, ls="--", lw=1.6, zorder=4))
        _band_lines(ax, axis="y")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nominal entitlement utilization, $u$")
        ax.set_title(title, loc="left")
    axes[0].axvline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)
    handles = [
        Line2D([0], [0], ls="none", marker="o", ms=7, markerfacecolor=fspace.ETEST_COLOR,
               markeredgecolor="none", alpha=0.55, label="E$_{test}$ Latin hypercube draws"),
        Line2D([0], [0], ls="--", lw=1.6, color=fspace.ETEST_COLOR,
               label="E$_{test}$ sampling box"),
        Line2D([0], [0], ls=":", lw=0.8, color="0.55", label="utilization-band edges"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    style.save_figure(fig, out_stub)
    plt.close(fig)

    rows = pd.DataFrame({
        "sow": np.arange(sample["n_sow"]), "em": em, "r1": r1,
        "r2": theta[:, names.index("r2")], "u": u,
        "k": np.asarray(sample["k"], dtype=float),
        "k_clipped": np.asarray(sample["k_clipped"], dtype=bool),
    })
    rows.to_csv(table_dir / "demand_utilization_sampling.csv", index=False)
    return {"n_sow": int(sample["n_sow"]), "u_range": [float(u.min()), float(u.max())],
            "n_clipped": int(rows["k_clipped"].sum())}


# ---------------------------------------------------------------------------
# Nominal vs realized utilization
# ---------------------------------------------------------------------------

def fig_demand_utilization_realized(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Nominal versus realized NYC entitlement utilization per E_test SOW.

    Realized ``u`` is each realization's staged demand series averaged over
    complete FFMP years and divided by the 800 MGD allocation
    (``src.demand_utilization.realized_utilization``), merged per SOW into
    ``demand_utilization_realized_by_sow.csv`` in the parent staging
    directory. Dry SOWs realize above nominal through the negative regression
    slopes and stochastic draws add scatter; the figure reports that coupling,
    it never corrects it. Points are coloured by the SOW's volume multiplier
    ``e^m`` when the staged sample is available.

    Raises:
        FileNotFoundError: If the merged per-SOW table is absent.
    """
    parent = config.STAGED_ENSEMBLE_DIR / ctx.tag
    path = parent / "demand_utilization_realized_by_sow.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing: stage the E_test chunks' demand series with the "
            f"utilization scales, then merge the realized tables into '{ctx.tag}' "
            f"(scripts/supplemental/merge_demand_utilization_realized.py)."
        )
    df = pd.read_csv(path).sort_values("sow").reset_index(drop=True)
    required = {"sow", "u_nominal", "u_realized_mean", "u_realized_min", "u_realized_max"}
    if not required <= set(df.columns):
        raise ValueError(f"{path} lacks columns {sorted(required - set(df.columns))}")

    em = None
    npz = parent / "forcing_profiles.npz"
    if npz.is_file():
        sample = fspace.load_etest_sample(parent)
        if sample["n_sow"] >= int(df["sow"].max()) + 1:
            m_col = sample["theta"][:, sample["theta_names"].index("m")]
            em = np.exp(m_col[df["sow"].to_numpy(dtype=int)])

    x = df["u_nominal"].to_numpy(dtype=float)
    y = df["u_realized_mean"].to_numpy(dtype=float)
    lo = y - df["u_realized_min"].to_numpy(dtype=float)
    hi = df["u_realized_max"].to_numpy(dtype=float) - y

    fig, ax = plt.subplots(figsize=style.FIGSIZE_SINGLE)
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="none", ecolor="0.75", elinewidth=0.6,
                zorder=1)
    if em is not None:
        sc = ax.scatter(x, y, c=em, cmap="viridis", s=14, linewidths=0, zorder=3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Water-year volume multiplier, $e^{m}$")
    else:
        ax.scatter(x, y, color=_REALIZED_COLOR, s=14, linewidths=0, zorder=3)
    span = (min(float(x.min()), float(np.nanmin(y - lo))) - 0.02,
            max(float(x.max()), float(np.nanmax(y + hi))) + 0.02)
    ax.plot(span, span, color="0.2", lw=1.0, ls="--", zorder=2, label="1:1")
    _band_lines(ax, axis="x")
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_xlabel("Nominal entitlement utilization, $u$")
    ax.set_ylabel("Realized utilization (mean over FFMP years / 800 MGD)")
    ax.set_title("Nominal versus realized NYC entitlement utilization", loc="left")
    handles = [
        Line2D([0], [0], ls="--", lw=1.0, color="0.2", label="1:1"),
        Line2D([0], [0], lw=0.6, color="0.75",
               label="min-max over the SOW's realizations"),
        Line2D([0], [0], ls=":", lw=0.8, color="0.55", label="utilization-band edges"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    style.save_figure(fig, out_stub)
    plt.close(fig)

    out = df.copy()
    if em is not None:
        out["em"] = em
    out["realized_minus_nominal"] = y - x
    out.to_csv(table_dir / "demand_utilization_realized.csv", index=False)
    return {"n_sow": int(len(df)),
            "mean_realized_minus_nominal": float(np.nanmean(y - x))}


# ---------------------------------------------------------------------------
# Regret and gain against the incumbent across utilization bins
# ---------------------------------------------------------------------------

def _objectives_of(df: pd.DataFrame) -> list[str]:
    """Objective names carried by the ``regret_mean__`` columns of a table."""
    return [c[len("regret_mean__"):] for c in df.columns
            if c.startswith("regret_mean__") and not c.startswith("regret_mean_best__")]


def fig_regret_vs_utilization(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Per-objective regret and gain against the incumbent across utilization bins.

    Small multiples, one panel per objective, from
    ``design_regret_by_utilization.csv`` (``scripts/main/compare_designs.py``):
    per design the median-over-solutions mean regret (solid) and mean gain
    (dashed) in natural units against the bin-centre utilization ``u``,
    draws and seeds averaged. A final panel carries the best-attained
    satisficing fraction (solid) and low-regret frequency (dashed) per bin.
    How much of any re-optimization gain is contingent on demand reads
    directly off the slope of these curves.

    Raises:
        FileNotFoundError: If the comparison table is absent.
    """
    path = ctx.comparison_dir() / "design_regret_by_utilization.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing: run scripts/main/compare_designs.py (workflow step 10) "
            f"on a re-evaluation whose E_test carries the utilization factor."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    objectives = _objectives_of(df)
    df["u_mid"] = 0.5 * (df["u_lo"] + df["u_hi"])
    agg_cols = ["sat_best", "no_harm_tau_best"] + [
        f"{pre}__{o}" for o in objectives for pre in ("regret_mean", "gain_mean")]
    agg = (df.groupby(["design", "u_bin", "u_mid"], as_index=False)[agg_cols]
             .mean(numeric_only=True))
    designs = [d for d in style.DESIGN_ORDER if d in set(agg["design"])]
    designs += sorted(set(agg["design"]) - set(designs))

    n_panels = len(objectives) + 1
    ncol = min(3, n_panels)
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.4 * nrow), squeeze=False)
    flat = axes.ravel()
    for k, obj in enumerate(objectives):
        ax = flat[k]
        for d in designs:
            g = agg[agg["design"] == d].sort_values("u_mid")
            ax.plot(g["u_mid"], g[f"regret_mean__{obj}"], "-o", ms=3.5, lw=1.6,
                    color=style.design_color(d))
            ax.plot(g["u_mid"], g[f"gain_mean__{obj}"], "--s", ms=3.0, lw=1.2,
                    color=style.design_color(d), alpha=0.8)
        _band_lines(ax, axis="x")
        ax.set_title(style.label_for(obj), fontsize=9)
        ax.set_xlabel("Nominal entitlement utilization, $u$")
        ax.set_ylabel("Regret (solid), gain (dashed)\nvs incumbent, natural units", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = flat[len(objectives)]
    for d in designs:
        g = agg[agg["design"] == d].sort_values("u_mid")
        ax.plot(g["u_mid"], g["sat_best"], "-o", ms=3.5, lw=1.6, color=style.design_color(d))
        ax.plot(g["u_mid"], g["no_harm_tau_best"], "--s", ms=3.0, lw=1.2,
                color=style.design_color(d), alpha=0.8)
    _band_lines(ax, axis="x")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Best policy per bin", fontsize=9)
    ax.set_xlabel("Nominal entitlement utilization, $u$")
    ax.set_ylabel("Satisficing (solid),\nlow-regret frequency (dashed)", fontsize=8)
    ax.grid(True, alpha=0.3)
    for k in range(n_panels, nrow * ncol):
        flat[k].axis("off")

    handles = [Line2D([0], [0], color=style.design_color(d), lw=2.0,
                      label=style.design_label(d)) for d in designs]
    handles += [Line2D([0], [0], color="0.3", lw=1.6, ls="-", label="regret / satisficing"),
                Line2D([0], [0], color="0.3", lw=1.2, ls="--", label="gain / low-regret")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    style.save_figure(fig, out_stub)
    plt.close(fig)

    agg.to_csv(table_dir / "regret_vs_utilization.csv", index=False)
    return {"objectives": objectives, "designs": designs,
            "n_bins": int(agg["u_bin"].nunique())}
