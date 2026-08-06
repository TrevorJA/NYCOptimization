"""Supporting Information figures for the forcing-space parameterization (Text S6).

Fits the interpretable 2-harmonic model to the 54 CMIP6 monthly change-factor
profiles and documents the choices the main-manuscript figure only summarises:
fit quality across the whole ensemble, the best and worst individual fits, the
full parameter space including the phases that are NOT sampled, and the two
sampling boxes side by side.

Writes to ``outputs/diagnostics/forcing_parameterization/``:

    cmip6_harmonic_params.csv                  per-run fitted params + shape R2
    SI_harmonic_fit.png                        decomposition, all fits, fit quality
    SI_harmonic_param_space.png                CMIP6 futures in parameter space
    SI_harmonic_lhs_sampling.png               LHS draws amid the CMIP6 params + boxes
    SI_harmonic_best_worst_fits.png            3 best / 3 worst fitting profiles
    SI_harmonic_monthly_flow_comparison.png    sampled monthly range vs the CMIP6 range

The harmonic fitting, the sampling boxes, and the as-built E_test sample all
come from :mod:`src.plotting.forcing_space`, shared with the main-manuscript
figure, so the two can never disagree about what was built.

TWO BOXES, NOT ONE. The search-side candidate pool samples a trimmed 5th-95th
percentile box with no widening, while E_test samples the FULL empirical CMIP6
range widened by 25% so it strictly contains every search box. An earlier
version of these figures drew the search box while describing E_test; each
panel here names the box it draws.

Run through the SI figure driver::

    sbatch workflow/supplemental/si_figures_design.sh
    # or, directly:
    python3 scripts/supplemental/figures_forcing_parameterization.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import config
from scengen import forcing_space as fs
from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
from src.load.historical_flows import load_historical_flows
from src.plotting import forcing_space as fspace
from src.plotting import style

WY = fspace.WY_LABELS
T = fspace.WY_T
W = fspace.OMEGA

#: SSP colouring is kept HERE (and only here): the main figure collapses CMIP6
#: to one colour, so the scenario breakdown is an SI-only refinement.
SSP_C = {"245": "#2c7fb8", "370": "#d95f0e"}
PER_M = fspace.PERIOD_MARKERS

OUT = config.OUTPUTS_DIR / "diagnostics" / "forcing_parameterization"


def _cmip_scatter(ax, df, x, y):
    """Scatter CMIP6 runs coloured by SSP, marker by projection period."""
    x, y = np.asarray(x), np.asarray(y)
    for ssp, colour in SSP_C.items():
        for period, marker in PER_M.items():
            sel = ((df.ssp == ssp) & (df.period == period)).to_numpy()
            if not sel.any():
                continue
            ax.scatter(x[sel], y[sel], c=colour, marker=marker, s=44,
                       edgecolor="white", linewidth=0.4, zorder=3)


def _box(ax, xlo, xhi, ylo, yhi, *, colour="k", ls="--", label=None):
    ax.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False,
                           ec=colour, ls=ls, lw=1.3, zorder=4, label=label))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_fit(fits):
    """Decomposition on a representative run, all fits, and the fit-quality histogram."""
    A, df, recon = fits["A"], fits["df"], fits["recon"]
    fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)

    rep = int(np.argmin(np.abs(df.r1_annual_amp - df.r1_annual_amp.median())
                        + np.abs(df.m_log - df.m_log.median())))
    ax = fig.add_subplot(1, 3, 1)
    level = np.exp(np.full(12, df.m_log[rep]))
    plus_annual = np.exp(df.m_log[rep]
                         + df.r1_annual_amp[rep] * np.cos(W * T - df.phase1_rad[rep]))
    ax.plot(T, A[rep], "o", color="k", ms=5, label="CMIP6 $a_j$", zorder=5)
    ax.plot(T, level, color="#999999", lw=1.6, ls="--", label="mean $e^{m}$")
    ax.plot(T, plus_annual, color="#1f77b4", lw=1.6, label="+ annual ($r_1,\\phi_1$)")
    ax.plot(T, recon[rep], color="#d62728", lw=2.0, label="+ semiannual ($r_2$)")
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="change factor $a_j$",
           title=f"(a) Harmonic decomposition\n{df.gcm[rep]} ssp{df.ssp[rep]} {df.period[rep]}")
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(1, 3, 2)
    for profile in recon:
        ax.plot(T, profile, color="#1f77b4", lw=0.5, alpha=0.35)
    ax.fill_between(T, A.min(0), A.max(0), color="0.85", alpha=0.6, zorder=0,
                    label="CMIP6 envelope")
    ax.plot(T, A.mean(0), "k-", lw=1.8, label="CMIP6 mean")
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="change factor $a_j$",
           title="(b) 2-harmonic fits, all CMIP6 futures")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    ax.hist(df.shape_R2, bins=np.linspace(0.0, 1.0, 21), color="#1f77b4",
            edgecolor="white")
    ax.axvline(df.shape_R2.median(), color="#d62728", lw=2,
               label=f"median={df.shape_R2.median():.2f}")
    ax.set(xlabel="per-profile shape $R^2$ (2 harmonics)", ylabel="# CMIP6 futures",
           title="(c) Fit quality")
    ax.legend(fontsize=8)

    style.save_figure(fig, OUT / "SI_harmonic_fit"); plt.close(fig)


def fig_param_space(fits):
    """The full fitted parameter space, including the phases that are not sampled."""
    df = fits["df"]
    fig = plt.figure(figsize=(11.5, 9.5), constrained_layout=True)

    ax = fig.add_subplot(2, 2, 1)
    _cmip_scatter(ax, df, df.vol_mult_expm, df.r1_annual_amp)
    ax.axvline(1, color="0.6", lw=0.7, ls=":")
    ax.set(xlabel="annual volume multiplier $e^{m}$", ylabel="annual amplitude $r_1$",
           title="(a) Seasonality vs volume")

    ax = fig.add_subplot(2, 2, 2, projection="polar")
    for ssp, colour in SSP_C.items():
        for period, marker in PER_M.items():
            sel = ((df.ssp == ssp) & (df.period == period)).to_numpy()
            ax.scatter(df.phase1_rad[sel], df.r1_annual_amp[sel], c=colour,
                       marker=marker, s=44, edgecolor="white", linewidth=0.4)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, 30))); ax.set_xticklabels(WY)
    ax.set_title("(b) Annual harmonic: peak month (angle)\n& amplitude $r_1$ (radius)",
                 pad=18)

    ax = fig.add_subplot(2, 2, 3)
    _cmip_scatter(ax, df, df.r1_annual_amp, df.r2_semiann_amp)
    ax.set(xlabel="annual amplitude $r_1$", ylabel="semiannual amplitude $r_2$",
           title="(c) Second- vs first-harmonic")

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    for ssp, colour in SSP_C.items():
        for period, marker in PER_M.items():
            sel = ((df.ssp == ssp) & (df.period == period)).to_numpy()
            ax.scatter(df.vol_mult_expm[sel], df.r1_annual_amp[sel],
                       df.r2_semiann_amp[sel], c=colour, marker=marker, s=40,
                       edgecolor="white", linewidth=0.3)
    ax.set_xlabel("$e^{m}$"); ax.set_ylabel("$r_1$"); ax.set_zlabel("$r_2$")
    ax.set_title("(d) Joint parameter cloud"); ax.view_init(elev=22, azim=-60)

    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SSP_C["245"],
                  markersize=9, label="SSP2-4.5"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=SSP_C["370"],
                  markersize=9, label="SSP3-7.0"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4",
                  markersize=9, label="2020-2059"),
           Line2D([0], [0], marker="^", color="w", markerfacecolor="0.4",
                  markersize=9, label="2060-2099")]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CMIP6 futures in interpretable harmonic-parameter space", fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_param_space"); plt.close(fig)


def fig_lhs_sampling(fits, sample):
    """The as-built E_test draws against the CMIP6 points, and both sampling boxes.

    Panels (a) and (b) draw the E_test box around the E_test draws actually
    used; panel (a) additionally shows the narrower search-pool box, so the
    containment relation between the two designs is visible rather than
    asserted. Panel (c) shows the phase coordinate that is held fixed.
    """
    df, env = fits["df"], fits["env"]
    e_lo, e_hi, names = fspace.etest_param_box(fits["fit"])
    s_lo, s_hi, _ = fspace.search_param_box(fits["fit"])
    fspace.assert_box_contains(sample["theta"], sample["theta_names"],
                               e_lo, e_hi, names)

    i_m = sample["theta_names"].index("m")
    i_r1 = sample["theta_names"].index("r1")
    i_r2 = sample["theta_names"].index("r2")
    k_m, k_r1, k_r2 = names.index("m"), names.index("r1"), names.index("r2")
    em, r1, r2 = np.exp(sample["theta"][:, i_m]), sample["theta"][:, i_r1], \
        sample["theta"][:, i_r2]
    peak_canon = (fs.canonical_phases(env)[0] / W) % 12

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)

    ax = axes[0]
    ax.scatter(em, r1, s=7, c="0.7", alpha=0.5, zorder=1)
    _cmip_scatter(ax, df, df.vol_mult_expm, df.r1_annual_amp)
    _box(ax, np.exp(e_lo[k_m]), np.exp(e_hi[k_m]), e_lo[k_r1], e_hi[k_r1])
    _box(ax, np.exp(s_lo[k_m]), np.exp(s_hi[k_m]), s_lo[k_r1], s_hi[k_r1],
         colour="0.35", ls=":")
    ax.axvline(1, color="0.6", lw=0.7, ls=":")
    ax.set(xlabel="annual volume multiplier $e^{m}$", ylabel="annual amplitude $r_1$",
           title="(a) volume vs seasonality")

    ax = axes[1]
    ax.scatter(r1, r2, s=7, c="0.7", alpha=0.5, zorder=1)
    _cmip_scatter(ax, df, df.r1_annual_amp, df.r2_semiann_amp)
    _box(ax, e_lo[k_r1], e_hi[k_r1], e_lo[k_r2], e_hi[k_r2])
    ax.set(xlabel="annual amplitude $r_1$", ylabel="semiannual amplitude $r_2$",
           title="(b) first vs second harmonic")

    ax = axes[2]
    _cmip_scatter(ax, df, df.peak1_wy_month, df.r1_annual_amp)
    ax.axvline(peak_canon, color="k", ls="--", lw=1.6, label="fixed canonical phase")
    ax.set(xlabel="annual-harmonic peak month (WY index)",
           ylabel="annual amplitude $r_1$",
           title="(c) phase fixed at canonical CMIP6 shape")
    ax.set_xticks(range(12)); ax.set_xticklabels(WY)
    ax.legend(fontsize=7.5, loc="upper right")

    leg = [Line2D([0], [0], marker="s", color="w", mfc="0.7", ms=9,
                  label=f"E$_{{test}}$ LHS draws (n={sample['n_sow']})"),
           Line2D([0], [0], marker="o", color="w", mfc=SSP_C["245"], ms=9,
                  label="CMIP6 SSP2-4.5"),
           Line2D([0], [0], marker="o", color="w", mfc=SSP_C["370"], ms=9,
                  label="CMIP6 SSP3-7.0"),
           Line2D([0], [0], marker="^", color="w", mfc="0.4", ms=9,
                  label="end-century"),
           Line2D([0], [0], ls="--", color="k",
                  label="E$_{test}$ box (full CMIP6 range +25%)"),
           Line2D([0], [0], ls=":", color="0.35",
                  label="search-pool box (CMIP6 90% range)")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Fixed-phase sampling amid the CMIP6 fitted harmonic parameters",
                 fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_lhs_sampling"); plt.close(fig)


def fig_best_worst(fits):
    """The best- and worst-fitting individual CMIP6 profiles, bounding fit quality."""
    A, df, recon = fits["A"], fits["df"], fits["recon"]
    order = np.argsort(df.shape_R2.to_numpy())
    best, worst = order[::-1][:3], order[:3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True,
                             sharey=True)
    cmap = plt.cm.tab10
    for ax, idxs, title in [(axes[0], best, "(a) 3 best-fitting"),
                            (axes[1], worst, "(b) 3 worst-fitting")]:
        for j, k in enumerate(idxs):
            c = cmap(j)
            ax.plot(T, A[k], "o", color=c, ms=5, zorder=3)
            ax.plot(T, recon[k], "-", color=c, lw=1.8,
                    label=f"{df.gcm[k]} ssp{df.ssp[k]} {df.period[k].split('_')[1]} "
                          f"($R^2$={df.shape_R2[k]:.2f})")
        ax.axhline(1, color="k", lw=0.6, ls=":")
        ax.set_xticks(range(12)); ax.set_xticklabels(WY)
        ax.set(xlabel="month (water year)", title=title)
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("change factor $a_j$  (points = CMIP6, line = fit)")
    fig.suptitle("Range of CMIP6 change-profile behaviors and their harmonic fits",
                 fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_best_worst_fits"); plt.close(fig)


def fig_monthly_flow(fits, sample):
    """The as-built monthly change/flow range against the raw CMIP6 monthly range."""
    A = fits["A"]
    a_sample = sample["a"]
    hist = load_historical_flows(gage=False, period="full")
    nyc = hist.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    base_wy = nyc.groupby(nyc.index.month).mean().reindex(
        list(fs.WATER_YEAR_MONTHS)).to_numpy()

    def band(ax, X, color, label):
        ax.fill_between(T, X.min(0), X.max(0), color=color, alpha=0.18, lw=0,
                        label=f"{label} min–max")
        ax.plot(T, np.percentile(X, 5, 0), color=color, lw=0.9, ls="--")
        ax.plot(T, np.percentile(X, 95, 0), color=color, lw=0.9, ls="--")
        ax.plot(T, np.median(X, 0), color=color, lw=2.0, label=f"{label} median")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    ax = axes[0]
    band(ax, A, "#7f7f7f", "CMIP6 (raw)")
    band(ax, a_sample, "#1f77b4", "E$_{test}$ sample")
    ax.axhline(1, color="k", lw=0.7, ls=":")
    ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="monthly mean change factor $a_j$",
           title="(a) Change factor: E$_{test}$ sample vs CMIP6")
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    ax = axes[1]
    band(ax, base_wy[None, :] * A, "#7f7f7f", "CMIP6 (raw)")
    band(ax, base_wy[None, :] * a_sample, "#1f77b4", "E$_{test}$ sample")
    ax.plot(T, base_wy, color="k", lw=2.0, label="historical baseline")
    ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="monthly mean NYC inflow (MGD)",
           title="(b) Resulting monthly flow: E$_{test}$ sample vs CMIP6")
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    fig.suptitle("Monthly flow change of the sampled forcing space vs the CMIP6 range",
                 fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_monthly_flow_comparison"); plt.close(fig)

    extend = (np.maximum(0, A.min(0) - a_sample.min(0))
              + np.maximum(0, a_sample.max(0) - A.max(0))).mean()
    print(f"[fig] mean monthly change-factor extension beyond CMIP6 min/max: {extend:.3f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    style.apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    from src import etest as etest_mod
    slug = etest_mod.E_TEST_VARIANTS[etest_mod.E_TEST_VARIANT].slug
    ensemble_dir = config.STAGED_ENSEMBLE_DIR / slug
    if not ensemble_dir.is_dir():
        raise FileNotFoundError(
            f"staged E_test not found: {ensemble_dir}. These figures describe the "
            "AS-BUILT sample and never re-run the sampler."
        )

    fits = fspace.load_cmip6_fits()
    sample = fspace.load_etest_sample(ensemble_dir)
    fits["df"].to_csv(OUT / "cmip6_harmonic_params.csv", index=False)

    fig_fit(fits)
    fig_param_space(fits)
    fig_lhs_sampling(fits, sample)
    fig_best_worst(fits)
    fig_monthly_flow(fits, sample)

    df = fits["df"]
    print(f"[fig] wrote 5 figures + cmip6_harmonic_params.csv to {OUT}")
    print(f"[fig] {len(df)} CMIP6 future runs; E_test = {slug} "
          f"({sample['n_sow']} SOWs)")
    print(f"[fig] fit shape R2 median={df.shape_R2.median():.2f} "
          f"(p5={df.shape_R2.quantile(.05):.2f}, p95={df.shape_R2.quantile(.95):.2f})")


if __name__ == "__main__":
    main()
