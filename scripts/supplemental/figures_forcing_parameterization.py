"""Regenerate all forcing-parameterization SI figures + the CMIP6 harmonic-parameter table.

Fits the interpretable 2-harmonic model to the CMIP6 monthly change factors, draws the CMIP6-based
DMDU harmonic hypercube (``scengen.forcing_space.sample_harmonic_forcing``, production default
``margin``), and writes to ``outputs/diagnostics/forcing_parameterization/``:

    cmip6_harmonic_params.csv             per-CMIP6-run fitted params + shape R2 + metadata
    SI_harmonic_fit.png                   decomposition, all fits, fit-quality histogram
    SI_harmonic_param_space.png           CMIP6 futures in interpretable parameter space
    SI_harmonic_lhs_sampling.png          LHS hypercube draws amid the CMIP6 fitted params + box
    SI_harmonic_best_worst_fits.png       3 best / 3 worst fitting CMIP6 profiles
    SI_harmonic_monthly_flow_comparison.png   LHS monthly change/flow vs the CMIP6 monthly range

Run from the repo root::

    PYTHONPATH=$(pwd -W) venv/Scripts/python.exe scripts/supplemental/figures_forcing_parameterization.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import config
from scengen import forcing_space as fs
from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
from src.load.historical_flows import load_historical_flows
from src.plotting import style

WY = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
T = np.arange(12)
W = 2 * np.pi / 12
N_LHS = 1000
SSP_C = {"245": "#2c7fb8", "370": "#d95f0e"}
PER_M = {"2020_2059": "o", "2060_2099": "^"}
OUT = config.OUTPUTS_DIR / "diagnostics" / "forcing_parameterization"
CMIP6 = (Path(config.PROJECT_DIR).parent / "CMIP6_multimodel_streamflow" / "stats"
         / "diff_relative_to_dataset_baseline"
         / "nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv")


def _meta(col):
    return (re.search(r"ssp(\d+)", col).group(1),
            re.search(r"(\d{4}_\d{4})$", col).group(1),
            re.search(r"RAPID_(.+?)_ssp", col).group(1))


def load_and_fit():
    env = fs.load_cmip6_envelope(CMIP6)
    cols = list(env.columns)
    A = env.values.T  # (K, 12) raw CMIP6 change factors, water-year
    fit = fs.fit_harmonic_params(env, order=2)
    m, r1, psi1 = fit["m"], fit["amp"][:, 0], fit["phase"][:, 0]
    r2, psi2 = fit["amp"][:, 1], fit["phase"][:, 1]
    params = np.column_stack([m, r1, psi1, r2, psi2])
    recon = fs.reconstruct_harmonic(params, order=2, floor=1e-6)
    L, pred = np.log(A), np.log(recon)
    shape_r2 = 1 - ((L - pred) ** 2).sum(1) / ((L - L.mean(1, keepdims=True)) ** 2).sum(1)
    ssp, per, gcm = zip(*[_meta(c) for c in cols])
    df = pd.DataFrame({
        "scenario": cols, "gcm": gcm, "ssp": ssp, "period": per,
        "m_log": m, "vol_mult_expm": np.exp(m),
        "r1_annual_amp": r1, "phase1_rad": psi1, "peak1_wy_month": (psi1 / W) % 12,
        "r2_semiann_amp": r2, "phase2_rad": psi2, "shape_R2": shape_r2,
    })
    return env, A, fit, df, recon


def _cmip_scatter(ax, df, x, y):
    for s in SSP_C:
        for p in PER_M:
            msk = ((df.ssp == s) & (df.period == p)).to_numpy()
            ax.scatter(np.asarray(x)[msk], np.asarray(y)[msk], c=SSP_C[s], marker=PER_M[p],
                       s=44, edgecolor="white", linewidth=0.4, zorder=3)


def fig_fit(A, df, recon):
    fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)
    rep = int(np.argmin(np.abs(df.r1_annual_amp - df.r1_annual_amp.median())
                        + np.abs(df.m_log - df.m_log.median())))
    ax = fig.add_subplot(1, 3, 1)
    mean_term = np.exp(np.full(12, df.m_log[rep]))
    h1 = np.exp(df.m_log[rep] + df.r1_annual_amp[rep] * np.cos(W * T - df.phase1_rad[rep]))
    ax.plot(T, A[rep], "o", color="k", ms=5, label="CMIP6 $a_j$", zorder=5)
    ax.plot(T, mean_term, color="#999999", lw=1.6, ls="--", label="mean $e^{m}$")
    ax.plot(T, h1, color="#1f77b4", lw=1.6, label="+ annual ($r_1,\\phi_1$)")
    ax.plot(T, recon[rep], color="#d62728", lw=2.0, label="+ semiannual ($r_2$)")
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="change factor $a_j$",
           title=f"(a) Harmonic decomposition\n{df.gcm[rep]} ssp{df.ssp[rep]} {df.period[rep]}")
    ax.legend(fontsize=7.5)
    ax = fig.add_subplot(1, 3, 2)
    for k in range(len(A)):
        ax.plot(T, recon[k], color="#1f77b4", lw=0.5, alpha=0.35)
    ax.fill_between(T, A.min(0), A.max(0), color="0.85", alpha=0.6, zorder=0, label="CMIP6 envelope")
    ax.plot(T, A.mean(0), "k-", lw=1.8, label="CMIP6 mean")
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="change factor $a_j$", title="(b) 2-harmonic fits, all CMIP6 futures")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 3, 3)
    ax.hist(df.shape_R2, bins=np.linspace(0.0, 1.0, 21), color="#1f77b4", edgecolor="white")
    ax.axvline(df.shape_R2.median(), color="#d62728", lw=2, label=f"median={df.shape_R2.median():.2f}")
    ax.set(xlabel="per-profile shape $R^2$ (2 harmonics)", ylabel="# CMIP6 futures", title="(c) Fit quality")
    ax.legend(fontsize=8)
    style.save_figure(fig, OUT / "SI_harmonic_fit"); plt.close(fig)


def fig_param_space(df):
    fig = plt.figure(figsize=(11.5, 9.5), constrained_layout=True)
    ax = fig.add_subplot(2, 2, 1)
    _cmip_scatter(ax, df, df.vol_mult_expm, df.r1_annual_amp)
    ax.axvline(1, color="0.6", lw=0.7, ls=":")
    ax.set(xlabel="annual volume multiplier $e^{m}$", ylabel="annual amplitude $r_1$", title="(a) Seasonality vs volume")
    ax = fig.add_subplot(2, 2, 2, projection="polar")
    for s in SSP_C:
        for p in PER_M:
            msk = ((df.ssp == s) & (df.period == p)).to_numpy()
            ax.scatter(df.phase1_rad[msk], df.r1_annual_amp[msk], c=SSP_C[s], marker=PER_M[p],
                       s=44, edgecolor="white", linewidth=0.4)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, 30))); ax.set_xticklabels(WY)
    ax.set_title("(b) Annual harmonic: peak month (angle)\n& amplitude $r_1$ (radius)", pad=18)
    ax = fig.add_subplot(2, 2, 3)
    _cmip_scatter(ax, df, df.r1_annual_amp, df.r2_semiann_amp)
    ax.set(xlabel="annual amplitude $r_1$", ylabel="semiannual amplitude $r_2$", title="(c) Second- vs first-harmonic")
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    for s in SSP_C:
        for p in PER_M:
            msk = ((df.ssp == s) & (df.period == p)).to_numpy()
            ax.scatter(df.vol_mult_expm[msk], df.r1_annual_amp[msk], df.r2_semiann_amp[msk],
                       c=SSP_C[s], marker=PER_M[p], s=40, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("$e^{m}$"); ax.set_ylabel("$r_1$"); ax.set_zlabel("$r_2$")
    ax.set_title("(d) Joint parameter cloud"); ax.view_init(elev=22, azim=-60)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SSP_C["245"], markersize=9, label="SSP2-4.5"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=SSP_C["370"], markersize=9, label="SSP3-7.0"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markersize=9, label="2020-2059"),
           Line2D([0], [0], marker="^", color="w", markerfacecolor="0.4", markersize=9, label="2060-2099")]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CMIP6 futures in interpretable harmonic-parameter space", fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_param_space"); plt.close(fig)


def fig_lhs_sampling(env, fit, df):
    # Default fixed-phase sampler: amplitudes [m, r1, r2] sampled, phases fixed at the canonical shape.
    _, p_lhs, _ = fs.sample_harmonic_forcing(800, env, seed=0, return_params=True)
    lo, hi, _ = fs.harmonic_param_box(fit)  # full box; amplitude indices m=0, r1=1, r2=3
    em_l, r1_l, r2_l = np.exp(p_lhs[:, 0]), p_lhs[:, 1], p_lhs[:, 2]
    peak_canon = (fs.canonical_phases(env)[0] / W) % 12
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)

    def box(ax, xlo, xhi, ylo, yhi):
        ax.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, ec="k", ls="--", lw=1.3, zorder=4))
    ax = axes[0]
    ax.scatter(em_l, r1_l, s=7, c="0.7", alpha=0.5, zorder=1)
    _cmip_scatter(ax, df, df.vol_mult_expm, df.r1_annual_amp); box(ax, np.exp(lo[0]), np.exp(hi[0]), lo[1], hi[1])
    ax.axvline(1, color="0.6", lw=0.7, ls=":")
    ax.set(xlabel="annual volume multiplier $e^{m}$", ylabel="annual amplitude $r_1$", title="(a) volume vs seasonality")
    ax = axes[1]
    ax.scatter(r1_l, r2_l, s=7, c="0.7", alpha=0.5, zorder=1)
    _cmip_scatter(ax, df, df.r1_annual_amp, df.r2_semiann_amp); box(ax, lo[1], hi[1], lo[3], hi[3])
    ax.set(xlabel="annual amplitude $r_1$", ylabel="semiannual amplitude $r_2$", title="(b) first vs second harmonic")
    ax = axes[2]
    _cmip_scatter(ax, df, df.peak1_wy_month, df.r1_annual_amp)
    ax.axvline(peak_canon, color="k", ls="--", lw=1.6, label="fixed canonical phase")
    ax.set(xlabel="annual-harmonic peak month (WY index)", ylabel="annual amplitude $r_1$",
           title="(c) phase fixed at canonical CMIP6 shape")
    ax.set_xticks(range(12)); ax.set_xticklabels(WY); ax.legend(fontsize=7.5, loc="upper right")
    leg = [Line2D([0], [0], marker="s", color="w", mfc="0.7", ms=9, label="LHS amplitude draws"),
           Line2D([0], [0], marker="o", color="w", mfc=SSP_C["245"], ms=9, label="CMIP6 SSP2-4.5"),
           Line2D([0], [0], marker="o", color="w", mfc=SSP_C["370"], ms=9, label="CMIP6 SSP3-7.0"),
           Line2D([0], [0], marker="^", color="w", mfc="0.4", ms=9, label="end-century"),
           Line2D([0], [0], ls="--", color="k", label="DMDU box (CMIP6 90% range)")]
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Fixed-phase LHS sampling amid the CMIP6 fitted harmonic parameters", fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_lhs_sampling"); plt.close(fig)


def fig_best_worst(A, df, recon):
    order = np.argsort(df.shape_R2.to_numpy())
    best, worst = order[::-1][:3], order[:3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharey=True)
    cmap = plt.cm.tab10
    for ax, idxs, title in [(axes[0], best, "(a) 3 best-fitting"), (axes[1], worst, "(b) 3 worst-fitting")]:
        for j, k in enumerate(idxs):
            c = cmap(j)
            ax.plot(T, A[k], "o", color=c, ms=5, zorder=3)
            ax.plot(T, recon[k], "-", color=c, lw=1.8,
                    label=f"{df.gcm[k]} ssp{df.ssp[k]} {df.period[k].split('_')[1]} ($R^2$={df.shape_R2[k]:.2f})")
        ax.axhline(1, color="k", lw=0.6, ls=":")
        ax.set_xticks(range(12)); ax.set_xticklabels(WY)
        ax.set(xlabel="month (water year)", title=title); ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("change factor $a_j$  (points = CMIP6, line = fit)")
    fig.suptitle("Range of CMIP6 change-profile behaviors and their harmonic fits", fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_best_worst_fits"); plt.close(fig)


def fig_monthly_flow(env, A):
    a_lhs = fs.sample_harmonic_forcing(N_LHS, env, seed=0)  # production default margin
    hist = load_historical_flows(gage=False, period="full")
    nyc = hist.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    base_wy = nyc.groupby(nyc.index.month).mean().reindex(list(fs.WATER_YEAR_MONTHS)).to_numpy()
    flow_cmip6, flow_lhs = base_wy[None, :] * A, base_wy[None, :] * a_lhs

    def band(ax, X, color, label):
        ax.fill_between(T, X.min(0), X.max(0), color=color, alpha=0.18, lw=0, label=f"{label} min–max")
        ax.plot(T, np.percentile(X, 5, 0), color=color, lw=0.9, ls="--")
        ax.plot(T, np.percentile(X, 95, 0), color=color, lw=0.9, ls="--")
        ax.plot(T, np.median(X, 0), color=color, lw=2.0, label=f"{label} median")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    ax = axes[0]
    band(ax, A, "#7f7f7f", "CMIP6 (raw)"); band(ax, a_lhs, "#1f77b4", "harmonic LHS")
    ax.axhline(1, color="k", lw=0.7, ls=":"); ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="monthly mean change factor $a_j$",
           title="(a) Change factor: harmonic-LHS vs CMIP6")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax = axes[1]
    band(ax, flow_cmip6, "#7f7f7f", "CMIP6 (raw)"); band(ax, flow_lhs, "#1f77b4", "harmonic LHS")
    ax.plot(T, base_wy, color="k", lw=2.0, label="historical baseline")
    ax.set_xticks(T); ax.set_xticklabels(WY)
    ax.set(xlabel="month (water year)", ylabel="monthly mean NYC inflow (MGD)",
           title="(b) Resulting monthly flow: harmonic-LHS vs CMIP6")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("Monthly flow change from harmonic-LHS sampling vs the CMIP6 monthly range", fontsize=12)
    style.save_figure(fig, OUT / "SI_harmonic_monthly_flow_comparison"); plt.close(fig)
    extend = (np.maximum(0, A.min(0) - a_lhs.min(0)) + np.maximum(0, a_lhs.max(0) - A.max(0))).mean()
    print(f"[fig] mean monthly change-factor extension beyond CMIP6 min/max: {extend:.3f}")


def main():
    style.apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    env, A, fit, df, recon = load_and_fit()
    df.to_csv(OUT / "cmip6_harmonic_params.csv", index=False)
    fig_fit(A, df, recon)
    fig_param_space(df)
    fig_lhs_sampling(env, fit, df)
    fig_best_worst(A, df, recon)
    fig_monthly_flow(env, A)
    print(f"[fig] wrote 5 figures + cmip6_harmonic_params.csv to {OUT}")
    print(f"[fig] fit shape R2 median={df.shape_R2.median():.2f} (p5={df.shape_R2.quantile(.05):.2f}, "
          f"p95={df.shape_R2.quantile(.95):.2f})")


if __name__ == "__main__":
    main()
