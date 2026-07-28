"""diagnose_persistence_axis.py - Feasibility of the persistence-tilted bootstrap DU axis.

Supplemental diagnostic for the interannual-persistence forcing axis
(``scengen.persistence``): the Kirsch bootstrap index matrix is drawn from a latent
annual AR(1) state through a Gaussian copula (loading ``lam``, memory ``phi``), which
leaves every cell's bootstrap marginal exactly uniform (monthly moments + cross-site
structure preserved in distribution) while making adjacent synthetic years resample
wetness-similar historical years.

This driver measures, on the full-record Kirsch fit, the trade the mechanism makes:

    gain        annual lag-1 autocorrelation of aggregate NYC inflow (the axis target),
                plus multi-year drought proxies (min 3-yr rolling total, below-median
                run length)
    distortion  within-year adjacent-month correlation inflation (the latent state is
                shared by all 12 months of a year), monthly mean/std drift, and NYC
                cross-site correlation drift - all relative to the lam=0 control

against CMIP6-anchored targets: the historical reconstruction's rho1 (~0.27) and the
CMIP6 future-run range (p5-p95 ~ [-0.10, 0.40], max ~0.49).

Configuration (no CLI value flags):

    NYCOPT_PERSDIAG_R        realizations per setting  (default 64)
    NYCOPT_PERSDIAG_YEARS    years per realization     (default 50)
    NYCOPT_PERSDIAG_SEED     root seed                 (default 20260728)

Outputs -> ``outputs/supplemental/persistence_axis/``:
    settings_summary.csv   per-(phi, lam) gain/distortion metrics
    persistence_axis.png   rho1 response curve + distortion panel
    findings.json          headline numbers
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from scengen.persistence import persistent_bootstrap_indices, wetness_rank_order

from src.ensemble_generation import _fit_kirsch, _load_masked_flows

R = int(os.environ.get("NYCOPT_PERSDIAG_R", "64"))
YEARS = int(os.environ.get("NYCOPT_PERSDIAG_YEARS", "50"))
SEED = int(os.environ.get("NYCOPT_PERSDIAG_SEED", "20260728"))

NYC = ["cannonsville", "pepacton", "neversink"]
#: (phi, lam) settings; (0.8, 0.0) is the exact-control reduction.
SETTINGS: list[tuple[float, float]] = [
    (0.8, 0.0), (0.8, 0.1), (0.8, 0.2), (0.8, 0.35), (0.8, 0.5), (0.8, 0.75), (0.8, 1.0),
    (0.5, 0.35), (0.9, 0.35),
]

OUT_DIR = config.OUTPUTS_DIR / "supplemental" / "persistence_axis"


def _annual_rho1(ann: np.ndarray) -> float:
    """Lag-1 autocorrelation of one realization's annual totals."""
    x = ann - ann.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else np.nan


def _realization_metrics(df: pd.DataFrame) -> dict:
    """Gain metrics of one generated monthly realization (aggregate NYC inflow)."""
    agg = df[NYC].sum(axis=1)
    ann = agg.groupby(agg.index.year).sum().to_numpy(dtype=float)
    below = ann < np.median(ann)
    runs, run = [], 0
    for b in below:
        run = run + 1 if b else 0
        runs.append(run)
    roll3 = np.convolve(ann, np.ones(3), mode="valid")
    return {
        "rho1": _annual_rho1(ann),
        "max_dry_run": int(np.max(runs)),
        "min_roll3_frac": float(roll3.min() / (3.0 * ann.mean())),
    }


def _distortion_metrics(frames: list[pd.DataFrame]) -> dict:
    """Structure metrics pooled over one setting's realizations (log space)."""
    logs = [np.log(df[NYC]) for df in frames]
    pooled = pd.concat(logs)
    month = pooled.index.month

    # Standardize per (month, site) so correlations are moment-free.
    z = pooled.copy()
    for m in range(1, 13):
        sel = month == m
        z.loc[sel] = (pooled.loc[sel] - pooled.loc[sel].mean()) / pooled.loc[sel].std()

    # Within-year adjacent-month correlation of the aggregate z (11 within + boundary).
    za = z.mean(axis=1)
    adj = []
    for df_z in np.array_split(za.to_numpy(), len(frames)):
        adj.append(np.corrcoef(df_z[:-1], df_z[1:])[0, 1])
    adjacent_corr = float(np.mean(adj))

    # Cross-site correlation (3 NYC pairs) of z.
    zc = z.to_numpy()
    cross = np.corrcoef(zc.T)
    cross_site = float(np.mean(cross[np.triu_indices(3, k=1)]))

    by_month = pooled.groupby(month)
    return {
        "adjacent_corr": adjacent_corr,
        "cross_site_corr": cross_site,
        "monthly_mean": by_month.mean(),
        "monthly_std": by_month.std(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Q_gage, _, _ = _load_masked_flows("pub_nhmv10_BC_withObsScaled")
    Q_gen = Q_gage.loc["1945-10-01":"2022-09-30"]
    kirsch = _fit_kirsch(Q_gen)
    order = wetness_rank_order(kirsch.Y)

    # Historical reference values (calendar years, aggregate NYC inflow).
    agg_h = Q_gen[NYC].sum(axis=1)
    ann_h = agg_h.groupby(agg_h.index.year).sum().iloc[1:-1].to_numpy(dtype=float)
    rho1_hist = _annual_rho1(ann_h)
    print(f"[persdiag] historical annual rho1 (fit record) = {rho1_hist:.3f}")

    rows = []
    baseline = None
    for phi, lam in SETTINGS:
        frames, gains = [], []
        for r in range(R):
            rng = np.random.default_rng(np.random.SeedSequence((SEED, int(phi * 100), int(lam * 100), r)))
            M = persistent_bootstrap_indices(
                YEARS + 1, kirsch.n_periods_per_year, order, phi=phi, lam=lam, rng=rng,
            )
            df = kirsch.generate_single_series(YEARS, M=M, as_array=False)
            frames.append(df)
            gains.append(_realization_metrics(df))
        g = pd.DataFrame(gains)
        d = _distortion_metrics(frames)
        row = {
            "phi": phi, "lam": lam,
            "rho1": g["rho1"].mean(), "rho1_se": g["rho1"].std() / np.sqrt(R),
            "max_dry_run": g["max_dry_run"].mean(),
            "min_roll3_frac_p10": g["min_roll3_frac"].quantile(0.10),
            "adjacent_corr": d["adjacent_corr"],
            "cross_site_corr": d["cross_site_corr"],
        }
        if baseline is None:
            baseline = d
        row["mean_drift_max"] = float(
            (d["monthly_mean"] - baseline["monthly_mean"]).abs().to_numpy().max()
        )
        row["std_drift_max_rel"] = float(
            ((d["monthly_std"] - baseline["monthly_std"]) / baseline["monthly_std"])
            .abs().to_numpy().max()
        )
        rows.append(row)
        print(f"[persdiag] phi={phi:.2f} lam={lam:.2f}: rho1={row['rho1']:+.3f} "
              f"(se {row['rho1_se']:.3f}), adj_corr={row['adjacent_corr']:.3f}, "
              f"cross_site={row['cross_site_corr']:.3f}, "
              f"std_drift={row['std_drift_max_rel']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "settings_summary.csv", index=False)

    main_curve = out[(out.phi == 0.8)]
    ctrl = out.iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.errorbar(main_curve["lam"], main_curve["rho1"], yerr=main_curve["rho1_se"],
                 marker="o", color="#c1272d", label=r"$\phi=0.8$")
    for phi in (0.5, 0.9):
        sub = out[(out.phi == phi) & (out.lam == 0.35)]
        ax1.scatter(sub["lam"], sub["rho1"], marker="s", label=rf"$\phi={phi}$")
    ax1.axhline(rho1_hist, ls="--", color="0.4", label=f"historic ({rho1_hist:.2f})")
    ax1.axhspan(-0.10, 0.40, color="0.85", zorder=0, label="CMIP6 future p5-p95")
    ax1.set_xlabel(r"copula loading $\lambda$")
    ax1.set_ylabel(r"annual lag-1 autocorrelation $\rho_1$")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("gain: interannual persistence")

    ax2.plot(main_curve["lam"], main_curve["adjacent_corr"], marker="o",
             label="adjacent-month corr")
    ax2.axhline(ctrl["adjacent_corr"], ls="--", color="0.4", label="control")
    ax2.plot(main_curve["lam"], main_curve["cross_site_corr"], marker="^",
             label="cross-site corr")
    ax2.set_xlabel(r"copula loading $\lambda$")
    ax2.set_ylabel("pooled correlation")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("distortion: within-year / cross-site structure")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "persistence_axis.png", dpi=200)

    findings = {
        "R": R, "years": YEARS, "seed": SEED,
        "rho1_hist_fit_record": rho1_hist,
        "cmip6_future_rho1_p5_p50_p95": [-0.10, 0.08, 0.40],
        "settings": out.round(4).to_dict("records"),
    }
    (OUT_DIR / "findings.json").write_text(json.dumps(findings, indent=2))
    print(out.round(3).to_string(index=False))
    print(f"[persdiag] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
