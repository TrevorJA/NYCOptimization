"""check_cmip6_rho1_anchor_noise.py - Is the CMIP6 delta-rho1 anchor sampling noise?

The persistence-axis anchor (``docs/notes/methods/persistence_axis_diagnostics.md``)
reads a positive tendency in interannual persistence from the CMIP6 ensemble:
sibling-matched delta-rho1 (future run minus the same hydro-model x GCM 1980-2019
baseline) with median ~ +0.08 and p95 ~ +0.4. Each rho1 is estimated from a 39-yr
water-year record, so a single estimate carries se(rho1) ~ 1/sqrt(39) ~ 0.16 and a
sibling difference ~ 0.23. This check asks whether the ensemble's tendency and tail
are distinguishable from that sampling noise.

Procedure:

    1. Recompute per-run rho1 of water-year aggregate NYC inflow (Cannonsville +
       Pepacton + Neversink catchment inflow) for every CMIP6 run and its sibling
       baseline, reproducing the anchor table from raw data.
    2. Monte Carlo null with the ensemble's exact dependence structure: per
       (hydro model, GCM) cell, one baseline AR(1) series plus one series per
       future run, all sharing a common true rho (no change); delta computed
       against the shared baseline exactly as observed. Two nulls are run:
       rho_true = 0 and rho_true = the bias-corrected pooled estimate.
    3. Test statistics: ensemble median/mean delta-rho1, GCM-level mean-of-means
       and sign count (7 GCMs; guards against treating 54 dependent runs as
       independent), p95/max/sd of future rho1 (is the upper tail a real signal
       or the expected extreme of ~54 noisy estimates?).

Configuration (no CLI value flags):

    NYCOPT_ANCHORNOISE_NMC     Monte Carlo replicates  (default 4000)
    NYCOPT_ANCHORNOISE_SEED    root seed               (default 20260729)

Outputs -> ``outputs/supplemental/persistence_axis/anchor_noise/``:
    run_rho1.csv        per-run water-year rho1 (baselines + futures)
    sibling_delta.csv   per-future-run sibling-matched delta-rho1
    anchor_noise.png    observed deltas vs null band; future-rho1 ECDF vs null
    findings.json       observed stats + p-values under both nulls
"""

from __future__ import annotations

import json
import os
import re
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

NMC = int(os.environ.get("NYCOPT_ANCHORNOISE_NMC", "4000"))
SEED = int(os.environ.get("NYCOPT_ANCHORNOISE_SEED", "20260729"))

NYC = ["cannonsville", "pepacton", "neversink"]
CMIP6_INPUTS = config.PROJECT_DIR.parent / "CMIP6_multimodel_streamflow" / "pywrdrb" / "inputs"
RUN_RE = re.compile(
    r"^(?P<hydro>PRMS|VIC5)_RAPID_(?P<gcm>.+)_(?P<ssp>ssp\d{3})_r\d+i\d+p\d+f\d+"
    r"_DBCCA_Daymet_(?P<y0>\d{4})_(?P<y1>\d{4})$"
)

OUT_DIR = config.OUTPUTS_DIR / "supplemental" / "persistence_axis" / "anchor_noise"


def _annual_rho1(ann: np.ndarray) -> float:
    """Lag-1 autocorrelation of annual totals (same estimator as diagnose_persistence_axis)."""
    x = ann - ann.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else np.nan


def _run_wy_rho1(run_dir: Path) -> tuple[float, int]:
    """(rho1, n_water_years) of aggregate NYC inflow for one CMIP6 run."""
    df = pd.read_csv(run_dir / "catchment_inflow_mgd.csv",
                     usecols=["datetime", *NYC], parse_dates=["datetime"])
    agg = df[NYC].sum(axis=1)
    wy = df["datetime"].dt.year + (df["datetime"].dt.month >= 10).astype(int)
    totals = agg.groupby(wy.to_numpy()).agg(["sum", "size"])
    complete = totals[totals["size"] >= 365]["sum"].to_numpy(dtype=float)
    return _annual_rho1(complete), len(complete)


def _discover_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(CMIP6_INPUTS.iterdir()):
        m = RUN_RE.match(d.name)
        if m is None:
            continue
        rows.append({
            "run": d.name, "hydro": m["hydro"], "gcm": m["gcm"], "ssp": m["ssp"],
            "period": f"{m['y0']}_{m['y1']}",
            "kind": "baseline" if m["y0"] == "1980" else "future",
        })
    return pd.DataFrame(rows)


def _null_stats(deltas_layout: list[int], n_years: int, rho: float,
                rng: np.random.Generator) -> dict:
    """One null replicate mirroring the ensemble structure.

    Args:
        deltas_layout: number of future runs per (hydro, gcm) cell.
        n_years: annual record length per series.
        rho: common true lag-1 autocorrelation (null: no change).
        rng: replicate RNG.
    """
    def ar1(k: int) -> np.ndarray:
        e = rng.standard_normal((k, n_years))
        x = np.empty_like(e)
        x[:, 0] = e[:, 0]
        c = np.sqrt(1.0 - rho * rho)
        for t in range(1, n_years):
            x[:, t] = rho * x[:, t - 1] + c * e[:, t]
        return x

    deltas, fut = [], []
    for n_fut in deltas_layout:
        series = ar1(n_fut + 1)
        r = np.array([_annual_rho1(s) for s in series])
        deltas.extend(r[1:] - r[0])
        fut.extend(r[1:])
    deltas, fut = np.array(deltas), np.array(fut)
    return {
        "median_delta": float(np.median(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "p95_future": float(np.quantile(fut, 0.95)),
        "max_future": float(np.max(fut)),
        "sd_future": float(np.std(fut, ddof=1)),
        "delta_abs_p95": float(np.quantile(np.abs(deltas), 0.95)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs()
    rho1, n_wy = zip(*(_run_wy_rho1(CMIP6_INPUTS / r) for r in runs["run"]))
    runs["rho1"], runs["n_wy"] = rho1, n_wy
    runs.to_csv(OUT_DIR / "run_rho1.csv", index=False)

    base = runs[runs.kind == "baseline"].set_index(["hydro", "gcm"])["rho1"]
    fut = runs[runs.kind == "future"].copy()
    fut["rho1_baseline"] = [base[(h, g)] for h, g in zip(fut["hydro"], fut["gcm"])]
    fut["delta_rho1"] = fut["rho1"] - fut["rho1_baseline"]
    fut.to_csv(OUT_DIR / "sibling_delta.csv", index=False)

    n_years = int(runs["n_wy"].mode().iloc[0])
    gcm_means = fut.groupby("gcm")["delta_rho1"].mean()
    obs = {
        "n_future": len(fut), "n_baseline": len(base), "n_years": n_years,
        "future_rho1_p5_p50_p95": [float(fut["rho1"].quantile(q)) for q in (0.05, 0.5, 0.95)],
        "baseline_rho1_p5_p50_p95": [float(base.quantile(q)) for q in (0.05, 0.5, 0.95)],
        "delta_p5_p50_p95": [float(fut["delta_rho1"].quantile(q)) for q in (0.05, 0.5, 0.95)],
        "median_delta": float(fut["delta_rho1"].median()),
        "mean_delta": float(fut["delta_rho1"].mean()),
        "gcm_means": gcm_means.round(4).to_dict(),
        "gcm_mean_of_means": float(gcm_means.mean()),
        "n_positive_gcm": int((gcm_means > 0).sum()),
        "p95_future": float(fut["rho1"].quantile(0.95)),
        "max_future": float(fut["rho1"].max()),
        "sd_future": float(fut["rho1"].std(ddof=1)),
    }

    # Bias-corrected pooled rho (estimator bias E[rho_hat] ~ rho - (1+3rho)/n).
    pooled = float(runs["rho1"].mean())
    rho_bc = max(0.0, (pooled + 1.0 / n_years) / (1.0 - 3.0 / n_years))

    # Observed-record control: same statistic on the reconstruction, full fit record
    # vs the CMIP6 baseline window vs the pre-1980 epoch (1960s drought).
    from src.ensemble_generation import _load_masked_flows

    Q_gage, _, _ = _load_masked_flows("pub_nhmv10_BC_withObsScaled")
    agg_o = Q_gage[NYC].sum(axis=1)
    wy_o = agg_o.index.year + (agg_o.index.month >= 10).astype(int)
    tot_o = agg_o.groupby(wy_o).agg(["sum", "size"])
    obs["observed_record_rho1"] = {}
    for lo_wy, hi_wy, label in [(1946, 2022, "wy1946_2022_fit_record"),
                                (1981, 2019, "wy1981_2019_cmip6_baseline_window"),
                                (1946, 1980, "wy1946_1980_pre_pluvial")]:
        sel = tot_o[(tot_o.index >= lo_wy) & (tot_o.index <= hi_wy)
                    & (tot_o["size"] >= 365)]["sum"].to_numpy(dtype=float)
        obs["observed_record_rho1"][label] = {"n": len(sel), "rho1": _annual_rho1(sel)}
    print("[anchornoise] observed record rho1 by window: "
          + ", ".join(f"{k}={v['rho1']:+.3f} (n={v['n']})"
                      for k, v in obs["observed_record_rho1"].items()))

    layout = [int(n) for _, n in fut.groupby(["hydro", "gcm"]).size().items()]
    nulls = {}
    for label, rho in (("rho0", 0.0), ("rho_pooled", rho_bc)):
        rng = np.random.default_rng(np.random.SeedSequence((SEED, int(rho * 1000))))
        reps = pd.DataFrame(_null_stats(layout, n_years, rho, rng) for _ in range(NMC))
        nulls[label] = {
            "rho_true": rho,
            "se_rho1_single_series": float(reps["sd_future"].mean()),
            # One-sided: P(null >= observed) for positive-tendency / tail stats.
            "p_median_delta": float((reps["median_delta"] >= obs["median_delta"]).mean()),
            "p_mean_delta": float((reps["mean_delta"] >= obs["mean_delta"]).mean()),
            "p_p95_future": float((reps["p95_future"] >= obs["p95_future"]).mean()),
            "p_max_future": float((reps["max_future"] >= obs["max_future"]).mean()),
            "p_sd_future": float((reps["sd_future"] >= obs["sd_future"]).mean()),
            "null_median_delta_p5_p95": [float(reps["median_delta"].quantile(q)) for q in (0.05, 0.95)],
            "null_p95_future_p5_p95": [float(reps["p95_future"].quantile(q)) for q in (0.05, 0.95)],
            "null_delta_abs_p95": float(reps["delta_abs_p95"].mean()),
            "_reps": reps,
        }
        print(f"[anchornoise] null {label} (rho={rho:.3f}): "
              f"p(median dRho1)={nulls[label]['p_median_delta']:.4f}, "
              f"p(mean)={nulls[label]['p_mean_delta']:.4f}, "
              f"p(p95 future)={nulls[label]['p_p95_future']:.4f}, "
              f"p(sd future)={nulls[label]['p_sd_future']:.4f}")

    # Sign test on GCM-level means (binomial, p=0.5).
    from math import comb
    k, n_g = obs["n_positive_gcm"], len(gcm_means)
    obs["p_sign_gcm"] = sum(comb(n_g, j) for j in range(k, n_g + 1)) / 2 ** n_g

    print(f"[anchornoise] observed: median dRho1={obs['median_delta']:+.3f}, "
          f"GCM mean-of-means={obs['gcm_mean_of_means']:+.3f} "
          f"({k}/{n_g} GCMs positive, sign p={obs['p_sign_gcm']:.3f}), "
          f"future p95={obs['p95_future']:.3f}")

    reps0 = nulls["rho_pooled"].pop("_reps")
    nulls["rho0"].pop("_reps")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    order = gcm_means.sort_values().index.tolist()
    xpos = {g: i for i, g in enumerate(order)}
    band = float(np.median(reps0["delta_abs_p95"]))
    ax1.axhspan(-band, band, color="0.88", zorder=0,
                label=f"null per-pair 90% band (rho={rho_bc:.2f})")
    ax1.axhline(0.0, color="0.4", lw=0.8)
    for h, mk in (("PRMS", "o"), ("VIC5", "^")):
        sub = fut[fut.hydro == h]
        ax1.scatter([xpos[g] for g in sub["gcm"]], sub["delta_rho1"],
                    marker=mk, s=28, color="#c1272d", alpha=0.7, label=h)
    ax1.scatter([xpos[g] for g in order], gcm_means[order], marker="_",
                s=300, color="k", label="GCM mean")
    ax1.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel(r"sibling $\Delta\rho_1$ (future $-$ baseline)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("per-run persistence change vs sampling-noise null")

    xs = np.sort(fut["rho1"].to_numpy())
    ax2.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post",
             color="#c1272d", label="observed future runs")
    ax2.axvline(obs["p95_future"], ls=":", color="#c1272d", lw=1)
    ax2.axvspan(*nulls["rho_pooled"]["null_p95_future_p5_p95"], color="0.88",
                zorder=0, label="null p95 range")
    ax2.set_xlabel(r"future-run $\rho_1$")
    ax2.set_ylabel("ECDF")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("future-run rho1 distribution vs null tail")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "anchor_noise.png", dpi=200)

    findings = {"NMC": NMC, "seed": SEED, "rho_pooled_bias_corrected": rho_bc,
                "observed": obs, "nulls": nulls}
    (OUT_DIR / "findings.json").write_text(json.dumps(findings, indent=2))
    print(f"[anchornoise] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
