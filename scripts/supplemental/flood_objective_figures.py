"""flood_objective_figures.py - Reductions for the flood-objective diagnostics.

Reduces the cube written by ``flood_objective_run.py`` into the tables and
figures behind the flood-objective definition decision
(``docs/notes/methods/flood_objective_diagnostics.md``):

  A. Sim-vs-obs / regulatory: per-candidate bias ratio, annual-series Pearson
     and Spearman, and bad-flood-year rank agreement on the gauge experiment's
     post-fix 2000-2023 run.
  B. Rating-curve exposure: how often, and how far, simulated flood-day
     discharges leave the rated range at each gauge (the risk a stage-based
     severity metric carries; the stage-vs-flow-basis evidence).
  C. Resolution / discriminating power (decisive): distinct values, tie-pair
     fraction, spread, and occupied epsilon-boxes across the baseline+random
     policy sample, on the historic trace and on the KN ensemble; plus the
     cross-candidate Spearman matrix.
  D. Monotone-response gate: candidate value vs the flood-release-scale ladder
     fraction t — Spearman |rho|, direction reversals, largest single-step
     share of range (cliff detection).
  E. Ensemble sampling noise: pooled unit-year distributions, zero-inflated
     fraction, top-unit / top-day share of each severity integral, bootstrap
     SE of the ensemble estimate vs number of realizations.
  F. Epsilon proposal: max(IQR/10, granularity, noise) per candidate on both
     the whole-window (§1) and pooled-annual (§2) scales.

Zero simulation — everything reads the cube and the run's sim-vs-obs tables.
Outputs -> outputs/supplemental/flood_objective/{tables,figures}
Configuration lives in supplemental_config.py (FLOODOBJ_* section) — no CLI
value flags.

Usage:
    python scripts/supplemental/flood_objective_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_floodobj_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.plotting.style import (  # noqa: E402
    annotated_corr_heatmap,
    apply_style,
    save_figure,
)

# Fixed candidate identity (Okabe-Ito, assignment order CVD-validated; never
# re-ordered or cycled).
CAND_STYLE = {
    "C1_days_minor":  {"color": "#0072B2", "label": "C1 days $\\geq$ minor (incumbent)", "unit": "days/yr"},
    "C2_days_action": {"color": "#E69F00", "label": "C2 days $\\geq$ cautionary",        "unit": "days/yr"},
    "C3_sum_ft":      {"color": "#009E73", "label": "C3 $\\Sigma$(stage$-$minor)$^+$, gauge sum", "unit": "ft·d/yr"},
    "C4_max_ft":      {"color": "#D55E00", "label": "C4 $\\Sigma$ max-gauge (stage$-$minor)$^+$", "unit": "ft·d/yr"},
    "C5_norm_ft":     {"color": "#56B4E9", "label": "C5 normalized $\\Sigma$(stage$-$minor)$^+$", "unit": "1/yr"},
    "C6_flow_mg":     {"color": "#CC79A7", "label": "C6 $\\Sigma$(Q$-$Q$_{minor}$)$^+$ volume",   "unit": "MG/yr"},
}
GAUGE_NAMES = {"01426500": "Hale Eddy", "01421000": "Fishs Eddy",
               "01436690": "Bridgeville"}

#: The incumbent's shipped §1 epsilon (src/objectives.py), the comparability
#: anchor for the block-F proposals.
INCUMBENT_EPS = 0.02


def load_cube() -> dict:
    path = scfg.floodobj_cube_path("flood_cube")
    if not path.exists():
        sys.exit(f"[flood_fig] cube not found: {path} — run "
                 "flood_objective_run.py first.")
    z = np.load(path, allow_pickle=False)
    cube = {k: z[k] for k in z.files}
    cube["candidates"] = [str(c) for c in cube["candidates"]]
    cube["gauges"] = [str(g) for g in cube["gauges"]]
    with open(scfg.FLOODOBJ_CUBE_DIR / "flood_run_manifest.json") as f:
        cube["manifest"] = json.load(f)
    return cube


def policy_values(cube: dict) -> dict:
    """Per-policy candidate values on the two scales the decision uses.

    ``hist``: §1 whole-window days-per-year-normalized values on the historic
    trace. ``ens``: mean over the pooled ensemble unit-years (the §2
    PooledMean search aggregation, in per-unit-year units).
    """
    return {
        "hist": cube["hist_window"],                      # (P, C)
        "ens": cube["ens_units"].mean(axis=(1, 2)),       # (P, C)
    }


###############################################################################
# Block A — sim vs obs
###############################################################################

def block_a(cands: list) -> pd.DataFrame:
    summary = pd.read_csv(scfg.floodobj_table_path("simobs_summary"),
                          index_col=0)
    annual = pd.read_csv(
        scfg.floodobj_table_path("simobs_annual_candidates"))

    rows = []
    for cand in cands:
        sub = annual[annual["candidate"] == cand].dropna(
            subset=["sim", "obs"]).sort_values("water_year")
        sim, obs = sub["sim"].to_numpy(), sub["obs"].to_numpy()
        pear = stats.pearsonr(sim, obs).statistic if len(sim) > 2 else np.nan
        spear = stats.spearmanr(sim, obs).statistic if len(sim) > 2 else np.nan
        # Bad-flood-year ranking: overlap of the top-5 observed water years
        # with the top-5 simulated ones.
        k = 5
        top_obs = set(sub.nlargest(k, "obs")["water_year"])
        top_sim = set(sub.nlargest(k, "sim")["water_year"])
        rows.append({
            "candidate": cand,
            "sim": summary.loc[cand, "sim"],
            "obs": summary.loc[cand, "obs"],
            "ratio_sim_obs": summary.loc[cand, "ratio_sim_obs"],
            "annual_pearson_r": pear,
            "annual_spearman_rho": spear,
            f"top{k}_year_overlap": len(top_obs & top_sim),
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.floodobj_table_path("A_sim_vs_obs"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), sharex=True)
    for ax, cand in zip(axes.ravel(), cands):
        sub = annual[annual["candidate"] == cand].sort_values("water_year")
        st = CAND_STYLE[cand]
        ax.plot(sub["water_year"], sub["obs"], color="black", ls="--", lw=1.4,
                label="observed")
        ax.plot(sub["water_year"], sub["sim"], color=st["color"], lw=1.6,
                label="simulated")
        r = df.loc[df["candidate"] == cand, "annual_spearman_rho"].iloc[0]
        ax.set_title(f"{st['label']}\n$\\rho_S$ = {r:.2f}", fontsize=9)
        ax.set_ylabel(st["unit"].replace("/yr", "/WY"))
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("water year")
    fig.suptitle("Annual candidate values, gauge-experiment run vs observed "
                 "(WY2001–WY2023)", y=1.02)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("A_sim_vs_obs_annual"))
    plt.close(fig)
    return df


###############################################################################
# Block B — rating-curve exposure
###############################################################################

def block_b(cube: dict) -> pd.DataFrame:
    from pywrdrb.utils.constants import cfs_to_mgd
    from pywrdrb.utils.rating_curves import load_all_flood_monitoring_curves

    curves = load_all_flood_monitoring_curves()
    gauges = cube["gauges"]
    rec_g, rec_q = cube["rec_gauge"], cube["rec_q_mgd"]
    rec_s = cube["rec_stage_ft"]

    rows = []
    for gi, g in enumerate(gauges):
        c = curves[g]
        rated_q_mgd = c.discharge_max * cfs_to_mgd
        qmax_all = max(float(cube["hist_max"][:, gi, 1].max()),
                       float(cube["ens_max"][:, :, gi, 1].max()))
        mask = rec_g == gi
        n_days = int(mask.sum())
        n_over = int((rec_q[mask] > rated_q_mgd).sum())
        rows.append({
            "gauge": f"{GAUGE_NAMES[g]} ({g})",
            "rated_q_max_mgd": rated_q_mgd,
            "rated_stage_max_ft": c.stage_max,
            "minor_ft": float(cube["minor_ft"][gi]),
            "major_ft": float(cube["major_ft"][gi]),
            "sim_q_max_mgd": qmax_all,
            "q_headroom_ratio": rated_q_mgd / qmax_all,
            "sim_stage_max_ft": max(
                float(cube["hist_max"][:, gi, 0].max()),
                float(cube["ens_max"][:, :, gi, 0].max())),
            "n_flood_gauge_days": n_days,
            "n_days_beyond_rated": n_over,
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.floodobj_table_path("B_rating_curve_exposure"),
              index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, (gi, g) in zip(axes, enumerate(gauges)):
        mask = rec_g == gi
        ax.scatter(cube["rec_q_mgd"][mask] / 1000.0, rec_s[mask], s=12,
                   color="#56B4E9", alpha=0.6, edgecolors="none",
                   label="flood gauge-days (all policies)")
        c = curves[g]
        for lev, ls in (("minor_ft", "-"), ("major_ft", ":")):
            ax.axhline(float(cube[lev][gi]), color="#555555", ls=ls, lw=1.0)
        ax.axvline(c.discharge_max * cfs_to_mgd / 1000.0, color="#D55E00",
                   ls="--", lw=1.2, label="rated-range max")
        ax.axhline(c.stage_max, color="#D55E00", ls="--", lw=1.2)
        ax.set_title(GAUGE_NAMES[g])
        ax.set_xlabel("discharge (1000 MGD)")
    axes[0].set_ylabel("stage (ft)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Flood-day stage vs discharge against the rated range "
                 "(solid grey: minor; dotted: major)", y=1.02)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("B_rating_curve_exposure"))
    plt.close(fig)
    return df


###############################################################################
# Blocks C+F — resolution / discriminating power + epsilon proposal
###############################################################################

def block_e_noise(cube: dict, cands: list) -> pd.DataFrame:
    """Ensemble sampling-noise reductions (baseline policy)."""
    rng = np.random.default_rng(scfg.FLOODOBJ_BOOTSTRAP_SEED)
    units = cube["ens_units"][0]          # (R, U, C) baseline policy
    n_real, n_units, _ = units.shape
    pooled = units.reshape(-1, len(cands))  # (R*U, C)

    rows = []
    for ci, cand in enumerate(cands):
        u = pooled[:, ci]
        total = u.sum()
        # bootstrap over realizations (the sampling unit of the design)
        idx = rng.integers(0, n_real, size=(scfg.FLOODOBJ_BOOTSTRAP_B, n_real))
        boot = units[:, :, ci][idx].mean(axis=(1, 2))
        se_real = float(boot.std(ddof=1))
        mean = float(u.mean())
        rows.append({
            "candidate": cand,
            "pooled_units": len(u),
            "mean_per_unit": mean,
            "zero_unit_frac": float((u == 0).mean()),
            "top_unit_share": float(u.max() / total) if total > 0 else np.nan,
            "bootstrap_se_5real": se_real,
            "bootstrap_cv_5real": se_real / mean if mean > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.floodobj_table_path("E_ensemble_noise"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.0))
    for ax, (ci, cand) in zip(axes.ravel(), enumerate(cands)):
        u = pooled[:, ci]
        st = CAND_STYLE[cand]
        nz = u[u > 0]
        bins = np.linspace(0, max(u.max(), 1e-9), 30)
        ax.hist(u, bins=bins, color=st["color"], edgecolor="white", lw=0.4)
        ax.set_yscale("log")
        z = float((u == 0).mean())
        ax.set_title(f"{st['label']}\nzero units: {z:.0%}   "
                     f"top unit: {u.max():.3g}", fontsize=9)
        ax.set_xlabel(f"unit-year value ({st['unit'].replace('/yr', '')})")
        if nz.size == 0:
            ax.text(0.5, 0.5, "all zero", transform=ax.transAxes,
                    ha="center")
    for ax in axes[:, 0]:
        ax.set_ylabel("unit-years (log)")
    fig.suptitle("Baseline policy: pooled ensemble unit-year distributions "
                 f"({pooled.shape[0]} unit-years, N=5 x 50 yr)", y=1.02)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("E_unit_distributions"))
    plt.close(fig)
    return df


def block_f_epsilon(vals: dict, noise: pd.DataFrame, cube: dict,
                    cands: list) -> pd.DataFrame:
    """Epsilon proposal per candidate: max(IQR/10, granularity, noise)."""
    kind = cube["kind"]
    sample = kind != "sweep"
    n_real, n_units = cube["ens_units"].shape[1:3]
    hist_years = None
    rows = []
    for ci, cand in enumerate(cands):
        v_hist = vals["hist"][sample, ci]
        v_ens = vals["ens"][sample, ci]
        # granularity: the smallest step the metric can take on each scale.
        # Counts move in whole days: 1/window-years (§1 historic) or
        # 1/(pooled unit-years) (§2 ensemble mean). Continuous integrals have
        # no intrinsic step (0).
        if cand.startswith(("C1", "C2")):
            gran_hist = 1.0 / 76.0          # 77-WY trace -> 76 metric units
            gran_ens = 1.0 / (n_real * n_units)
        else:
            gran_hist = gran_ens = 0.0
        iqr_h = stats.iqr(v_hist)
        iqr_e = stats.iqr(v_ens)
        noise_e = float(noise.loc[noise["candidate"] == cand,
                                  "bootstrap_se_5real"].iloc[0])
        rows.append({
            "candidate": cand,
            "hist_iqr10": iqr_h / 10.0,
            "hist_granularity": gran_hist,
            "eps_hist": max(iqr_h / 10.0, gran_hist),
            "ens_iqr10": iqr_e / 10.0,
            "ens_granularity": gran_ens,
            "ens_noise_se": noise_e,
            "eps_ens": max(iqr_e / 10.0, gran_ens),
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.floodobj_table_path("F_epsilon_proposal"), index=False)
    return df


def block_c(vals: dict, eps: pd.DataFrame, cube: dict,
            cands: list) -> pd.DataFrame:
    kind = cube["kind"]
    sample = kind != "sweep"
    n_pol = int(sample.sum())

    rows = []
    for domain in ("hist", "ens"):
        eps_col = "eps_hist" if domain == "hist" else "eps_ens"
        for ci, cand in enumerate(cands):
            v = vals[domain][sample, ci]
            e = float(eps.loc[eps["candidate"] == cand, eps_col].iloc[0])
            uniq = np.unique(np.round(v, 12))
            # tie-pair fraction: fraction of policy pairs with identical value
            _, counts = np.unique(np.round(v, 12), return_counts=True)
            n_pairs = n_pol * (n_pol - 1) / 2
            tie_pairs = float((counts * (counts - 1) / 2).sum() / n_pairs)
            boxes = np.unique(np.floor(v / e).astype(np.int64)) if e > 0 \
                else uniq
            rows.append({
                "domain": domain,
                "candidate": cand,
                "n_policies": n_pol,
                "n_distinct": len(uniq),
                "tie_pair_frac": tie_pairs,
                "range": float(v.max() - v.min()),
                "iqr": stats.iqr(v),
                "epsilon": e,
                "n_epsilon_boxes": len(boxes),
            })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.floodobj_table_path("C_resolution"), index=False)

    # Figure: per-domain strip of policy values normalized to each candidate's
    # own range (annotated with distinct-value and box counts).
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True)
    rng = np.random.default_rng(3)
    for ax, domain in zip(axes, ("hist", "ens")):
        for ci, cand in enumerate(cands):
            v = vals[domain][sample, ci]
            lo, hi = v.min(), v.max()
            vn = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
            x = ci + rng.uniform(-0.13, 0.13, size=len(v))
            st = CAND_STYLE[cand]
            ax.scatter(x, vn, s=18, color=st["color"], alpha=0.75,
                       edgecolors="white", linewidths=0.4)
            row = df[(df["domain"] == domain) & (df["candidate"] == cand)]
            ax.text(ci, 1.06, f"{int(row['n_distinct'].iloc[0])} distinct\n"
                    f"{int(row['n_epsilon_boxes'].iloc[0])} $\\epsilon$-boxes",
                    ha="center", fontsize=7.5)
        ax.set_xticks(range(len(cands)))
        ax.set_xticklabels([c.split("_")[0] for c in cands])
        ax.set_ylim(-0.05, 1.18)
        ax.set_title("historic trace" if domain == "hist"
                     else "KN ensemble (pooled-annual mean)")
    axes[0].set_ylabel("policy value, normalized to candidate range")
    fig.suptitle(f"Discriminating power across {n_pol} baseline+random "
                 "policies", y=1.02)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("C_discriminating_power"))
    plt.close(fig)

    # Cross-candidate agreement (ensemble scale, sample policies).
    m = len(cands)
    corr = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(m):
            vi, vj = vals["ens"][sample, i], vals["ens"][sample, j]
            if np.ptp(vi) > 0 and np.ptp(vj) > 0:
                corr[i, j] = stats.spearmanr(vi, vj).statistic
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = annotated_corr_heatmap(
        ax, corr, cands,
        label_fn=lambda n: CAND_STYLE[n]["label"].replace("$", ""),
        fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman $\\rho_S$")
    ax.set_title("Candidate agreement across policies (KN ensemble)")
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("C_candidate_agreement"))
    plt.close(fig)
    pd.DataFrame(corr, index=cands, columns=cands).to_csv(
        scfg.floodobj_table_path("C_candidate_spearman"))
    return df


###############################################################################
# Block D — monotone-response gate
###############################################################################

def block_d(vals: dict, cube: dict, cands: list) -> pd.DataFrame:
    kind = cube["kind"]
    sweep = kind == "sweep"
    ts = cube["sweep_t"][sweep]
    order = np.argsort(ts)
    ts = ts[order]

    rows = []
    for domain in ("hist", "ens"):
        v_all = vals[domain][sweep][order]
        for ci, cand in enumerate(cands):
            v = v_all[:, ci]
            rng_v = float(v.max() - v.min())
            d = np.diff(v)
            dominant = np.sign(np.median(d[d != 0])) if (d != 0).any() else 0.0
            reversals = int(((np.sign(d) == -dominant) & (d != 0)).sum())
            rho = stats.spearmanr(ts, v).statistic if rng_v > 0 else np.nan
            rows.append({
                "domain": domain,
                "candidate": cand,
                "range": rng_v,
                "rel_range": rng_v / v.mean() if v.mean() > 0 else np.nan,
                "spearman_rho_vs_t": rho,
                "n_reversals": reversals,
                "max_step_share": float(np.abs(d).max() / rng_v)
                if rng_v > 0 else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.floodobj_table_path("D_monotone_response"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, domain in zip(axes, ("hist", "ens")):
        v_all = vals[domain][sweep][order]
        for ci, cand in enumerate(cands):
            v = v_all[:, ci]
            lo, hi = v.min(), v.max()
            vn = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
            st = CAND_STYLE[cand]
            ax.plot(ts, vn, color=st["color"], lw=1.8, marker="o", ms=4,
                    label=st["label"])
        ax.set_xlabel("flood-release multiplier ladder fraction $t$\n"
                      "(both zones, all reservoirs; 0 = scale 0.5, "
                      "1 = L1a upper bound)")
        ax.set_title("historic trace" if domain == "hist"
                     else "KN ensemble (pooled-annual mean)")
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_ylabel("candidate value, normalized to own range")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Monotone response to flood-release aggressiveness", y=1.02)
    fig.tight_layout()
    save_figure(fig, scfg.floodobj_figure_path("D_monotone_response"))
    plt.close(fig)
    return df


###############################################################################
# Main
###############################################################################

def main() -> None:
    apply_style()
    cube = load_cube()
    cands = cube["candidates"]
    scfg.FLOODOBJ_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scfg.FLOODOBJ_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    vals = policy_values(cube)
    dfa = block_a(cands)
    dfb = block_b(cube)
    dfe = block_e_noise(cube, cands)
    dff = block_f_epsilon(vals, dfe, cube, cands)
    dfc = block_c(vals, dff, cube, cands)
    dfd = block_d(vals, cube, cands)

    pd.set_option("display.width", 160)
    print("\n=== A. sim vs obs (2000-2023, gauge-experiment run) ===")
    print(dfa.round(3).to_string(index=False))
    print("\n=== B. rating-curve exposure ===")
    print(dfb.round(2).to_string(index=False))
    print("\n=== C. discriminating power (baseline + random policies) ===")
    print(dfc.round(4).to_string(index=False))
    print("\n=== D. monotone response (flood-release ladder) ===")
    print(dfd.round(3).to_string(index=False))
    print("\n=== E. ensemble sampling noise (baseline policy) ===")
    print(dfe.round(4).to_string(index=False))
    print("\n=== F. epsilon proposal ===")
    print(dff.round(5).to_string(index=False))
    print(f"\n[flood_fig] tables -> {scfg.FLOODOBJ_TABLES_DIR}")
    print(f"[flood_fig] figures -> {scfg.FLOODOBJ_FIGURES_DIR}")


if __name__ == "__main__":
    main()
