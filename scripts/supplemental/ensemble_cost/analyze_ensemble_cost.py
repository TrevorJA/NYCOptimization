"""
analyze_ensemble_cost.py - Tables and figures for the ensemble-cost surface.

Reads the per-rank CSV shards written by ``bench_eval_worker.py`` across the
ensemble-cost sweep's cells and produces the numbers that price the MOEA
campaign:

    tables/cell_evals.csv          every timed evaluation (the raw measurement)
    tables/cost_surface.csv        per (N, L, model): warm cost, throughput, SU, RSS
    tables/scaling_fits.csv        the empirical exponents in N and in L
    tables/model_ratio.csv         full / trimmed cost ratio across the surface
    tables/campaign_projection.csv SU for search + re-evaluation vs the allocation

    figures/F1_cost_surface.png    warm eval time vs N (the money figure)
    figures/F2_memory.png          RSS per eval and the node total vs 256 GB
    figures/F3_model_ratio.png     full / trimmed ratio
    figures/F4_campaign_su.png     projected SU vs test-ensemble size vs 1M SU

Cost accounting follows the packing sweep (analyze_scaling.py), so the two
experiments' numbers are directly comparable:
    evals_per_node_hr   = 3600 * K / t_warm
    su_per_1000_evals   = 1000 * NODE_CORES * t_warm / (3600 * K)
Anvil's wholenode partition charges all 128 cores of a node per node-hour
regardless of how many ranks are used, so K (ranks per node) is what converts a
per-eval second into an SU.

Runs anywhere (login node or laptop) - no MPI, no SLURM; it only reads files.
Degrades gracefully when cells are missing: a cell whose ranks were OOM-killed
simply has no shard, and is reported as absent rather than silently imputed.

Usage:
    python scripts/supplemental/ensemble_cost/analyze_ensemble_cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_ensemble_cost_env()  # must precede any config-importing module

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.plotting.style import (  # noqa: E402
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    apply_style,
    save_figure,
)

# Fixed identity colors, matching analyze_scaling.py.
C_TRIMMED = "steelblue"
C_FULL = "#D55E00"
C_REF = "0.45"          # dashed reference lines (ideal / linear)
C_LIMIT = "#B2182B"     # hard limits (256 GB, 1M SU)
MODEL_COLORS = {"trimmed": C_TRIMMED, "full": C_FULL}

#: One marker per realization length, so L is readable without a legend lookup.
L_MARKERS = {5: "o", 10: "s", 20: "^", 30: "D"}

#: The design point every campaign SU number is derived from.
N_STAR, L_STAR = scfg.ENSEMBLE_COST_DESIGN_POINT


###############################################################################
# Load
###############################################################################

def load_cells() -> pd.DataFrame:
    """Concatenate every per-rank shard in the ensemble-cost cells directory."""
    shards = sorted(scfg.ENSEMBLE_COST_CELLS_DIR.glob("n*_L*_rank*_*.csv"))
    if not shards:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    for col in ("n_realizations", "realization_years", "k_concurrent",
                "eval_index", "rank"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("wall_seconds", "ru_maxrss_mb", "mem_available_mb",
                "obj0", "obj1", "obj2", "t_start_epoch"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_step_manifests() -> pd.DataFrame:
    """One row per attempted cell: its exit code and wall time."""
    rows = []
    for p in sorted(scfg.ENSEMBLE_COST_CELLS_DIR.glob("step_n*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    if not rows:
        return pd.DataFrame()
    steps = pd.DataFrame(rows)
    steps["step_wall_s"] = steps["t1"] - steps["t0"]
    return steps


def select_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the newest code epoch, refusing to pool across code versions.

    A SLURM job runs the code as it exists when the job STARTS, not when it was
    submitted, so a merge that lands while cells sit in the queue silently
    changes the model under test. Pooling those measurements would average two
    different models' costs into one surface. Shards record the git SHA they ran
    against; if more than one appears, the newest is kept and the rest are named
    in a warning, so the fix is to re-run the dropped cells rather than to
    quietly trust a contaminated number.
    """
    if df.empty or "code_sha" not in df.columns:
        return df
    epochs = df["code_sha"].dropna().unique()
    if len(epochs) <= 1:
        return df
    newest = (df.groupby("code_sha")["t_start_epoch"].max().idxmax())
    dropped = [e for e in epochs if e != newest]
    kept = df[df["code_sha"] == newest]
    lost = df[df["code_sha"] != newest]
    lost_cells = sorted(
        {(int(r.n_realizations), int(r.realization_years), r.model_variant)
         for r in lost.itertuples()}
    )
    sys.stderr.write(
        f"WARNING: shards span {len(epochs)} code epochs; keeping '{newest}' and "
        f"dropping {dropped}. Cells present only in the dropped epochs must be "
        f"re-run: {lost_cells}\n"
    )
    return kept


###############################################################################
# Cost surface
###############################################################################

def summarize_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Per (N, L, model) warm/cold cost, throughput, SU, and memory.

    The headline is the WARM median: Borg workers evaluate warm (the model dict
    is cached after the first eval, ``src/simulation.py::_get_cached_model_dict``),
    so warm is the cost the campaign actually pays. Cold is reported separately
    because it is what each worker pays once at startup. The spread (IQR across
    every rank x warm eval in the cell) is part of the result, not noise to be
    averaged away.
    """
    rows = []
    for (n, ell, model), grp in df.groupby(
        ["n_realizations", "realization_years", "model_variant"]
    ):
        warm = grp.loc[grp["kind"] == "warm", "wall_seconds"].dropna()
        cold = grp.loc[grp["kind"] == "cold", "wall_seconds"].dropna()
        if warm.empty:
            continue
        k = int(grp["k_concurrent"].max())
        # Peak RSS is per-process: take each rank's max, then summarize ranks.
        rss_per_rank = grp.groupby("rank")["ru_maxrss_mb"].max()
        rss_max_mb = float(rss_per_rank.max())
        t = float(warm.median())
        rows.append({
            "n_realizations": int(n),
            "realization_years": int(ell),
            "model_variant": model,
            "scenario_years": int(n) * int(ell),
            "k_concurrent": k,
            "n_ranks": int(grp["rank"].nunique()),
            "n_warm_evals": int(warm.size),
            "warm_median_s": t,
            "warm_q25_s": float(warm.quantile(0.25)),
            "warm_q75_s": float(warm.quantile(0.75)),
            "warm_p90_s": float(warm.quantile(0.90)),
            "warm_min_s": float(warm.min()),
            "warm_max_s": float(warm.max()),
            "cold_median_s": float(cold.median()) if not cold.empty else np.nan,
            "evals_per_node_hr": 3600.0 * k / t,
            "su_per_1000_evals": (1000.0 * scfg.SCALING_NODE_CORES * t
                                  / (3600.0 * k)),
            "rss_median_mb": float(rss_per_rank.median()),
            "rss_max_mb": rss_max_mb,
            "node_gb_at_k": k * rss_max_mb / 1024.0,
            "node_gb_at_128": 128 * rss_max_mb / 1024.0,
            "fits_128_ranks": bool(128 * rss_max_mb / 1024.0
                                   <= scfg.SCALING_NODE_MEM_GB),
            "objs_ok_frac": float(grp["objs_ok"].astype(str).eq("True").mean()),
            "obj0": float(grp["obj0"].median()),
            "code_sha": grp["code_sha"].iloc[0],
        })
    surf = pd.DataFrame(rows)
    return surf.sort_values(
        ["model_variant", "realization_years", "n_realizations"]
    ).reset_index(drop=True)


###############################################################################
# Scaling fits
###############################################################################

def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Ordinary least squares ``y = a + b*x``; returns (a, b, se_b, r2).

    ``se_b`` is NaN with fewer than 3 points: a two-point line has no residual
    degrees of freedom, and reporting a slope without its uncertainty is exactly
    the failure this experiment exists to correct.
    """
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if n < 3:
        return a, b, np.nan, r2
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_b = float(np.sqrt(ss_res / (n - 2) / sxx)) if sxx > 0 else np.nan
    return a, b, se_b, r2


def _offset_power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit ``t = c + b * x**alpha``; returns (c, b, alpha, se_alpha).

    The plain log-log fit below assumes cost is a pure power of x. It is not:
    every evaluation carries a fixed cost that does not scale with the ensemble
    (model write/load, objective reduction, MPI hand-off). That offset flattens
    the low-x end of the log-log curve and drags the fitted exponent DOWN — it
    manufactures sub-linearity. Dropping N=1 reduces the bias but does not remove
    it, so the offset is fitted explicitly here and reported alongside, and it is
    this exponent that describes the marginal cost of another realization.
    """
    from scipy.optimize import curve_fit

    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan

    def model(xx, c, b, alpha):
        return c + b * xx ** alpha

    # Seed from the log-log solution; bound the offset non-negative and the
    # exponent to a physically meaningful range.
    a0, alpha0, _, _ = _ols(np.log(x), np.log(y))
    try:
        popt, pcov = curve_fit(
            model, x, y,
            p0=[max(float(y.min()) * 0.1, 1e-3), float(np.exp(a0)), alpha0],
            bounds=([0.0, 1e-9, 0.0], [float(y.min()), np.inf, 3.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, np.nan
    c, b, alpha = (float(v) for v in popt)
    se_alpha = float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else np.nan
    return c, b, alpha, se_alpha


def fit_scaling(surf: pd.DataFrame) -> pd.DataFrame:
    """Empirical exponents: cost vs N at fixed L, and cost vs L at fixed N.

    An exponent of 1 is linear; below 1 is sub-linear. That is the whole question
    for N — pywrdrb runs realizations as pywr scenarios inside one model,
    vectorizing per-timestep work, so cost should grow more slowly than the
    realization count, and how much more slowly is what decides the N-vs-L trade.

    Each axis is fitted twice, because the two fits answer different questions
    and disagree in a way that matters:

    ``power_law``  ``log t = log a + alpha * log x``, ordinary least squares in
        log-log space. This is the exponent a reader sees as a slope on the F1
        figure. N=1 is excluded from the N fits: a one-realization evaluation is
        mostly fixed overhead, so keeping it bends the low-N end and biases the
        slope downward.

    ``offset_power_law``  ``t = c + b * x**alpha``, fitted in natural space with
        the fixed per-eval overhead c as a free parameter. This is the honest
        MARGINAL exponent — what another realization actually costs — and it is
        the one to quote. The pure power law understates alpha whenever c > 0,
        which is always; the gap between the two rows is exactly the size of that
        artifact.
    """
    rows = []

    def _add(axis: str, model: str, held: str, g: pd.DataFrame,
             xcol: str, drop_ones: bool) -> None:
        if drop_ones:
            g = g[g[xcol] > 1]
        g = g.sort_values(xcol)
        if len(g) < 2:
            return
        x = g[xcol].to_numpy(float)
        y = g["warm_median_s"].to_numpy(float)
        rng = f"{'N' if axis == 'N' else 'L'}={int(x.min())}..{int(x.max())}"

        a, alpha, se, r2 = _ols(np.log(x), np.log(y))
        rows.append({
            "axis": axis, "model_variant": model, "held_fixed": held,
            "fit_form": "power_law", "n_points": int(len(g)), "range": rng,
            "exponent": alpha, "exponent_se": se, "r2": r2,
            "prefactor_s": float(np.exp(a)), "offset_s": 0.0,
            "verdict": _verdict(alpha, se),
        })

        c, b, alpha_o, se_o = _offset_power_fit(x, y)
        if np.isfinite(alpha_o):
            rows.append({
                "axis": axis, "model_variant": model, "held_fixed": held,
                "fit_form": "offset_power_law", "n_points": int(len(g)),
                "range": rng, "exponent": alpha_o, "exponent_se": se_o,
                "r2": np.nan, "prefactor_s": b, "offset_s": c,
                "verdict": _verdict(alpha_o, se_o),
            })

    for model, mgrp in surf.groupby("model_variant"):
        for ell, grp in mgrp.groupby("realization_years"):
            _add("N", model, f"L={int(ell)}", grp, "n_realizations",
                 drop_ones=True)
        for n, grp in mgrp.groupby("n_realizations"):
            _add("L", model, f"N={int(n)}", grp, "realization_years",
                 drop_ones=False)
    return pd.DataFrame(rows)


def _verdict(alpha: float, se: float) -> str:
    """State plainly whether the exponent is sub-linear, linear, or super-linear.

    "Plainly" means against the measurement's own uncertainty: an exponent is
    only called sub-linear when it sits more than 2 standard errors below 1.
    """
    if not np.isfinite(alpha):
        return "unfitted"
    if not np.isfinite(se):
        return f"alpha={alpha:.2f} (2 points; no uncertainty)"
    if alpha < 1.0 - 2 * se:
        return f"sub-linear ({100 * (1 - alpha):.0f}% below linear)"
    if alpha > 1.0 + 2 * se:
        return f"super-linear ({100 * (alpha - 1):.0f}% above linear)"
    return "linear (within 2 se)"


###############################################################################
# Full / trimmed ratio
###############################################################################

def model_ratio(surf: pd.DataFrame) -> pd.DataFrame:
    """Full ÷ trimmed cost and memory ratio at every (N, L) measured in both.

    Every re-evaluation SU estimate is this ratio times a trimmed number, so it
    is measured at as many points as the sweep covers rather than assumed
    constant.

    Also cross-checks the experiment's one silent failure mode: if
    ``NYCOPT_USE_TRIMMED_MODEL`` never reached ``config``, the "full" cells would
    have re-run the trimmed model and produced a ratio near 1.0 with no error
    anywhere. The two models route the lower-basin reservoirs differently, so
    their objective vectors MUST differ; identical objectives mean the variant
    switch did not take, and the cell is flagged rather than reported.
    """
    rows = []
    piv = surf.set_index(["n_realizations", "realization_years", "model_variant"])
    pairs = sorted(
        {(n, ell) for n, ell, m in piv.index if
         (n, ell, "trimmed") in piv.index and (n, ell, "full") in piv.index}
    )
    for n, ell in pairs:
        tr = piv.loc[(n, ell, "trimmed")]
        fu = piv.loc[(n, ell, "full")]
        objs_differ = not np.isclose(tr["obj0"], fu["obj0"], rtol=1e-9, atol=0.0)
        rows.append({
            "n_realizations": n,
            "realization_years": ell,
            "scenario_years": n * ell,
            "trimmed_warm_s": tr["warm_median_s"],
            "full_warm_s": fu["warm_median_s"],
            "time_ratio": fu["warm_median_s"] / tr["warm_median_s"],
            "trimmed_rss_mb": tr["rss_max_mb"],
            "full_rss_mb": fu["rss_max_mb"],
            "rss_ratio": fu["rss_max_mb"] / tr["rss_max_mb"],
            "trimmed_k": int(tr["k_concurrent"]),
            "full_k": int(fu["k_concurrent"]),
            "su_ratio": fu["su_per_1000_evals"] / tr["su_per_1000_evals"],
            "variant_switch_ok": bool(objs_differ),
        })
    ratio = pd.DataFrame(rows)
    if not ratio.empty and not ratio["variant_switch_ok"].all():
        bad = ratio.loc[~ratio["variant_switch_ok"],
                        ["n_realizations", "realization_years"]].to_dict("records")
        sys.stderr.write(
            "WARNING: trimmed and full produced IDENTICAL objectives at "
            f"{bad}. NYCOPT_USE_TRIMMED_MODEL did not reach config for those "
            "cells, so their 'full' timings are trimmed timings. Re-run them.\n"
        )
    return ratio


###############################################################################
# Campaign projection
###############################################################################

def project_campaign(surf: pd.DataFrame, ratio: pd.DataFrame) -> pd.DataFrame:
    """SU for the search campaign and the re-evaluation, against the allocation.

    Search: ``designs x draws x seeds`` independent MM-Borg runs, each of NFE
    evaluations at the campaign design point (N=100, L=10) on the trimmed model.
    Wall time per run is ``NFE * t / (workers * efficiency)``; SU is node-hours
    times the node's 128 charged cores. Efficiency is the measured Anvil Stage-B
    value, which was measured on ~13 s evaluations — coordination overhead is a
    larger share of wall time there than at the campaign's minute-scale ensemble
    evaluations, so this over-estimates search wall time and the search SU here
    is a conservative upper bound.

    Re-evaluation: ``n_policies x N_theta`` full-model simulations of an R x
    L_test test ensemble. This is a task farm, not a search: no islands, no
    generations, no coordination — so it is priced at a utilization factor, not
    at the Borg efficiency. Its per-eval cost is the measured FULL-model cost at
    the (R, L_test) shape when that cell was measured, and otherwise the
    measured trimmed cost at that shape scaled by the measured full/trimmed
    ratio (recorded per row in ``cost_basis``, so no projected number's
    provenance is ambiguous).
    """
    design_pt = surf[
        (surf["n_realizations"] == N_STAR)
        & (surf["realization_years"] == L_STAR)
        & (surf["model_variant"] == "trimmed")
    ]
    if design_pt.empty:
        sys.stderr.write(
            f"WARNING: the campaign design point (N={N_STAR}, L={L_STAR}, "
            "trimmed) was not measured; skipping the projection. It is the one "
            "cell the projection cannot be built without.\n"
        )
        return pd.DataFrame()
    t_star = float(design_pt["warm_median_s"].iloc[0])
    k_star = int(design_pt["k_concurrent"].iloc[0])

    nodes = scfg.ENSEMBLE_COST_PROJ_NODES
    islands = scfg.ENSEMBLE_COST_PROJ_ISLANDS
    eff = scfg.ENSEMBLE_COST_PROJ_EFFICIENCY
    # MM-Borg rank budget: one controller, one master per island, the rest work.
    workers = nodes * k_star - 1 - islands

    # Median measured ratio, the fallback when a re-eval cell has no full-model
    # measurement of its own.
    fallback_ratio = float(ratio["time_ratio"].median()) if not ratio.empty else np.nan

    def _reeval_cost(r: int, l_test: int) -> tuple[float, int, str]:
        """(seconds, ranks_per_node, provenance) for one full-model E_test sim."""
        cell = surf[(surf["n_realizations"] == r)
                    & (surf["realization_years"] == l_test)
                    & (surf["model_variant"] == "full")]
        if not cell.empty:
            return (float(cell["warm_median_s"].iloc[0]),
                    int(cell["k_concurrent"].iloc[0]), "measured_full")
        cell = surf[(surf["n_realizations"] == r)
                    & (surf["realization_years"] == l_test)
                    & (surf["model_variant"] == "trimmed")]
        if not cell.empty and np.isfinite(fallback_ratio):
            return (float(cell["warm_median_s"].iloc[0]) * fallback_ratio,
                    int(cell["k_concurrent"].iloc[0]),
                    "trimmed x median_full_trimmed_ratio")
        return np.nan, 0, "unmeasured"

    n_pol = scfg.ENSEMBLE_COST_REEVAL_POLICIES
    util = scfg.ENSEMBLE_COST_REEVAL_UTILIZATION
    cores = scfg.SCALING_NODE_CORES
    alloc = scfg.ENSEMBLE_COST_ALLOCATION_SU

    rows = []
    for draws in scfg.ENSEMBLE_COST_PROJ_DRAWS:
        for seeds in scfg.ENSEMBLE_COST_PROJ_SEEDS:
            for nfe in scfg.ENSEMBLE_COST_PROJ_NFE:
                n_runs = scfg.ENSEMBLE_COST_PROJ_DESIGNS * draws * seeds
                run_hr = nfe * t_star / (workers * eff) / 3600.0
                search_su = n_runs * nodes * cores * run_hr
                for n_theta in scfg.ENSEMBLE_COST_ETEST_NTHETA:
                    for r in scfg.ENSEMBLE_COST_ETEST_R:
                        for l_test in scfg.ENSEMBLE_COST_ETEST_LTEST:
                            t_re, k_re, basis = _reeval_cost(r, l_test)
                            n_sims = n_pol * n_theta
                            if np.isfinite(t_re) and k_re > 0:
                                reeval_su = (n_sims * t_re * cores
                                             / (3600.0 * k_re * util))
                            else:
                                reeval_su = np.nan
                            total = search_su + reeval_su
                            rows.append({
                                "n_designs": scfg.ENSEMBLE_COST_PROJ_DESIGNS,
                                "n_draws": draws,
                                "n_seeds": seeds,
                                "nfe": nfe,
                                "n_borg_runs": n_runs,
                                "search_t_eval_s": t_star,
                                "search_ranks_per_node": k_star,
                                "search_nodes": nodes,
                                "search_islands": islands,
                                "search_workers": workers,
                                "search_efficiency": eff,
                                "search_hr_per_run": run_hr,
                                "search_su": search_su,
                                "etest_n_theta": n_theta,
                                "etest_r": r,
                                "etest_l_test": l_test,
                                "etest_scenarios": n_theta * r,
                                "reeval_policies": n_pol,
                                "reeval_sims": n_sims,
                                "reeval_t_eval_s": t_re,
                                "reeval_ranks_per_node": k_re,
                                "reeval_cost_basis": basis,
                                "reeval_su": reeval_su,
                                "total_su": total,
                                "frac_of_allocation": total / alloc,
                            })
    return pd.DataFrame(rows)


###############################################################################
# Figures
###############################################################################

def fig_cost_surface(surf: pd.DataFrame, out) -> None:
    """F1 - warm eval time vs N, log-log, one line per L, one panel per model.

    The money figure. A reference line of slope 1 (cost proportional to N) is
    drawn in each panel: every measured line sitting flatter than it is the
    sub-linearity in N, visible without reading a table. The campaign design
    point is marked because it is the number the whole experiment exists to
    supply.
    """
    models = [m for m in ("trimmed", "full") if m in set(surf["model_variant"])]
    fig, axes = plt.subplots(1, len(models), figsize=FIGSIZE_WIDE,
                             sharey=True, squeeze=False, layout="constrained")
    for ax, model in zip(axes[0], models):
        sub = surf[surf["model_variant"] == model]
        for ell, grp in sub.groupby("realization_years"):
            g = grp.sort_values("n_realizations")
            x = g["n_realizations"].to_numpy(float)
            y = g["warm_median_s"].to_numpy(float)
            ax.fill_between(x, g["warm_q25_s"], g["warm_q75_s"],
                            color=MODEL_COLORS[model], alpha=0.15, lw=0)
            ax.plot(x, y, marker=L_MARKERS.get(int(ell), "o"), ms=4,
                    color=MODEL_COLORS[model], lw=1.4)
            # Label the line at its right end instead of spending a legend on L.
            ax.annotate(f"L={int(ell)} yr", xy=(x[-1], y[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MODEL_COLORS[model])
        # Slope-1 reference, anchored under the cheapest measured line.
        if not sub.empty:
            x0 = float(sub["n_realizations"].min())
            x1 = float(sub["n_realizations"].max())
            y0 = float(sub["warm_median_s"].min()) * 0.75
            ref_x = np.array([max(x0, 1.0), x1])
            ref_y = y0 * ref_x / ref_x[0]
            ax.plot(ref_x, ref_y, ls="--", color=C_REF, lw=1.0)
            ax.annotate("slope 1 (cost ∝ N)", xy=(ref_x[-1], ref_y[-1]),
                        xytext=(-4, 4), textcoords="offset points",
                        ha="right", fontsize=8, color=C_REF)
        star = sub[(sub["n_realizations"] == N_STAR)
                   & (sub["realization_years"] == L_STAR)]
        if not star.empty:
            ts = float(star["warm_median_s"].iloc[0])
            ax.plot([N_STAR], [ts], marker="*", ms=15, mfc="none",
                    mec="black", mew=1.2, ls="none", zorder=5)
            ax.annotate(f"campaign design point\nN={N_STAR}, L={L_STAR}: {ts:.0f} s",
                        xy=(N_STAR, ts), xytext=(-8, -30),
                        textcoords="offset points", ha="right", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Realizations per evaluation, N")
        ax.set_title(f"{model} model "
                     f"({'search' if model == 'trimmed' else 're-evaluation'} path)")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel("Warm evaluation wall time (s)")
    fig.suptitle(
        "Per-evaluation cost grows sub-linearly in N: pywrdrb runs realizations "
        "as pywr scenarios inside one model.\n"
        "Bands are the interquartile range across ranks x warm evaluations. "
        "Each cell ran at its memory-feasible packing density (<= 128 ranks/node).",
        fontsize=9,
    )
    save_figure(fig, out)
    plt.close(fig)


def fig_memory(surf: pd.DataFrame, out) -> None:
    """F2 - RSS per evaluation, and the node total it implies at 128 ranks.

    Left: the per-rank peak. Right: what 128 of those ranks would need on one
    node, against the 256 GB the node has. Where the right panel crosses the
    limit, memory - not contention - is what caps the campaign's packing
    density, and the SU-optimal 128 ranks/node stops being reachable.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE,
                                   layout="constrained")
    for model in ("trimmed", "full"):
        sub = surf[surf["model_variant"] == model]
        for ell, grp in sub.groupby("realization_years"):
            g = grp.sort_values("n_realizations")
            x = g["n_realizations"].to_numpy(float)
            ax1.plot(x, g["rss_median_mb"] / 1024.0, lw=1.3,
                     marker=L_MARKERS.get(int(ell), "o"), ms=4,
                     color=MODEL_COLORS[model],
                     ls="-" if model == "trimmed" else "--")
            ax1.fill_between(x, g["rss_median_mb"] / 1024.0,
                             g["rss_max_mb"] / 1024.0,
                             color=MODEL_COLORS[model], alpha=0.12, lw=0)
            ax2.plot(x, g["node_gb_at_128"], lw=1.3,
                     marker=L_MARKERS.get(int(ell), "o"), ms=4,
                     color=MODEL_COLORS[model],
                     ls="-" if model == "trimmed" else "--")
            if len(g):
                ax2.annotate(f"L={int(ell)}",
                             xy=(x[-1], float(g["node_gb_at_128"].iloc[-1])),
                             xytext=(4, 0), textcoords="offset points",
                             va="center", fontsize=7,
                             color=MODEL_COLORS[model])
    ax2.axhline(scfg.SCALING_NODE_MEM_GB, color=C_LIMIT, ls="-", lw=1.4)
    ax2.annotate(f"{scfg.SCALING_NODE_MEM_GB} GB node limit",
                 xy=(0.02, scfg.SCALING_NODE_MEM_GB), xycoords=("axes fraction", "data"),
                 xytext=(0, 4), textcoords="offset points",
                 fontsize=8, color=C_LIMIT)
    for ax, ylab, title in (
        (ax1, "Peak RSS per evaluating rank (GB)",
         "Memory per evaluation\n(band: median to max across ranks)"),
        (ax2, "Projected node total at 128 ranks (GB)",
         "Can the campaign pack 128 ranks/node?"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("Realizations per evaluation, N")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    ax1.plot([], [], color=C_TRIMMED, ls="-", label="trimmed (search)")
    ax1.plot([], [], color=C_FULL, ls="--", label="full (re-evaluation)")
    ax1.legend(frameon=False, loc="upper left")
    save_figure(fig, out)
    plt.close(fig)


def fig_model_ratio(ratio: pd.DataFrame, out) -> None:
    """F3 - the full/trimmed cost ratio across the surface.

    One number decides how much more a re-evaluation costs than a search
    evaluation. Plotted against N with a line per L so a reader can see whether
    it is a constant (in which case one number suffices) or drifts with the
    ensemble shape (in which case it does not).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, layout="constrained")
    for ell, grp in ratio.groupby("realization_years"):
        g = grp.sort_values("n_realizations")
        x = g["n_realizations"].to_numpy(float)
        y = g["time_ratio"].to_numpy(float)
        ax.plot(x, y, marker=L_MARKERS.get(int(ell), "o"), ms=5, lw=1.4,
                color=C_FULL)
        ax.annotate(f"L={int(ell)} yr", xy=(x[-1], y[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=C_FULL)
    med = float(ratio["time_ratio"].median())
    ax.axhline(med, color=C_REF, ls="--", lw=1.0)
    ax.annotate(f"median {med:.2f}x", xy=(0.02, med), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", fontsize=8, color=C_REF)
    ax.axhline(1.0, color="0.75", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Realizations per evaluation, N")
    ax.set_ylabel("Full ÷ trimmed warm evaluation time")
    ax.set_title("A re-evaluation costs this many search evaluations\n"
                 "(full model simulates the lower-basin reservoirs the trimmed "
                 "model reads from presimulated releases)", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    save_figure(fig, out)
    plt.close(fig)


def fig_campaign_su(proj: pd.DataFrame, out) -> None:
    """F4 - projected SU vs test-ensemble size, against the 1M SU allocation.

    The search campaign is a fixed base (it does not depend on the test
    ensemble); the re-evaluation stacks on top and grows with E_test. Where a
    bar crosses the allocation line, that E_test is unaffordable. A reader
    should be able to point at this figure and pick a test-ensemble size.

    Drawn at the median search configuration so the E_test axis is readable;
    ``campaign_projection.csv`` carries the full search grid.
    """
    if proj.empty:
        return
    # Middle-of-the-road search config: the table has the rest.
    draws = sorted(proj["n_draws"].unique())[0]
    seeds = sorted(proj["n_seeds"].unique())[0]
    nfe_grid = sorted(proj["nfe"].unique())

    fig, axes = plt.subplots(1, len(nfe_grid), figsize=FIGSIZE_WIDE,
                             sharey=True, squeeze=False, layout="constrained")
    alloc = scfg.ENSEMBLE_COST_ALLOCATION_SU
    for ax, nfe in zip(axes[0], nfe_grid):
        sub = proj[(proj["n_draws"] == draws) & (proj["n_seeds"] == seeds)
                   & (proj["nfe"] == nfe)].copy()
        sub = sub.sort_values(["etest_l_test", "etest_r", "etest_n_theta"])
        labels = [f"{int(r.etest_n_theta)}x{int(r.etest_r)}, L={int(r.etest_l_test)}"
                  for r in sub.itertuples()]
        idx = np.arange(len(sub))
        search = sub["search_su"].to_numpy(float)
        reeval = np.nan_to_num(sub["reeval_su"].to_numpy(float), nan=0.0)
        ax.bar(idx, search, color=C_TRIMMED, label="search (Borg, trimmed)")
        ax.bar(idx, reeval, bottom=search, color=C_FULL,
               label="re-evaluation (full model)")
        ax.axhline(alloc, color=C_LIMIT, lw=1.5)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
        ax.set_xlabel("Test ensemble: N_theta x R realizations, L_test yr",
                      fontsize=8)
        ax.set_title(f"NFE = {int(nfe):,}", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0][0].set_ylabel("Projected cost (SU)")
    axes[0][-1].annotate(f"{alloc:,} SU allocation",
                         xy=(0.98, alloc), xycoords=("axes fraction", "data"),
                         xytext=(0, 5), textcoords="offset points",
                         ha="right", fontsize=8, color=C_LIMIT)
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(
        f"Campaign cost vs test-ensemble size: "
        f"{scfg.ENSEMBLE_COST_PROJ_DESIGNS} designs x {draws} draws x {seeds} "
        f"seeds of MM-Borg search, plus {scfg.ENSEMBLE_COST_REEVAL_POLICIES} "
        f"policies re-evaluated on the full model.",
        fontsize=9,
    )
    save_figure(fig, out)
    plt.close(fig)


###############################################################################
# Main
###############################################################################

def main() -> int:
    """Build every table and figure from whatever cells have been measured."""
    apply_style()
    scfg.ENSEMBLE_COST_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scfg.ENSEMBLE_COST_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_cells()
    if df.empty:
        sys.stderr.write(
            f"ERROR: no shards under {scfg.ENSEMBLE_COST_CELLS_DIR}. Run "
            "workflow/supplemental/ensemble_cost_sweep.sh first.\n"
        )
        return 2
    df = select_epoch(df)
    df.to_csv(scfg.ensemble_cost_table_path("cell_evals"), index=False)

    surf = summarize_cells(df)
    surf.to_csv(scfg.ensemble_cost_table_path("cost_surface"), index=False)

    fits = fit_scaling(surf)
    fits.to_csv(scfg.ensemble_cost_table_path("scaling_fits"), index=False)

    ratio = model_ratio(surf)
    ratio.to_csv(scfg.ensemble_cost_table_path("model_ratio"), index=False)

    proj = project_campaign(surf, ratio)
    if not proj.empty:
        proj.to_csv(scfg.ensemble_cost_table_path("campaign_projection"),
                    index=False)

    # Cells attempted but with no surviving shard: an OOM-killed cell is a
    # measurement of the memory ceiling, so name it rather than let it vanish.
    steps = load_step_manifests()
    if not steps.empty:
        measured = {(int(r.n_realizations), int(r.realization_years),
                     r.model_variant) for r in surf.itertuples()}
        lost = [
            (int(r.n_realizations), int(r.realization_years), r.model_variant,
             int(r.k), int(r.rc))
            for r in steps.itertuples()
            if (int(r.n_realizations), int(r.realization_years),
                r.model_variant) not in measured
        ]
        if lost:
            sys.stderr.write(
                "NOTE: cells attempted with no surviving shard "
                "(N, L, model, K, rc): "
                f"{lost}\n"
            )

    fig_cost_surface(surf, scfg.ensemble_cost_figure_path("F1_cost_surface"))
    fig_memory(surf, scfg.ensemble_cost_figure_path("F2_memory"))
    if not ratio.empty:
        fig_model_ratio(ratio, scfg.ensemble_cost_figure_path("F3_model_ratio"))
    fig_campaign_su(proj, scfg.ensemble_cost_figure_path("F4_campaign_su"))

    # Headline numbers, for the log and the note.
    star = surf[(surf["n_realizations"] == N_STAR)
                & (surf["realization_years"] == L_STAR)]
    print(f"cells measured: {len(surf)} "
          f"({surf['n_warm_evals'].sum()} warm evals)")
    for _, r in star.iterrows():
        print(f"  design point N={N_STAR} L={L_STAR} {r['model_variant']:>7}: "
              f"{r['warm_median_s']:.1f}s warm "
              f"(IQR {r['warm_q25_s']:.1f}-{r['warm_q75_s']:.1f}), "
              f"K={int(r['k_concurrent'])}, "
              f"{r['evals_per_node_hr']:.0f} evals/node-hr, "
              f"{r['su_per_1000_evals']:.1f} SU/1000 evals, "
              f"RSS {r['rss_max_mb']:.0f} MB")
    for _, r in fits[(fits["axis"] == "N")
                     & (fits["fit_form"] == "offset_power_law")].iterrows():
        print(f"  N exponent [{r['model_variant']}, {r['held_fixed']}]: "
              f"{r['exponent']:.3f} +/- {r['exponent_se']:.3f} "
              f"(marginal, offset {r['offset_s']:.1f}s) -> {r['verdict']}")
    if not ratio.empty:
        print(f"  full/trimmed time ratio: median {ratio['time_ratio'].median():.2f}x "
              f"(range {ratio['time_ratio'].min():.2f}-{ratio['time_ratio'].max():.2f}, "
              f"n={len(ratio)} paired cells)")
    print(f"wrote tables -> {scfg.ENSEMBLE_COST_TABLES_DIR}")
    print(f"wrote figures -> {scfg.ENSEMBLE_COST_FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
