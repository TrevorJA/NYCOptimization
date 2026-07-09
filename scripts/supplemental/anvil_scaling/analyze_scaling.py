"""
analyze_scaling.py - Stage C of the Anvil scaling experiment: tables + figures.

Combines the Stage A packing shards and Stage B Borg timing CSVs under
``outputs/supplemental/anvil_scaling_experiment/`` into summary tables and the
manuscript-supplement figures. Runs anywhere (login node or laptop) — no MPI,
no SLURM; it only reads files. Degrades gracefully when a stage is missing or
partial (e.g. smoke-only data): each table/figure is produced from whatever
data exists and skipped otherwise.

Outputs (under the experiment root):
    tables/packing_combined.csv   all shard rows, concatenated
    tables/packing_summary.csv    per (K, batch): timing/throughput/SU/memory
    tables/borg_jobs.csv          one row per Stage B job
    tables/borg_summary.csv       per geometry: wall stats, speedup, efficiency
    tables/projection.csv         production-campaign walltime/SU projections
    figures/F1_packing_eval_time  warm/cold eval time vs ranks-per-node
    figures/F2_packing_throughput_cost  evals/node-hr + SU/1000 evals vs K
    figures/F3_packing_memory     per-rank RSS + projected node total vs 256 GB
    figures/F4_borg_strong_scaling  wall time / speedup-vs-ideal / efficiency
    figures/F5_production_projection  nodes x islands walltime heatmap + SU

Usage:
    python scripts/supplemental/anvil_scaling/analyze_scaling.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_anvil_scaling_env()  # before any config-importing module

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.plotting.style import (  # noqa: E402
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    apply_style,
    save_figure,
)

# Fixed series colors (colorblind-safe pair + neutrals; identity, not rank).
C_PRIMARY = "steelblue"     # measured series
C_SECONDARY = "#D55E00"     # cost / secondary measured series
C_IDEAL = "0.45"            # dashed reference lines
C_BAND = "steelblue"        # replicate bands (alpha-shaded)
C_LIMIT = "#B2182B"         # hard limits (256 GB line)

#: Distinct markers for the 64-slot island-decomposition arms (identity-fixed).
ISLAND_MARKERS = {1: "o", 2: "s", 4: "D"}

_EVAL_LINE_RE = re.compile(r"Eval #\d+: ([\d.]+)s this eval")


# ---------------------------------------------------------------------------
# Stage A — packing
# ---------------------------------------------------------------------------

def load_packing() -> "tuple[pd.DataFrame | None, pd.DataFrame | None]":
    """Load all packing shards and step manifests.

    Returns:
        ``(rows, steps)`` — the concatenated shard rows and the per-step
        manifest table (``k, batch, rc, wall_s``), either ``None`` if absent.
    """
    shard_paths = sorted(scfg.SCALING_PACKING_DIR.glob("k*_b*_rank*_*.csv"))
    rows = None
    if shard_paths:
        rows = pd.concat([pd.read_csv(p) for p in shard_paths],
                         ignore_index=True)
    step_paths = sorted(scfg.SCALING_PACKING_DIR.glob("step_k*_b*_*.json"))
    steps = None
    if step_paths:
        recs = []
        for p in step_paths:
            d = json.loads(p.read_text())
            d["wall_s"] = d["t1"] - d["t0"]
            recs.append(d)
        steps = pd.DataFrame(recs)
    return rows, steps


def summarize_packing(rows: pd.DataFrame,
                      steps: "pd.DataFrame | None") -> pd.DataFrame:
    """Aggregate shard rows to one summary row per (K, realization_batch).

    Missing ranks (fewer shards than K) and nonzero step exit codes are
    retained as evidence — they mark the memory ceiling, not bad data.

    Smoke-mode shards are excluded whenever measurement-mode (ladder/spot)
    shards exist: smoke runs may execute on non-exclusive shared nodes, and
    mixing them into the same (K, batch) groups would bias the K=1 baseline
    every slowdown and the K* choice normalize against. With only smoke data
    (the Step-1 dry run) they are summarized as-is.
    """
    measured = rows[rows["mode"] != "smoke"]
    if len(measured):
        n_dropped = len(rows) - len(measured)
        if n_dropped:
            print(f"[analyze] excluding {n_dropped} smoke-mode shard rows "
                  f"from the packing summary (ladder/spot data present)")
        rows = measured
    out = []
    for (k, batch), grp in rows.groupby(["k_concurrent", "realization_batch"]):
        warm = grp.loc[grp["kind"] == "warm", "wall_seconds"].astype(float)
        cold = grp.loc[grp["kind"] == "cold", "wall_seconds"].astype(float)
        rss_per_rank = grp.groupby("rank")["ru_maxrss_mb"].max()
        warm_med = warm.median() if len(warm) else np.nan
        rc = 0
        if steps is not None:
            m = steps[(steps["k"] == k) & (steps["batch"] == batch)]
            if len(m):
                rc = int(m["rc"].iloc[0])
        out.append({
            "k": int(k),
            "realization_batch": int(batch),
            "n_ranks_reported": grp["rank"].nunique(),
            "n_warm": len(warm),
            "warm_median_s": warm_med,
            "warm_q25_s": warm.quantile(0.25) if len(warm) else np.nan,
            "warm_q75_s": warm.quantile(0.75) if len(warm) else np.nan,
            # Straggler tail: Borg throughput is bounded by the slowest
            # evaluator, so the p90 complements the median in the K* choice.
            "warm_p90_s": warm.quantile(0.90) if len(warm) else np.nan,
            "cold_median_s": cold.median() if len(cold) else np.nan,
            "evals_per_node_hr": (3600.0 * k / warm_med
                                  if np.isfinite(warm_med) else np.nan),
            "su_per_1000_evals": (1000.0 * scfg.SCALING_NODE_CORES * warm_med
                                  / (3600.0 * k)
                                  if np.isfinite(warm_med) else np.nan),
            "rss_median_mb": rss_per_rank.median(),
            "rss_max_mb": rss_per_rank.max(),
            "node_projected_gb": k * rss_per_rank.max() / 1024.0,
            # Explicit string map: astype(bool) would turn "False" into True
            # if a partial shard ever makes the column object-dtype.
            "objs_ok_frac": grp["objs_ok"].astype(str).eq("True").mean(),
            # Identical DVs must give identical objectives on every rank/eval;
            # >1 distinct vector flags a correctness problem, not noise.
            "n_distinct_obj_vectors": (
                grp[["obj0", "obj1", "obj2"]].astype(float).round(9)
                .drop_duplicates().shape[0]),
            "step_rc": rc,
            "completed": rc == 0 and grp["rank"].nunique() == k,
        })
    df = pd.DataFrame(out).sort_values(["realization_batch", "k"])
    base = df[(df["k"] == 1) & (df["realization_batch"] == 0)]
    ref = base["warm_median_s"].iloc[0] if len(base) else np.nan
    df["slowdown_vs_k1"] = df["warm_median_s"] / ref
    return df.reset_index(drop=True)


def choose_kstar(summary: pd.DataFrame) -> "int | None":
    """Pick the packing density K* minimizing SU cost within the memory margin.

    Candidates: completed unbatched steps whose projected node memory stays
    under 90% of the node's 256 GB. Falls back to the cheapest completed step.
    """
    ok = summary[(summary["realization_batch"] == 0) & summary["completed"]]
    if not len(ok):
        return None
    safe = ok[ok["node_projected_gb"] < 0.9 * scfg.SCALING_NODE_MEM_GB]
    pick = safe if len(safe) else ok
    return int(pick.loc[pick["su_per_1000_evals"].idxmin(), "k"])


def summarize_batch(rows: pd.DataFrame) -> "pd.DataFrame | None":
    """Per-(K, B) summary of the batched-evaluation sweep (mode "batch" only).

    Self-contained: uses batch-mode shards exclusively, so it is valid even
    when the ladder ran on a different node/partition. ``B=0`` (all
    realizations in one pywr scenario block) is reported as
    ``b_effective = n_realizations`` so the sweep plots on one numeric axis.
    ``time_vs_all`` normalizes each warm median to the same-K B=0 point —
    the per-run-overhead amortization curve.
    """
    b = rows[rows["mode"] == "batch"]
    if not len(b):
        return None
    n_real = int(b["n_realizations"].mode().iloc[0])
    out = []
    for (k, batch), grp in b.groupby(["k_concurrent", "realization_batch"]):
        warm = grp.loc[grp["kind"] == "warm", "wall_seconds"].astype(float)
        med = warm.median() if len(warm) else np.nan
        out.append({
            "k": int(k),
            "realization_batch": int(batch),
            "b_effective": int(batch) if batch > 0 else n_real,
            "n_model_runs_per_eval": int(np.ceil(n_real / (batch or n_real))),
            "n_ranks_reported": grp["rank"].nunique(),
            "n_warm": len(warm),
            "warm_median_s": med,
            "warm_q25_s": warm.quantile(0.25) if len(warm) else np.nan,
            "warm_q75_s": warm.quantile(0.75) if len(warm) else np.nan,
            "rss_max_mb": grp.groupby("rank")["ru_maxrss_mb"].max().max(),
            "su_per_1000_evals": (1000.0 * scfg.SCALING_NODE_CORES * med
                                  / (3600.0 * k) if np.isfinite(med) else np.nan),
            "objs_ok_frac": grp["objs_ok"].astype(str).eq("True").mean(),
            "n_distinct_obj_vectors": (
                grp[["obj0", "obj1", "obj2"]].astype(float).round(9)
                .drop_duplicates().shape[0]),
        })
    df = pd.DataFrame(out).sort_values(["k", "b_effective"]).reset_index(drop=True)
    ref = df[df["realization_batch"] == 0].set_index("k")["warm_median_s"]
    df["time_vs_all"] = df.apply(
        lambda r: r["warm_median_s"] / ref.get(r["k"], np.nan), axis=1)
    return df


# ---------------------------------------------------------------------------
# Stage B — Borg strong scaling
# ---------------------------------------------------------------------------

def load_borg() -> "pd.DataFrame | None":
    """Load all Stage B one-row timing CSVs into a jobs table."""
    paths = sorted(scfg.SCALING_BORG_DIR.glob("timing_*.csv"))
    if not paths:
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _log_eval_seconds() -> "dict[str, float]":
    """Median per-eval wall time per geometry, parsed from Stage B job logs.

    Defensive enrichment: returns an empty dict when logs are absent (e.g.
    when analyzing on a laptop). Cold first evals are excluded by the median.
    """
    out: dict[str, list[float]] = {}
    for p in (PROJECT_DIR / "logs").glob("anvil_scaling_borg_ansb_*_seed*.out"):
        m = re.search(r"ansb_(scale_\w+?)_\d+_seed", p.name)
        if not m:
            continue
        try:
            vals = [float(v) for v in _EVAL_LINE_RE.findall(p.read_text())]
        except OSError:
            continue
        out.setdefault(m.group(1), []).extend(vals)
    return {cfg: float(np.median(v)) for cfg, v in out.items() if v}


def summarize_borg(jobs: pd.DataFrame) -> pd.DataFrame:
    """Per-geometry strong-scaling summary (speedup vs the smallest geometry).

    ``scale_smoke`` (different NFE) and failed jobs are excluded; all included
    geometries must share the same fixed total NFE for wall times to compare.
    """
    ok = jobs[(jobs["rc"] == 0) & (jobs["config"] != "scale_smoke")]
    if not len(ok):
        return pd.DataFrame()
    total_nfe = ok["total_nfe"].mode().iloc[0]
    ok = ok[ok["total_nfe"] == total_nfe]
    t_eval = _log_eval_seconds()

    g = ok.groupby("config").agg(
        islands=("islands", "first"),
        workers_per_island=("workers_per_island", "first"),
        total_slots=("total_slots", "first"),
        ranks=("ranks", "first"),
        total_nfe=("total_nfe", "first"),
        n_seeds=("seed", "nunique"),
        wall_mean_s=("wall_seconds", "mean"),
        wall_min_s=("wall_seconds", "min"),
        wall_max_s=("wall_seconds", "max"),
    ).reset_index()

    base = g.loc[g["total_slots"].idxmin()]
    g["speedup"] = base["wall_mean_s"] / g["wall_mean_s"]
    g["speedup_min"] = base["wall_mean_s"] / g["wall_max_s"]
    g["speedup_max"] = base["wall_mean_s"] / g["wall_min_s"]
    g["ideal_speedup"] = g["total_slots"] / base["total_slots"]
    g["efficiency"] = g["speedup"] / g["ideal_speedup"]
    g["eval_median_s"] = g["config"].map(t_eval).astype(float)
    # Fraction of slot-time NOT spent evaluating (Borg coordination + idle).
    g["overhead_frac"] = 1.0 - (g["total_nfe"] * g["eval_median_s"]
                                / (g["total_slots"] * g["wall_mean_s"]))
    return g.sort_values("total_slots").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Projection (Stage A throughput x Stage B efficiency)
# ---------------------------------------------------------------------------

def project_production(pack: pd.DataFrame, borg: pd.DataFrame,
                       kstar: int) -> pd.DataFrame:
    """Predict campaign walltime/SU for candidate (nodes, islands) geometries.

    ``walltime = NFE * t_warm(K*) / (workers * efficiency(slots))`` with
    efficiency interpolated from the **single-island** Stage B curve only
    (clamped at its measured ends; clamped rows are flagged
    ``eff_extrapolated``). The multi-island 64-slot arms are excluded: they
    duplicate the x-value (breaking interpolation) and the 4x16 arm is the
    deliberately overhead-dominated short-trajectory comparison, not a
    scaling measurement. The Stage B efficiency comes from short-window
    single-trace runs (~13 s/eval), where coordination overhead is a larger
    fraction than in production ensemble evals — the projection is therefore
    conservative (over-estimates walltime).
    """
    row = pack[(pack["k"] == kstar) & (pack["realization_batch"] == 0)]
    t_warm = float(row["warm_median_s"].iloc[0])
    curve = borg[borg["islands"] == 1].sort_values("total_slots")
    slots_meas = curve["total_slots"].to_numpy(dtype=float)
    eff_meas = curve["efficiency"].to_numpy(dtype=float)

    recs = []
    for nodes in scfg.SCALING_PROJECTION_NODES:
        for islands in scfg.SCALING_PROJECTION_ISLANDS:
            ranks = nodes * kstar
            workers = ranks - 1 - islands  # controller + island masters
            if workers < islands:
                continue
            eff = float(np.interp(workers, slots_meas, eff_meas))
            hours = (scfg.SCALING_PROJECTION_TOTAL_NFE * t_warm
                     / (workers * eff) / 3600.0)
            recs.append({
                "nodes": nodes,
                "islands": islands,
                "ranks_per_node": kstar,
                "ranks": ranks,
                "workers": workers,
                "efficiency": eff,
                "eff_extrapolated": bool(workers > slots_meas.max()
                                         or workers < slots_meas.min()),
                "walltime_hr": hours,
                "su": nodes * scfg.SCALING_NODE_CORES * hours,
            })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_packing_eval_time(pack: pd.DataFrame) -> None:
    """F1: warm per-eval wall time vs ranks-per-node, IQR band, cold markers."""
    d = pack[pack["realization_batch"] == 0]
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.fill_between(d["k"], d["warm_q25_s"], d["warm_q75_s"],
                    color=C_BAND, alpha=0.2, label="warm IQR across ranks")
    ax.plot(d["k"], d["warm_median_s"], "o-", color=C_PRIMARY,
            label="warm median")
    ax.plot(d["k"], d["cold_median_s"], "o", mfc="none", color=C_SECONDARY,
            label="cold median (first eval)")
    for i, (_, r) in enumerate(d.iterrows()):
        if np.isfinite(r["slowdown_vs_k1"]):
            # Alternate the offset so labels at closely spaced K don't collide.
            ax.annotate(f"{r['slowdown_vs_k1']:.2f}x",
                        (r["k"], r["warm_median_s"]),
                        textcoords="offset points",
                        xytext=(0, 8 + 10 * (i % 2)),
                        ha="center", fontsize=8, color="0.3")
    ax.set_xlabel("Concurrent evaluator ranks per node (K)")
    ax.set_ylabel("Per-evaluation wall time (s)")
    ax.set_title("Ensemble evaluation time vs node packing density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, scfg.SCALING_FIGURES_DIR / "F1_packing_eval_time")
    plt.close(fig)


def fig_packing_throughput_cost(pack: pd.DataFrame,
                                kstar: "int | None") -> None:
    """F2: node throughput and SU cost vs K (two panels — the decision figure)."""
    d = pack[pack["realization_batch"] == 0]
    batched = pack[pack["realization_batch"] > 0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(d["k"], d["evals_per_node_hr"], "o-", color=C_PRIMARY)
    ax1.set_ylabel("Warm evaluations per node-hour")
    ax1.set_title("Node throughput and SU cost vs packing density")
    ax2.plot(d["k"], d["su_per_1000_evals"], "o-", color=C_SECONDARY)
    if len(batched):
        ax2.plot(batched["k"], batched["su_per_1000_evals"], "s", mfc="none",
                 color=C_SECONDARY,
                 label=f"batched (B={int(batched['realization_batch'].iloc[0])})")
        ax2.legend(frameon=False)
    ax2.set_ylabel("SU per 1,000 evaluations")
    ax2.set_xlabel("Concurrent evaluator ranks per node (K)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        if kstar is not None:
            ax.axvline(kstar, color="0.45", ls="--", lw=1)
    if kstar is not None:
        ax1.annotate(f"K* = {kstar}", (kstar, ax1.get_ylim()[1]),
                     ha="left", va="top", fontsize=9, color="0.3",
                     textcoords="offset points", xytext=(4, -2))
    save_figure(fig, scfg.SCALING_FIGURES_DIR / "F2_packing_throughput_cost")
    plt.close(fig)


def fig_packing_memory(pack: pd.DataFrame) -> None:
    """F3: per-rank peak RSS and projected node total vs the 256 GB limit."""
    d = pack[pack["realization_batch"] == 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    ax1.plot(d["k"], d["rss_median_mb"] / 1024.0, "o-", color=C_PRIMARY,
             label="median rank")
    ax1.plot(d["k"], d["rss_max_mb"] / 1024.0, "o--", mfc="none",
             color=C_PRIMARY, label="max rank")
    ax1.set_xlabel("Concurrent evaluator ranks per node (K)")
    ax1.set_ylabel("Per-rank peak RSS (GB)")
    ax1.set_title("Per-rank memory")
    ax1.legend(frameon=False)
    ax2.plot(d["k"], d["node_projected_gb"], "o-", color=C_PRIMARY,
             label="projected node total (K x max RSS)")
    ax2.axhline(scfg.SCALING_NODE_MEM_GB, color=C_LIMIT, ls="--", lw=1.2,
                label=f"node memory ({scfg.SCALING_NODE_MEM_GB} GB)")
    incomplete = d[~d["completed"]]
    if len(incomplete):
        ax2.plot(incomplete["k"], incomplete["node_projected_gb"], "x",
                 color=C_LIMIT, ms=10, label="incomplete step (OOM/killed)")
    ax2.set_xlabel("Concurrent evaluator ranks per node (K)")
    ax2.set_ylabel("Node memory (GB)")
    ax2.set_title("Projected node memory")
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    save_figure(fig, scfg.SCALING_FIGURES_DIR / "F3_packing_memory")
    plt.close(fig)


def fig_batched_eval(bsum: pd.DataFrame) -> None:
    """F6: batched-evaluation trade — time, memory, and SU cost vs B, per K.

    x is realizations per pywr ``model.run()`` (B; the B=0 all-in-one point
    plots at B=N). One line per measured density K, so the (K, B) interaction
    is visible: whether small-B memory savings survive contention at K*.
    """
    n_real = int(bsum["b_effective"].max())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    for i, (k, d) in enumerate(bsum.groupby("k")):
        d = d.sort_values("b_effective")
        color = C_PRIMARY if i == 0 else C_SECONDARY
        label = f"K={k}"
        ax1.fill_between(d["b_effective"], d["warm_q25_s"], d["warm_q75_s"],
                         color=color, alpha=0.15)
        ax1.plot(d["b_effective"], d["warm_median_s"], "o-", color=color,
                 label=label)
        ax2.plot(d["b_effective"], d["rss_max_mb"] / 1024.0, "o-",
                 color=color, label=label)
        ax3.plot(d["b_effective"], d["su_per_1000_evals"], "o-",
                 color=color, label=label)
    ax1.set_ylabel("Warm per-eval wall time (s)")
    ax1.set_title("Evaluation time")
    ax2.set_ylabel("Per-rank peak RSS (GB)")
    ax2.set_title("Memory")
    ax3.set_ylabel("SU per 1,000 evaluations")
    ax3.set_title("Cost")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Realizations per model.run() (B)")
        ax.grid(alpha=0.25)
        ax.axvline(n_real, color=C_IDEAL, ls=":", lw=1)
        ax.legend(frameon=False, fontsize=8)
        # Mark the production default (all realizations in one run).
        ax.annotate("all", (n_real, ax.get_ylim()[0]), fontsize=7,
                    color="0.4", ha="center", va="bottom")
    save_figure(fig, scfg.SCALING_FIGURES_DIR / "F6_batched_eval")
    plt.close(fig)


def fig_borg_strong_scaling(jobs: pd.DataFrame, borg: pd.DataFrame) -> None:
    """F4: wall time, speedup vs ideal, and parallel efficiency vs eval slots."""
    ok = jobs[(jobs["rc"] == 0) & jobs["config"].isin(borg["config"])]
    single = borg[borg["islands"] == 1]
    multi = borg[borg["islands"] > 1]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)

    for _, r in borg.iterrows():
        seeds = ok[ok["config"] == r["config"]]
        ax1.plot(seeds["total_slots"], seeds["wall_seconds"],
                 ISLAND_MARKERS.get(int(r["islands"]), "o"), mfc="none",
                 color=C_PRIMARY, ms=6)
    ax1.plot(single["total_slots"], single["wall_mean_s"], "-",
             color=C_PRIMARY, label="1-island mean")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("Evaluation slots (islands x workers)")
    ax1.set_ylabel("Wall time to fixed NFE (s)")
    ax1.set_title("Time to solution")
    ax1.legend(frameon=False)

    ax2.plot(single["total_slots"], single["ideal_speedup"], "--",
             color=C_IDEAL, label="ideal (linear)")
    ax2.fill_between(single["total_slots"], single["speedup_min"],
                     single["speedup_max"], color=C_BAND, alpha=0.2,
                     label="seed range")
    ax2.plot(single["total_slots"], single["speedup"], "o-", color=C_PRIMARY,
             label="measured (1 island)")
    for _, r in multi.iterrows():
        ax2.plot(r["total_slots"], r["speedup"],
                 ISLAND_MARKERS.get(int(r["islands"]), "o"), color=C_SECONDARY,
                 ms=7, label=f"{int(r['islands'])}x{int(r['workers_per_island'])}")
    ax2.set_xlabel("Evaluation slots")
    ax2.set_ylabel("Speedup vs smallest geometry")
    ax2.set_title("Strong-scaling speedup")
    ax2.legend(frameon=False, fontsize=8)

    ax3.plot(single["total_slots"], single["efficiency"], "o-",
             color=C_PRIMARY, label="1 island")
    for _, r in multi.iterrows():
        ax3.plot(r["total_slots"], r["efficiency"],
                 ISLAND_MARKERS.get(int(r["islands"]), "o"), color=C_SECONDARY,
                 ms=7, label=f"{int(r['islands'])}x{int(r['workers_per_island'])}")
    ax3.axhline(1.0, color=C_IDEAL, ls="--", lw=1)
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel("Evaluation slots")
    ax3.set_ylabel("Parallel efficiency (speedup / ideal)")
    ax3.set_title("Parallel efficiency")
    ax3.legend(frameon=False, fontsize=8)

    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.25)
    save_figure(fig, scfg.SCALING_FIGURES_DIR / "F4_borg_strong_scaling")
    plt.close(fig)


def fig_production_projection(proj: pd.DataFrame) -> None:
    """F5: nodes x islands heatmap of projected campaign walltime, SU annotated."""
    nodes = sorted(proj["nodes"].unique())
    islands = sorted(proj["islands"].unique())
    grid = np.full((len(islands), len(nodes)), np.nan)
    for _, r in proj.iterrows():
        grid[islands.index(r["islands"]), nodes.index(r["nodes"])] = r["walltime_hr"]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_yticks(range(len(islands)))
    ax.set_yticklabels([str(i) for i in islands])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Islands")
    kstar = int(proj["ranks_per_node"].iloc[0])
    nfe_k = scfg.SCALING_PROJECTION_TOTAL_NFE // 1000
    ax.set_title(f"Projected walltime, {nfe_k}k-NFE campaign "
                 f"at K*={kstar} ranks/node")
    vmax = np.nanmax(grid)
    for _, r in proj.iterrows():
        i, j = islands.index(r["islands"]), nodes.index(r["nodes"])
        flag = "*" if r["eff_extrapolated"] else ""
        ax.text(j, i, f"{r['walltime_hr']:.1f} h{flag}\n{r['su']:,.0f} SU",
                ha="center", va="center", fontsize=8,
                color="white" if r["walltime_hr"] > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, label="Walltime (hours)")
    fig.text(0.99, 0.005,
             "* efficiency extrapolated beyond measured slots; efficiency from "
             "short-window single-trace runs (conservative)",
             ha="right", va="bottom", fontsize=7, color="0.4")
    save_figure(fig, scfg.SCALING_FIGURES_DIR / "F5_production_projection")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Build every table and figure the available data supports."""
    apply_style()
    scfg.SCALING_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scfg.SCALING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows, steps = load_packing()
    pack = None
    kstar = None
    if rows is not None:
        rows.to_csv(scfg.SCALING_TABLES_DIR / "packing_combined.csv",
                    index=False)
        pack = summarize_packing(rows, steps)
        pack.to_csv(scfg.SCALING_TABLES_DIR / "packing_summary.csv",
                    index=False)
        kstar = choose_kstar(pack)
        print(f"[analyze] packing: {len(pack)} (K, batch) points, K*={kstar}")
        fig_packing_eval_time(pack)
        fig_packing_throughput_cost(pack, kstar)
        fig_packing_memory(pack)
        bsum = summarize_batch(rows)
        if bsum is not None:
            bsum.to_csv(scfg.SCALING_TABLES_DIR / "batch_summary.csv",
                        index=False)
            print(f"[analyze] batch sweep: {len(bsum)} (K, B) points "
                  f"at K = {sorted(bsum['k'].unique())}")
            fig_batched_eval(bsum)
        else:
            print("[analyze] no batch-mode shards -- skipping F6")
    else:
        print("[analyze] no packing shards found -- skipping Stage A outputs")

    jobs = load_borg()
    borg = None
    if jobs is not None:
        jobs.to_csv(scfg.SCALING_TABLES_DIR / "borg_jobs.csv", index=False)
        borg = summarize_borg(jobs)
        if len(borg):
            borg.to_csv(scfg.SCALING_TABLES_DIR / "borg_summary.csv",
                        index=False)
            print(f"[analyze] borg: {len(borg)} geometries "
                  f"({borg['n_seeds'].min()}-{borg['n_seeds'].max()} seeds)")
            if len(borg) >= 2:
                fig_borg_strong_scaling(jobs, borg)
            else:
                print("[analyze] <2 geometries -- skipping F4")
        else:
            borg = None
            print("[analyze] no comparable Borg runs -- skipping Stage B outputs")
    else:
        print("[analyze] no Borg timing CSVs found -- skipping Stage B outputs")

    if pack is not None and borg is not None and kstar is not None:
        proj = project_production(pack, borg, kstar)
        proj.to_csv(scfg.SCALING_TABLES_DIR / "projection.csv", index=False)
        fig_production_projection(proj)
        print(f"[analyze] projection: {len(proj)} candidate geometries")
    else:
        print("[analyze] projection needs both stages -- skipped")

    print(f"[analyze] outputs under {scfg.SCALING_OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
