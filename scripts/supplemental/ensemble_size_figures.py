"""ensemble_size_figures.py - Statistics, decision table, and figures for the
ensemble-size diagnostics.

Pure post-processing of the Layer-A tables (``ensemble_size_hazard.py``) and
the Layer-B library (``ensemble_size_library_run.py``); nothing simulates.
Design, pre-registered criteria, and the figure list:
``docs/notes/methods/ensemble_size_diagnostics.md``.

Layer A (from tables):  A1 hf_tail_share_vs_n, A2 hf_coverage_vs_n,
                        A3 np_ladder, A4 ps_tail_sampling, A5 descriptor_convergence
Layer B (from library): B1 level_se_vs_n, B2 paired_se_vs_n, B3 flip_rate_vs_n,
                        B4 optimism_vs_n, B5 tail_replicate_bands,
                        B6 effective_sample_size, B7 epscube_crosscheck,
                        B8 nfe_asymptote (existing runtime archives),
                        B9 cost_pricing
Tables: ``layer_b_stats`` (long), ``decision_table``, ``n_min``, ``n_eff``,
``epscube_crosscheck``, ``nfe_asymptote``, ``cost_pricing``, ``np_tail_share``.

Every figure follows ``src/plotting/style.py``, is PNG only, and carries the
epsilon line (where an epsilon applies) and the campaign point
(``ESD_N_CAMPAIGN``, N = 300).
Settings in ``supplemental_config.py`` (``ESD_*``); no CLI value flags.
Wrapper: ``workflow/supplemental/ensemble_size_analysis.sh``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_esd_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src.ensemble_size_stats import (  # noqa: E402
    bootstrap_sd, compose_objectives, epsilon_relations, flip_rate, level_se,
    majority_relation, n_eff_ratio, optimism, paired_se, supplemented_replicates,
    summarize_over_pairs,
)
from src.objectives_ensemble import ENSEMBLE_OBJECTIVES  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_style, design_color, design_label, save_figure, short_label_for,
)

PS, HF = "fixed_probabilistic", "hazard_filling_stationary"
EPS_COLOR = "#009E73"
CAMPAIGN_COLOR = "0.35"
NULL_COLOR = "0.55"

#: Axis label used on every N axis (linear).
N_LABEL = "ensemble size N (10-yr realizations per evaluation)"

#: Objectives whose operator is a pooled percentile (the tail operators).
def _is_tail(name: str) -> bool:
    return type(ENSEMBLE_OBJECTIVES[name].unit_operator).__name__ == "PooledPercentileOp"


def _axis_label(name: str) -> str:
    try:
        return short_label_for(name)
    except Exception:
        return name


def _mark_campaign(ax, eps: float | None = None, eps_frac: float = 1.0,
                   eps_label: str | None = None, campaign_label: bool = False) -> None:
    """The campaign point (``ESD_N_CAMPAIGN``) and (optionally) the epsilon line."""
    ax.axvline(scfg.ESD_N_CAMPAIGN, color=CAMPAIGN_COLOR, lw=1.0, ls=":",
               label=f"campaign N = {scfg.ESD_N_CAMPAIGN}" if campaign_label else None)
    if eps is not None:
        ax.axhline(eps * eps_frac, color=EPS_COLOR, lw=1.4, ls="--",
                   label=eps_label or (r"$\varepsilon$" if eps_frac == 1.0
                                       else rf"{eps_frac:g}$\varepsilon$"))


def _table(name: str) -> pd.DataFrame | None:
    path = scfg.esd_table_path(name)
    if not path.exists():
        print(f"[esd:fig] table missing: {path}")
        return None
    return pd.read_csv(path)


###############################################################################
# Layer A figures
###############################################################################

def fig_a1_tail_share(ladder: pd.DataFrame) -> None:
    """A1: per-axis P90 and P99 tail shares of the HF selection vs N, with the null."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for p, (col, ttl, ref, q) in enumerate((
        ("tail_share_min", "share of members above the pool's 90th percentile\n(minimum over the 6 selection axes)", 0.10, "P90"),
        ("tail_share_p99_min", "share of members above the pool's 99th percentile\n(minimum over the 6 selection axes)", 0.01, "P99"),
    )):
        a = ax[p]
        for draw, g in ladder.groupby("pool_draw"):
            sel = g[g.selector == "lhs_nn"].groupby("n")[col]
            lab = None
            if draw == 0:
                lab = "hazard-filling selection, pool d0: mean over 10 anchor plans (bars = range)"
            elif draw == 1:
                lab = "hazard-filling selection, pools d1 and d2 (lighter)"
            a.errorbar(sel.mean().index, sel.mean().values,
                       yerr=[sel.mean().values - sel.min().values, sel.max().values - sel.mean().values],
                       fmt="o-", ms=4, capsize=2, color=design_color(HF),
                       alpha=1.0 if draw == 0 else 0.45, label=lab)
        null = ladder[(ladder.selector == "random") & (ladder.pool_draw == 0)].groupby("n")[col]
        a.plot(null.mean().index, null.mean().values, "s--", ms=3, color=NULL_COLOR,
               label="random selection of the same N (mean over 50 seeds)")
        a.axhline(ref, color="0.6", lw=1, ls="--",
                  label=f"expected share for a random sample ({ref:g})")
        _mark_campaign(a, campaign_label=True)
        a.set_xlabel(N_LABEL)
        a.set_ylabel(ttl, fontsize=8.5)
        a.set_ylim(bottom=0)
        a.set_title(f"({'ab'[p]}) tail enrichment above the pool {q}")
        a.legend(fontsize=6.5, loc="center right" if p == 0 else "upper right")
    fig.suptitle("How much of the hazard-filling selection sits in the severe tail of the 10⁶-member candidate pool, as N grows")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, scfg.esd_figure_path("A1_hf_tail_share_vs_n"))
    plt.close(fig)


def fig_a2_coverage(ladder: pd.DataFrame) -> None:
    """A2: joint geometry of the HF selection vs the matched random design, vs N."""
    metrics = (("L2_star_abs", "(a) joint L2-star discrepancy",
                "L2-star discrepancy in the unit hazard box\n(lower = more uniform coverage, log)"),
               ("mst_edge_mean", "(b) mean minimum-spanning-tree edge",
                "mean MST edge length in the unit hazard box\n(larger = members more spread out, log)"),
               ("nn_min_abs", "(c) closest pair of members",
                "smallest distance between any two members\n(near 0 = near-duplicates, log)"),
               ("ks_mean", "(d) per-axis stratification",
                "Kolmogorov–Smirnov distance to uniform,\nmean over the 6 axes (lower = better stratified)"))
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    d0 = ladder[ladder.pool_draw == 0]
    for p, (col, ttl, ylab) in enumerate(metrics):
        a = ax[p]
        sel = d0[d0.selector == "lhs_nn"].groupby("n")[col]
        null = d0[d0.selector == "random"].groupby("n")[col]
        a.plot(sel.mean().index, sel.mean().values, "o-", ms=4, color=design_color(HF),
               label="hazard-filling selection: mean, band = range over 10 anchor plans")
        a.fill_between(sel.mean().index, sel.min().values, sel.max().values,
                       color=design_color(HF), alpha=0.2, lw=0)
        a.plot(null.mean().index, null.mean().values, "s--", ms=3, color=NULL_COLOR,
               label="random selection of the same N: mean, band = 5–95 % over 50 seeds")
        a.fill_between(null.mean().index, null.quantile(0.05).values, null.quantile(0.95).values,
                       color=NULL_COLOR, alpha=0.15, lw=0)
        _mark_campaign(a, campaign_label=(p == 0))
        if col != "ks_mean":
            a.set_yscale("log")
        else:
            a.set_ylim(bottom=0)
        a.set_xlabel(N_LABEL if p == 0 else "N")
        a.set_ylabel(ylab, fontsize=8)
        a.set_title(ttl, fontsize=9.5)
        if p == 0:
            a.legend(fontsize=6, loc="center left")
    fig.suptitle("Joint coverage geometry of the hazard-filling selection vs a random selection of the same size (pool d0)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, scfg.esd_figure_path("A2_hf_coverage_vs_n"))
    plt.close(fig)


def fig_a3_np_ladder(npl: pd.DataFrame) -> pd.DataFrame:
    """A3: tail enrichment over the (N, P′) plane; returns the seed-mean table.

    Panel (a) is the min per-axis P90 tail share against N, one line per prefix
    size; panel (b) is the same statistic against P′ (log), one line per N, so
    the saturation with pool size and its independence of N are read directly.
    """
    g = npl[npl.selector == "lhs_nn"].groupby(["P", "n"])["tail_share_min"]
    tab = pd.DataFrame({"tail_share_min_seed_mean": g.mean(),
                        "tail_share_min_seed_sd": g.std(ddof=1)}).reset_index()
    null_label = "expected share for a random sample (0.10)"
    ylab = ("share of members above the pool's 90th percentile\n"
            "(minimum over the 6 selection axes, mean over anchor plans)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    a = ax[0]
    Ps = sorted(tab.P.unique())
    for i, P in enumerate(Ps):
        t = tab[tab.P == P]
        a.plot(t.n, t.tail_share_min_seed_mean, "o-", ms=4,
               color=cmap(0.15 + 0.7 * i / max(1, len(Ps) - 1)), label=f"P′ = {P:,}")
    a.axhline(0.10, color="0.6", lw=1, ls="--", label=null_label)
    _mark_campaign(a, campaign_label=True)
    a.set_xlabel(N_LABEL)
    a.set_ylabel(ylab, fontsize=8.5)
    a.set_ylim(bottom=0)
    a.set_title("(a) tail enrichment vs N, one line per prefix size P′")
    a.legend(fontsize=6.5, loc="lower right")
    a2 = ax[1]
    Ns = sorted(tab.n.unique())
    for i, n in enumerate(Ns):
        t = tab[tab.n == n].sort_values("P")
        a2.errorbar(t.P, t.tail_share_min_seed_mean, yerr=t.tail_share_min_seed_sd,
                    fmt="o-", ms=3, capsize=2, elinewidth=0.7,
                    color=cmap(0.15 + 0.7 * i / max(1, len(Ns) - 1)), label=f"N = {n}")
    a2.axhline(0.10, color="0.6", lw=1, ls="--", label=null_label)
    a2.plot([], [], " ", label="bars = SD over 10 anchor plans")
    a2.set_xscale("log")
    a2.set_xlabel("pool size P′ (prefix of pool d0; log)")
    a2.set_ylabel(ylab, fontsize=8.5)
    a2.set_ylim(bottom=0)
    a2.set_title("(b) saturation with pool size, one line per N")
    a2.legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.suptitle("Tail enrichment of the hazard-filling selection over the (N, pool size) plane — prefixes of pool d0 are exact i.i.d. pools of their size")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, scfg.esd_figure_path("A3_np_ladder"))
    plt.close(fig)
    return tab


def fig_a4_ps_tail(ps: pd.DataFrame) -> None:
    """A4: the i.i.d. law of a size-N subset: tail counts, quantile error, closed form."""
    axes_names = list(dict.fromkeys(ps.axis))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    a = ax[0]
    for k, axis in enumerate(axes_names):
        g = ps[ps.axis == axis].sort_values("n")
        a.errorbar(g.n + 4.0 * (k - len(axes_names) / 2), g.count_p99_mean,
                   yerr=[g.count_p99_mean - g.count_p99_p05, g.count_p99_p95 - g.count_p99_mean],
                   fmt="o-", ms=3, lw=1, capsize=2, elinewidth=0.7, alpha=0.85,
                   label=_axis_label(axis))
    a.axhline(1.0, color="0.6", lw=1, ls="--", label="one member")
    _mark_campaign(a, campaign_label=True)
    a.set_xlabel(N_LABEL)
    a.set_ylabel("members above the pool's 99th percentile, per axis\n(mean and 5–95 % range over 200 random size-N subsets)", fontsize=8)
    a.set_ylim(bottom=0)
    a.set_title("(a) how many severe members a random sample of N holds")
    a.legend(fontsize=6, ncol=2, title="hazard axis", title_fontsize=6.5, loc="upper left")
    a2 = ax[1]
    g0 = ps[ps.axis == axes_names[0]].sort_values("n")
    a2.plot(g0.n, g0.prob_ge1_p99_closed_form, "-", color="black",
            label=r"beyond the pool's P99: closed form $1-0.99^N$")
    a2.plot(g0.n, 1.0 - g0.prob_zero_p99_empirical, "o", ms=4, color=design_color(PS),
            label=f"beyond P99: empirical over 200 subsets ({_axis_label(axes_names[0])})")
    a2.plot(g0.n, g0.prob_ge1_p90_closed_form, "-", color="0.5",
            label=r"beyond the pool's P90: closed form $1-0.90^N$")
    _mark_campaign(a2, campaign_label=True)
    a2.set_xlabel(N_LABEL)
    a2.set_ylabel("probability that a random size-N sample holds\nat least one member beyond the pool quantile", fontsize=8)
    a2.set_ylim(0, 1.02)
    a2.set_title("(b) chance of holding at least one severe member")
    a2.legend(fontsize=6.5, loc="lower right")
    a3 = ax[2]
    for q, ls in ((50, "-"), (90, "--"), (99, ":")):
        col = f"relerr_p{q}_rms"
        m = ps.groupby("n")[col].mean()
        a3.plot(m.index, m.values, ls, marker="o", ms=3, color=design_color(PS),
                label=f"sample P{q} vs pool P{q} (mean over the 6 axes)")
    _mark_campaign(a3, campaign_label=True)
    a3.set_xlabel(N_LABEL)
    a3.set_ylabel("RMS relative error of the sample's quantile\nagainst the pool's (200 subsets)", fontsize=8)
    a3.set_ylim(bottom=0)
    a3.set_title("(c) how far a sample's quantiles sit from the pool's")
    a3.legend(fontsize=6.5)
    fig.suptitle("What an i.i.d. sample of N pool members contains, by construction (the fixed-probabilistic design), pool d0")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, scfg.esd_figure_path("A4_ps_tail_sampling"))
    plt.close(fig)


def fig_a5_descriptor_convergence(cv: pd.DataFrame) -> None:
    """A5: pooled-mean vs ensemble-max descriptors vs N (PS bands, HF plans)."""
    axes_names = list(dict.fromkeys(cv.axis))
    fig, ax = plt.subplots(2, len(axes_names), figsize=(2.6 * len(axes_names), 6.2),
                           sharex=True)
    for k, axis in enumerate(axes_names):
        for r, stat in enumerate(("pooled_mean", "ensemble_max")):
            a = ax[r, k]
            for design in (PS, HF):
                g = cv[(cv.axis == axis) & (cv.statistic == stat) & (cv.design == design)].sort_values("n")
                if g.empty:
                    continue
                lab = None
                if k == 0 and r == 0:
                    lab = ("fixed-probabilistic (i.i.d.): median, band = 5–95 % over 200 subsets"
                           if design == PS else "hazard-filling: median, band = range over 10 anchor plans")
                a.plot(g.n, g.p50, "o-", ms=3, color=design_color(design), label=lab)
                a.fill_between(g.n, g.p05, g.p95, color=design_color(design), alpha=0.18, lw=0)
            pool_val = cv[(cv.axis == axis) & (cv.statistic == stat)].pool_value.iloc[0]
            a.axhline(pool_val, color="black", lw=0.9, ls="--",
                      label="value over the full 10⁶-member pool" if k == 0 and r == 0 else None)
            _mark_campaign(a, campaign_label=(k == 0 and r == 0))
            if r == 0:
                a.set_title(_axis_label(axis), fontsize=8)
            if k == 0:
                a.set_ylabel({"pooled_mean": "mean of the descriptor\nover the ensemble's members",
                              "ensemble_max": "maximum of the descriptor\nover the ensemble's members"}[stat], fontsize=8)
            if r == 1:
                a.set_xlabel(N_LABEL if k == 0 else "N", fontsize=8)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle("Hazard descriptors of the ensemble vs N: means converge to the pool value, maxima keep drifting")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(fig, scfg.esd_figure_path("A5_descriptor_convergence"))
    plt.close(fig)


###############################################################################
# Layer B: library loading and replicate composition
###############################################################################

class Library:
    """The merged per-realization annual-unit library plus its replicate index."""

    def __init__(self, path: Path, plan: dict):
        with h5py.File(path, "r") as f:
            self.units = f["units"][:]                             # (P, R, M, U)
            self.policy_ids = [s.decode() for s in f["policy_ids"][:]]
            self.policy_labels = [s.decode() for s in f["policy_labels"][:]]
            self.obj_names = [s.decode() for s in f["objective_names"][:]]
            self.active_names = [s.decode() for s in f["active_objective_names"][:]]
            self.real_kind = np.array([s.decode() for s in f["real_kind"][:]])
            self.real_design = np.array([s.decode() for s in f["real_design"][:]])
            self.real_draw = f["real_draw"][:]
            self.real_gid = f["real_global_id"][:]
            self.p_ref = int(f.attrs["p_ref"])
        self.plan = plan
        self.active_idx = [self.obj_names.index(a) for a in self.active_names]
        self.objs = [ENSEMBLE_OBJECTIVES[a] for a in self.active_names]
        self.operators = [o.unit_operator for o in self.objs]
        self.directions = [o.direction for o in self.objs]
        self.epsilons = np.array([o.epsilon for o in self.objs])
        self.units_active = self.units[:, :, self.active_idx, :]
        pool = self.real_kind == "pool"
        self.pool_row = {int(g): i for i, g in zip(np.flatnonzero(pool), self.real_gid[pool])}
        missing = [g for g in range(self.p_ref) if g not in self.pool_row]
        if missing:
            raise RuntimeError(f"reference prefix rows missing from the library: {missing[:10]}")
        self.ref_rows = np.array([self.pool_row[g] for g in range(self.p_ref)])

    @property
    def n_policy(self) -> int:
        return len(self.policy_ids)

    def compose(self, rows) -> np.ndarray:
        return compose_objectives(self.units_active, rows, self.operators)

    def replicates(self, design: str, n: int) -> tuple[list[np.ndarray], list[str]]:
        """Member rows and a replicate label per replicate of (design, n)."""
        reps, labels = [], []
        if design == PS:
            blocks, flags = supplemented_replicates(self.p_ref, n, scfg.ESD_PS_MIN_REPLICATES,
                                                    scfg.ESD_REPLICATE_SEED)
            for b, fl in zip(blocks, flags):
                reps.append(self.ref_rows[b])
                labels.append("random_overlapping" if fl else "disjoint_prefix")
            if n == scfg.ESD_N_CAMPAIGN:
                for d in sorted(set(self.real_draw[self.real_design == PS])):
                    rows = np.flatnonzero((self.real_design == PS) & (self.real_draw == d))
                    if len(rows) == n:
                        reps.append(rows)
                        labels.append(f"staged_draw{d}")
        elif design == HF:
            members = self.plan["hf_members"].get(str(n), {})
            for ad, gids in members.items():
                reps.append(np.array([self.pool_row[int(g)] for g in gids]))
                labels.append(f"anchor_plan{ad}")
            if n == scfg.ESD_N_CAMPAIGN:
                for d in sorted(set(self.real_draw[self.real_design == HF])):
                    if d == self.plan["pool_draw"]:
                        continue  # identical members to anchor plan 0
                    rows = np.flatnonzero((self.real_design == HF) & (self.real_draw == d))
                    if len(rows) == n:
                        reps.append(rows)
                        labels.append(f"staged_draw{d}")
        return reps, labels


def layer_b_statistics(lib: Library) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All per-(design, N, objective) statistics, the tail bands, and the raw values."""
    ref_values = lib.compose(lib.ref_rows)                        # (P, M) PS reference
    ref_codes = epsilon_relations(ref_values, lib.epsilons, lib.directions)
    rows, band_rows, values_store = [], [], {"ref": ref_values}
    rng = np.random.default_rng(scfg.ESD_BOOTSTRAP_SEED)
    tail_k = [k for k, nm in enumerate(lib.active_names) if _is_tail(nm)]
    for design in (PS, HF):
        for n in scfg.ESD_N_LADDER:
            reps, labels = lib.replicates(design, n)
            if len(reps) < 2:
                print(f"[esd:B] {design} N={n}: {len(reps)} replicate(s); skipped")
                continue
            values = np.stack([lib.compose(r) for r in reps])     # (R, P, M)
            values_store[(design, n)] = {"values": values, "labels": labels}
            lse = level_se(values)                                  # (P, M)
            pse = summarize_over_pairs(paired_se(values))           # dict (M,)
            codes = np.stack([epsilon_relations(v, lib.epsilons, lib.directions) for v in values])
            reference = ref_codes if design == PS else majority_relation(codes)
            fr = flip_rate(codes, reference)
            gap = optimism(values, ref_values, lib.directions)      # (R, P, M)
            gap_mean, gap_sd = gap.mean(axis=0), gap.std(axis=0, ddof=1)
            # Realization-level bootstrap SD of the tail operators inside the
            # first few replicates (the honest block bootstrap, never sigma/sqrt N).
            boot = np.full((lib.n_policy, len(lib.active_names)), np.nan)
            for k in tail_k:
                vals = []
                for r in reps[:min(len(reps), 5)]:
                    for p in range(lib.n_policy):
                        vals.append(bootstrap_sd(lib.units_active[p, r, k, :], lib.operators[k],
                                                 scfg.ESD_BOOTSTRAP_B, rng))
                boot[:, k] = np.array(vals).reshape(-1, lib.n_policy).mean(axis=0)
            n_disjoint = sum(1 for l in labels if l == "disjoint_prefix")
            for k, name in enumerate(lib.active_names):
                rows.append({
                    "design": design, "n": n, "objective": name, "epsilon": lib.epsilons[k],
                    "n_replicates": len(reps), "n_disjoint": n_disjoint,
                    "n_supplemented": sum(1 for l in labels if l == "random_overlapping"),
                    "n_staged": sum(1 for l in labels if l.startswith("staged")),
                    "level_se_max": float(np.nanmax(lse[:, k])),
                    "level_se_median": float(np.nanmedian(lse[:, k])),
                    "paired_se_max": float(pse["max"][k]),
                    "paired_se_p90": float(pse["p90"][k]),
                    "paired_se_median": float(pse["median"][k]),
                    "flip_rate": fr,
                    "optimism_mean_maxabs": float(np.nanmax(np.abs(gap_mean[:, k]))),
                    "optimism_mean_median": float(np.nanmedian(gap_mean[:, k])),
                    "shift_sd_max": float(np.nanmax(gap_sd[:, k])),
                    "bootstrap_sd_realization_median": float(np.nanmedian(boot[:, k]))
                    if k in tail_k else np.nan,
                })
                for p in range(lib.n_policy):
                    v = values[:, p, k]
                    band_rows.append({
                        "design": design, "n": n, "objective": name, "policy_id": lib.policy_ids[p],
                        "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                        "p95": float(np.percentile(v, 95)), "sd": float(lse[p, k]),
                        "reference": float(ref_values[p, k]),
                        "optimism_mean": float(gap_mean[p, k]), "shift_sd": float(gap_sd[p, k]),
                        "bootstrap_sd_realization": float(boot[p, k]),
                    })
            print(f"[esd:B] {design} N={n}: {len(reps)} replicates "
                  f"(flip rate {fr:.3f})", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(band_rows), values_store


def decision_table(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the pre-registered thresholds; return (per-row table, N_min table)."""
    t = stats.copy()
    t["pass_level_se"] = t.level_se_max <= scfg.ESD_LEVEL_SE_EPS_FRAC * t.epsilon
    t["pass_paired_se"] = t.paired_se_max <= scfg.ESD_PAIRED_SE_EPS_FRAC * t.epsilon
    t["pass_flip_rate"] = t.flip_rate <= scfg.ESD_FLIP_RATE_MAX
    t["pass_bias"] = np.where(
        t.design == PS,
        t.optimism_mean_maxabs <= scfg.ESD_OPTIMISM_EPS_FRAC * t.epsilon,
        t.shift_sd_max <= scfg.ESD_LEVEL_SE_EPS_FRAC * t.epsilon)
    t["pass_all"] = t.pass_level_se & t.pass_paired_se & t.pass_flip_rate & t.pass_bias
    rows = []
    for design, g in t.groupby("design"):
        by_n = g.groupby("n")
        passing = [n for n, gg in by_n if bool(gg.pass_all.all())]
        n_min = min(passing) if passing else None
        n_noise_pass = [n for n, gg in by_n if bool(gg.pass_paired_se.all())]
        n_noise = min(n_noise_pass) if n_noise_pass else None
        largest = max(g.n)
        last = g[g.n == largest]
        binding = last[~last.pass_all]
        rows.append({
            "design": design, "n_min": n_min, "n_noise": n_noise,
            "largest_n_tested": largest,
            "binding_objectives_at_largest_n": ";".join(sorted(binding.objective.unique())),
            "binding_criteria_at_largest_n": ";".join(
                c for c in ("pass_level_se", "pass_paired_se", "pass_flip_rate", "pass_bias")
                if not bool(last[c].all())),
        })
    nmin = pd.DataFrame(rows)
    vals = [v for v in nmin.n_min if v is not None and np.isfinite(v)]
    # N_common is defined only when EVERY design passes somewhere on the ladder.
    n_common = int(max(vals)) if len(vals) == len(nmin) else None
    nmin["n_common"] = n_common if n_common is not None else np.nan
    return t, nmin


def n_eff_table(lib: Library) -> pd.DataFrame:
    """Realization-level vs naive unit-level bootstrap at the campaign point."""
    rng = np.random.default_rng(scfg.ESD_BOOTSTRAP_SEED + 1)
    rows = []
    for design in (PS, HF):
        reps, labels = lib.replicates(design, scfg.ESD_N_CAMPAIGN)
        for r_i, (r, lab) in enumerate(zip(reps[:5], labels[:5])):
            for p in range(lib.n_policy):
                for k, name in enumerate(lib.active_names):
                    u = lib.units_active[p, r, k, :]
                    sd_r = bootstrap_sd(u, lib.operators[k], scfg.ESD_BOOTSTRAP_B, rng, "realization")
                    sd_u = bootstrap_sd(u, lib.operators[k], scfg.ESD_BOOTSTRAP_B, rng, "unit")
                    rows.append({"design": design, "replicate": lab, "policy_id": lib.policy_ids[p],
                                 "objective": name, "sd_realization": sd_r, "sd_unit": sd_u,
                                 "n_eff_ratio": n_eff_ratio(sd_u, sd_r),
                                 "n_units": int(u.size)})
    return pd.DataFrame(rows)


###############################################################################
# Layer B figures
###############################################################################

def _grid(n_obj: int):
    ncol = 4
    nrow = int(np.ceil(n_obj / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow))
    return fig, np.atleast_2d(ax).reshape(-1)


def fig_b_stat_vs_n(stats: pd.DataFrame, col: str, ylabel: str, stem: str, eps_frac: float,
                    aux_col: str | None = None, aux_label: str | None = None,
                    log_y: bool = True, main_label: str = "", suptitle: str = "") -> None:
    """One panel per objective: a statistic vs N for both designs, with its epsilon criterion."""
    objs = list(dict.fromkeys(stats.objective))
    fig, ax = _grid(len(objs))
    for k, name in enumerate(objs):
        a = ax[k]
        eps = float(stats[stats.objective == name].epsilon.iloc[0])
        for design in (PS, HF):
            g = stats[(stats.objective == name) & (stats.design == design)].sort_values("n")
            if g.empty:
                continue
            a.plot(g.n, g[col], "o-", ms=4, color=design_color(design),
                   label=f"{design_label(design)}: {main_label}")
            if aux_col:
                a.plot(g.n, g[aux_col], "o:", ms=3, color=design_color(design), alpha=0.6,
                       label=f"{design_label(design)}: {aux_label}")
        a.axhline(eps * eps_frac, color=EPS_COLOR, lw=1.4, ls="--",
                  label=("ε, the archive's precision on this objective" if eps_frac == 1.0
                         else f"{eps_frac:g} ε, the pre-registered criterion"))
        a.axvline(scfg.ESD_N_CAMPAIGN, color=CAMPAIGN_COLOR, lw=1.0, ls=":",
                  label=f"campaign N = {scfg.ESD_N_CAMPAIGN}")
        if log_y:
            a.set_yscale("log")
            lo, hi = a.get_ylim()
            a.set_ylim(bottom=max(lo, 1e-3 * eps), top=max(hi, 2.0 * eps * eps_frac))
        a.set_title(f"({'abcdefgh'[k]}) {_axis_label(name)}", fontsize=9)
        a.set_xlabel(N_LABEL if k % 4 == 0 else "N")
        if k % 4 == 0:
            a.set_ylabel(ylabel, fontsize=8)
    for j in range(len(objs), len(ax)):
        ax[j].axis("off")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    save_figure(fig, scfg.esd_figure_path(stem))
    plt.close(fig)


def fig_b3_flip(stats: pd.DataFrame) -> None:
    """Flip rate of pairwise epsilon-dominance verdicts vs N."""
    fig, a = plt.subplots(figsize=(7.0, 4.4))
    for design in (PS, HF):
        g = stats[stats.design == design].groupby("n")["flip_rate"].first()
        a.plot(g.index, g.values, "o-", ms=4, color=design_color(design),
               label=(f"{design_label(design)}: verdict at N vs the 5,000-member reference" if design == PS
                      else f"{design_label(design)}: verdict at N vs the majority of its 3–5 constructions"))
    a.axhline(scfg.ESD_FLIP_RATE_MAX, color="#c1272d", lw=1.2, ls=":",
              label=f"pre-registered criterion (≤ {scfg.ESD_FLIP_RATE_MAX:g})")
    a.axvline(scfg.ESD_N_CAMPAIGN, color=CAMPAIGN_COLOR, lw=1.0, ls=":",
              label=f"campaign N = {scfg.ESD_N_CAMPAIGN}")
    a.set_ylim(bottom=0)
    a.set_xlabel(N_LABEL)
    a.set_ylabel("fraction of the 45 policy pairs whose ε-dominance verdict\n(A beats B / B beats A / incomparable) differs from the reference", fontsize=8)
    a.set_title("How often a size-N ensemble reverses an ε-dominance verdict between two policies", fontsize=10)
    a.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save_figure(fig, scfg.esd_figure_path("B3_flip_rate_vs_n"))
    plt.close(fig)


def fig_b4_optimism(bands: pd.DataFrame, stats: pd.DataFrame) -> None:
    """PS estimator bias vs N per operator; HF construction shift ± SD beside it."""
    objs = list(dict.fromkeys(stats.objective))
    fig, ax = _grid(len(objs))
    for k, name in enumerate(objs):
        a = ax[k]
        eps = float(stats[stats.objective == name].epsilon.iloc[0])
        for design in (PS, HF):
            g = bands[(bands.objective == name) & (bands.design == design)]
            m = g.groupby("n")["optimism_mean"].median()
            lo = g.groupby("n")["optimism_mean"].min()
            hi = g.groupby("n")["optimism_mean"].max()
            a.plot(m.index, m.values, "o-", ms=4, color=design_color(design),
                   label=(f"{design_label(design)}: mean over replicates, median policy (band = range over 10 policies)" if k == 0 else None))
            a.fill_between(m.index, lo.values, hi.values, color=design_color(design), alpha=0.18, lw=0)
            if design == HF:
                sd = g.groupby("n")["shift_sd"].max()
                a.plot(sd.index, sd.values, "s:", ms=3, color=design_color(design), alpha=0.7,
                       label="hazard-filling: SD across constructions, worst policy" if k == 0 else None)
        a.axhline(0.0, color="black", lw=0.8)
        a.axhline(scfg.ESD_OPTIMISM_EPS_FRAC * eps, color=EPS_COLOR, lw=1.2, ls="--",
                  label=f"± {scfg.ESD_OPTIMISM_EPS_FRAC:g} ε, the pre-registered bias criterion" if k == 0 else None)
        a.axhline(-scfg.ESD_OPTIMISM_EPS_FRAC * eps, color=EPS_COLOR, lw=1.2, ls="--")
        a.axvline(scfg.ESD_N_CAMPAIGN, color=CAMPAIGN_COLOR, lw=1.0, ls=":",
                  label=f"campaign N = {scfg.ESD_N_CAMPAIGN}" if k == 0 else None)
        a.set_title(f"({'abcdefgh'[k]}) {_axis_label(name)}", fontsize=9)
        a.set_xlabel(N_LABEL if k % 4 == 0 else "N")
        if k % 4 == 0:
            a.set_ylabel("objective at N minus its value on the\n5,000-member i.i.d. reference (+ = looks better)", fontsize=8)
    for j in range(len(objs), len(ax)):
        ax[j].axis("off")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Bias of the i.i.d. estimate (PS) and the intended shift of the hazard-filling design (HF), both vs the i.i.d. reference")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    save_figure(fig, scfg.esd_figure_path("B4_optimism_vs_n"))
    plt.close(fig)


def fig_b5_tail_bands(bands: pd.DataFrame, lib: Library) -> None:
    tail = [nm for nm in lib.active_names if _is_tail(nm)]
    if not tail:
        return
    fig, ax = plt.subplots(2, len(tail), figsize=(3.6 * len(tail), 6.4), sharex=True)
    ax = np.atleast_2d(ax)
    cmap = plt.get_cmap("tab10")
    for r, design in enumerate((PS, HF)):
        for c, name in enumerate(tail):
            a = ax[r, c]
            eps = float(lib.epsilons[lib.active_names.index(name)])
            for p, pid in enumerate(lib.policy_ids):
                g = bands[(bands.design == design) & (bands.objective == name) & (bands.policy_id == pid)].sort_values("n")
                if g.empty:
                    continue
                a.plot(g.n, g.p50, "-", color=cmap(p % 10), lw=1.1,
                       label=(f"{pid} {lib.policy_labels[p].replace('_', ' ')[:34]}" if r == 0 and c == 0 else None))
                a.fill_between(g.n, g.p05, g.p95, color=cmap(p % 10), alpha=0.13, lw=0)
            ref = bands[(bands.design == design) & (bands.objective == name)].reference
            a.axvline(scfg.ESD_N_CAMPAIGN, color=CAMPAIGN_COLOR, lw=1.0, ls=":",
                      label=f"campaign N = {scfg.ESD_N_CAMPAIGN}" if r == 0 and c == 0 else None)
            y0 = float(np.nanmedian(bands[(bands.objective == name)].p50))
            a.plot([scfg.ESD_N_LADDER[0]] * 2, [y0, y0 + eps], "-", color=EPS_COLOR, lw=3,
                   solid_capstyle="butt",
                   label="bar height = ε, the archive's precision" if r == 0 and c == 0 else None)
            if r == 0:
                a.set_title(_axis_label(name), fontsize=9)
            if c == 0:
                a.set_ylabel(f"{design_label(design)}\nobjective value: median line, band = 5–95 %\nacross replicate ensembles", fontsize=7.5)
            if r == 1:
                a.set_xlabel(N_LABEL if c == 0 else "N")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False,
               title="one line per policy (median across replicate ensembles; band = 5–95 %)", title_fontsize=7.5)
    fig.suptitle("The three pooled-percentile (tail) objectives, policy by policy: where each policy's value sits and how wide its spread is at each N")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    save_figure(fig, scfg.esd_figure_path("B5_tail_replicate_bands"))
    plt.close(fig)


def fig_b6_neff(neff: pd.DataFrame) -> None:
    fig, a = plt.subplots(figsize=(7.5, 3.8))
    objs = list(dict.fromkeys(neff.objective))
    x = np.arange(len(objs))
    for i, design in enumerate((PS, HF)):
        g = neff[neff.design == design].groupby("objective")["n_eff_ratio"]
        med = np.array([g.median().get(o, np.nan) for o in objs])
        lo = np.array([g.quantile(0.1).get(o, np.nan) for o in objs])
        hi = np.array([g.quantile(0.9).get(o, np.nan) for o in objs])
        a.errorbar(x + (i - 0.5) * 0.18, med, yerr=[med - lo, hi - med], fmt="o", ms=5, capsize=3,
                   color=design_color(design), label=f"{design_label(design)} (median, 10–90%)")
    a.axhline(1.0, color="0.5", lw=1, ls="--", label="1 = every annual unit counts as independent")
    a.set_xticks(x)
    a.set_xticklabels([_axis_label(o) for o in objs], rotation=35, ha="right", fontsize=7)
    a.set_ylabel("effective sample size ÷ pooled annual units N(L−1)\n= (unit-level bootstrap SD / realization-level bootstrap SD)²", fontsize=8)
    a.set_title(f"How many of the {scfg.ESD_N_CAMPAIGN * 9} pooled annual units act as independent samples (N = {scfg.ESD_N_CAMPAIGN})")
    a.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, scfg.esd_figure_path("B6_effective_sample_size"))
    plt.close(fig)


###############################################################################
# Epsilon-cube cross-check
###############################################################################

def epscube_crosscheck(lib: Library) -> pd.DataFrame | None:
    rows = []
    for design in (PS, HF):
        path = scfg.epsilon_cube_path(design)
        if not path.exists():
            print(f"[esd:B7] epsilon cube missing for {design}: {path}")
            continue
        with h5py.File(path, "r") as f:
            units = f["units"][:]                                   # (D, 100, 10, 9)
            names = [s.decode() for s in f["objective_names"][:]]
            sids = f["sample_ids"][:]
        feas = sids >= 0
        units = units[feas][:, :, [names.index(a) for a in lib.active_names], :]
        n_real = units.shape[1]
        rng = np.random.default_rng(scfg.ESD_BOOTSTRAP_SEED + 2)
        for n in scfg.ESD_EPSCUBE_N:
            vals = np.stack([
                compose_objectives(units, np.sort(rng.choice(n_real, size=n, replace=False)),
                                   lib.operators)
                for _ in range(scfg.ESD_EPSCUBE_SUBSETS)])          # (S, D, M)
            se = np.std(vals, axis=0, ddof=1)                        # (D, M)
            for k, name in enumerate(lib.active_names):
                rows.append({"design": design, "n": n, "objective": name,
                             "level_se_median": float(np.nanmedian(se[:, k])),
                             "level_se_p90": float(np.nanpercentile(se[:, k], 90)),
                             "n_policies": int(units.shape[0]), "source": "epsilon_cube"})
    return pd.DataFrame(rows) if rows else None


def fig_b7_crosscheck(cross: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Library level SE (policy median) vs the same statistic on the epsilon-calibration cubes."""
    objs = list(dict.fromkeys(stats.objective))
    fig, ax = _grid(len(objs))
    for k, name in enumerate(objs):
        a = ax[k]
        eps = float(stats[stats.objective == name].epsilon.iloc[0])
        for design in (PS, HF):
            g = stats[(stats.objective == name) & (stats.design == design)].sort_values("n")
            a.plot(g.n, g.level_se_median, "o-", ms=4, color=design_color(design),
                   label=f"{design_label(design)}: this library (10 rule-selected policies)")
            c = cross[(cross.objective == name) & (cross.design == design)].sort_values("n")
            a.plot(c.n, c.level_se_median, "^--", ms=5, color=design_color(design), alpha=0.7,
                   label=f"{design_label(design)}: ε-calibration cube (512 random policies), subsampled")
        a.axhline(eps, color=EPS_COLOR, lw=1.4, ls="--", label="ε, the archive's precision")
        a.axvline(scfg.ESD_N_CAMPAIGN, color=CAMPAIGN_COLOR, lw=1.0, ls=":",
                  label=f"campaign N = {scfg.ESD_N_CAMPAIGN}")
        a.set_yscale("log")
        lo, hi = a.get_ylim()
        a.set_ylim(bottom=max(lo, 1e-3 * eps))
        a.set_title(f"({'abcdefgh'[k]}) {_axis_label(name)}", fontsize=9)
        a.set_xlabel(N_LABEL if k % 4 == 0 else "N")
        if k % 4 == 0:
            a.set_ylabel("level SE, median over policies (log)", fontsize=8)
    for j in range(len(objs), len(ax)):
        ax[j].axis("off")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Cross-check: the library's estimator noise vs the same statistic on the independent ε-calibration policy population")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    save_figure(fig, scfg.esd_figure_path("B7_epscube_crosscheck"))
    plt.close(fig)


###############################################################################
# NFE asymptote (existing runtime archives)
###############################################################################

def _read_island(metrics_file: Path) -> pd.DataFrame | None:
    """One island's per-snapshot NFE, hypervolume, ε-progress count, archive size."""
    from src.plotting.hypervolume_convergence import _load_metrics_file, _read_runtime_nfe

    df = _load_metrics_file(metrics_file)
    if df is None or df.empty or "Hypervolume" not in df.columns:
        return None
    nfe = _read_runtime_nfe(metrics_file, len(df))
    runtime = metrics_file.parent.parent / "runtime" / f"{metrics_file.stem}.runtime"
    imp, arch = [], []
    if runtime.exists():
        with open(runtime) as fh:
            for line in fh:
                if line.startswith("//Improvements="):
                    imp.append(int(line.strip().split("=")[1]))
                elif line.startswith("//ArchiveSize="):
                    arch.append(int(line.strip().split("=")[1]))
    out = pd.DataFrame({"snapshot": np.arange(len(df)), "hypervolume": df["Hypervolume"].values})
    out["nfe"] = nfe if nfe else np.arange(1, len(df) + 1)
    out["improvements"] = imp if len(imp) == len(df) else np.nan
    out["archive_size"] = arch if len(arch) == len(df) else np.nan
    return out


def nfe_asymptote() -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for design, slug, seed in scfg.ESD_NFE_ARCHIVES:
        mdir = config.OUTPUTS_DIR / design / slug / "metrics"
        files = sorted(mdir.glob(f"seed_{seed:02d}_{slug}_*.metrics"))
        if not files:
            print(f"[esd:B8] no metrics for {design}/{slug} seed {seed}")
            continue
        for mf in files:
            isl = _read_island(mf)
            if isl is None:
                continue
            island = mf.stem.split("_")[-1]
            curves[(design, slug, seed, island)] = isl
            hv = isl.hypervolume.values
            final = hv[-1]
            rec = {"design": design, "slug": slug, "seed": seed, "island": island,
                   "nfe_final": int(isl.nfe.iloc[-1]), "hv_final": float(final),
                   "archive_final": float(isl.archive_size.iloc[-1])}
            for frac in scfg.ESD_NFE_HV_FRACTIONS:
                idx = np.flatnonzero(hv >= frac * final)
                rec[f"nfe_at_{int(frac * 100)}pct_hv"] = int(isl.nfe.iloc[idx[0]]) if len(idx) else np.nan
            fifth = max(1, len(isl) // 5)
            if isl.improvements.notna().all():
                d_imp = np.diff(isl.improvements.values)
                d_nfe = np.diff(isl.nfe.values)
                rate = d_imp / np.maximum(d_nfe, 1)
                rec["eps_progress_per_nfe_first_fifth"] = float(rate[:fifth].mean())
                rec["eps_progress_per_nfe_last_fifth"] = float(rate[-fifth:].mean())
                rec["eps_progress_last_over_first"] = float(rate[-fifth:].mean() / max(rate[:fifth].mean(), 1e-12))
            d_hv = np.diff(hv)
            rec["hv_gain_last_fifth_frac_of_final"] = float(d_hv[-fifth:].sum() / max(final, 1e-12))
            rows.append(rec)
    return pd.DataFrame(rows), curves


def fig_b8_nfe(curves: dict, table: pd.DataFrame) -> None:
    if not curves:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    seen = set()
    for (design, slug, seed, island), isl in curves.items():
        lab = design_label(design) if design not in seen else None
        seen.add(design)
        ax[0].plot(isl.nfe, isl.hypervolume / isl.hypervolume.iloc[-1], "-", lw=1.0,
                   color=design_color(design), alpha=0.8, label=lab)
        if isl.improvements.notna().all():
            rate = np.diff(isl.improvements.values) / np.maximum(np.diff(isl.nfe.values), 1)
            ax[1].plot(isl.nfe.values[1:], rate, "-", lw=1.0, color=design_color(design), alpha=0.8)
    for frac in scfg.ESD_NFE_HV_FRACTIONS:
        ax[0].axhline(frac, color="0.6", lw=0.9, ls="--")
    ax[0].set_xlabel("function evaluations per island (4 islands per search)")
    ax[0].set_ylabel("hypervolume ÷ the island's own final value")
    ax[0].set_title("(a) runtime hypervolume, one line per island\n(dashed = 95 % and 99 % of final)", fontsize=10)
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].set_xlabel("function evaluations per island")
    ax[1].set_ylabel("ε-progress improvements per evaluation (log)")
    ax[1].set_yscale("log")
    ax[1].set_title("(b) ε-progress improvements per evaluation\nbetween consecutive snapshots", fontsize=10)
    fig.suptitle("Are the existing 500k-NFE searches converged? Runtime archives of the go/no-go cells (seed 1) and the 50k-NFE historic runs")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, scfg.esd_figure_path("B8_nfe_asymptote"))
    plt.close(fig)


###############################################################################
# Cost pricing
###############################################################################

def cost_pricing(n_common: int | None) -> pd.DataFrame:
    a_mb, b_mb = scfg.ENSEMBLE_COST_RSS_MB["trimmed"]
    L = 10
    ns = sorted(set(scfg.ESD_N_LADDER) | {scfg.ESD_N_CAMPAIGN} | ({n_common} if n_common else set()))
    rows = []
    for n in ns:
        # ESD_SU_PER_SEARCH_N100 is priced at N = 100 (measured basis, 8 x 128,
        # 21.3 h per 500k-NFE search); the campaign is two matched designs at
        # 750k + 500k NFE = 5 search-equivalents (campaign_design.md §6).
        scale = (n / 100.0) ** scfg.ESD_COST_N_EXPONENT
        gb_rank = (a_mb + b_mb * n * L) / 1024.0
        node_gb = gb_rank * scfg.ESD_RANKS_PER_NODE
        max_ranks = int((scfg.ESD_NODE_MEM_GB * scfg.ESD_MEM_SAFETY) // gb_rank)
        rows.append({
            "n": n, "su_per_search": scfg.ESD_SU_PER_SEARCH_N100 * scale,
            "wall_hours_8nodes": 21.3 * scale,
            "matched_campaign_search_su": 5 * scfg.ESD_SU_PER_SEARCH_N100 * scale,
            "gb_per_rank": gb_rank, "node_gb_at_128_ranks": node_gb,
            "fits_128_ranks_at_safety": node_gb <= scfg.ESD_NODE_MEM_GB * scfg.ESD_MEM_SAFETY,
            "max_ranks_per_node_at_safety": min(scfg.ESD_RANKS_PER_NODE, max_ranks),
            "unit_years_per_eval": n * (L - 1),
            "etest_sow_units_1225_ge_9x_n": 1225 >= 9 * n,
        })
    return pd.DataFrame(rows)


def fig_b9_cost(cost: pd.DataFrame, n_common: int | None) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(cost.n, cost.su_per_search / 1000.0, "o-", color="black",
               label=f"{scfg.ESD_SU_PER_SEARCH_N100:,.0f} SU × (N/100)^0.951 (measured basis, 8 nodes × 128)")
    ax[0].set_xlabel(N_LABEL)
    ax[0].set_ylabel("thousand SU per 500,000-evaluation search")
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("(a) cost of one search")
    ax[1].plot(cost.n, cost.node_gb_at_128_ranks, "o-", color="black",
               label="128 ranks × (601 + 0.394·N·L) MB (measured RSS model)")
    ax[1].axhline(scfg.ESD_NODE_MEM_GB, color="#c1272d", lw=1.2, ls=":", label="Anvil node memory (256 GB)")
    ax[1].axhline(scfg.ESD_NODE_MEM_GB * scfg.ESD_MEM_SAFETY, color="#c1272d", lw=1.0, ls="--",
                  label=f"{scfg.ESD_MEM_SAFETY:.0%} of node memory (safety margin)")
    ax[1].set_xlabel(N_LABEL)
    ax[1].set_ylabel("memory per node with one evaluator per core (GB)")
    ax[1].set_ylim(bottom=0)
    ax[1].set_title("(b) memory at full 128-rank packing")
    for a in ax:
        a.axvline(scfg.ESD_N_CAMPAIGN, color=CAMPAIGN_COLOR, lw=1.0, ls=":",
                  label=f"campaign N = {scfg.ESD_N_CAMPAIGN}")
        if n_common:
            a.axvline(n_common, color=design_color(HF), lw=1.2, ls="-.", label=f"N_common = {n_common}")
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle("Cost and memory of a larger N (a fact table for the budget decision, not a criterion)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, scfg.esd_figure_path("B9_cost_pricing"))
    plt.close(fig)


###############################################################################
# Driver
###############################################################################

def main() -> None:
    apply_style()
    scfg.ESD_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    scfg.ESD_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Layer A ----
    ladder = _table("hf_ladder")
    if ladder is not None:
        fig_a1_tail_share(ladder)
        fig_a2_coverage(ladder)
    npl = _table("np_ladder")
    if npl is not None:
        fig_a3_np_ladder(npl).to_csv(scfg.esd_table_path("np_tail_share"), index=False)
    ps = _table("ps_tail_sampling")
    if ps is not None:
        fig_a4_ps_tail(ps)
    cv = _table("descriptor_convergence")
    if cv is not None:
        fig_a5_descriptor_convergence(cv)

    # ---- Layer B ----
    lib_path = scfg.esd_library_path()
    plan_path = scfg.esd_json_path("library_plan")
    n_common = None
    qc_path = scfg.esd_json_path("library_qc")
    if lib_path.exists() and plan_path.exists():
        qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
        if not qc.get("library_valid", False):
            sys.exit(f"[esd:B] {lib_path} failed its build QC ({qc_path}); refusing to analyze.")
        lib = Library(lib_path, json.loads(plan_path.read_text()))
        stats, bands, _ = layer_b_statistics(lib)
        stats.to_csv(scfg.esd_table_path("layer_b_stats"), index=False)
        bands.to_csv(scfg.esd_table_path("layer_b_policy_bands"), index=False)
        dec, nmin = decision_table(stats)
        dec.to_csv(scfg.esd_table_path("decision_table"), index=False)
        nmin.to_csv(scfg.esd_table_path("n_min"), index=False)
        n_common = None if nmin.n_common.isna().all() else int(nmin.n_common.iloc[0])
        largest = int(max(scfg.ESD_N_LADDER))
        print("[esd:B] N_min table:\n" + nmin.to_string(index=False), flush=True)
        neff = n_eff_table(lib)
        neff.to_csv(scfg.esd_table_path("n_eff"), index=False)
        fig_b_stat_vs_n(stats, "level_se_max",
                        "SD across replicate ensembles",
                        "B1_level_se_vs_n", scfg.ESD_LEVEL_SE_EPS_FRAC,
                        aux_col="level_se_median", aux_label="median policy",
                        main_label="worst policy",
                        suptitle="Level standard error: how much one policy's objective value moves from one size-N ensemble to another (solid = worst of 10 policies, dotted = median)")
        fig_b_stat_vs_n(stats, "paired_se_max",
                        "SD of the pairwise difference\nacross replicate ensembles",
                        "B2_paired_se_vs_n", scfg.ESD_PAIRED_SE_EPS_FRAC,
                        aux_col="paired_se_median", aux_label="median pair",
                        main_label="worst pair",
                        suptitle="Paired standard error, the binding criterion: noise in the difference between two policies (solid = worst of 45 pairs, dotted = median pair) vs ½ ε")
        fig_b3_flip(stats)
        fig_b4_optimism(bands, stats)
        fig_b5_tail_bands(bands, lib)
        fig_b6_neff(neff)
        cross = epscube_crosscheck(lib)
        if cross is not None:
            cross.to_csv(scfg.esd_table_path("epscube_crosscheck"), index=False)
            fig_b7_crosscheck(cross, stats)
    else:
        print(f"[esd:B] library not built yet ({lib_path}); Layer-B figures skipped")

    nfe, curves = nfe_asymptote()
    if not nfe.empty:
        nfe.to_csv(scfg.esd_table_path("nfe_asymptote"), index=False)
        fig_b8_nfe(curves, nfe)
    cost = cost_pricing(n_common)
    cost.to_csv(scfg.esd_table_path("cost_pricing"), index=False)
    fig_b9_cost(cost, n_common)
    print(f"[esd:fig] done -> {scfg.ESD_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
