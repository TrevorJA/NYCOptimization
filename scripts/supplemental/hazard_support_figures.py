"""hazard_support_figures.py - SI figures for the hazard-support decomposition.

Pure post-processing of the tables persisted by
``scripts/supplemental/hazard_support_run.py`` (no cube, pool, or ensemble is
touched): the support map over the forcing plane, the support-score
distribution across pool draws, the axis-excursion attribution, the per-axis
reach view, and - once stage B has run - the stratified design contrast, the
partition-agreement view, and the pool-deficit failure association. Stage-B
figures are skipped with a message when their tables are absent.

Settings in ``supplemental_config.py`` (``HSD_*``); figures follow
``src/plotting/style.py`` (PNG only during iteration, no in-panel text
annotations). Run after the run script::

    python scripts/supplemental/hazard_support_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_hsd_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.plotting.etest_hazard_overlay import HAZARD_AXIS_LABELS  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_style, design_color, design_label, save_figure,
)

#: Fixed categorical hue per support stratum (identity job, never cycled).
STRATUM_COLORS: dict = {
    "in_support": "#7f7f7f",
    "boundary": "#e6a03c",
    "out_of_support": "#7a4fa3",   # the reserved E_test purple: beyond-pool
    "beyond_support": "#7a4fa3",   # two-way fallback label
}

#: Display names for the m-terciles (dominant forcing factor, dry -> wet).
TERCILE_LABELS: tuple = ("dry tercile", "middle tercile", "wet tercile")


def _table(name: str, tagged: bool = False) -> pd.DataFrame | None:
    path = scfg.hsd_table_path(name, tagged=tagged)
    if not path.exists():
        print(f"[hsd-fig] {name} not found ({path}); skipping its figures.")
        return None
    return pd.read_csv(path)


def _manifest() -> dict:
    path = scfg.HSD_TABLES_DIR / (
        ("smoke_" if scfg.HSD_SMOKE else "") + "hsd_manifest.json")
    return json.loads(path.read_text()) if path.exists() else {}


def _primary_score_col(sow: pd.DataFrame) -> str:
    return f"out_frac_q{scfg.HSD_SELF_NN_QUANTILE:g}_d0"


def fig_support_map(sow: pd.DataFrame) -> None:
    """F1: strata and support score over the (e^m, r1) forcing plane."""
    col = _primary_score_col(sow)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for name in scfg.HSD_STRATA_NAMES:
        sel = sow["stratum"] == name
        ax1.scatter(sow.loc[sel, "em"], sow.loc[sel, "r1"], s=12,
                    c=STRATUM_COLORS[name], label=name.replace("_", " "),
                    linewidths=0)
    ax1.set_xlabel("water-year volume multiplier $e^m$")
    ax1.set_ylabel("annual harmonic amplitude $r_1$")
    ax1.set_title(f"(a) Support stratum (n = {len(sow)})")
    ax1.legend(frameon=False, loc="best")
    sc = ax2.scatter(sow["em"], sow["r1"], s=12, c=sow[col], cmap="viridis",
                     vmin=0.0, vmax=1.0, linewidths=0)
    ax2.set_xlabel("water-year volume multiplier $e^m$")
    ax2.set_title("(b) Out-of-support sub-window fraction")
    fig.colorbar(sc, ax=ax2, label="out_frac")
    fig.tight_layout()
    save_figure(fig, scfg.hsd_figure_path("F1_support_map_theta"))
    plt.close(fig)


def fig_score_distribution(sow: pd.DataFrame) -> None:
    """F2: ECDF of the per-SOW support score, one line per pool re-roll."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k in range(len(scfg.HSD_POOL_SLUGS)):
        col = f"out_frac_q{scfg.HSD_SELF_NN_QUANTILE:g}_d{k}"
        if col not in sow.columns:
            continue
        v = np.sort(sow[col].to_numpy(dtype=float))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                label=f"pool draw {k}")
    for cut in scfg.HSD_STRATA_CUTS:
        ax.axvline(cut, color="0.35", lw=0.9, ls="--")
    ax.set_xlabel("out-of-support sub-window fraction (out_frac)")
    ax.set_ylabel("fraction of SOWs")
    ax.set_title(f"Support-score ECDF (n = {len(sow)} SOWs)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, scfg.hsd_figure_path("F2_support_score_distribution"))
    plt.close(fig)


def fig_axis_excursion(axis_tab: pd.DataFrame) -> None:
    """F3: which axes carry the excursion, by forcing tercile (pool d0)."""
    d0 = axis_tab[axis_tab["draw"] == 0]
    axes = [a for a in dict.fromkeys(d0["axis"])]
    x = np.arange(len(axes))
    width = 0.8 / scfg.HSD_TERCILE_BINS
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for t in range(scfg.HSD_TERCILE_BINS):
        sel = d0[d0["m_tercile"] == t].set_index("axis")
        vals = [sel.loc[a, "dominant_share"] if a in sel.index else np.nan
                for a in axes]
        ax.bar(x + (t - (scfg.HSD_TERCILE_BINS - 1) / 2) * width, vals,
               width=width, label=TERCILE_LABELS[t],
               color=plt.get_cmap("cividis")(t / max(1, scfg.HSD_TERCILE_BINS - 1)))
    ax.set_xticks(x)
    ax.set_xticklabels([HAZARD_AXIS_LABELS.get(a, a).replace("\n", " ")
                        for a in axes], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("share of out-of-support sub-windows\nwith largest excursion on axis")
    ax.set_title("Excursion attribution by forcing tercile")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, scfg.hsd_figure_path("F3_axis_excursion"))
    plt.close(fig)


def fig_reach_by_tercile(reach: pd.DataFrame) -> None:
    """F4: per axis, E_test sub-window quantiles by tercile vs the pool band."""
    axes = [a for a in dict.fromkeys(reach["axis"])]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.4))
    markers = ("o", "s", "^")
    for j, (axis, ax) in enumerate(zip(axes, axs.ravel())):
        sub = reach[reach["axis"] == axis]
        lo = float(sub["pool_lo"].iloc[0])
        hi = float(sub["pool_hi"].iloc[0])
        ax.axhspan(lo, hi, color="0.85", zorder=0)
        for t in range(scfg.HSD_TERCILE_BINS):
            row = sub[sub["m_tercile"] == t]
            if row.empty:
                continue
            for q, mk in zip(("q50", "q90", "q99"), markers):
                ax.plot(t, float(row[q].iloc[0]), mk, color="#7a4fa3",
                        ms=6, mfc="none" if q != "q99" else "#7a4fa3")
        ax.set_xticks(range(scfg.HSD_TERCILE_BINS))
        ax.set_xticklabels([l.split()[0] for l in TERCILE_LABELS])
        ax.set_title(HAZARD_AXIS_LABELS.get(axis, axis).replace("\n", " "),
                     fontsize=9)
    handles = [Line2D([], [], marker=mk, ls="", color="#7a4fa3",
                      mfc="none" if q != "q99" else "#7a4fa3",
                      label=f"E_test {q}")
               for q, mk in zip(("q50", "q90", "q99"), markers)]
    handles.append(Line2D([], [], marker="s", ls="", color="0.85", ms=10,
                          label="pool p1-p99"))
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=4)
    fig.suptitle("E_test sub-window reach vs the stationary pool band")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, scfg.hsd_figure_path("F4_reach_by_tercile"))
    plt.close(fig)


def _group_order(partition: str, groups: list) -> list:
    if partition == "support_stratum":
        order = list(scfg.HSD_STRATA_NAMES) + ["beyond_support"]
        return [g for g in order if g in groups]
    return sorted(groups)


def fig_contrast(contrast: pd.DataFrame, boot: pd.DataFrame,
                 noharm: pd.DataFrame | None) -> None:
    """F5/F6: HF - PS differences per group, both partitions side by side."""
    focal_key = [k for k in dict.fromkeys(boot["criterion_key"])
                 if k != "all_axes"][0]
    panels = [("sat_fixed", focal_key,
               "Starr satisficing (focal set, fixed policy)")]
    if noharm is not None and (boot["endpoint"] == "noharm_fixed").any():
        panels.append(("noharm_fixed", "all_axes",
                       "No-harm frequency (adopted tau, fixed policy)"))

    for figname, partition in (("F5_contrast_by_stratum", "support_stratum"),
                               ("F6_partition_agreement", "m_tercile")):
        sub_b = boot[boot["partition"] == partition]
        if sub_b.empty:
            continue
        fig, axs = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4),
                                squeeze=False)
        for ax, (endpoint, ckey, title) in zip(axs.ravel(), panels):
            b = sub_b[(sub_b["endpoint"] == endpoint)
                      & (sub_b["criterion_key"] == ckey)]
            groups = _group_order(partition, list(dict.fromkeys(b["group"])))
            x = np.arange(len(groups))
            for i, g in enumerate(groups):
                row = b[b["group"] == g].iloc[0]
                ax.errorbar(i, row["delta_hf_minus_ps"],
                            yerr=[[row["delta_hf_minus_ps"] - row["ci_lo"]],
                                  [row["ci_hi"] - row["delta_hf_minus_ps"]]],
                            fmt="o", color="#222222", capsize=4, zorder=3)
            # Draw-level points behind the CI (seed means per design draw).
            src_tab = contrast if endpoint == "sat_fixed" else noharm
            val_col = "fixed_frac" if endpoint == "sat_fixed" else "noharm_fixed"
            sel = src_tab[src_tab["partition"] == partition]
            if endpoint == "sat_fixed":
                sel = sel[sel["criterion_key"] == ckey]
            for i, g in enumerate(groups):
                gg = sel[sel["group"] == g]
                hf = (gg[gg["design"] == "hazard_filling_stationary"]
                      .groupby("draw")[val_col].mean())
                ps_mean = (gg[gg["design"] == "fixed_probabilistic"]
                           .groupby("draw")[val_col].mean().mean())
                ax.scatter(np.full(len(hf), i) + 0.12, hf - ps_mean, s=22,
                           color=design_color("hazard_filling_stationary"),
                           zorder=2)
            ax.axhline(0.0, color="0.4", lw=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [g.replace("_", " ") if partition == "support_stratum"
                 else TERCILE_LABELS[int(g.split("_")[1])] for g in groups],
                rotation=15, ha="right")
            ax.set_ylabel("HF - PS difference")
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        save_figure(fig, scfg.hsd_figure_path(figname, tagged=True))
        plt.close(fig)


def fig_pool_deficit(deficit: pd.DataFrame) -> None:
    """F7: failure rate vs pool-deficit decile, per design (fixed policy)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for design in dict.fromkeys(deficit["design"]):
        sub = (deficit[deficit["design"] == design]
               .groupby("bin")[["deficit_mid", "failure_rate"]].mean())
        ax.plot(sub["deficit_mid"], sub["failure_rate"], "-o", ms=4,
                color=design_color(design), label=design_label(design))
    ax.set_xlabel("distance to nearest POOL member (E_test rank space)")
    ax.set_ylabel("failure rate (focal criterion, fixed policy)")
    ax.set_title("Pool-level coverage deficit vs failure")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, scfg.hsd_figure_path("F7_pool_vs_design_deficit", tagged=True))
    plt.close(fig)


def main() -> None:
    apply_style()
    scfg.HSD_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    if manifest:
        print(f"[hsd-fig] manifest: n_sow={manifest.get('n_sow')}, "
              f"strata={manifest.get('stratum_counts')}")

    sow = _table("hsd_sow_support")
    if sow is None:
        sys.exit("[hsd-fig] stage A has not run; nothing to draw.")
    fig_support_map(sow)
    fig_score_distribution(sow)
    axis_tab = _table("hsd_axis_excursion")
    if axis_tab is not None and not axis_tab.empty:
        fig_axis_excursion(axis_tab)
    reach = _table("hsd_reach_by_tercile")
    if reach is not None and not reach.empty:
        fig_reach_by_tercile(reach)

    contrast = _table("hsd_stratum_contrast", tagged=True)
    boot = _table("hsd_contrast_bootstrap", tagged=True)
    if contrast is not None and boot is not None and not boot.empty:
        noharm = _table("hsd_stratum_noharm", tagged=True)
        fig_contrast(contrast, boot, noharm)
    deficit = _table("hsd_pool_deficit_failure", tagged=True)
    if deficit is not None and not deficit.empty:
        fig_pool_deficit(deficit)
    print(f"[hsd-fig] figures -> {scfg.HSD_FIGURES_DIR}")


if __name__ == "__main__":
    main()
