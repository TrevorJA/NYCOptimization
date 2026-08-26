"""hazard_support_figures.py - SI figures for the hazard-support decomposition.

Pure post-processing of the tables persisted by
``scripts/supplemental/hazard_support_run.py`` (no cube, pool, or ensemble is
touched): where the test futures sit relative to the stationary candidate
pool's hazard range, which hazard metrics carry the excursion, the per-metric
reach view, and - once stage B has run - the stratified design contrast, the
dry/wet partition of the same contrast, and the pool-coverage vs failure view.
Stage-B figures are skipped with a message when their tables are absent.

Labeling is deliberately plain-language: every axis, legend, and title is
written for a reader with only a basic picture of the experiment (a test
ensemble of climate futures, a stationary candidate pool of synthetic decades,
two ways of picking search scenarios from that pool). Internal names
(``out_frac``, ``in_support``, ``tercile_0``) never appear on a figure.

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
from matplotlib.patches import Patch  # noqa: E402

from src.plotting.style import apply_style, design_color, save_figure  # noqa: E402

###############################################################################
# Plain-language vocabulary (the only place figure wording is defined)
###############################################################################

#: Fixed categorical hue per group (identity job, never cycled).
GROUP_COLORS: dict = {
    "in_support": "#7f7f7f",
    "boundary": "#e6a03c",
    "out_of_support": "#7a4fa3",   # the reserved E_test purple: beyond-pool
    "beyond_support": "#7a4fa3",   # two-way fallback label
}

#: Plain-language names for the support groups (a "future" = one E_test SOW;
#: its "decades" = its 125 ten-year sub-windows).
GROUP_LABELS: dict = {
    "in_support": "hazards typical of the stationary pool",
    "boundary": "some decades beyond the stationary pool",
    "out_of_support": "most decades beyond the stationary pool",
    "beyond_support": "some or most decades beyond the stationary pool",
}

#: Plain-language names for the thirds of the mean-flow forcing factor.
THIRD_LABELS: tuple = ("driest third of futures", "middle third of futures",
                       "wettest third of futures")
THIRD_SHORT: tuple = ("driest third", "middle third", "wettest third")

#: Compact two-line tick labels for the six hazard metrics (bar charts).
METRIC_TICKS: dict = {
    "drought_magnitude": "drought\nmagnitude",
    "drought_severity": "drought\nseverity",
    "drought_onset_rate": "drought\nonset speed",
    "drought_recovery_rate": "drought\nrecovery speed",
    "flood_peak_discharge": "flood\npeak flow",
    "flood_pulse_duration": "flood\npulse length",
}

#: Plain-language names for the six hazard metrics.
METRIC_LABELS: dict = {
    "drought_magnitude": "drought magnitude\n(cumulative dryness, |ΣSSI|)",
    "drought_severity": "drought severity\n(peak dryness, |min SSI|)",
    "drought_onset_rate": "drought onset speed\n(SSI per month)",
    "drought_recovery_rate": "drought recovery speed\n(SSI per month)",
    "flood_peak_discharge": "flood peak flow\n(× long-term mean daily flow)",
    "flood_pulse_duration": "flood pulse length\n(days above threshold)",
}

#: Plain-language design names for the contrast figures.
DESIGN_PLAIN: dict = {
    "hazard_filling_stationary": "hazard filling",
    "fixed_probabilistic": "random sampling",
    "historic": "historic record only",
}

X_MEANFLOW = "mean-flow change of the future\n(multiplier on annual volume; 1 = no change)"
Y_SEASONAL = "seasonal-cycle change of the future\n(annual harmonic amplitude)"
SCORE_LABEL = ("fraction of the future's decades whose hazards\n"
               "lie beyond the stationary pool's range")


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


def _primary_score_col() -> str:
    return f"out_frac_q{scfg.HSD_SELF_NN_QUANTILE:g}_d0"


def _metric_label(axis: str, oneline: bool = False) -> str:
    lab = METRIC_LABELS.get(axis, axis.replace("_", " "))
    return lab.replace("\n", " ") if oneline else lab


###############################################################################
# Stage-A figures
###############################################################################

def fig_support_map(sow: pd.DataFrame) -> None:
    """F1: which test futures carry hazards the stationary pool never showed."""
    col = _primary_score_col()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.9), sharey=True)
    for name in scfg.HSD_STRATA_NAMES:
        sel = sow["stratum"] == name
        ax1.scatter(sow.loc[sel, "em"], sow.loc[sel, "r1"], s=12,
                    c=GROUP_COLORS[name], linewidths=0,
                    label=f"{GROUP_LABELS[name]} (n = {int(sel.sum())})")
    ax1.set_xlabel(X_MEANFLOW)
    ax1.set_ylabel(Y_SEASONAL)
    ax1.set_title(f"(a) Test futures grouped by hazard novelty (n = {len(sow)})")
    sc = ax2.scatter(sow["em"], sow["r1"], s=12, c=sow[col], cmap="viridis",
                     vmin=0.0, vmax=float(np.ceil(sow[col].max() * 10) / 10),
                     linewidths=0)
    ax2.set_xlabel(X_MEANFLOW)
    ax2.set_title("(b) Share of each future's decades with novel hazards")
    fig.colorbar(sc, ax=ax2, label=SCORE_LABEL)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=3,
               fontsize=8, title="each point is one test future", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, scfg.hsd_figure_path("F1_support_map_theta"))
    plt.close(fig)


def fig_score_distribution(sow: pd.DataFrame) -> None:
    """F2: distribution of the novelty score, and its stability across pools."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for k in range(len(scfg.HSD_POOL_SLUGS)):
        col = f"out_frac_q{scfg.HSD_SELF_NN_QUANTILE:g}_d{k}"
        if col not in sow.columns:
            continue
        v = np.sort(sow[col].to_numpy(dtype=float))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                label=f"scored against candidate pool draw {k}")
    lo, hi = scfg.HSD_STRATA_CUTS
    ax.axvline(lo, color="0.35", lw=0.9, ls="--")
    ax.axvline(hi, color="0.35", lw=0.9, ls="--")
    ax.set_xlabel(SCORE_LABEL)
    ax.set_ylabel("cumulative share of test futures")
    ax.set_title(f"How many test futures carry novel hazards (n = {len(sow)})")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.35", lw=0.9, ls="--"))
    labels.append(f"group boundaries ({lo:g} and {hi:g} of a future's decades)")
    ax.legend(handles, labels, frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, scfg.hsd_figure_path("F2_support_score_distribution"))
    plt.close(fig)


def fig_axis_excursion(axis_tab: pd.DataFrame) -> None:
    """F3: which hazard metric the novel decades exceed the pool on."""
    d0 = axis_tab[axis_tab["draw"] == 0]
    axes = [a for a in dict.fromkeys(d0["axis"])]
    x = np.arange(len(axes))
    width = 0.8 / scfg.HSD_TERCILE_BINS
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    cmap = plt.get_cmap("cividis")
    for t in range(scfg.HSD_TERCILE_BINS):
        sel = d0[d0["m_tercile"] == t].set_index("axis")
        vals = [sel.loc[a, "dominant_share"] if a in sel.index else np.nan
                for a in axes]
        ax.bar(x + (t - (scfg.HSD_TERCILE_BINS - 1) / 2) * width, vals,
               width=width, label=THIRD_LABELS[t],
               color=cmap(t / max(1, scfg.HSD_TERCILE_BINS - 1)))
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_TICKS.get(a, a.replace("_", " ")) for a in axes],
                       fontsize=9)
    ax.set_ylabel("share of novel decades whose largest\nexceedance is on this hazard metric")
    ax.set_title("Which hazard metric carries the novelty, by how wet the future is")
    ax.legend(frameon=False, title="test futures split by mean-flow change")
    fig.tight_layout()
    save_figure(fig, scfg.hsd_figure_path("F3_axis_excursion"))
    plt.close(fig)


def fig_reach_by_tercile(reach: pd.DataFrame) -> None:
    """F4: per metric, how far test decades reach beyond the pool's range."""
    axes = [a for a in dict.fromkeys(reach["axis"])]
    fig, axs = plt.subplots(2, 3, figsize=(12.0, 7.0))
    markers = ("o", "s", "^")
    qlabels = {"q50": "median test decade", "q90": "90th-percentile test decade",
               "q99": "99th-percentile test decade"}
    for axis, ax in zip(axes, axs.ravel()):
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
        ax.set_xticklabels(THIRD_SHORT, fontsize=8)
        ax.set_xlim(-0.5, scfg.HSD_TERCILE_BINS - 0.5)
        ax.set_title(_metric_label(axis, oneline=True), fontsize=9)
    handles = [Line2D([], [], marker=mk, ls="", color="#7a4fa3",
                      mfc="none" if q != "q99" else "#7a4fa3", label=qlabels[q])
               for q, mk in zip(("q50", "q90", "q99"), markers)]
    handles.append(Patch(color="0.85",
                         label="range covered by the stationary candidate pool (1st–99th percentile)"))
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=2, fontsize=9)
    fig.suptitle("How far test-future decades reach beyond the stationary pool, by hazard metric")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_figure(fig, scfg.hsd_figure_path("F4_reach_by_tercile"))
    plt.close(fig)


###############################################################################
# Stage-B figures
###############################################################################

def _group_order(partition: str, groups: list) -> list:
    if partition == "support_stratum":
        order = list(scfg.HSD_STRATA_NAMES) + ["beyond_support"]
        return [g for g in order if g in groups]
    return sorted(groups)


def _group_tick(partition: str, g: str) -> str:
    if partition == "support_stratum":
        return GROUP_LABELS[g].replace(" the ", "\nthe ", 1)
    return THIRD_LABELS[int(g.split("_")[1])].replace(" of ", "\nof ")


def fig_contrast(contrast: pd.DataFrame, boot: pd.DataFrame,
                 noharm: pd.DataFrame | None) -> None:
    """F5/F6: hazard filling minus random sampling, by group of test futures."""
    focal_key = [k for k in dict.fromkeys(boot["criterion_key"])
                 if k != "all_axes"][0]
    panels = [("sat_fixed", focal_key,
               "(a) Fraction of futures where the policy meets\nthe compromise performance criteria")]
    if noharm is not None and (boot["endpoint"] == "noharm_fixed").any():
        panels.append(("noharm_fixed", "all_axes",
                       "(b) Fraction of futures where the policy is\nno worse than current FFMP operations"))

    titles = {
        "support_stratum": "Does hazard filling's advantage survive on futures with novel hazards?",
        "m_tercile": "The same advantage, split by how wet the future is",
    }
    for figname, partition in (("F5_contrast_by_stratum", "support_stratum"),
                               ("F6_partition_agreement", "m_tercile")):
        sub_b = boot[boot["partition"] == partition]
        if sub_b.empty:
            continue
        fig, axs = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 5.0),
                                squeeze=False)
        for ax, (endpoint, ckey, title) in zip(axs.ravel(), panels):
            b = sub_b[(sub_b["endpoint"] == endpoint)
                      & (sub_b["criterion_key"] == ckey)]
            groups = _group_order(partition, list(dict.fromkeys(b["group"])))
            x = np.arange(len(groups))
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
                ax.scatter(np.full(len(hf), i) + 0.12, hf - ps_mean, s=26,
                           color=design_color("hazard_filling_stationary"),
                           zorder=2)
                row = b[b["group"] == g].iloc[0]
                ax.errorbar(i, row["delta_hf_minus_ps"],
                            yerr=[[row["delta_hf_minus_ps"] - row["ci_lo"]],
                                  [row["ci_hi"] - row["delta_hf_minus_ps"]]],
                            fmt="o", color="#222222", capsize=4, zorder=3)
            ax.axhline(0.0, color="0.4", lw=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{_group_tick(partition, g)}\n"
                 f"(n = {int(b[b['group'] == g]['n_sow'].iloc[0])})"
                 for g in groups], fontsize=8)
            ax.set_ylabel("advantage of hazard filling over random sampling\n"
                          "(difference in fraction of futures; > 0 favours hazard filling)")
            ax.set_title(title, fontsize=10)
        handles = [
            Line2D([], [], marker="o", ls="", color="#222222",
                   label="difference between the two designs, with 95% interval\n"
                         "from resampling the test futures"),
            Line2D([], [], marker="o", ls="", color=design_color("hazard_filling_stationary"),
                   label="one independent ensemble draw"),
        ]
        fig.legend(handles=handles, loc="lower center", frameon=False, ncol=2, fontsize=8)
        fig.suptitle(titles[partition])
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        save_figure(fig, scfg.hsd_figure_path(figname, tagged=True))
        plt.close(fig)


def fig_pool_deficit(deficit: pd.DataFrame) -> None:
    """F7: do failures sit where the stationary pool is sparse?"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for design in dict.fromkeys(deficit["design"]):
        sub = (deficit[deficit["design"] == design]
               .groupby("bin")[["deficit_mid", "failure_rate"]].mean())
        ax.plot(sub["deficit_mid"], sub["failure_rate"], "-o", ms=4,
                color=design_color(design),
                label=f"best policy found with {DESIGN_PLAIN.get(design, design)}")
    ax.set_xlabel("how far a future's typical hazards sit from the nearest\n"
                  "stationary-pool decade (rank units; larger = sparser pool coverage)")
    ax.set_ylabel("fraction of futures where the policy fails\nthe compromise performance criteria")
    ax.set_title("Do policy failures sit where the stationary pool is sparse?\n"
                 "(test futures binned into tenths by pool sparseness)")
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=1)
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
