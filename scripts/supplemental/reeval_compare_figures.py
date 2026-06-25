"""reeval_compare_figures.py — clean, minimalist comparison of two scenario
designs' performance under common-ensemble re-evaluation.

Each design's Pareto policies were re-evaluated on ONE common (held-out)
streamflow ensemble; every objective is a satisficing fraction in [0, 1]
(higher = the policy meets the decree goalpost in more realizations). This
script compares the distribution of those fractions across each design's
policies, objective by objective.

Usage:
    python scripts/supplemental/reeval_compare_figures.py \
        --reeval-tag kn_78yr_n100 \
        --arms hazard_filling fixed_probabilistic_short
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import OUTPUTS_DIR  # noqa: E402

# 5 discriminating satisficing objectives (trenton reliability & minor-flood are
# degenerate/non-discriminating and excluded). Short, clean axis labels.
OBJECTIVES = [
    ("nyc_delivery_reliability_weekly__sat95",  "NYC delivery\nreliability"),
    ("nyc_delivery_deficit_cvar90_pct__sat10",  "NYC deficit\nCVaR90"),
    ("montague_flow_reliability_weekly__sat85",  "Montague\nreliability"),
    ("montague_flow_deficit_cvar90_pct__sat25",  "Montague deficit\nCVaR90"),
    ("nyc_storage_p5_pct__sat25",                "NYC storage\n5th pctile"),
]

# Muted, colorblind-friendly palette; clean labels.
STYLE = {
    "hazard_filling":            ("#2c7fb8", "hazard_filling (faithful, cdf)"),
    "hazard_filling_absolute":   ("#d95f0e", "hazard_filling_absolute (abs)"),
    "fixed_probabilistic_short": ("#636363", "fixed_probabilistic_short (baseline)"),
}


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.9", lw=0.8)
    ax.set_axisbelow(True)


def load_arm(arm, slug, tag):
    csv = OUTPUTS_DIR / arm / slug / "reeval" / tag / "objectives_summary.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, index_col="solution_id")
    cols = [c for c, _ in OBJECTIVES]
    df = df[cols].dropna()
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slug", default="ffmp_obj7_pilot")
    p.add_argument("--reeval-tag", required=True)
    p.add_argument("--arms", nargs="+",
                   default=["hazard_filling", "fixed_probabilistic_short"])
    p.add_argument("--outdir", default=None)
    args = p.parse_args(argv)

    data = {}
    for arm in args.arms:
        df = load_arm(arm, args.slug, args.reeval_tag)
        if df is None:
            print(f"[fig] WARNING: no re-eval CSV for '{arm}' (tag={args.reeval_tag}); skipping.")
            continue
        data[arm] = df
        print(f"[fig] {arm}: {len(df)} re-evaluated policies")
    if len(data) < 2:
        print("[fig] need >=2 arms with data; aborting.")
        return 1

    outdir = Path(args.outdir) if args.outdir else (
        OUTPUTS_DIR / "diagnostics" / "scenario_design_pilot" / f"reeval_{args.reeval_tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    arms = list(data.keys())
    labels = [c[1] for c in OBJECTIVES]
    keys = [c[0] for c in OBJECTIVES]

    # ---- Figure 1: grouped box plots, per objective, by design ----
    fig, ax = plt.subplots(figsize=(9, 4.2))
    n = len(arms)
    width = 0.8 / n
    for ai, arm in enumerate(arms):
        color = STYLE.get(arm, ("#333333", arm))[0]
        positions = np.arange(len(keys)) + (ai - (n - 1) / 2) * width
        vals = [data[arm][k].values for k in keys]
        bp = ax.boxplot(vals, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.3),
                        whiskerprops=dict(color=color, lw=1.0),
                        capprops=dict(color=color, lw=1.0),
                        boxprops=dict(facecolor=color, edgecolor=color, alpha=0.55, lw=1.0))
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("satisficing fraction  (higher = better)", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Performance under common-ensemble re-evaluation  ({args.reeval_tag})",
                 fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=STYLE.get(a, ("#333", a))[0], alpha=0.55)
               for a in arms]
    ax.legend(handles, [STYLE.get(a, (None, a))[1] for a in arms],
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=len(arms),
              frameon=False, fontsize=9)
    _clean_ax(ax)
    fig.tight_layout()
    f1 = outdir / "reeval_boxplots.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {f1}")

    # ---- Figure 2: parallel coordinates (faint policies + bold median) ----
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(keys))
    for arm in arms:
        color = STYLE.get(arm, ("#333333", arm))[0]
        M = data[arm][keys].values
        for row in M:
            ax.plot(xs, row, color=color, alpha=0.06, lw=0.7)
        ax.plot(xs, np.median(M, axis=0), color=color, lw=2.4,
                marker="o", ms=5, label=STYLE.get(arm, (None, arm))[1])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("satisficing fraction  (higher = better)", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Pareto-policy performance by objective  ({args.reeval_tag}; bold = median)",
                 fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=len(arms), frameon=False, fontsize=9)
    _clean_ax(ax)
    fig.tight_layout()
    f2 = outdir / "reeval_parallel.png"
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {f2}")

    # ---- Median table (numbers behind the figures) ----
    print("\n[fig] median satisficing fraction per objective:")
    hdr = "  " + "objective".ljust(34) + "".join(a[:22].rjust(24) for a in arms)
    print(hdr)
    for k, lab in OBJECTIVES:
        row = "  " + lab.replace("\n", " ").ljust(34)
        for arm in arms:
            row += f"{np.median(data[arm][k].values):24.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
