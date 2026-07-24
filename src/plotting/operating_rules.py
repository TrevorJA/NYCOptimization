"""operating_rules.py - Visualize the operational rules encoded by a policy's DVs.

The 24 FFMP decision variables ARE the operating policy: minimum-release-flow (MRF)
targets, the NYC drought-level delivery-reduction schedule, storage-zone boundary
shifts, reservoir flood-release caps, and the seasonal MRF profile scaling. This
module draws those rules as an interpretable "fingerprint" so a handful of chosen
Pareto policies can be compared against each other and the status-quo FFMP baseline
in the language of operations rather than raw DV vectors.

The headline panel is the NYC drought-level delivery schedule: a policy that keeps
the curtailment factors high prioritizes NYC diversions, while one that cuts them
hard is buying downstream (Montague/Trenton) flow -- exactly the trade-off the front
spans.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _val(dv, idx, name, default=np.nan):
    """DV value by name (NaN if the formulation lacks it)."""
    j = idx.get(name)
    return float(dv[j]) if j is not None else default


def plot_operating_rules(policies, out_file, formulation: str = "ffmp",
                         figsize: tuple = (15, 8.5)):
    """Draw a 6-panel operating-rules comparison for a small set of policies.

    Args:
        policies: List of dicts, each ``{"label": str, "dv": (n_vars,) array,
            "color": str}``. Draw order is as given; the FFMP baseline is added
            automatically as a black dashed reference.
        out_file: PNG path.
        formulation: Formulation name (fixes the DV name->index map).
        figsize: Figure size.
    """
    from src.formulations import get_var_names, get_baseline_values

    vnames = get_var_names(formulation)
    idx = {n: i for i, n in enumerate(vnames)}

    baseline = {"label": "FFMP baseline", "dv": get_baseline_values(formulation),
                "color": "black", "baseline": True}
    series = list(policies) + [baseline]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    def _style(s):
        return dict(color=s["color"],
                    lw=2.6 if s.get("baseline") else 2.0,
                    ls="--" if s.get("baseline") else "-",
                    marker="o", ms=5,
                    zorder=6 if s.get("baseline") else 4,
                    alpha=0.95)

    def _grouped_bars(ax, cats, value_fn, ylabel, title):
        n = len(series)
        x = np.arange(len(cats))
        w = 0.8 / n
        for si, s in enumerate(series):
            vals = [value_fn(s["dv"], c) for c in cats]
            ax.bar(x + (si - (n - 1) / 2) * w, vals, width=w,
                   color=s["color"], alpha=0.9,
                   edgecolor="black" if s.get("baseline") else "none",
                   hatch="//" if s.get("baseline") else None,
                   label=s["label"])
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    # -- Panel 1: NYC drought-level delivery schedule (the diversion rule) ---- #
    ax = axes[0, 0]
    levels = ["Normal\n(L1a-L2)", "L3", "L4", "L5"]
    xl = np.arange(len(levels))
    for s in series:
        curve = [1.0,
                 _val(s["dv"], idx, "nyc_drought_factor_L3"),
                 _val(s["dv"], idx, "nyc_drought_factor_L4"),
                 _val(s["dv"], idx, "nyc_drought_factor_L5")]
        ax.plot(xl, curve, label=s["label"], **_style(s))
    ax.set_xticks(xl)
    ax.set_xticklabels(levels, fontsize=8)
    ax.set_ylabel("NYC delivery factor\n(fraction of full delivery)", fontsize=9)
    ax.set_title("1. NYC drought-level delivery schedule\n(higher = prioritize NYC diversions)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # -- Panel 2: downstream MRF flow targets -------------------------------- #
    _grouped_bars(axes[0, 1], ["mrf_montague", "mrf_trenton"],
                  lambda dv, c: _val(dv, idx, c),
                  "MRF target (MGD)",
                  "2. Downstream flow targets\n(higher = prioritize Montague/Trenton)")
    axes[0, 1].set_xticklabels(["Montague", "Trenton"], fontsize=9)

    # -- Panel 3: NYC diversion cap + NYC-reservoir MRF baselines ------------ #
    _grouped_bars(axes[0, 2],
                  ["max_nyc_delivery", "mrf_cannonsville", "mrf_pepacton",
                   "mrf_neversink"],
                  lambda dv, c: _val(dv, idx, c),
                  "MGD",
                  "3. NYC diversion cap & reservoir MRF baselines")
    axes[0, 2].set_xticklabels(["Max NYC\ndelivery", "Cannons.\nMRF",
                                "Pepacton\nMRF", "Neversink\nMRF"], fontsize=7.5)

    # -- Panel 4: storage-zone boundary shifts ------------------------------- #
    ax = axes[1, 0]
    zones = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
    xz = np.arange(len(zones))
    for s in series:
        shifts = [_val(s["dv"], idx, f"zone_shift_{z}") for z in zones]
        ax.plot(xz, shifts, label=s["label"], **_style(s))
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(xz)
    ax.set_xticklabels(zones, fontsize=8, rotation=20)
    ax.set_ylabel("Zone boundary shift\n(fraction of capacity)", fontsize=9)
    ax.set_title("4. Storage-zone boundary shifts\n(+ raises drought triggers = more conservative)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

    # -- Panel 5: flood-release caps ----------------------------------------- #
    _grouped_bars(axes[1, 1],
                  ["flood_max_cannonsville", "flood_max_pepacton",
                   "flood_max_neversink"],
                  lambda dv, c: _val(dv, idx, c),
                  "Max flood release (CFS)",
                  "5. Reservoir flood-release caps")
    axes[1, 1].set_xticklabels(["Cannons.", "Pepacton", "Neversink"], fontsize=8)

    # -- Panel 6: seasonal MRF profile scaling ------------------------------- #
    ax = axes[1, 2]
    seasons = ["winter", "spring", "summer", "fall"]
    xs = np.arange(len(seasons))
    for s in series:
        scale = [_val(s["dv"], idx, f"mrf_profile_scale_{sn}") for sn in seasons]
        ax.plot(xs, scale, label=s["label"], **_style(s))
    ax.axhline(1.0, color="0.6", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.capitalize() for s in seasons], fontsize=8)
    ax.set_ylabel("MRF profile multiplier", fontsize=9)
    ax.set_title("6. Seasonal MRF profile scaling", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Operating rules of representative Pareto policies vs status-quo FFMP",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
