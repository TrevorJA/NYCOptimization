"""
factor_map_surfaces.py - Manuscript Figure 9: success/failure surfaces.

The scenario-discovery closer, in the Hadjimichael et al. (2020) / Lau et al.
(2023) idiom: for each design's focal-criterion compromise policy (plus the
FFMP incumbent), the boosted-tree classifier's predicted probability of
SUCCESS over the two most important DU forcing axes, drawn as a diverging
field (red = predicted failure, blue = predicted success, neutral at the
P = 0.5 boundary, which is also drawn as a contour), under the raw E_test
SOW labels (white circles = pass, black crosses = fail) and the reference-SOW
markers (triangle = expected conditions, star = deepest sampled dry state).

Data: the ``factor_mapping/{criterion}/`` artifacts written by
``scripts/main/factor_mapping_run.py`` (surfaces NPZ + labels/fits/reference
CSVs) -- figures render anywhere, no cubes needed. CV skill per fit goes to
the companion CSV, never into the panels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.layout import (WIDTH_DOUBLE_COL, add_colorbar, criteria_footer,
                                 panel_label)
from src.plotting.style import (FACTOR_MAP_CMAP, FACTOR_MAP_MARKS,
                                design_label, save_manuscript_figure)
from src.satisficing_criteria import focal_criterion

#: Axis display names for the theta features.
_THETA_LABELS = {"em": "Water-year volume multiplier, $e^{m}$",
                 "r1": "Seasonal amplitude, $r_1$",
                 "r2": "Semiannual amplitude, $r_2$"}


def _load_artifacts(ctx, criterion: str) -> dict:
    root = ctx.comparison_dir() / "factor_mapping" / criterion
    if not (root / "factor_map_fits.csv").exists():
        raise FileNotFoundError(
            f"factor-mapping artifacts not found under {root} -- run "
            f"scripts/main/factor_mapping_run.py first (Anvil)."
        )
    out = {
        "fits": pd.read_csv(root / "factor_map_fits.csv"),
        "labels": pd.read_csv(root / "factor_map_labels.csv"),
        "meta": json.loads((root / "factor_mapping_meta.json").read_text()),
        "surfaces": {},
        "refs": None,
    }
    npz = root / "factor_map_surfaces.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            out["surfaces"] = {k: z[k] for k in z.files}
    ref_csv = root / "factor_map_reference_sows.csv"
    if ref_csv.exists():
        out["refs"] = pd.read_csv(ref_csv)
    return out


def fig_success_failure_surfaces(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Theta-space GBM probability surfaces per design policy + incumbent."""
    focal = focal_criterion()
    art = _load_artifacts(ctx, focal.key)
    fits = art["fits"]
    theta = fits[fits["space"] == "theta"]

    # One column per (design, policy); the incumbent column drawn once.
    panels = []
    seen_incumbent = False
    for _, row in theta.iterrows():
        if row["policy"] == "incumbent":
            if seen_incumbent:
                continue
            seen_incumbent = True
        panels.append(row)
    if not panels:
        raise FileNotFoundError("no theta-space fits in the artifacts")

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(WIDTH_DOUBLE_COL, WIDTH_DOUBLE_COL / n * 1.15),
        constrained_layout=True, sharex=False, sharey=False)
    axes = np.atleast_1d(axes)

    mappable = None
    csv_rows = []
    for i, (ax, row) in enumerate(zip(axes, panels)):
        design, policy = row["design"], str(row["policy"])
        key = f"{design}__{policy}__theta"
        if f"{key}__P" not in art["surfaces"]:
            ax.axis("off")
            continue
        g1 = art["surfaces"][f"{key}__g1"]
        g2 = art["surfaces"][f"{key}__g2"]
        P = art["surfaces"][f"{key}__P"]
        ax_names = [str(a) for a in art["surfaces"][f"{key}__axes"]]

        mappable = ax.pcolormesh(g1, g2, P, cmap=FACTOR_MAP_CMAP, vmin=0.0,
                                 vmax=1.0, alpha=0.85, shading="auto",
                                 rasterized=True)
        ax.contour(g1, g2, P, levels=[0.5], colors="0.15", linewidths=1.0)

        lab = art["labels"]
        sub = lab[(lab["design"] == design) & (lab["policy"] == policy)
                  & (lab["criterion"] == focal.key)]
        if not sub.empty and all(a in sub.columns for a in ax_names):
            ok = sub["pass"].astype(bool).to_numpy()
            x, y = sub[ax_names[0]].to_numpy(), sub[ax_names[1]].to_numpy()
            m_ok = FACTOR_MAP_MARKS["success"]
            ax.scatter(x[ok], y[ok], s=12, marker=m_ok["marker"],
                       facecolors=m_ok["facecolor"],
                       edgecolors=m_ok["edgecolor"], linewidths=0.5, zorder=4)
            m_no = FACTOR_MAP_MARKS["failure"]
            ax.scatter(x[~ok], y[~ok], s=14, marker=m_no["marker"],
                       color=m_no["color"], linewidths=0.8, zorder=4)
            n_pass = int(ok.sum())
        else:
            n_pass = -1

        if art["refs"] is not None and all(a in art["refs"].columns
                                           for a in ax_names):
            exp = art["refs"][art["refs"]["role"] == "expected"].iloc[0]
            dry = art["refs"][art["refs"]["role"] == "dry"].iloc[0]
            ax.scatter(exp[ax_names[0]], exp[ax_names[1]], marker="^", s=70,
                       facecolors="none", edgecolors="0.1", linewidths=1.4,
                       zorder=5)
            ax.scatter(dry[ax_names[0]], dry[ax_names[1]], marker="*", s=110,
                       facecolors="none", edgecolors="0.1", linewidths=1.2,
                       zorder=5)

        who = ("FFMP incumbent" if policy == "incumbent"
               else f"{design_label(design).split(' (')[0]} #{policy}")
        title = who if n_pass < 0 else f"{who}\n({n_pass}/{len(sub)} pass)"
        ax.set_title(title, fontsize=8)
        panel_label(ax, chr(ord("a") + i))
        ax.set_xlabel(_THETA_LABELS.get(ax_names[0], ax_names[0]), fontsize=8)
        if i == 0:
            ax.set_ylabel(_THETA_LABELS.get(ax_names[1], ax_names[1]),
                          fontsize=8)
        csv_rows.append({
            "design": design, "policy": policy, "criterion": focal.key,
            "axis_1": ax_names[0], "axis_2": ax_names[1],
            "cv_auc": row.get("cv_auc"), "cv_auc_std": row.get("cv_auc_std"),
            "train_accuracy": row.get("train_accuracy"),
            "backend": row.get("backend"), "n_pos": row.get("n_pos"),
            "n_neg": row.get("n_neg"),
        })

    if mappable is not None:
        add_colorbar(fig, mappable, list(axes),
                     label="Predicted probability the SOW meets the focal set")

    meta = art["meta"]
    if "criterion_thresholds" in meta and "criterion_kinds" in meta:
        criteria_footer(
            fig, focal, dict(meta["criterion_thresholds"]),
            dict(meta["criterion_kinds"]),
            obj_order=list(meta["criterion_thresholds"]), y=-0.10,
            provenance=(f"Per-set compromise policies + incumbent on "
                        f"{meta.get('n_sow', '?')} E_test SOWs; boosted-tree "
                        f"surfaces (CV skill in the companion table)."))

    save_manuscript_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(
        table_dir / f"success_failure_surfaces_{focal.key}.csv", index=False)
    return {"criterion": focal.key, "n_panels": n}
