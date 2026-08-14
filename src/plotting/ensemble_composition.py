"""
ensemble_composition.py - Manuscript Figure 4: hazard-space ensemble composition.

Reduced, manuscript-grade companion of the SI corner overlay
(``src.plotting.etest_hazard_overlay``): the two most decision-relevant
pairwise hazard-space panels, showing where each realized search ensemble sits
relative to the candidate pool it was drawn from and to the held-out E_test
cloud. Layers per panel (bottom to top): candidate-pool grayscale log-density,
robust selection box, one categorical scatter per realized search ensemble
(entity-stable design colors), and E_test's sub-window mass contours.

Data contracts (all staged under ``config.STAGED_ENSEMBLE_DIR``):

* pool + search layers: ``{slug}/hazard_image.npz`` written by
  ``scengen.diagnostics.save_hazard_image`` (search layers use
  ``selected_rows`` when non-empty);
* E_test: ``{etest_slug}/hazard_image_subwindows.npz`` written by
  ``scripts/main/compute_etest_hazard_image.py``.

Configuration is via environment variables (no CLI value flags):

    NYCOPT_OVERLAY_POOL_SLUG      staged candidate-pool slug
                                  (default ``statpool_10yr_n1000000_d0``)
    NYCOPT_OVERLAY_SEARCH_SLUGS   comma-separated staged search-ensemble slugs;
                                  empty = auto-discover staged dirs carrying a
                                  hazard_image.npz with a non-empty selection
    NYCOPT_ETEST_VARIANT          E_test variant (resolved by ``src.etest``)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import config
from src.ensembles import staged_ensemble_dir
from src.plotting import style
from src.plotting.etest_hazard_overlay import (HAZARD_AXIS_LABELS,
                                               LAYER_COLORS, OverlayLayer,
                                               _col, draw_pair_panel)
from src.plotting.layout import panel_grid, panel_label, shared_legend
from src.plotting.style import DESIGN_ORDER, design_color, design_label

#: The two decision-relevant hazard-axis pairs shown in the main text, as
#: ``(x_axis, y_axis)``: the dry-severity vs wet-severity trade-off plane and
#: the drought intensity-vs-kinetics plane.
PANEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("drought_deficit_volume", "flood_peak_magnitude"),
    ("drought_peak_depth", "drought_onset_rate"),
)

#: Default staged candidate-pool slug (matches the SI overlay driver).
DEFAULT_POOL_SLUG = "statpool_10yr_n1000000_d0"


def _search_layers(pool_slug: str) -> list[OverlayLayer]:
    """Load the realized search ensembles as overlay layers, in design order.

    Slugs come from ``NYCOPT_OVERLAY_SEARCH_SLUGS`` when set; otherwise every
    staged dir (other than the pool) carrying a ``hazard_image.npz`` with a
    non-empty selection is used. Each layer's ``name`` is its design key (from
    ``_meta.json``) and its color is the entity-stable design color.

    Args:
        pool_slug: Staged candidate-pool slug (excluded from auto-discovery).
    """
    from scengen.diagnostics import load_hazard_image

    slugs = [s for s in
             os.environ.get("NYCOPT_OVERLAY_SEARCH_SLUGS", "").split(",") if s]
    root = Path(config.STAGED_ENSEMBLE_DIR)
    if not slugs and root.is_dir():
        for d in sorted(root.iterdir()):
            p = d / "hazard_image.npz"
            if d.name == pool_slug or not p.exists():
                continue
            if len(load_hazard_image(p)["selected_rows"]):
                slugs.append(d.name)

    layers = []
    for slug in slugs:
        haz = load_hazard_image(staged_ensemble_dir(slug) / "hazard_image.npz")
        rows = haz["selected_rows"]
        H = haz["H"][rows] if len(rows) else haz["H"]
        design = slug
        meta_path = staged_ensemble_dir(slug) / "_meta.json"
        if meta_path.exists():
            design = json.loads(meta_path.read_text()).get("design") or slug
        layers.append(OverlayLayer(name=design, H=H,
                                   axes=[str(a) for a in haz["hazard_axes"]],
                                   color=design_color(design)))
    order = {d: i for i, d in enumerate(DESIGN_ORDER)}
    layers.sort(key=lambda layer: order.get(layer.name, len(order)))
    return layers


def fig_ensemble_composition(ctx, out_stub: Path,
                             table_dir: Path) -> list[Path]:
    """Figure 4: realized hazard-space composition of the search ensembles.

    Two pairwise hazard-space panels: (a) drought deficit volume vs flood peak
    magnitude, (b) drought peak depth vs drought onset rate. Each overlays the
    candidate pool (grayscale log-density), the robust p1/p99 selection box,
    the realized search ensembles (design-colored scatters), and E_test's
    sub-window cloud (mass contours). The exact per-layer point counts and
    plotted axis ranges go to ``ensemble_composition.csv`` in ``table_dir``.

    Args:
        ctx: Figure context (unused; this builder reads only staged npz files).
        out_stub: Output path without extension.
        table_dir: Directory for the companion CSV.

    Returns:
        The written figure paths (PNG + PDF).

    Raises:
        FileNotFoundError: If the pool or E_test hazard image, or every staged
            search ensemble, is absent (staged data is Anvil-side).
    """
    from scengen.diagnostics import load_hazard_image
    from scengen.subsample import ROBUST_HI_PCT, ROBUST_LO_PCT
    from src.etest import E_TEST_VARIANT, get_etest_variant

    pool_slug = os.environ.get("NYCOPT_OVERLAY_POOL_SLUG", DEFAULT_POOL_SLUG)
    pool_path = staged_ensemble_dir(pool_slug) / "hazard_image.npz"
    if not pool_path.exists():
        raise FileNotFoundError(
            f"candidate-pool hazard image not found: {pool_path} "
            f"(stage the pool with workflow step 02 first)")
    variant = get_etest_variant(E_TEST_VARIANT)
    etest_path = staged_ensemble_dir(variant.slug) / "hazard_image_subwindows.npz"
    if not etest_path.exists():
        raise FileNotFoundError(
            f"E_test sub-window hazard image not found: {etest_path} "
            f"(stage E_test, then run "
            f"scripts/main/compute_etest_hazard_image.py)")

    pool = load_hazard_image(pool_path)
    pool_H, pool_axes = pool["H"], [str(a) for a in pool["hazard_axes"]]
    with np.load(etest_path, allow_pickle=True) as et:
        etest_H = et["H"]
        etest_axes = [str(a) for a in et["hazard_axes"]]

    layers = _search_layers(pool_slug)
    if not layers:
        raise FileNotFoundError(
            f"no staged search ensembles with a selected hazard_image.npz "
            f"under {config.STAGED_ENSEMBLE_DIR} (workflow steps 02-03)")
    missing_designs = [d for d in DESIGN_ORDER
                       if d not in {layer.name for layer in layers}
                       and not style.design_style(d)["reference"]]
    if missing_designs:
        print(f"[fig04] warning: no staged search ensemble for "
              f"{missing_designs}; drawing the rest")

    needed = sorted({a for pair in PANEL_PAIRS for a in pair})
    for name, axis_names in (("pool", pool_axes), ("etest", etest_axes),
                             *((layer.name, layer.axes) for layer in layers)):
        absent = [a for a in needed if a not in axis_names]
        if absent:
            raise ValueError(f"hazard image '{name}' lacks axes {absent}; "
                             f"recompute it")

    # Per-axis panel limits spanning every drawn layer, padded 3%.
    lims: dict[str, tuple[float, float]] = {}
    for a in needed:
        v = np.concatenate(
            [_col(pool_H, pool_axes, a), _col(etest_H, etest_axes, a)]
            + [_col(layer.H, layer.axes, a) for layer in layers])
        lo, hi = float(v.min()), float(v.max())
        pad = 0.03 * (hi - lo or 1.0)
        lims[a] = (lo - pad, hi + pad)
    box = {a: tuple(np.percentile(_col(pool_H, pool_axes, a),
                                  [ROBUST_LO_PCT, ROBUST_HI_PCT]))
           for a in needed}

    fig, axes = panel_grid(1, len(PANEL_PAIRS), panel_aspect=0.95)
    rows = []
    for k, (ax, (ax_x, ax_y)) in enumerate(zip(np.ravel(axes), PANEL_PAIRS)):
        draw_pair_panel(ax, pool_H, pool_axes, layers, etest_H, etest_axes,
                        ax_x, ax_y, xlim=lims[ax_x], ylim=lims[ax_y],
                        box_x=box[ax_x], box_y=box[ax_y], scatter_size=9.0)
        ax.set_xlabel(HAZARD_AXIS_LABELS.get(ax_x, ax_x))
        ax.set_ylabel(HAZARD_AXIS_LABELS.get(ax_y, ax_y))
        letter = chr(ord("a") + k)
        panel_label(ax, letter)
        for lname, H, laxes in (
                [("candidate_pool", pool_H, pool_axes)]
                + [(layer.name, layer.H, layer.axes) for layer in layers]
                + [("etest", etest_H, etest_axes)]):
            x, y = _col(H, laxes, ax_x), _col(H, laxes, ax_y)
            rows.append({"panel": letter, "x_axis": ax_x, "y_axis": ax_y,
                         "layer": lname, "n_points": int(len(x)),
                         "x_min": float(x.min()), "x_max": float(x.max()),
                         "y_min": float(y.min()), "y_max": float(y.max())})

    handles = [
        Line2D([], [], marker="s", ls="", color="0.7",
               label="Candidate pool (log density)"),
        Line2D([], [], color="0.25", ls="--", lw=1.0,
               label=f"Pool p{ROBUST_LO_PCT:g}/p{ROBUST_HI_PCT:g} "
                     f"selection box"),
    ] + [
        Line2D([], [], marker="o", ls="", color=layer.color, markersize=6,
               markeredgecolor="white", label=design_label(layer.name))
        for layer in layers
    ] + [
        Line2D([], [], color=LAYER_COLORS["etest"], lw=1.6,
               label="E_test sub-windows (50/90/99% mass)"),
    ]
    shared_legend(fig, handles, ncol=2)

    pd.DataFrame(rows).to_csv(table_dir / "ensemble_composition.csv",
                              index=False)
    written = style.save_manuscript_figure(fig, out_stub)
    plt.close(fig)
    return written
