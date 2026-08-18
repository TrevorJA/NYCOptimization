"""plot_etest_hazard_overlay.py - Overlay E_test on the pool + search-ensemble hazard image.

Candidate main-text figure: the staged candidate pool's hazard image as a grayscale
density field, the robust p1/p99 selection box, each realized search ensemble as a
categorical scatter layer, and E_test's disjoint 10-yr sub-window cloud as density
contours — all over the campaign hazard selection axes. A companion
``overlap_stats.json`` quantifies per-axis containment (fractions of E_test
sub-windows beyond the pool's robust bounds and outside its hull), feeding the
hazard-restricted composition-sensitivity checks and the generalization claim.

Prerequisites: the pool image (step 02), the staged search-ensemble draws (step 03),
and ``hazard_image_subwindows.npz`` in the staged E_test directory
(``scripts/main/compute_etest_hazard_image.py``).

Configuration is via environment variables (no CLI value flags):

    NYCOPT_ETEST_VARIANT          E_test variant to overlay (default "kn")
    NYCOPT_OVERLAY_POOL_SLUG      staged pool slug (default statpool_10yr_n1000000_d0)
    NYCOPT_OVERLAY_SEARCH_SLUGS   comma-separated staged search-ensemble slugs; empty =
                                  auto-discover staged dirs carrying a hazard_image.npz
                                  with a non-empty selection (hazfill_*/fixedprob_*)
    NYCOPT_OVERLAY_AXES           comma-separated axis subset (default: the campaign
                                  selection set, config.HAZARD_SELECTION_AXES)

Outputs -> ``outputs/supplemental/etest_hazard_overlay/{etest_slug}__{pool_slug}/``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

import config  # noqa: E402
from src.ensembles import staged_ensemble_dir  # noqa: E402
from src.etest import E_TEST_VARIANT, get_etest_variant  # noqa: E402
from src.plotting.etest_hazard_overlay import (  # noqa: E402
    LAYER_COLORS, _FALLBACK_COLORS, OverlayLayer, build_overlay_figure, overlap_stats,
)
from src.plotting.style import apply_style, save_figure  # noqa: E402

POOL_SLUG = os.environ.get("NYCOPT_OVERLAY_POOL_SLUG", "statpool_10yr_n1000000_d0")
SEARCH_SLUGS = [s for s in os.environ.get("NYCOPT_OVERLAY_SEARCH_SLUGS", "").split(",") if s]
AXES_OVERRIDE = [a for a in os.environ.get("NYCOPT_OVERLAY_AXES", "").split(",") if a]


def _discover_search_slugs() -> list[str]:
    """Staged search-ensemble dirs carrying a hazard image with a non-empty selection."""
    from scengen.diagnostics import load_hazard_image

    root = Path(config.STAGED_ENSEMBLE_DIR)
    found = []
    for d in sorted(root.iterdir()):
        p = d / "hazard_image.npz"
        if d.name == POOL_SLUG or not p.exists():
            continue
        if len(load_hazard_image(p)["selected_rows"]):
            found.append(d.name)
    return found


def _layer_for(slug: str, color: str) -> OverlayLayer:
    """Build a search-ensemble overlay layer from a staged dir's hazard image."""
    from scengen.diagnostics import load_hazard_image

    haz = load_hazard_image(staged_ensemble_dir(slug) / "hazard_image.npz")
    rows = haz["selected_rows"]
    H = haz["H"][rows] if len(rows) else haz["H"]
    meta_path = staged_ensemble_dir(slug) / "_meta.json"
    name = slug
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("design") is not None:
            name = f"{meta['design']} (draw {meta.get('draw', '?')})"
    return OverlayLayer(name=name, H=H, axes=haz["hazard_axes"], color=color)


def main() -> None:
    """Build the overlay figure + per-axis containment stats."""
    from scengen.diagnostics import load_hazard_image
    from scengen.subsample import ROBUST_HI_PCT, ROBUST_LO_PCT

    variant = get_etest_variant(E_TEST_VARIANT)
    etest_path = staged_ensemble_dir(variant.slug) / "hazard_image_subwindows.npz"
    if not etest_path.exists():
        print(f"[overlay] E_test sub-window hazard image not found: {etest_path}.\n"
              f"[overlay] Stage E_test, then run "
              f"scripts/main/compute_etest_hazard_image.py first.")
        sys.exit(1)
    pool_path = staged_ensemble_dir(POOL_SLUG) / "hazard_image.npz"
    if not pool_path.exists():
        print(f"[overlay] Pool hazard image not found: {pool_path}.")
        sys.exit(1)

    pool = load_hazard_image(pool_path)
    et = np.load(etest_path, allow_pickle=True)
    if "reference_start" not in et:
        sys.exit(
            f"[overlay] {etest_path} lacks 'reference_start' provenance: it predates "
            f"the truthful January date convention and is stale. Regenerate it with "
            f"scripts/main/compute_etest_hazard_image.py."
        )
    etest_H, etest_axes = et["H"], [str(a) for a in et["hazard_axes"]]

    slugs = SEARCH_SLUGS or _discover_search_slugs()
    fallback = iter(_FALLBACK_COLORS)
    layers = []
    for slug in slugs:
        # Stable identity hue per design (from _meta.json); fallback order otherwise.
        meta_path = staged_ensemble_dir(slug) / "_meta.json"
        design = (json.loads(meta_path.read_text()).get("design")
                  if meta_path.exists() else None)
        color = LAYER_COLORS.get(design) or next(fallback, "#666666")
        layers.append(_layer_for(slug, color))

    axes_names = AXES_OVERRIDE or [
        a for a in config.HAZARD_SELECTION_AXES if a in pool["hazard_axes"]
    ]
    missing = [a for a in axes_names if a not in etest_axes]
    if missing:
        raise ValueError(f"E_test image lacks axes {missing}; recompute it.")

    apply_style()
    fig = build_overlay_figure(
        pool["H"], pool["hazard_axes"], layers, etest_H, etest_axes, axes_names,
        lo_pct=float(ROBUST_LO_PCT), hi_pct=float(ROBUST_HI_PCT),
    )
    out_dir = (Path(config.OUTPUTS_DIR) / "supplemental" / "etest_hazard_overlay"
               / f"{variant.slug}__{POOL_SLUG}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "etest_hazard_overlay")

    stats = overlap_stats(
        pool["H"], pool["hazard_axes"], etest_H, etest_axes, axes_names,
        float(ROBUST_LO_PCT), float(ROBUST_HI_PCT),
    )
    (out_dir / "overlap_stats.json").write_text(json.dumps({
        "etest_slug": variant.slug, "pool_slug": POOL_SLUG,
        "search_slugs": slugs, "axes": axes_names,
        "n_etest_subwindows": int(etest_H.shape[0]),
        "per_axis": stats,
    }, indent=2))
    print(f"[overlay] wrote figure + overlap_stats.json -> {out_dir}")
    for a, s in stats.items():
        print(f"[overlay]   {a}: E_test beyond pool p-hi {s['etest_frac_above_hi']:.3f}, "
              f"below p-lo {s['etest_frac_below_lo']:.3f}, "
              f"outside hull {s['etest_frac_outside_hull']:.3f}")


if __name__ == "__main__":
    main()
