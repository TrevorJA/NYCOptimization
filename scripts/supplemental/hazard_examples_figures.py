"""hazard_examples_figures.py - The hazard metrics on example HF realizations.

Renders the hazard-metric illustration of ``src/plotting/hazard_examples.py``
in each geometry of ``HEX_GEOMETRIES``: the HF search ensemble's hazard
characteristics with a few example realizations highlighted (left) and each
example's SSI-6 series and annual peak discharge (right). Reads the staged HF
ensemble of draw ``NYCOPT_ENSEMBLE_DRAW`` (hazard image + daily traces); no
simulation, no pool.

Settings in ``supplemental_config.py`` (``HEX_*``); outputs under
``outputs/supplemental/hazard_examples/{figures,tables}``. Run after workflow
steps 02-03 have staged the HF ensemble::

    python scripts/supplemental/hazard_examples_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_hex_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src.plotting import hazard_examples as hex_  # noqa: E402
from src.plotting.style import apply_manuscript_style, save_figure  # noqa: E402


def write_examples_table(data: dict, targets: list, path: Path) -> None:
    """The chosen examples: identity, ensemble row, pool id, target, and each
    selection axis's value and ensemble percentile."""
    H, axes = data["H"], data["axes"]
    rows = []
    for k, (i, g, t) in enumerate(zip(data["chosen"], data["global_ids"], targets)):
        row = {"example": k, "color": hex_.EXAMPLE_COLORS[k],
               "marker": hex_.EXAMPLE_MARKERS[k], "ensemble_row": i,
               "pool_realization_id": g, "target": json.dumps(t)}
        for a in config.HAZARD_SELECTION_AXES:
            row[a] = H[i, axes.index(a)]
            row[f"{a}_pctl"] = hex_.ensemble_percentile(H, axes, a)[i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    slug = hex_.hf_ensemble_slug()
    missing = [p for p in hex_.required_files(slug) if not p.exists()]
    if missing:
        print(f"[hex] staged HF ensemble files not found (workflow steps 02-03): {missing}")
        sys.exit(1)

    targets = hex_.validate_targets(scfg.HEX_EXAMPLE_TARGETS)
    data = hex_.load_examples(slug, targets)
    print(f"[hex] '{slug}': N={data['H'].shape[0]}, example rows={data['chosen']} "
          f"(pool ids {data['global_ids']})")

    scfg.HEX_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    scfg.HEX_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    write_examples_table(data, targets, scfg.HEX_TABLES_DIR / f"examples_{slug}.csv")

    apply_manuscript_style()
    for geometry in scfg.HEX_GEOMETRIES:
        fig = hex_.build_hazard_examples_figure(
            data["H"], data["axes"], data["sequences"],
            left=geometry, triple=scfg.HEX_SCATTER_TRIPLE,
        )
        stub = scfg.HEX_FIGURES_DIR / f"hazard_examples_{geometry}_{slug}"
        # Fixed canvas: the builder's margins hold every label, and the tight
        # bounding box would crop a 3-D axes' z label.
        with plt.rc_context({"savefig.bbox": None}):
            save_figure(fig, stub)
        plt.close(fig)
        print(f"[hex]   -> {stub}.png")


if __name__ == "__main__":
    main()
