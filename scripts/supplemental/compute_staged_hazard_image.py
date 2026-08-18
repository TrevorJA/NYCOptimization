"""compute_staged_hazard_image.py - Hazard image of a staged L=SCENARIO_YEARS ensemble.

Scores every realization of a staged search ensemble on the 8-axis candidate hazard
image under the exact pool convention: SSI-6 controlling-event run-theory dry axes on
the monthly aggregate NYC inflow, POT wet axes on the daily series with the leading
``METRIC_EXCLUSION_MONTHS`` cut by date, and the SSI fit / POT threshold / reference
mean fitted once on the full historical record. The scoring loop is the E_test
sub-window scorer (``scripts/main/compute_etest_hazard_image.py``) run with a single
window per realization, so the coordinates are commensurable with the candidate-pool
images, the realized hazard-filling ensembles, and the E_test sub-window image.

Intended for ensembles that never receive a hazard image on their generation path —
``fixed_probabilistic`` stages flows only (its generator skips the image because no
selection happens), yet the realized-composition diagnostics need its coordinates.

Writes ``hazard_image.npz`` into the staged directory in the
``scengen.diagnostics.save_hazard_image`` format with ``selected_rows`` empty: per the
overlay contract, an empty selection means every row IS the ensemble.

Configuration is via environment variables (no CLI value flags):

    NYCOPT_HAZIMG_SLUG   staged slug to score. Default: the active design's
                         ``search_ensemble_slug(SCENARIO_ENSEMBLE_DRAW)``.

Run after the ensemble is staged (workflow step 02)::

    NYCOPT_SCENARIO_DESIGN=fixed_probabilistic \
        python scripts/supplemental/compute_staged_hazard_image.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

from config import ACTIVE_SCENARIO_DESIGN, SCENARIO_ENSEMBLE_DRAW, SCENARIO_YEARS  # noqa: E402
from scripts.main.compute_etest_hazard_image import _reference_series, _score_chunk  # noqa: E402
from src.ensembles import pool_chunk_specs, staged_ensemble_dir  # noqa: E402


def main() -> None:
    """Score the staged ensemble's realizations and persist its hazard image."""
    from scengen.diagnostics import save_hazard_image

    slug = os.environ.get("NYCOPT_HAZIMG_SLUG", "").strip() or (
        ACTIVE_SCENARIO_DESIGN.search_ensemble_slug(SCENARIO_ENSEMBLE_DRAW)
    )
    if not slug:
        raise ValueError(
            f"Active design '{ACTIVE_SCENARIO_DESIGN.name}' has no search-ensemble "
            f"slug; set NYCOPT_HAZIMG_SLUG explicitly."
        )
    out_dir = staged_ensemble_dir(slug)
    meta_path = out_dir / "_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"'{slug}' is not staged ({meta_path} missing). Run workflow step 02 first."
        )
    out_path = out_dir / "hazard_image.npz"
    if out_path.exists():
        print(f"[hazimg] already computed: {out_path}. Delete it to recompute.")
        return

    meta = json.loads(meta_path.read_text())
    L = int(meta.get("realization_years") or meta.get("n_years"))
    if L != SCENARIO_YEARS:
        raise ValueError(
            f"'{slug}' has L={L} yr but the pool convention scores "
            f"SCENARIO_YEARS={SCENARIO_YEARS} yr windows; for longer realizations use "
            f"scripts/main/compute_etest_hazard_image.py (disjoint sub-windows)."
        )
    flowtype = meta.get("flowtype", "pub_nhmv10_BC_withObsScaled")
    reference_monthly, reference_daily = _reference_series(flowtype)

    chunks = pool_chunk_specs(slug)
    print(f"[hazimg] '{slug}': {len(chunks)} chunk(s), 1 window of {L} yr per realization.")
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    axes: list[str] = []
    for spec, gids in chunks:
        H, rid, _win, axes = _score_chunk(
            staged_ensemble_dir(spec.inflow_type), list(range(len(gids))),
            [int(g) for g in gids], 1, reference_monthly, reference_daily,
        )
        parts.append((H, rid))
    H = np.vstack([p[0] for p in parts])
    rid = np.concatenate([p[1] for p in parts])
    order = np.argsort(rid)

    from scengen.hazard_metrics import _REFERENCE_START

    save_hazard_image(
        out_path, H=H[order], hazard_axes=axes,
        realization_ids=rid[order], selected_rows=[],
        reference_start=_REFERENCE_START,
    )
    print(f"[hazimg] wrote {out_path} ({H.shape[0]} rows x {H.shape[1]} axes).")


if __name__ == "__main__":
    main()
