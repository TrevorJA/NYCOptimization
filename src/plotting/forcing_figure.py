"""
forcing_figure.py - Manuscript Figure 3: the deeply uncertain forcing space.

Composes the four panels of ``src.plotting.forcing_space`` (harmonic model,
fitted CMIP6 parameters + sampled box, implied monthly change factors, change in
the flow duration curve); rendered via the registry (``scripts/main/figures.py``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from src.plotting import forcing_space as fspace
from src.plotting import style

#: Smallest type size in this figure (points), above the project floor
#: :data:`style.MIN_FONTSIZE`: the four square panels are
#: reduced further on the page than the single-row figures.
MIN_FONTSIZE: int = 14

#: Figure size (inches). Each panel is forced square, so the height is set to
#: what four square panels and their labels need at :data:`MIN_FONTSIZE`;
#: a taller figure only opens a gap between the rows.
FIGSIZE: tuple[float, float] = (11.0, 10.0)


def build_forcing_space(ctx, out_stub: Path, table_dir: Path) -> list[Path]:
    """Figure 3: construction of the deeply uncertain forcing space.

    Four square panels reading left to right, top to bottom, as the
    parameterization is built and then checked, each carrying its own legend:

        (a) the harmonic model, one term at a time, on the best-fitting CMIP6 run
        (b) the fitted CMIP6 parameters, the E_test draws, and the sampling box
        (c) the monthly change factors the sample implies, against the CMIP6 fits
        (d) the flow duration curves of the E_test realizations, against the
            historical record and that record scaled by each CMIP6 run's
            monthly change factors

    Panels (a)-(c) come from the CMIP6 change-factor table and the as-built
    ``forcing_profiles.npz``; panel (d) reads the E_test rows and the historical
    record from the FDC cache produced by ``scripts/main/forcing_fdc_cache.py``,
    keeping only the staged E_test's own realizations, and scales the record by
    the CMIP6 change factors at plot time. The cache must have been built on
    that E_test or on the generated parent it was cut from, under the same
    forcing hash.
    """
    import json

    from src import etest as etest_mod
    from scripts.main.forcing_fdc_cache import DEFAULT_CACHE

    slug = etest_mod.E_TEST_VARIANTS[etest_mod.E_TEST_VARIANT].slug
    ensemble_dir = config.STAGED_ENSEMBLE_DIR / slug
    if not ensemble_dir.is_dir():
        raise FileNotFoundError(f"staged E_test not found: {ensemble_dir}")
    if not DEFAULT_CACHE.is_file():
        raise FileNotFoundError(
            f"FDC cache not found: {DEFAULT_CACHE}\n"
            "Build it first: sbatch workflow/13_main_figures.sh (or "
            "python3 -m scripts.main.forcing_fdc_cache)"
        )
    meta = json.loads((ensemble_dir / "_meta.json").read_text())

    fits = fspace.load_cmip6_fits()
    sample = fspace.load_etest_sample(ensemble_dir)
    pctl = fspace.envelope_pctl()

    # The floor covers the save too: tick labels are sized when they are drawn.
    with plt.rc_context(style.font_rc(MIN_FONTSIZE)):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
        fspace.panel_harmonic_decomposition(axes[0, 0], fits)
        fspace.panel_parameter_space(axes[0, 1], fits, sample)
        fspace.panel_monthly_change(axes[1, 0], fits, sample, pctl)
        with np.load(DEFAULT_CACHE, allow_pickle=False) as cache:
            built_on = str(cache["ensemble_slug"])
            if built_on not in {slug, meta.get("subset_of")}:
                raise ValueError(
                    f"FDC cache was built on {built_on!r}, which is neither E_test "
                    f"{slug!r} nor its parent {meta.get('subset_of')!r}."
                )
            if str(cache["forcing_hash"]) != meta.get("forcing_hash"):
                raise ValueError(
                    f"FDC cache forcing_hash {str(cache['forcing_hash'])!r} does not "
                    f"match the staged E_test's {meta.get('forcing_hash')!r}."
                )
            cmip6_fdc = fspace.cmip6_shifted_historic_fdcs(
                fits, cache["exceedance"], tuple(str(n) for n in cache["nyc_nodes"]))
            fspace.panel_flow_duration(axes[1, 1], cache, cmip6_fdc, pctl,
                                       realization_ids=sample["realization_ids"])

        for ax in axes.flat:
            ax.set_box_aspect(1)

        print(f"[fig03] {len(fits['df'])} CMIP6 runs, {sample['n_sow']} SOWs "
              f"({sample['realization_ids'].size} realizations in panel d; cache "
              f"built on {built_on}), envelope={pctl}, "
              f"median shape R2={fits['df'].shape_R2.median():.2f}")
        written = style.save_manuscript_figure(fig, out_stub)
        plt.close(fig)
    return written
