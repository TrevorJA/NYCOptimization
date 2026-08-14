"""
forcing_figure.py - Manuscript Figure 3: the deeply uncertain forcing space.

Composes the four panels of ``src.plotting.forcing_space`` (harmonic model,
fitted CMIP6 parameters + sampled box, implied monthly change factors, flow
duration curves). Moved from the retired ``scripts/main/manuscript_figures.py``
driver into the builder layer; rendering is orchestrated by
``scripts/main/figures.py`` via the registry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from src.plotting import forcing_space as fspace
from src.plotting import style


def build_forcing_space(ctx, out_stub: Path, table_dir: Path) -> list[Path]:
    """Figure 3: construction of the deeply uncertain forcing space.

    Four square panels reading left to right, top to bottom, as the
    parameterization is built and then checked:

        (a) the harmonic model, one term at a time, on a representative CMIP6 run
        (b) the fitted CMIP6 parameters, the E_test draws, and the sampling box
        (c) the monthly change factors the sample implies, against the CMIP6 fits
        (d) the resulting flow duration curves, against raw CMIP6 flows and history

    Panels (a)-(c) come from the CMIP6 change-factor table and the as-built
    ``forcing_profiles.npz``; panel (d) reads the FDC cache produced by
    ``scripts/main/forcing_fdc_cache.py``.
    """
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

    fits = fspace.load_cmip6_fits()
    sample = fspace.load_etest_sample(ensemble_dir)
    pctl = fspace.envelope_pctl()

    fig, axes = plt.subplots(2, 2, figsize=style.FIGSIZE_MANUSCRIPT_2X2,
                             constrained_layout=True)
    fspace.panel_harmonic_decomposition(axes[0, 0], fits)
    fspace.panel_parameter_space(axes[0, 1], fits, sample)
    fspace.panel_monthly_change(axes[1, 0], fits, sample, pctl)
    with np.load(DEFAULT_CACHE, allow_pickle=False) as cache:
        fspace.panel_flow_duration(axes[1, 1], cache, pctl)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    handles = fspace.shared_legend_handles(pctl)
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, handlelength=1.8, columnspacing=1.8)

    print(f"[fig03] {len(fits['df'])} CMIP6 runs, {sample['n_sow']} SOWs, "
          f"envelope={pctl}, median shape R2={fits['df'].shape_R2.median():.2f}")
    written = style.save_manuscript_figure(fig, out_stub)
    plt.close(fig)
    return written
