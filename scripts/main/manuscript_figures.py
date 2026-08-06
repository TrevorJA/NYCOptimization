"""Render the main-manuscript figure sequence.

This is the single home for the POLISHED figures that appear in the manuscript
body. Diagnostic and Supporting Information figures stay in
``scripts/supplemental/`` at their own (smaller, denser) style; the two are kept
apart deliberately, because main figures must satisfy the journal's typographic
floor while SI figures are optimised for information density.

The :data:`FIGURES` registry is the source of truth for the sequence: adding a
figure means adding one builder and one registry entry, and ``--all`` picks it
up. Output goes to ``figures/manuscript/`` as both PNG (to look at) and PDF
(to submit); unlike ``outputs/**``, that tree is tracked by git, so a rendered
main figure is a syncable deliverable.

Run through srun/sbatch, never on a login node::

    sbatch workflow/13_main_figures.sh                       # the whole sequence
    python3 -m scripts.main.manuscript_figures --list
    python3 -m scripts.main.manuscript_figures --figure fig03_forcing_space

Figure-level settings come from the environment, never from CLI value flags
(repo convention)::

    NYCOPT_FIG_ENVELOPE_PCTL="5,95"   # envelope definition in panels (c) and (d)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.plotting import forcing_space as fspace
from src.plotting import style

#: Where the manuscript-ready figures land (tracked by git).
MANUSCRIPT_FIGURES_DIR = config.PROJECT_DIR / "figures" / "manuscript"


# ---------------------------------------------------------------------------
# Figure 3 - the deeply uncertain forcing space
# ---------------------------------------------------------------------------

def build_fig03_forcing_space(out_stub: Path) -> list[Path]:
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


#: The manuscript figure sequence. Key = output stem, value = builder.
FIGURES: dict[str, Callable[[Path], list[Path]]] = {
    "fig03_forcing_space": build_fig03_forcing_space,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--figure", action="append", choices=sorted(FIGURES),
                   help="render one figure (repeatable); default is --all")
    p.add_argument("--all", action="store_true", help="render every figure")
    p.add_argument("--list", action="store_true", help="list the sequence and exit")
    p.add_argument("--out-dir", type=Path, default=MANUSCRIPT_FIGURES_DIR)
    args = p.parse_args(argv)

    if args.list:
        for name, fn in sorted(FIGURES.items()):
            print(f"{name:28s} {(fn.__doc__ or '').splitlines()[0]}")
        return 0

    names = sorted(FIGURES) if (args.all or not args.figure) else args.figure

    style.apply_manuscript_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"[manuscript_figures] building {name}")
        for path in FIGURES[name](args.out_dir / name):
            print(f"[manuscript_figures]   wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
