"""
search_diagnostics.py - Standard post-search MOEA diagnostics figures.

Rendered automatically at the end of step 09 (``run_full_diagnostics``) for
every completed MM-Borg run; needs only the search outputs (``sets/``,
``runtime_coordinator/``, ``metrics/``) plus the step-06 baseline CSV and the
seed-1 equal-NFE extraction (``extract_runtime_archive.py --seed 1``), nothing
from the step 10 re-evaluation (that suite lives in
``scripts/main/plot_run_results.py``).

Every convergence line is a seed's coordinator archive (the epsilon-box merge
of its island archives at each snapshot, ``src.diagnostics``); islands are
never drawn individually. The NFE axis is total NFE across islands.

Figures (under ``figures/{scenario}/{slug}/``):

  search_01_parallel_axes_seeds.png
      Each seed's equal-NFE archive (its controller ``.set`` when the seed ran
      to the reporting budget, otherwise its runtime snapshot extraction at
      that budget) overlaid on common axes (up = preferred) with the FFMP
      baseline bold. The baseline vector is the scenario-matched one from
      ``config.baseline_objectives_csv``; scenarios not yet scored draw none.
  search_02_hypervolume_convergence.png
      Coordinator-archive hypervolume vs total NFE, one line per seed, with
      the reporting budget marked.
  search_03_runtime_indicators.png
      All six MOEAFramework runtime indicators vs total NFE (hypervolume,
      generational distance, inverted generational distance, spacing,
      additive epsilon-indicator, maximum Pareto-front error), one panel
      each, one line per seed.

Metrics are scored against the design's diagnostics reference set
(``sets/{slug}_diagnostics_reference.set``), so indicator values are
comparable across seeds of one design and never across designs.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import (ACTIVE_MOEA_CONFIG, FIGURES_DIR, active_scenario_name,
                    baseline_objectives_csv, run_output_dir)
from src.formulations import get_obj_names, get_obj_directions, get_n_vars, get_n_objs
from src.load.reference_set import load_reference_set
from src.plotting.style import apply_style
from src.plotting.hypervolume_convergence import (
    SEED_COLORS, plot_hypervolume_convergence, seed_metrics_files,
    _load_metrics_file, _read_runtime_nfe,
)
from src.runtime_archive import equal_nfe_seed_set

BASELINE_COLOR = "firebrick"

#: The six MOEAFramework indicator columns with preferred direction.
INDICATORS = [
    ("Hypervolume", "↑ better"),
    ("GenerationalDistance", "↓ better"),
    ("InvertedGenerationalDistance", "↓ better"),
    ("Spacing", "↓ more uniform"),
    ("EpsilonIndicator", "↓ better"),
    ("MaximumParetoFrontError", "↓ better"),
]


def reporting_total_nfe() -> int | None:
    """Total NFE of the active config's reporting budget (islands x per-island)."""
    cfg = ACTIVE_MOEA_CONFIG
    if cfg.n_islands is None or cfg.max_evaluations is None:
        return None
    return cfg.n_islands * cfg.max_evaluations


def seed_equal_nfe_sets(slug: str, scenario: str, seeds: list[int]) -> dict[int, Path]:
    """Resolve each seed's equal-NFE archive file under the active config.

    Raises:
        FileNotFoundError: When a seed's file is missing (for a seed run past
            the reporting budget this means ``extract_runtime_archive.py
            --seed N`` has not been run).
    """
    cfg = ACTIVE_MOEA_CONFIG
    sets_dir = run_output_dir(scenario, slug, "sets")
    out = {}
    for seed in seeds:
        path = equal_nfe_seed_set(sets_dir, slug, seed,
                                  cfg.max_evaluations_for_seed(seed),
                                  cfg.max_evaluations)
        if not path.exists():
            hint = (f"run `python scripts/main/extract_runtime_archive.py --seed {seed}`"
                    if path.name.endswith(f"_nfe{cfg.max_evaluations}.set")
                    else "the seed has not completed")
            raise FileNotFoundError(f"{path} missing ({hint})")
        out[seed] = path
    return out


def plot_seed_parallel_axes(
    slug: str,
    scenario: str,
    formulation: str,
    output_file: Path,
    seeds: list[int],
    baseline_csv: Path = None,
):
    """Overlay every seed's equal-NFE archive on parallel axes.

    Follows the repo parallel-coordinates convention (up = preferred);
    normalization spans all seeds plus the baseline. Seed draw order is
    interleaved (fixed permutation) so no seed systematically paints over
    another.
    """
    set_files = seed_equal_nfe_sets(slug, scenario, seeds)

    n_vars = get_n_vars(formulation)
    obj_names = get_obj_names()
    directions = np.array(get_obj_directions())

    labels, groups = [], []
    for seed, f in set_files.items():
        _, objs = load_reference_set(f, n_vars, n_objs=get_n_objs())
        raw = objs.copy()
        raw[:, directions == 1] *= -1.0
        labels.append(f"seed {seed:02d} (n = {len(raw)})")
        groups.append(raw)

    # A baseline vector is only overlaid when it shares the front's evaluation
    # substrate: config.baseline_objectives_csv points at the historic-record
    # vector for historic fronts and at the scenario's search-ensemble-scored
    # vector (step 06 --search-ensemble) otherwise. A scenario not yet scored
    # simply has no file and gets no line — never the historic values.
    baseline_raw = None
    if baseline_csv is None:
        baseline_csv = baseline_objectives_csv(formulation, scenario)
    if baseline_csv.exists():
        header, values = baseline_csv.read_text().splitlines()[:2]
        vals = dict(zip(header.split(","), map(float, values.split(","))))
        if set(obj_names) <= set(vals):
            baseline_raw = np.array([vals[n] for n in obj_names])

    import pandas as pd
    from src.plotting.parallel_coordinates import (custom_parallel_coordinates,
                                                   minmaxs_from_directions,
                                                   wrap_axis_labels)
    rows = np.vstack(groups)
    which = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    order = np.random.default_rng(0).permutation(len(rows))
    frame = pd.DataFrame(rows[order], columns=obj_names)
    frame["seed"] = [labels[i] for i in which[order]]
    color_dict = {lab: SEED_COLORS[i % len(SEED_COLORS)]
                  for i, lab in enumerate(labels)}

    apply_style()
    fig, ax = custom_parallel_coordinates(
        frame, columns_axes=obj_names, axis_labels=wrap_axis_labels(obj_names),
        minmaxs=minmaxs_from_directions(directions),
        color_by_categorical="seed", color_dict_categorical=color_dict,
        alpha_base=0.06, lw_base=0.5, fontsize=13, figsize=(13.5, 5.5),
        baseline=baseline_raw, baseline_label="FFMP baseline")
    ax.set_title(f"Equal-NFE archives by seed, {scenario}/{slug}",
                 loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_indicators(
    metrics_dir: Path,
    slug: str,
    output_file: Path,
    figsize: tuple = (13, 7),
    reporting_nfe: int | None = None,
):
    """Panel of the six runtime indicators vs total NFE, one line per seed."""
    files = seed_metrics_files(metrics_dir, slug)
    if not files:
        raise FileNotFoundError(f"no per-seed .metrics files for {slug} under {metrics_dir}")

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    axes = axes.ravel()

    used_nfe_axis = False
    for i, (seed, mf) in enumerate(files):
        df = _load_metrics_file(mf)
        if df is None or df.empty:
            continue
        nfe = _read_runtime_nfe(mf, len(df))
        x = nfe if nfe is not None else df.index.values
        used_nfe_axis = used_nfe_axis or nfe is not None
        color = SEED_COLORS[i % len(SEED_COLORS)]
        for k, (col, _) in enumerate(INDICATORS):
            if col in df.columns:
                axes[k].plot(x, df[col], color=color, linewidth=1.4,
                             label=f"seed {seed:02d}" if k == 0 else None)

    for k, (col, direction) in enumerate(INDICATORS):
        axes[k].set_title(f"{col}  ({direction})", fontsize=9)
        axes[k].grid(True, alpha=0.3)
        if reporting_nfe is not None and used_nfe_axis:
            axes[k].axvline(reporting_nfe, color="0.4", lw=1.0, ls="--")
    xlabel = ("Function evaluations (total across islands)"
              if used_nfe_axis else "Runtime snapshot index")
    for k in (3, 4, 5):
        axes[k].set_xlabel(xlabel)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Coordinator-archive runtime indicators vs the diagnostics "
                 "reference set, one line per seed", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_search_diagnostics(slug: str, scenario: str = None) -> Path:
    """Render the standard post-search figure suite for one run.

    Called at the end of ``run_full_diagnostics`` (step 09); each figure is
    wrapped so one failure never blocks the others. Returns the figure dir.
    """
    from src.diagnostics import discover_seeds, formulation_for

    if scenario is None:
        scenario = active_scenario_name()
    formulation = formulation_for(slug)
    run_dir = Path("outputs") / scenario / slug
    out_dir = FIGURES_DIR / scenario / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = discover_seeds(slug, scenario=scenario)
    total_nfe = reporting_total_nfe()

    tasks = [
        ("search_01_parallel_axes_seeds.png",
         lambda p: plot_seed_parallel_axes(slug, scenario, formulation, p, seeds)),
        ("search_02_hypervolume_convergence.png",
         lambda p: plot_hypervolume_convergence(run_dir / "metrics", slug, p,
                                                reporting_nfe=total_nfe)),
        ("search_03_runtime_indicators.png",
         lambda p: plot_runtime_indicators(run_dir / "metrics", slug, p,
                                           reporting_nfe=total_nfe)),
    ]
    for name, fn in tasks:
        try:
            fn(out_dir / name)
            print(f"  [figure] {out_dir / name}")
        except Exception as e:
            print(f"  [figure] FAILED {name}: {e}")
    return out_dir
