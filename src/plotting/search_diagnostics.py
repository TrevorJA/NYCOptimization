"""
search_diagnostics.py - Standard post-search MOEA diagnostics figures.

Rendered automatically at the end of step 07 (``run_full_diagnostics``) for
every completed MM-Borg run; needs only the search outputs (``runtime/``,
``metrics/``, ``sets/``) plus the step-05 baseline CSV — nothing from the
step 08/09 re-evaluation (that suite lives in
``scripts/main/plot_run_results.py``).

Figures (under ``figures/{scenario}/{slug}/``):

  search_01_parallel_axes_seeds.png
      Per-seed merged Pareto-approximate sets overlaid on common axes (up =
      preferred, raw best/worst annotated at the axis ends) with the FFMP
      baseline bold — cross-seed agreement and dominance of the baseline are
      readable directly. The baseline vector is the scenario-matched one from
      ``config.baseline_objectives_csv`` (historic record for historic runs,
      the search-ensemble-scored vector for ensemble scenario designs);
      scenarios not yet scored draw no baseline.
  search_02_hypervolume_convergence.png
      Hypervolume vs NFE, one line per island x seed.
  search_03_runtime_indicators.png
      All six MOEAFramework runtime indicators vs NFE (hypervolume,
      generational distance, inverted generational distance, spacing,
      additive epsilon-indicator, maximum Pareto-front error), one panel
      each, islands drawn in their seed's color.

Metrics are scored against the cross-seed merged reference set
(src/diagnostics.py), so indicator values are comparable across seeds.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import (active_scenario_name, baseline_objectives_csv,
                    run_output_dir)
from src.formulations import get_obj_names, get_obj_directions, get_n_vars, get_n_objs
from src.load.reference_set import load_reference_set
from src.plotting.style import apply_style
from src.plotting.hypervolume_convergence import (
    plot_hypervolume_convergence, _load_metrics_file, _read_runtime_nfe,
)

#: Per-seed line colors (CVD-safe as adjacent pairs); baseline keeps the
#: repo's bold firebrick.
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
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


def _seed_of(stem: str) -> int:
    m = re.match(r"seed_(\d+)_", stem)
    return int(m.group(1)) if m else 0


def plot_seed_parallel_axes(
    slug: str,
    scenario: str,
    formulation: str,
    output_file: Path,
    baseline_csv: Path = None,
):
    """Overlay every seed's merged Pareto-approximate set on parallel axes.

    Follows the repo parallel-coordinates convention (up = preferred, raw
    min/max annotated at the axis ends); normalization spans all seeds plus
    the baseline. Seed draw order is interleaved (fixed permutation) so no
    seed systematically paints over another.
    """
    sets_dir = run_output_dir(scenario, slug, "sets")
    set_files = sorted(Path(sets_dir).glob(f"{slug}_seed*_merged.set"))
    if not set_files:
        raise FileNotFoundError(f"no per-seed merged sets under {sets_dir}")

    n_vars = get_n_vars(formulation)
    obj_names = get_obj_names()
    directions = np.array(get_obj_directions())
    n_objs = len(obj_names)

    seeds, groups = [], []
    for f in set_files:
        _, objs = load_reference_set(f, n_vars, n_objs=get_n_objs())
        raw = objs.copy()
        raw[:, directions == 1] *= -1.0
        seeds.append(f.stem.replace(f"{slug}_", "").replace("_merged", ""))
        groups.append(raw)

    # A baseline vector is only overlaid when it shares the front's evaluation
    # substrate: config.baseline_objectives_csv points at the historic-record
    # vector for historic fronts and at the scenario's search-ensemble-scored
    # vector (step 05 --search-ensemble) otherwise. A scenario not yet scored
    # simply has no file and gets no line — never the historic values.
    baseline_raw = None
    if baseline_csv is None:
        baseline_csv = baseline_objectives_csv(formulation, scenario)
    if baseline_csv.exists():
        header, values = baseline_csv.read_text().splitlines()[:2]
        vals = dict(zip(header.split(","), map(float, values.split(","))))
        if set(obj_names) <= set(vals):
            baseline_raw = np.array([vals[n] for n in obj_names])

    # One renderer for every parallel-axis figure in the repo. Seed draw order
    # is interleaved (fixed permutation) so no seed systematically paints over
    # another; the categorical legend carries each seed's set size.
    import pandas as pd
    from src.plotting.parallel_coordinates import (custom_parallel_coordinates,
                                                   minmaxs_from_directions,
                                                   wrap_axis_labels)
    rows = np.vstack(groups)
    which = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    order = np.random.default_rng(0).permutation(len(rows))
    seed_labels = [f"{s} (n={len(g)})" for s, g in zip(seeds, groups)]
    frame = pd.DataFrame(rows[order], columns=obj_names)
    frame["seed"] = [seed_labels[i] for i in which[order]]
    color_dict = {lab: SEED_COLORS[i % len(SEED_COLORS)]
                  for i, lab in enumerate(seed_labels)}

    apply_style()
    fig, ax = custom_parallel_coordinates(
        frame, columns_axes=obj_names, axis_labels=wrap_axis_labels(obj_names),
        minmaxs=minmaxs_from_directions(directions),
        color_by_categorical="seed", color_dict_categorical=color_dict,
        alpha_base=0.06, lw_base=0.5, fontsize=13, figsize=(13.5, 5.5),
        baseline=baseline_raw, baseline_label="FFMP baseline")
    ax.set_title(f"Pareto-approximate sets, {scenario}/{slug} "
                 f"(per-seed merged reference sets)", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_indicators(
    metrics_dir: Path,
    output_file: Path,
    figsize: tuple = (13, 7),
):
    """Panel of the six runtime indicators vs NFE, islands colored by seed."""
    metrics_files = sorted(Path(metrics_dir).glob("*.metrics"))
    if not metrics_files:
        raise FileNotFoundError(f"no .metrics files under {metrics_dir}")

    seed_ids = sorted({_seed_of(f.stem) for f in metrics_files})
    color_of = {s: SEED_COLORS[i % len(SEED_COLORS)] for i, s in enumerate(seed_ids)}

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    axes = axes.ravel()

    used_nfe_axis = False
    for mf in metrics_files:
        df = _load_metrics_file(mf)
        if df is None or df.empty:
            continue
        nfe = _read_runtime_nfe(mf, len(df))
        x = nfe if nfe is not None else df.index.values
        used_nfe_axis = used_nfe_axis or nfe is not None
        for k, (col, _) in enumerate(INDICATORS):
            if col in df.columns:
                axes[k].plot(x, df[col], color=color_of[_seed_of(mf.stem)],
                             alpha=0.7, linewidth=1.0)

    for k, (col, direction) in enumerate(INDICATORS):
        axes[k].set_title(f"{col}  ({direction})", fontsize=9)
        axes[k].grid(True, alpha=0.3)
    xlabel = ("NFE (per island)" if used_nfe_axis else "Runtime snapshot index")
    for k in (3, 4, 5):
        axes[k].set_xlabel(xlabel)
    for s in seed_ids:
        axes[0].plot([], [], color=color_of[s], lw=2, label=f"seed {s:02d}")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("MOEA runtime indicators vs cross-seed reference set "
                 "(one line per island)", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_search_diagnostics(slug: str, scenario: str = None) -> Path:
    """Render the standard post-search figure suite for one run.

    Called at the end of ``run_full_diagnostics`` (step 07); each figure is
    wrapped so one failure never blocks the others. Returns the figure dir.
    """
    from src.diagnostics import problem_name_for

    if scenario is None:
        scenario = active_scenario_name()
    formulation = problem_name_for(slug).removeprefix("drb_")
    run_dir = Path("outputs") / scenario / slug
    out_dir = Path("figures") / scenario / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("search_01_parallel_axes_seeds.png",
         lambda p: plot_seed_parallel_axes(slug, scenario, formulation, p)),
        ("search_02_hypervolume_convergence.png",
         lambda p: plot_hypervolume_convergence(run_dir / "metrics", formulation, p)),
        ("search_03_runtime_indicators.png",
         lambda p: plot_runtime_indicators(run_dir / "metrics", p)),
    ]
    for name, fn in tasks:
        try:
            fn(out_dir / name)
            print(f"  [figure] {out_dir / name}")
        except Exception as e:
            print(f"  [figure] FAILED {name}: {e}")
    return out_dir
