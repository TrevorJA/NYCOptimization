"""explore_results.py - Select and visualize solutions from a Pareto front.

Screens one completed search's reference set against the FFMP baseline, picks
representatives with :mod:`src.solution_selection`, and renders the figure suite.

Figures (under ``outputs/figures/_exploratory/{scenario}/{slug}/explore/``):

    explore_01_baseline_dominance.png        front vs baseline on parallel axes
    explore_02_selected_policies.png         the chosen representatives highlighted
    explore_03_objective_tradeoffs.png       Spearman trade-off structure
    explore_04_policy_rules_*.png            operating rules of each representative
    explore_05_historic_timeseries.png       simulated behaviour vs the baseline
    explore_06_historic_timeseries_drought.png   the same, zoomed on the 1960s drought
    explore_07_tradeoff_scatter.png          headline pairwise tradeoff scatters
    explore_08_tradeoff_scatter_matrix.png   lower-triangle scatter matrix
    explore_09_dv_ranges.png                 bound-normalized DV ranges of
                                             criterion-satisfying subsets vs the front

Tables (same directory):

    selected_solutions.csv     one row per selection rule: reference-set row
                               index, rule, natural-scale objective vector
    selected_dv_distances.csv  pairwise bound-normalized DV distance between the
                               highlighted policies

Figures 05/06 read the full-model HDF5 runs written by
``workflow/supplemental/sim_selected_policies.sh`` (``src.policy_simulation``)
under ``outputs/{scenario}/{slug}/timeseries/selected_policies/``; the baseline
and the highlighted representatives are simulated by that job in one model mode.

Usage (from repo root, venv active):
    python3 -m scripts.main.explore_results --slug <slug>     # selection, cheap figures
    sbatch --export=ALL,NYCOPT_ENV_FILE=<scenario env> \
        workflow/supplemental/sim_selected_policies.sh --slug <slug>   # HDF5 runs
    python3 -m scripts.main.explore_results --slug <slug>     # adds figures 05/06
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import config
from src.formulations import get_bounds, get_n_vars
from src.plotting.style import apply_style
from src.solution_selection import (
    Selection, best_single, compromise_scores, dominance_mask,
    n_objectives_beaten, normalized_dv, pairwise_distances, select_by_rules,
    select_diverse,
)

#: Compromise rules reported for every run. Each entry is
#: ``(label, rule, method, kwargs)``; all are direction-aware and work for any
#: objective set, so nothing here is specific to the FFMP 8-objective problem.
COMPROMISE_RULES = [
    ("Balanced (mean scaled)", "mean_scaled", "mean_scaled", {}),
    ("Compromise L1", "distance_to_ideal_p1", "distance_to_ideal", {"p": 1}),
    ("Compromise L2", "distance_to_ideal_p2", "distance_to_ideal", {"p": 2}),
    ("Minimax regret", "distance_to_ideal_pinf", "distance_to_ideal",
     {"p": float("inf")}),
    ("Best worst-case", "maximin", "maximin", {}),
]

#: Timeseries panels only need these results keys/columns of the HDF5 runs.
_CACHE_SPEC = {
    "res_storage": None,          # filled with config.NYC_RESERVOIRS
    "major_flow": ["delMontague", "delTrenton"],
    "ibt_demands": ["demand_nyc"],
    "ibt_diversions": ["delivery_nyc"],
}

#: Drought of record on the historic trace: the window where the policies
#: differ most, so the full-record figure gets a companion zoom (shared with
#: the policy-dynamics figures).
from src.plotting.policy_dynamics import DROUGHT_PERIODS
DROUGHT_WINDOW = DROUGHT_PERIODS["drought_1960s"]


def _slugify(label: str) -> str:
    """Filesystem-safe stem for a policy label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _load_baseline(formulation: str, scenario: str,
                   obj_names: list) -> np.ndarray:
    """Scenario-matched baseline vector aligned to the front's objectives.

    Resolves through ``config.baseline_objectives_csv`` so an ensemble
    scenario reads the baseline scored on its own search ensemble (step 05
    ``--search-ensemble``), never the historic-record vector. The header must
    equal the active objective set exactly (hard error).

    Raises:
        FileNotFoundError: If the scenario's baseline CSV is missing.
        ValueError: If its header is not exactly ``obj_names``, in order.
    """
    path = config.baseline_objectives_csv(formulation, scenario)
    if not path.exists():
        raise FileNotFoundError(
            f"baseline objectives not found: {path}\n"
            f"Score the baseline on this scenario's search ensemble first:\n"
            f"  sbatch --export=ALL,NYCOPT_ENV_FILE=<scenario env>,"
            f"NYCOPT_BASELINE_SKIP_REEVAL=1 \\\n"
            f"      workflow/05_run_baseline.sh --search-ensemble\n"
            f"(or pass --no-baseline to skip the comparison)")
    df = pd.read_csv(path)
    if list(df.columns) != list(obj_names):
        raise ValueError(
            f"{path}: baseline objective columns do not match the active "
            f"objective set.\n  baseline: {list(df.columns)}\n"
            f"  expected: {list(obj_names)}\n"
            f"Re-run workflow/05_run_baseline.sh under this objective set."
        )
    return df.iloc[0][list(obj_names)].to_numpy(dtype=float)


def _report_dominance(natural, baseline, directions, obj_names) -> np.ndarray:
    """Print the baseline comparison and return the dominance mask."""
    mask = dominance_mask(natural, baseline, directions)
    beaten = n_objectives_beaten(natural, baseline, directions)
    n_objs = len(obj_names)
    strict = int((beaten == n_objs).sum())
    print(f"front size                       : {natural.shape[0]}")
    print(f"dominate the baseline (weak)     : {int(mask.sum())}")
    print(f"beat it on all {n_objs} with no ties : {strict}")
    print(f"objectives-beaten distribution   : "
          f"{np.bincount(beaten, minlength=n_objs + 1).tolist()}")
    for name, val, d in zip(obj_names, baseline, directions):
        worst = "worst possible" if (d == 1 and val == 0.0) else ""
        print(f"  baseline {name:34s} = {val:10.4f} {worst}")
    return mask


def _build_selections(natural, directions, obj_names, dom_mask,
                      n_diverse: int) -> tuple:
    """Assemble the full selection table and the highlighted subset.

    Returns:
        ``(all_selections, highlighted)``. ``all_selections`` reports every
        rule at its own true optimum (rules may collide — a collision is
        information). ``highlighted`` is the distinct, spread-out subset that
        gets plotted and simulated.
    """
    all_sel: list[Selection] = []
    for k, name in enumerate(obj_names):
        idx = best_single(natural, directions, k)
        all_sel.append(Selection(f"Best {name}", f"best_{name}", idx))
    for label, rule, method, kwargs in COMPROMISE_RULES:
        scores = compromise_scores(natural, directions, method=method, **kwargs)
        all_sel.append(Selection(label, rule, int(np.argmax(scores))))

    # The highlighted set: the balanced compromise plus spread-out members of
    # the baseline-dominating subset, so the plotted policies are genuinely
    # different operating strategies rather than neighbours on one axis. Falls
    # back to the whole front when nothing dominates the baseline. A None
    # dom_mask means no baseline screening at all (ensemble-evaluated fronts
    # have no comparable baseline vector): spread over the whole front.
    if dom_mask is None:
        pool = np.arange(natural.shape[0])
    else:
        pool = np.where(dom_mask)[0]
        if pool.size == 0:
            print("[selection] nothing dominates the baseline — "
                  "spreading over the whole front instead")
            pool = np.arange(natural.shape[0])
    balanced = compromise_scores(natural, directions, method="mean_scaled")
    seed = int(np.argmax(np.where(dom_mask, balanced, -np.inf))
               if dom_mask is not None and dom_mask.any()
               else np.argmax(balanced))
    spread = select_diverse(natural, directions, n_diverse,
                            seed_indices=[seed], candidates=pool)
    first_label = "Balanced" if dom_mask is None else "Balanced (dominating)"
    first_rule = ("mean_scaled_balanced" if dom_mask is None
                  else "mean_scaled_dominating")
    highlighted = [
        Selection(first_label if i == 0 else f"Diverse {i}",
                  first_rule if i == 0 else f"diverse_{i}", idx)
        for i, idx in enumerate(spread)
    ]
    all_sel.extend(highlighted)
    return all_sel, highlighted


def _write_selection_csv(path: Path, selections, highlighted, natural,
                         obj_names, baseline, directions) -> pd.DataFrame:
    """Write one row per selection: index, rule, and natural objective vector.

    ``highlighted`` flags the plotted (and simulated) subset so downstream
    consumers (``src.policy_simulation.read_selection_labels``) can recover it
    without re-running the selection. The baseline-comparison columns are
    omitted when ``baseline`` is None (no comparable baseline vector for
    ensemble-evaluated fronts).
    """
    highlighted_rules = {sel.rule for sel in highlighted}
    if baseline is not None:
        beaten = n_objectives_beaten(natural, baseline, directions)
        dom = dominance_mask(natural, baseline, directions)
    rows = []
    for sel in selections:
        row = {"row_index": sel.index, "rule": sel.rule, "label": sel.label,
               "highlighted": sel.rule in highlighted_rules}
        if baseline is not None:
            row["dominates_baseline"] = bool(dom[sel.index])
            row["n_objectives_beaten"] = int(beaten[sel.index])
        row.update({n: float(v) for n, v in zip(obj_names, natural[sel.index])})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[table] {path}  ({len(df)} rows)")
    return df


def _write_dv_distances(path: Path, dv, selections, formulation: str) -> None:
    """Pairwise bound-normalized DV distance between highlighted policies."""
    if len(selections) < 2:
        return
    idx = [s.index for s in selections]
    nd = normalized_dv(dv[idx], get_bounds(formulation))
    labels = [f"{s.label} (row {s.index})" for s in selections]
    d = pd.DataFrame(pairwise_distances(nd), index=labels, columns=labels)
    d.to_csv(path)
    off = d.to_numpy()[np.triu_indices(len(idx), k=1)]
    print(f"[table] {path}  (normalized DV distance: min {off.min():.3f}, "
          f"max {off.max():.3f}, {len(idx)} policies in "
          f"{get_n_vars(formulation)}-D DV space)")


# --------------------------------------------------------------------------- #
# Timeseries traces (full-model HDF5 runs written by src.policy_simulation)
# --------------------------------------------------------------------------- #
def _reduce_results(data: dict) -> dict:
    """Keep only the columns the timeseries figure reads."""
    spec = dict(_CACHE_SPEC)
    spec["res_storage"] = list(config.NYC_RESERVOIRS)
    return {key: data[key][cols].copy() for key, cols in spec.items()}


def _load_traces(run_dir: Path, selections, baseline_label: str) -> tuple:
    """Load the baseline and highlighted traces from a policy-set run directory.

    Policies are keyed by reference-set row (``row_{index}``; ``baseline`` for
    the FFMP policy), so relabelling a selection never orphans a trace.

    Returns:
        ``(results, model_mode)`` with ``results`` ordered baseline first, or
        ``({}, None)`` when the directory or any trace is missing.
    """
    from src.policy_simulation import (BASELINE_NAME, MANIFEST_NAME, PolicySet,
                                       load_policy_runs)

    wanted = [(baseline_label, BASELINE_NAME)]
    wanted += [(f"{s.label} (row {s.index})", PolicySet.row_name(s.index))
               for s in selections]
    if not (run_dir / MANIFEST_NAME).exists():
        return {}, None
    manifest, runs = load_policy_runs(run_dir, results_sets=list(_CACHE_SPEC),
                                      policies=[name for _l, name in wanted])
    missing = [name for _l, name in wanted if name not in runs]
    if missing:
        print(f"[timeseries] runs missing under {run_dir}: {missing}")
        return {}, None
    modes = set(manifest["model_mode"])
    if len(modes) > 1:
        raise ValueError(f"traces under {run_dir} mix model modes "
                         f"{sorted(modes)}; re-run the policy-set job so every "
                         f"trace shares one")
    results = {label: _reduce_results(runs[name]) for label, name in wanted}
    return results, modes.pop()


#: Figure names selected by ``--figures``; None renders the whole suite.
_FIGURE_FILTER: set | None = None


def _figure(name: str, fn, out_dir: Path) -> None:
    """Render one figure, reporting rather than aborting the suite on failure."""
    import matplotlib.pyplot as plt
    if _FIGURE_FILTER is not None and name not in _FIGURE_FILTER:
        return
    try:
        fn(out_dir / name)
        print(f"[figure] {out_dir / name}.png")
    except Exception as exc:                      # one panel must not block the rest
        import traceback
        print(f"[figure] FAILED {name}: {exc}")
        traceback.print_exc()
    finally:
        plt.close("all")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--slug", default="ffmp_obj8_mm_full")
    ap.add_argument("--formulation", default="ffmp")
    ap.add_argument("--set-file", default=None,
                    help="Reference set override (default: the run's merged set)")
    ap.add_argument("--out-dir", default=None,
                    help="Figure directory override (default: the exploratory "
                         "tree, outputs/figures/_exploratory/{scenario}/{slug}/explore)")
    ap.add_argument("--n-diverse", type=int, default=3,
                    help="Highlighted representatives to spread over the front")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip every baseline-objective comparison (figure 01, "
                         "overlays, CSV columns). The default loads the "
                         "scenario-matched baseline from "
                         "config.baseline_objectives_csv — for ensemble "
                         "scenarios that is the search-ensemble-scored vector "
                         "(step 05 --search-ensemble), never the historic-"
                         "record one.")
    ap.add_argument("--figures", nargs="*", default=None,
                    help="Render only these figure names (e.g. "
                         "explore_01_baseline_dominance); default: all")
    ap.add_argument("--skip-timeseries", action="store_true",
                    help="Never render the simulation-dependent timeseries figures")
    ap.add_argument("--timeseries-dir", default=None,
                    help="Policy-set run directory holding the HDF5 traces "
                         "(default outputs/{scenario}/{slug}/timeseries/"
                         "selected_policies, written by "
                         "workflow/supplemental/sim_selected_policies.sh)")
    args = ap.parse_args()
    global _FIGURE_FILTER
    _FIGURE_FILTER = set(args.figures) if args.figures else None

    from src.solution_selection import load_natural_front

    set_file = Path(args.set_file) if args.set_file else (
        Path("outputs") / args.scenario / args.slug / "sets"
        / f"{args.slug}_merged.set")
    import config
    out_dir = Path(args.out_dir) if args.out_dir else config.figure_dir_for(
        args.scenario, args.slug, "explore")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"set file : {set_file}")
    print(f"out dir  : {out_dir}\n")
    dv, natural, obj_names, directions = load_natural_front(set_file,
                                                            args.formulation)
    if args.no_baseline:
        baseline, dom_mask = None, None
        print(f"front size                       : {natural.shape[0]}")
        print("[baseline] skipped (--no-baseline)")
    else:
        baseline = _load_baseline(args.formulation, args.scenario, obj_names)
        print(f"[baseline] "
              f"{config.baseline_objectives_csv(args.formulation, args.scenario)}")
        dom_mask = _report_dominance(natural, baseline, directions, obj_names)
    print()

    all_sel, highlighted = _build_selections(natural, directions, obj_names,
                                             dom_mask, args.n_diverse)
    for s in highlighted:
        print(f"[selection] {s.rule:26s} -> row {s.index}")
    print()

    _write_selection_csv(out_dir / "selected_solutions.csv", all_sel,
                         highlighted, natural, obj_names, baseline, directions)
    _write_dv_distances(out_dir / "selected_dv_distances.csv", dv, highlighted,
                        args.formulation)

    apply_style()
    from src.plotting.front_overview import (plot_baseline_dominance,
                                             plot_objective_tradeoffs,
                                             plot_selected_policies)

    if baseline is not None:
        _figure("explore_01_baseline_dominance",
                lambda p: plot_baseline_dominance(natural, directions,
                                                  obj_names, baseline,
                                                  output_file=p), out_dir)
    _figure("explore_02_selected_policies",
            lambda p: plot_selected_policies(natural, directions, obj_names,
                                             highlighted, baseline=baseline,
                                             output_file=p), out_dir)
    _figure("explore_03_objective_tradeoffs",
            lambda p: plot_objective_tradeoffs(natural, directions, obj_names,
                                               output_file=p), out_dir)

    from src.plotting.dv_ranges import default_criteria, plot_dv_ranges
    from src.plotting.tradeoff_scatter import (plot_key_tradeoffs,
                                               plot_scatter_matrix)
    color_by = ("nyc_delivery_reliability_annual"
                if "nyc_delivery_reliability_annual" in obj_names
                else obj_names[0])
    _figure("explore_07_tradeoff_scatter",
            lambda p: plot_key_tradeoffs(natural, obj_names, directions,
                                         color_by=color_by, baseline=baseline,
                                         output_file=p), out_dir)
    _figure("explore_08_tradeoff_scatter_matrix",
            lambda p: plot_scatter_matrix(natural, obj_names, directions,
                                          color_by=color_by, baseline=baseline,
                                          output_file=p), out_dir)
    criteria = default_criteria(natural, obj_names, directions,
                                baseline=baseline, dom_mask=dom_mask)
    if criteria:
        _figure("explore_09_dv_ranges",
                lambda p: plot_dv_ranges(dv, args.formulation, criteria,
                                         output_file=p), out_dir)
    else:
        print("[figure] skipping explore_09_dv_ranges: no computable criteria "
              "(no baseline and no NYC-storage objective)")

    if args.formulation == "ffmp":
        from src.plotting.policy_rules import plot_policy_rules
        for s in highlighted:
            _figure(f"explore_04_policy_rules_{_slugify(s.label)}",
                    lambda p, s=s: plot_policy_rules(
                        dv[s.index], args.formulation, show_baseline=True,
                        candidate_label=f"{s.label} (row {s.index})",
                        output_file=p), out_dir)
    else:
        print(f"[figure] skipping policy-rule panels: plot_policy_rules is "
              f"tuned for the ffmp panel structure, not '{args.formulation}'")

    if args.skip_timeseries:
        print("\n[timeseries] skipped (--skip-timeseries)")
        return 0

    baseline_label = "FFMP baseline"
    from src.policy_simulation import default_run_dir
    run_dir = (Path(args.timeseries_dir) if args.timeseries_dir
               else default_run_dir(args.scenario, args.slug))
    results, mode = _load_traces(run_dir, highlighted, baseline_label)
    if not results:
        print(f"\n[timeseries] no complete trace set under {run_dir} for the "
              f"current selection; skipped.\n  Produce it with: "
              f"sbatch --export=ALL,NYCOPT_ENV_FILE=<scenario env> "
              f"workflow/supplemental/sim_selected_policies.sh --slug {args.slug}")
        return 0

    from src.plotting.historic_timeseries import plot_historic_timeseries
    _figure("explore_05_historic_timeseries",
            lambda p: plot_historic_timeseries(results, output_file=p,
                                               model_mode=mode,
                                               baseline_label=baseline_label),
            out_dir)
    _figure("explore_06_historic_timeseries_drought",
            lambda p: plot_historic_timeseries(results, output_file=p,
                                               model_mode=mode,
                                               date_range=DROUGHT_WINDOW,
                                               baseline_label=baseline_label),
            out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
