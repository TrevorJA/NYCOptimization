"""explore_results.py - Select and visualize solutions from a Pareto front.

One driver over one completed search: it screens the reference set against the
FFMP baseline, picks traceable representatives with :mod:`src.solution_selection`,
and renders the figure suite. Everything except the historic-timeseries figure
is pure post-processing and runs instantly on any run (any scenario / slug /
formulation).

Figures (under ``figures/{scenario}/{slug}/``):

    explore_01_baseline_dominance.png   front vs baseline on parallel axes,
                                        the dominating subset highlighted
    explore_02_selected_policies.png    the chosen representatives highlighted
    explore_03_objective_tradeoffs.png  Spearman trade-off structure
    explore_04_policy_rules_*.png       operating rules of each representative
    explore_05_historic_timeseries.png  simulated behaviour vs the baseline
    explore_06_historic_timeseries_drought.png   the same, zoomed on the
                                        1960s drought of record

Tables (same directory):

    selected_solutions.csv     one row per selection rule: the reference-set
                               row index, the rule that picked it, and its full
                               natural-scale objective vector. Those indices are
                               how the policies get referenced in any later
                               re-evaluation, so they are the point of the file.
    selected_dv_distances.csv  pairwise bound-normalized decision-variable
                               distance between the highlighted policies — the
                               check that "distinct" representatives really are
                               operationally distinct.

Simulation:
    Figures 05/06 need a Pywr-DRB run per policy plus one for the baseline
    (~31 s each on Anvil), so they are OFF unless a cached timeseries exists.
    Produce the cache with ``--simulate-timeseries`` from a batch job — see
    ``workflow/supplemental/sim_selected_policies.sh``. Baseline and candidates
    are always simulated together in one job and in one model mode, because the
    persisted ``outputs/baseline/ffmp_baseline.hdf5`` was written with the FULL
    model while search uses the TRIMMED model.

Usage (from repo root, venv active):
    python3 -m scripts.main.explore_results                       # cheap figures
    python3 -m scripts.main.explore_results --skip-timeseries
    sbatch workflow/supplemental/sim_selected_policies.sh         # + timeseries
"""
from __future__ import annotations

import argparse
import os
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

#: Timeseries panels only need these results keys/columns; the cache stores
#: exactly them so a re-render never needs another simulation.
_CACHE_SPEC = {
    "res_storage": None,          # filled with config.NYC_RESERVOIRS
    "major_flow": ["delMontague", "delTrenton"],
    "ibt_demands": ["demand_nyc"],
    "ibt_diversions": ["delivery_nyc"],
}

#: Drought of record on the historic trace — the window where the policies
#: actually differ, so the full-record figure gets a companion zoom.
DROUGHT_WINDOW = ("1961-01-01", "1970-01-01")


def _slugify(label: str) -> str:
    """Filesystem-safe stem for a policy label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _load_baseline(formulation: str, scenario: str,
                   obj_names: list) -> np.ndarray:
    """Scenario-matched baseline vector aligned to the front's objectives.

    Resolves through ``config.baseline_objectives_csv`` so an ensemble
    scenario reads the baseline scored on ITS search ensemble (step 05
    ``--search-ensemble``) — never the historic-record vector, which is not
    comparable to ensemble-evaluated fronts. Stale baselines from earlier
    objective sets exist in this project's history, so a header that is not
    exactly the active objective set is a hard error rather than a silent
    positional read.

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


def _write_selection_csv(path: Path, selections, natural, obj_names, baseline,
                         directions) -> pd.DataFrame:
    """Write one row per selection: index, rule, and natural objective vector.

    The baseline-comparison columns are omitted when ``baseline`` is None
    (no comparable baseline vector for ensemble-evaluated fronts).
    """
    if baseline is not None:
        beaten = n_objectives_beaten(natural, baseline, directions)
        dom = dominance_mask(natural, baseline, directions)
    rows = []
    for sel in selections:
        row = {"row_index": sel.index, "rule": sel.rule, "label": sel.label}
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
# Timeseries cache (avoids re-simulating just to re-render)
# --------------------------------------------------------------------------- #
def _cache_dir(scenario: str, slug: str) -> Path:
    return Path("outputs") / scenario / slug / "timeseries"


def _reduce_results(data: dict) -> dict:
    """Keep only the columns the timeseries figure reads."""
    spec = dict(_CACHE_SPEC)
    spec["res_storage"] = list(config.NYC_RESERVOIRS)
    return {key: data[key][cols].copy() for key, cols in spec.items()}


def _trace_key(index) -> str:
    """Cache subdirectory for one trace.

    Keyed on the reference-set ROW INDEX (``"baseline"`` for the FFMP policy),
    never the display label: the row index is the solution's identity, so
    relabelling a selection must not silently orphan a simulated trace.
    """
    return "baseline" if index is None else f"row_{int(index)}"


def _save_timeseries(cache: Path, index, data: dict, model_mode: str) -> None:
    out = cache / _trace_key(index)
    out.mkdir(parents=True, exist_ok=True)
    for key, df in _reduce_results(data).items():
        df.to_parquet(out / f"{key}.parquet")
    (out / "model_mode.txt").write_text(f"{model_mode}\n")


def _load_timeseries(cache: Path, traces: list) -> tuple:
    """Load cached traces; returns ``(results, model_mode)`` or ``({}, None)``.

    Args:
        cache: Trace cache directory.
        traces: Sequence of ``(label, index)`` where ``index`` is the
            reference-set row (``None`` for the baseline).
    """
    results, modes = {}, set()
    for label, index in traces:
        d = cache / _trace_key(index)
        if not (d / "res_storage.parquet").exists():
            return {}, None
        results[label] = {key: pd.read_parquet(d / f"{key}.parquet")
                          for key in _CACHE_SPEC}
        mode_file = d / "model_mode.txt"
        modes.add(mode_file.read_text().strip() if mode_file.exists() else "unknown")
    if len(modes) > 1:
        raise ValueError(f"cached traces mix model modes {sorted(modes)} — "
                         f"re-run --simulate-timeseries so every trace shares one")
    return results, modes.pop() if modes else None


def _simulate(dv, selections, formulation: str, use_trimmed: bool,
              baseline_label: str) -> dict:
    """Simulate the baseline and every highlighted policy in ONE process.

    One process, one model mode, one set of cached model dicts — which is the
    only way the baseline trace and the candidate traces are comparable.
    """
    import time

    from src.formulations import get_baseline_values
    from src.simulation import dvs_to_config, run_simulation_inmemory

    jobs = [(baseline_label, None, get_baseline_values(formulation))]
    jobs += [(f"{s.label} (row {s.index})", s.index, dv[s.index])
             for s in selections]
    mode = "trimmed" if use_trimmed else "full"
    results = {}
    for label, index, vector in jobs:
        t0 = time.perf_counter()
        results[label] = (index, run_simulation_inmemory(
            dvs_to_config(np.asarray(vector, dtype=float), formulation),
            use_trimmed=use_trimmed,
        ))
        print(f"[sim] {label}: {time.perf_counter() - t0:.1f}s ({mode} model)",
              flush=True)
    return results


def _figure(name: str, fn, out_dir: Path) -> None:
    """Render one figure, reporting rather than aborting the suite on failure."""
    import matplotlib.pyplot as plt
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
                    help="Figure directory override (default: figures/{scenario}/{slug})")
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
    ap.add_argument("--skip-timeseries", action="store_true",
                    help="Never render the simulation-dependent timeseries figures")
    ap.add_argument("--simulate-timeseries", action="store_true",
                    help="Simulate baseline + representatives and cache the traces "
                         "(batch job only — see workflow/supplemental/)")
    ap.add_argument("--full-model", action="store_true",
                    help="Simulate with the FULL model instead of config.USE_TRIMMED_MODEL")
    ap.add_argument("--force-local-sim", action="store_true",
                    help="Allow --simulate-timeseries outside SLURM (login nodes "
                         "must not run simulations; for laptops/interactive nodes)")
    args = ap.parse_args()

    from src.solution_selection import load_natural_front

    set_file = Path(args.set_file) if args.set_file else (
        Path("outputs") / args.scenario / args.slug / "sets"
        / f"{args.slug}_merged.set")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("figures") / args.scenario / args.slug)
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

    _write_selection_csv(out_dir / "selected_solutions.csv", all_sel, natural,
                         obj_names, baseline, directions)
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
    cache = _cache_dir(args.scenario, args.slug)
    traces = ([(baseline_label, None)]
              + [(f"{s.label} (row {s.index})", s.index) for s in highlighted])

    if args.simulate_timeseries:
        if not os.environ.get("SLURM_JOB_ID") and not args.force_local_sim:
            print("\nERROR: --simulate-timeseries runs Pywr-DRB simulations, which "
                  "must not run on a login node. Submit instead:\n"
                  "  sbatch workflow/supplemental/sim_selected_policies.sh\n"
                  "(or pass --force-local-sim if this really is a compute node).",
                  file=sys.stderr)
            return 2
        use_trimmed = False if args.full_model else config.USE_TRIMMED_MODEL
        results = _simulate(dv, highlighted, args.formulation, use_trimmed,
                            baseline_label)
        cache.mkdir(parents=True, exist_ok=True)
        mode = "trimmed" if use_trimmed else "full"
        for label, (index, data) in results.items():
            _save_timeseries(cache, index, data, mode)
        results = {k: _reduce_results(v) for k, (_i, v) in results.items()}
        print(f"[timeseries] cached traces under {cache}")
    else:
        results, mode = _load_timeseries(cache, traces)
        if not results:
            print(f"\n[timeseries] no cached traces under {cache} for the current "
                  f"selection — skipped.\n  Produce them with: "
                  f"sbatch workflow/supplemental/sim_selected_policies.sh")
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
