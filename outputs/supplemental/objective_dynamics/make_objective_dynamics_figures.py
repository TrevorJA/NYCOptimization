"""make_objective_dynamics_figures.py - Driver for the objective-dynamics figures.

Builds the five "objective-anatomy" figures for the historical single trace:
the default FFMP baseline vs one interpretable, storage-conservative contrasting
policy. For each of the two policies it obtains a full-model historical
simulation (reusing the artifact-of-record baseline HDF5 when it is current,
else simulating and caching), computes the §1 whole-trace **performance
metrics** via ``build_objective_set(config.ACTIVE_OBJECTIVES).compute(data)``,
and renders the figures defined in the sibling ``objective_dynamics`` module.

These §1 whole-trace metrics are the interpretable historical narrative and the
per-realization base of the re-evaluation layer; they are NOT the optimization
objectives. The optimizer targets the annual-unit (§2) versions even on the
historic trace (``simulation.evaluate`` -> ``compute_for_borg_ensemble``), shown
here as the per-water-year "annual-unit view" strips.

Self-contained under ``outputs/supplemental/objective_dynamics/`` (code +
figures are version-controlled; the cached ``*.hdf5`` simulations are not).

Run from the repo root with the project venv active:

    python outputs/supplemental/objective_dynamics/make_objective_dynamics_figures.py

Env toggles (no CLI value flags, per project convention):
    NYCOPT_OBJDYN_NOSTRICT=1   disable the figure-vs-score self-check assertions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE / "figures"
PROJECT_DIR = HERE.parents[2]  # objective_dynamics -> supplemental -> outputs -> root
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src.formulations import (  # noqa: E402
    get_baseline_values,
    get_bounds,
    get_constraint_names,
    get_var_names,
)
from src.objectives import build_objective_set  # noqa: E402
from src.plotting.policy_rules import plot_policy_rules  # noqa: E402
from src.plotting.style import apply_style  # noqa: E402
from src.simulation import (  # noqa: E402
    compute_constraint_violations,
    dvs_to_config,
    run_simulation_to_disk,
)
from src.load.results import load_simulation_results  # noqa: E402

import objective_dynamics as od  # noqa: E402  (sibling module)

FORMULATION = "ffmp"
STRICT = os.environ.get("NYCOPT_OBJDYN_NOSTRICT", "").strip() not in ("1", "true", "True")

# Storage-conservative contrast levers (baseline reproduces the FFMP exactly):
#   * release less toward the Montague/Trenton flow targets, and
#   * enter conservation zones at higher storage (raise the boundary curves).
_CONTRAST_MRF_SCALE = 0.80    # mrf_target_scale_* multiplier (baseline 1.0)
_CONTRAST_ZONE_VSHIFT = 0.03  # zone_vshift_* additive offset, frac of capacity (baseline 0.0)


def build_contrast_dv(formulation: str = FORMULATION) -> np.ndarray:
    """Interpretable storage-conservative perturbation of the FFMP baseline.

    Lowers the downstream flow-target scaling and raises the storage-zone
    boundary curves so the system holds more NYC storage at the cost of
    downstream Decree flow (and delivery) reliability. Values are clamped to the
    formulation bounds; delivery/flood DVs are left at baseline.

    Args:
        formulation: Formulation name.

    Returns:
        DV vector (float ndarray) in ``get_var_names`` order.
    """
    names = get_var_names(formulation)
    lower, upper = get_bounds(formulation)
    dv = get_baseline_values(formulation).astype(float).copy()
    for i, name in enumerate(names):
        if name.startswith("mrf_target_scale_"):
            dv[i] = _CONTRAST_MRF_SCALE
        elif name.startswith("zone_vshift_"):
            dv[i] = _CONTRAST_ZONE_VSHIFT
    return np.clip(dv, lower, upper)


def _baseline_is_current(hdf5: Path) -> bool:
    """True if the artifact-of-record baseline matches the active objective set.

    Reads the sibling ``*_objectives.csv`` header and compares its columns to
    ``config.ACTIVE_OBJECTIVES``; a mismatch means the on-disk baseline predates
    the current objective set / DV design and must be regenerated.
    """
    csv = hdf5.with_name(f"{FORMULATION}_baseline_objectives.csv")
    if not (hdf5.exists() and csv.exists()):
        return False
    header = csv.read_text(encoding="utf-8").splitlines()[0].strip()
    cols = [c.strip() for c in header.split(",") if c.strip()]
    return set(cols) == set(config.ACTIVE_OBJECTIVES)


def _get_or_simulate(dv: np.ndarray, cache_path: Path, tag: str) -> dict:
    """Load a cached full-model simulation, or run + cache one, and return data.

    Args:
        dv: DV vector to simulate (ignored if ``cache_path`` already exists).
        cache_path: HDF5 cache location.
        tag: Short label for progress messages.

    Returns:
        pywrdrb results dict (results-set name -> daily DataFrame).
    """
    if cache_path.exists():
        print(f"[{tag}] reusing cached simulation: {cache_path.name}")
        return load_simulation_results(cache_path, results_sets=config.RESULTS_SETS)
    print(f"[{tag}] simulating full model (historical, {config.START_DATE}"
          f" -> {config.END_DATE}) -> {cache_path.name} ...")
    cfg = dvs_to_config(dv, formulation_name=FORMULATION)
    data = run_simulation_to_disk(cfg, cache_path, use_trimmed=False)
    return data


def _load_baseline_data(baseline_dv: np.ndarray) -> dict:
    """Baseline data: prefer the current artifact-of-record HDF5, else simulate."""
    official = config.OUTPUT_BASELINE_DIR / f"{FORMULATION}_baseline.hdf5"
    if _baseline_is_current(official):
        print(f"[baseline] reusing current artifact-of-record: {official}")
        return load_simulation_results(official, results_sets=config.RESULTS_SETS)
    if official.exists():
        print(f"[baseline] artifact-of-record is stale "
              f"(objective columns != ACTIVE_OBJECTIVES); simulating fresh. "
              f"Run workflow/05_run_baseline.sh to refresh it.")
    return _get_or_simulate(baseline_dv, HERE / f"baseline_{FORMULATION}.hdf5",
                            "baseline")


def _scores(data: dict) -> dict:
    """§1 whole-trace performance-metric values as name -> value.

    Diagnostic metrics, not the optimization objectives (which are the §2
    annual-unit versions; see the module docstring).
    """
    obj_set = build_objective_set(config.ACTIVE_OBJECTIVES)
    return dict(zip(obj_set.names, obj_set.compute(data)))


def _print_score_table(baseline: "od.Policy", contrast: "od.Policy") -> None:
    """Print the two policies' whole-trace performance metrics side by side."""
    print("\nWhole-trace performance metrics (historical single trace; "
          "NOT the optimization objectives):")
    print(f"  {'metric':38s} {'baseline':>12s} {'contrast':>12s}")
    for name in config.ACTIVE_OBJECTIVES:
        print(f"  {name:38s} {baseline.scores[name]:12.4f} "
              f"{contrast.scores[name]:12.4f}")


def main() -> None:
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_dv = get_baseline_values(FORMULATION)
    contrast_dv = build_contrast_dv(FORMULATION)

    viol = compute_constraint_violations(contrast_dv, formulation_name=FORMULATION)
    print(f"[contrast] constraint violations {get_constraint_names()}: "
          f"{[round(float(v), 6) for v in viol]}")
    if any(float(v) > 0 for v in viol):
        print("[contrast] WARNING: contrast policy is infeasible; retune the "
              "levers in build_contrast_dv().")

    baseline_data = _load_baseline_data(baseline_dv)
    contrast_data = _get_or_simulate(
        contrast_dv, HERE / "contrast_storage_conservative.hdf5", "contrast")

    baseline = od.Policy("Baseline (FFMP)", baseline_data, _scores(baseline_data),
                         od.BASELINE_COLOR, "-")
    contrast = od.Policy("Storage-conservative", contrast_data,
                         _scores(contrast_data), od.CONTRAST_COLOR, "--")
    policies = [baseline, contrast]

    _print_score_table(baseline, contrast)

    figures = [
        ("figA_nyc_delivery", od.plot_delivery_anatomy),
        ("figB_montague_flow", od.plot_montague_anatomy),
        ("figC_trenton_flow", od.plot_trenton_anatomy),
        ("figD_nyc_storage", od.plot_storage_anatomy),
        ("figE_downstream_flooding", od.plot_flood_anatomy),
    ]
    print()
    for stem, fn in figures:
        fn(policies, output_file=FIGURES_DIR / stem, strict=STRICT)
        print(f"[figure] wrote {stem}.png")

    # Policy-rules visual of the contrast (candidate solid + baseline dashed).
    plot_policy_rules(
        contrast_dv, formulation=FORMULATION, show_baseline=True,
        candidate_label="Storage-conservative",
        output_file=FIGURES_DIR / "figF_policy_rules_storage_conservative")
    print("[figure] wrote figF_policy_rules_storage_conservative.png")

    # Parallel-axis comparison of the whole-trace performance metrics (these are
    # diagnostic metrics, NOT the optimization objectives — hence the title).
    obj_set = build_objective_set(config.ACTIVE_OBJECTIVES)
    od.plot_objective_parallel_axes(
        policies, obj_set.names, obj_set.directions,
        title=("Whole-trace performance metrics — baseline vs storage-conservative "
               "  (diagnostic; NOT the optimization objectives — the optimizer "
               "targets the annual-unit versions; top = better)"),
        output_file=FIGURES_DIR / "figG_performance_metric_parallel_axes")
    print("[figure] wrote figG_performance_metric_parallel_axes.png")

    print(f"\nDone. Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
