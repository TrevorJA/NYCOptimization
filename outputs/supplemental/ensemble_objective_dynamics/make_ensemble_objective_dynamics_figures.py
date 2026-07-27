"""make_ensemble_objective_dynamics_figures.py - Driver for the ensemble suite.

The ensemble analog of ``make_objective_dynamics_figures.py``. It evaluates the
same two policies — the default FFMP baseline and one interpretable
storage-conservative contrast — but under a small **stationary Kirsch-Nowak
ensemble** (5 realizations x 50 years; see ``src.local_test_ensemble``) instead
of the single historical trace, and scores them with the two-layer annual-unit
(§2) objectives Borg optimizes during an ensemble search
(``build_ensemble_objective_set``). It then renders the anatomy figures from the
sibling ``ensemble_objective_dynamics`` module, reuses the DV-agnostic
policy-rules figure, and draws the objective parallel-axes comparison on the §2
scores.

The ensemble is generated + staged on first run (a couple of minutes) and reused
thereafter; the two policies' per-realization simulations are cached to
``cache/`` so re-runs only re-plot.

Run from the repo root with the project venv active:

    python outputs/supplemental/ensemble_objective_dynamics/make_ensemble_objective_dynamics_figures.py

Env toggles (no CLI value flags, per project convention):
    NYCOPT_ENSOBJDYN_NOSTRICT=1   disable the figure-vs-score self-check assertions.
    NYCOPT_ENSOBJDYN_REFRESH=1    ignore the simulation cache and re-simulate.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

# Salinity / temperature LSTMs are unused by the §2 objective set and slow the
# ensemble down; disable them before config reads the environment at import.
os.environ.setdefault("NYCOPT_SALINITY_ON", "0")
os.environ.setdefault("NYCOPT_TEMPERATURE_ON", "0")

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE / "figures"
CACHE_DIR = HERE / "cache"
PROJECT_DIR = HERE.parents[2]  # ensemble_objective_dynamics -> supplemental -> outputs -> root
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src.formulations import (  # noqa: E402
    get_baseline_values,
    get_bounds,
    get_constraint_names,
    get_var_names,
)
from src.local_test_ensemble import (  # noqa: E402
    LOCAL_TEST_ENSEMBLE_SLUG,
    ensure_local_test_ensemble,
)
from src.objectives_ensemble import build_ensemble_objective_set  # noqa: E402
from src.plotting.policy_rules import plot_policy_rules  # noqa: E402
from src.plotting.style import apply_style  # noqa: E402
from src.simulation import (  # noqa: E402
    compute_constraint_violations,
    dvs_to_config,
    run_simulation_ensemble_inmemory,
)

import ensemble_objective_dynamics as eod  # noqa: E402  (sibling module)

# Reuse the single-trace parallel-axes renderer verbatim (it reads only
# scores/colour/style off each policy, so an EnsemblePolicy works unchanged).
sys.path.insert(0, str(HERE.parent / "objective_dynamics"))
import objective_dynamics as od  # noqa: E402

FORMULATION = "ffmp"
STRICT = os.environ.get("NYCOPT_ENSOBJDYN_NOSTRICT", "").strip() not in ("1", "true", "True")
REFRESH = os.environ.get("NYCOPT_ENSOBJDYN_REFRESH", "").strip() in ("1", "true", "True")

# Storage-conservative contrast levers (identical to the single-trace suite so
# the two are directly comparable): release less toward the Montague/Trenton
# flow targets and enter conservation zones at higher storage.
_CONTRAST_MRF_SCALE = 0.80    # mrf_target_scale_* multiplier (baseline 1.0)
_CONTRAST_ZONE_VSHIFT = 0.03  # zone_vshift_* additive offset, frac of capacity (baseline 0.0)


def build_contrast_dv(formulation: str = FORMULATION) -> np.ndarray:
    """Interpretable storage-conservative perturbation of the FFMP baseline.

    Lowers the downstream flow-target scaling and raises the storage-zone
    boundary curves so the system holds more NYC storage at the cost of
    downstream Decree flow (and delivery). Values are clamped to the formulation
    bounds; delivery/flood DVs stay at baseline.

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


def _get_or_simulate_ensemble(dv: np.ndarray, spec, tag: str) -> list:
    """Load a cached per-realization ensemble simulation, or run + cache one.

    The cache key includes the ensemble slug and a hash of the DV vector, so a
    changed policy or ensemble never silently reuses a stale simulation.

    Args:
        dv: DV vector to simulate.
        spec: The staged ``EnsembleSpec`` to simulate across.
        tag: Short label for progress messages and the cache filename.

    Returns:
        list[dict] of per-realization pywrdrb results.
    """
    key = hashlib.md5(np.asarray(dv, dtype=float).tobytes()).hexdigest()[:8]
    cache_path = CACHE_DIR / f"{tag}_{spec.inflow_type}_{key}.pkl"
    if cache_path.exists() and not REFRESH:
        print(f"[{tag}] reusing cached ensemble simulation: {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"[{tag}] simulating {spec.n_realizations} realizations "
          f"({spec.realization_years}-yr, {spec.inflow_type}) ...")
    cfg = dvs_to_config(dv, formulation_name=FORMULATION)
    data_per_real = run_simulation_ensemble_inmemory(cfg, spec)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data_per_real, f)
    return data_per_real


def _scores(data_per_real: list) -> dict:
    """True §2 annual-unit objective scores as annual-name -> value."""
    obj_set = build_ensemble_objective_set(config.ACTIVE_OBJECTIVES)
    return dict(zip(obj_set.names, obj_set.compute(data_per_real)))


def _print_score_table(baseline, contrast, obj_names) -> None:
    """Print the two policies' §2 objective scores side by side."""
    print(f"\nEnsemble (§2) objective scores  ({LOCAL_TEST_ENSEMBLE_SLUG}):")
    print(f"  {'objective':38s} {'baseline':>12s} {'contrast':>12s}")
    for name in obj_names:
        print(f"  {name:38s} {baseline.scores[name]:12.4f} "
              f"{contrast.scores[name]:12.4f}")


def main() -> None:
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    spec = ensure_local_test_ensemble()

    baseline_dv = get_baseline_values(FORMULATION)
    contrast_dv = build_contrast_dv(FORMULATION)

    viol = compute_constraint_violations(contrast_dv, formulation_name=FORMULATION)
    print(f"[contrast] constraint violations {get_constraint_names()}: "
          f"{[round(float(v), 6) for v in viol]}")
    if any(float(v) > 0 for v in viol):
        print("[contrast] WARNING: contrast policy is infeasible; retune the "
              "levers in build_contrast_dv().")

    baseline_data = _get_or_simulate_ensemble(baseline_dv, spec, "baseline")
    contrast_data = _get_or_simulate_ensemble(contrast_dv, spec, "contrast")

    baseline = eod.EnsemblePolicy(
        "Baseline (FFMP)", baseline_data, _scores(baseline_data),
        eod.BASELINE_COLOR, "-")
    contrast = eod.EnsemblePolicy(
        "Storage-conservative", contrast_data, _scores(contrast_data),
        eod.CONTRAST_COLOR, "--")
    policies = [baseline, contrast]

    obj_set = build_ensemble_objective_set(config.ACTIVE_OBJECTIVES)
    _print_score_table(baseline, contrast, obj_set.names)

    figures = [
        ("figA_nyc_delivery", eod.plot_delivery_anatomy),
        ("figB_montague_flow", eod.plot_montague_anatomy),
        ("figC_trenton_flow", eod.plot_trenton_anatomy),
        ("figD_nyc_storage", eod.plot_storage_anatomy),
        ("figE_downstream_flooding", eod.plot_flood_anatomy),
    ]
    print()
    for stem, fn in figures:
        fn(policies, output_file=FIGURES_DIR / stem, strict=STRICT)
        print(f"[figure] wrote {stem}.png")

    # Policy-rules visual of the contrast (DV-agnostic; identical to single-trace).
    plot_policy_rules(
        contrast_dv, formulation=FORMULATION, show_baseline=True,
        candidate_label="Storage-conservative",
        output_file=FIGURES_DIR / "figF_policy_rules_storage_conservative")
    print("[figure] wrote figF_policy_rules_storage_conservative.png")

    # Parallel-axis objective comparison on the §2 ensemble scores.
    od.plot_objective_parallel_axes(
        policies, obj_set.names, obj_set.directions,
        output_file=FIGURES_DIR / "figG_objective_parallel_axes")
    print("[figure] wrote figG_objective_parallel_axes.png")

    print(f"\nDone. Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
