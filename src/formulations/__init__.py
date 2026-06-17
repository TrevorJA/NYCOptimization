"""
src/formulations/__init__.py - Formulation registry for NYCOptimization.

This module is the single source of truth for problem formulations: decision
variable specifications, bounds, names, and the objective function factory.
Supports FFMP and variable-resolution FFMP only.

Exported API
------------
    get_formulation(name)       -> formulation dict  (supports "ffmp_N" variants)
    get_bounds(name)            -> (lower_array, upper_array)
    get_var_names(name)         -> list of DV names
    get_n_vars(name)            -> int
    get_baseline_values(name)   -> np.ndarray of baseline DVs
    get_n_objs()                -> int
    get_obj_names()             -> list of objective names
    get_obj_directions()        -> list of direction ints (+1 max, -1 min)
    get_objective_set(name)     -> ObjectiveSet instance
    make_objective_function(name) -> callable for Borg evaluation
    generate_ffmp_formulation(n_zones) -> formulation dict

Circular-import note
--------------------
    objectives.py imports config constants (NYC_RESERVOIRS, etc.) at module
    level.  config.py re-exports from this module at module level.  To break
    the cycle, functions here that need src.objectives or config use *local*
    imports executed at call time, not at import time.
"""

import numpy as np

from .ffmp import FFMP_FORMULATION, generate_ffmp_formulation, _merge_salt_front_dvs

# Sentinel: track which formulation dicts have already had salt-front DVs
# merged so repeated `get_formulation` calls don't double-merge.
_SALT_FRONT_MERGED_FLAGS: dict = {}

__all__ = [
    "FORMULATIONS",
    "get_formulation",
    "get_bounds",
    "get_var_names",
    "get_n_vars",
    "get_baseline_values",
    "get_n_objs",
    "get_obj_names",
    "get_obj_directions",
    "get_objective_set",
    "make_objective_function",
    "generate_ffmp_formulation",
]


###############################################################################
# Formulation registry
###############################################################################

FORMULATIONS = {
    "ffmp": FFMP_FORMULATION,
}


###############################################################################
# DV accessors
###############################################################################

def get_formulation(name: str = "ffmp") -> dict:
    """Return the formulation dict for *name*.

    Supports dynamic N-zone formulations via the pattern "ffmp_N" where N is
    the number of storage zone boundary curves (e.g. "ffmp_3", "ffmp_10").
    N=6 produces a zone count equivalent to the standard 7-level FFMP.

    Args:
        name: Formulation name.

    Returns:
        Dict with "description" and "decision_variables" keys.

    Raises:
        ValueError: If *name* is not in the registry and is not an ffmp_N pattern.
    """
    if name in FORMULATIONS:
        formulation = FORMULATIONS[name]
        # Lazily merge salt-front DVs into the registered formulation on
        # first access. Idempotent via the _SALT_FRONT_MERGED_FLAGS sentinel.
        # Defer to call time so we don't trigger a partial-import cycle
        # against config.py at module load.
        if name in ("ffmp",) and not _SALT_FRONT_MERGED_FLAGS.get(name):
            _merge_salt_front_dvs(formulation["decision_variables"])
            _SALT_FRONT_MERGED_FLAGS[name] = True
        return formulation
    if name.startswith("ffmp_"):
        try:
            n = int(name.split("_")[1])
        except (IndexError, ValueError):
            pass
        else:
            if n >= 2:
                # generate_ffmp_formulation already merges salt-front DVs.
                return generate_ffmp_formulation(n)
    raise ValueError(
        f"Unknown formulation '{name}'. "
        f"Available: {list(FORMULATIONS.keys())} or 'ffmp_N' for N-zone variants."
    )


def get_var_names(formulation_name: str = "ffmp") -> list:
    """Ordered list of decision variable names."""
    return list(get_formulation(formulation_name)["decision_variables"].keys())


def get_n_vars(formulation_name: str = "ffmp") -> int:
    """Number of decision variables."""
    return len(get_formulation(formulation_name)["decision_variables"])


def get_bounds(formulation_name: str = "ffmp") -> tuple:
    """Decision variable bounds as a pair of numpy arrays.

    Returns:
        (lower, upper) each of shape (n_vars,).
    """
    dvs = get_formulation(formulation_name)["decision_variables"]
    lower = [spec["bounds"][0] for spec in dvs.values()]
    upper = [spec["bounds"][1] for spec in dvs.values()]
    return np.array(lower), np.array(upper)


def get_baseline_values(formulation_name: str = "ffmp") -> np.ndarray:
    """Default (baseline FFMP) decision variable values."""
    dvs = get_formulation(formulation_name)["decision_variables"]
    return np.array([spec["baseline"] for spec in dvs.values()])


###############################################################################
# Objective accessors (lazy imports to avoid circular dependency)
###############################################################################

def get_objective_set(items=None):
    """Return an ObjectiveSet built from the given (or active) list of items.

    Resolves names against the **ensemble** registry (``src.objectives_ensemble``)
    when ``config.SEARCH_ENSEMBLE_SPEC.is_ensemble`` is True; otherwise resolves
    against the legacy single-trace registry (``src.objectives``). The legacy
    path is byte-identical to the manuscript baseline.

    Args:
        items: List of objective names (str) and/or Objective instances.
               If None, reads `config.ACTIVE_OBJECTIVES`.

    Returns:
        ObjectiveSet instance.
    """
    if items is None:
        from config import ACTIVE_OBJECTIVES
        items = ACTIVE_OBJECTIVES

    # SEARCH_ENSEMBLE_SPEC is None when the active scenario design has no search
    # ensemble wired yet; the single-trace registry (objective *definitions* are
    # identical across paths) is the safe default for non-optimization tasks
    # such as diagnostics and re-evaluation.
    from config import SEARCH_ENSEMBLE_SPEC
    if SEARCH_ENSEMBLE_SPEC is not None and SEARCH_ENSEMBLE_SPEC.is_ensemble:
        from src.objectives_ensemble import build_ensemble_objective_set
        return build_ensemble_objective_set(items)

    from src.objectives import build_objective_set
    return build_objective_set(items)


def get_n_objs(items=None) -> int:
    """Number of objectives in the active (or given) list."""
    return get_objective_set(items).n_objs


def get_obj_names(items=None) -> list:
    """Ordered list of objective names."""
    return get_objective_set(items).names


def get_obj_directions(items=None) -> list:
    """Objective directions: +1 for maximise, -1 for minimise.

    Borg minimises all objectives; ObjectiveSet.compute_for_borg() applies
    the sign flip automatically — callers should not negate manually.
    """
    return get_objective_set(items).directions


###############################################################################
# Objective function factory
###############################################################################

def make_objective_function(formulation_name: str = "ffmp"):
    """Return a Borg-compatible evaluation callable for a FFMP formulation.

    Args:
        formulation_name: "ffmp" or "ffmp_N" (N-zone variable-resolution FFMP).

    Returns:
        Callable: dv_vector -> list of floats (Borg-compatible, all minimised).
    """
    n_objs = get_n_objs()
    _penalty = [1e6] * n_objs

    from src.simulation import evaluate

    def _ffmp_fn(dv_vector):
        try:
            return evaluate(np.asarray(dv_vector),
                            formulation_name=formulation_name)
        except Exception:
            return _penalty

    return _ffmp_fn
