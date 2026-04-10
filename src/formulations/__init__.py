"""
src/formulations/__init__.py - Formulation registry for NYCOptimization.

This module is the single source of truth for problem formulations: decision
variable specifications, bounds, names, and the objective function factory.

Exported API
------------
    get_formulation(name)       -> formulation dict
    get_bounds(name)            -> (lower_array, upper_array)
    get_var_names(name)         -> list of DV names
    get_n_vars(name)            -> int
    get_baseline_values(name)   -> np.ndarray of baseline DVs
    get_n_objs()                -> int
    get_obj_names()             -> list of objective names
    get_obj_directions()        -> list of direction ints (+1 max, -1 min)
    get_objective_set(name)     -> ObjectiveSet instance
    make_objective_function(formulation_name, objective_set) -> callable
    is_external_policy(name)    -> bool
    generate_ffmp_formulation() -> formulation dict

Circular-import note
--------------------
    objectives.py imports config constants (NYC_RESERVOIRS, etc.) at module
    level.  config.py re-exports from this module at module level.  To break
    the cycle, functions here that need src.objectives or config use *local*
    imports executed at call time, not at import time.
"""

import numpy as np

from .ffmp import FFMP_FORMULATION, generate_ffmp_formulation
from .external import is_external_policy, get_architecture, register_architecture

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
    "is_external_policy",
    "get_architecture",
    "register_architecture",
    "generate_ffmp_formulation",
]


###############################################################################
# Formulation registry
###############################################################################

FORMULATIONS = {
    "ffmp": FFMP_FORMULATION,
}


###############################################################################
# DV accessors (no circular-import risk — only touch FORMULATIONS)
###############################################################################

def get_formulation(name: str = "ffmp") -> dict:
    """Return the formulation dict for *name*.

    Args:
        name: Formulation name.  Currently only "ffmp" is defined.

    Returns:
        Dict with "description" and "decision_variables" keys.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in FORMULATIONS:
        raise ValueError(
            f"Unknown formulation '{name}'. "
            f"Available: {list(FORMULATIONS.keys())}"
        )
    return FORMULATIONS[name]


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

def get_objective_set(name: str = None):
    """Return the active ObjectiveSet instance.

    Args:
        name: Objective set name.  If None, reads ACTIVE_OBJECTIVE_SET from
              config (the default is "default").

    Returns:
        ObjectiveSet instance.

    Raises:
        ValueError: If *name* is not in the OBJECTIVE_SETS registry.
    """
    from src.objectives import OBJECTIVE_SETS
    if name is None:
        from config import ACTIVE_OBJECTIVE_SET
        name = ACTIVE_OBJECTIVE_SET
    if name not in OBJECTIVE_SETS:
        raise ValueError(
            f"Unknown objective set '{name}'. "
            f"Available: {list(OBJECTIVE_SETS.keys())}"
        )
    return OBJECTIVE_SETS[name]


def get_n_objs(objective_set_name: str = None) -> int:
    """Number of objectives in the active (or named) set."""
    return get_objective_set(objective_set_name).n_objs


def get_obj_names(objective_set_name: str = None) -> list:
    """Ordered list of objective names."""
    return get_objective_set(objective_set_name).names


def get_obj_directions(objective_set_name: str = None) -> list:
    """Objective directions: +1 for maximise, -1 for minimise.

    Borg minimises all objectives; ObjectiveSet.compute_for_borg() applies
    the sign flip automatically — callers should not negate manually.
    """
    return get_objective_set(objective_set_name).directions


###############################################################################
# Objective function factory
###############################################################################

def make_objective_function(formulation_name: str = "ffmp", objective_set=None):
    """Return a Borg-compatible evaluation callable.

    The returned function maps a flat DV vector to a list of objective values
    (all sign-adjusted for minimisation).  Suitable for passing directly to
    Borg as the objective function.

    Args:
        formulation_name: Formulation to evaluate.
        objective_set: ObjectiveSet instance to use.  If None, uses the
                       active set from config.ACTIVE_OBJECTIVE_SET.

    Returns:
        Callable: dv_vector -> list of floats (Borg-compatible).
    """
    from src.simulation import evaluate

    def _fn(dv_vector):
        return evaluate(
            dv_vector,
            formulation_name=formulation_name,
            objective_set=objective_set,
        )

    return _fn
