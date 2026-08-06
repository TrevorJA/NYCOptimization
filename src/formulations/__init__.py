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
    get_n_constrs()             -> int (formal Borg constraint count)
    get_constraint_names()      -> list of constraint names
    make_constraint_function(name) -> callable: dv_vector -> DV-space violation list
    make_post_sim_constraint_function() -> callable: borg_objs -> violation list
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
    "CONSTRAINT_NAMES",
    "DV_CONSTRAINT_NAMES",
    "POST_SIM_CONSTRAINT_NAMES",
    "RELIABILITY_FLOOR_OBJECTIVE",
    "resolve_objective_index",
    "reliability_floor_objective_index",
    "get_n_constrs",
    "get_constraint_names",
    "make_constraint_function",
    "make_post_sim_constraint_function",
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

    Resolves names against the **annual-unit** registry
    (``src.objectives_ensemble``) whenever a scenario design is wired
    (``config.SEARCH_ENSEMBLE_SPEC is not None``) — the historic single-trace
    design searches under the same annual-unit objective as the ensembles
    (N=1). Resolves against the single-trace registry (``src.objectives``)
    only when no design is wired.

    Args:
        items: List of objective names (str) and/or Objective instances.
               If None, reads `config.ACTIVE_OBJECTIVES`.

    Returns:
        ObjectiveSet instance.
    """
    if items is None:
        from config import ACTIVE_OBJECTIVES
        items = ACTIVE_OBJECTIVES

    # All wired scenario designs — the single-trace historic design
    # (is_ensemble=False) and the multi-realization ensembles alike — now search
    # under the SAME annual-unit (§2) objective function
    # (objective_definitions.md §2/§3: the objective is held fixed across
    # designs so the only factor that varies is the scenario set). The
    # single-trace case is evaluated as N=1 over its L-1 water-year units; the
    # dispatch in src.simulation.evaluate wraps the one data dict as [data].
    # The §1 single-trace registry is returned ONLY when no design is wired
    # (a pure diagnostic / non-optimization context). No pipeline script may
    # build the §1 set for a wired design: a §1 vector is a DIFFERENT objective
    # function from the one Borg optimizes, so mixing the two silently makes
    # baseline / Pareto / re-eval vectors non-comparable. The two §1 consumers
    # that remain are correct by construction — src.reeval_core takes the §1
    # base metrics PER REALIZATION before its own annual/satisficing layer, and
    # the supplemental diagnostics in scripts/supplemental deliberately report
    # the whole-trace timescale.
    from config import SEARCH_ENSEMBLE_SPEC
    if SEARCH_ENSEMBLE_SPEC is not None:
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
# Formal Borg constraint accessors
###############################################################################

#: DV-space formal Borg constraints (violation magnitude convention: 0.0 =
#: feasible, positive scales linearly with the degree of violation). Computed
#: from pure DV arithmetic BEFORE any simulation. The same conditions are also
#: enforced by the apply-time clamps in src/simulation.py — the constraints
#: give Borg a direct pre-simulation feasibility signal; the clamps remain as
#: intentional redundancy.
#: Zone-curve crossings are clamp-only (not a constraint): the monotonicity
#: clamp resolves them at apply time and the clamped geometry is the
#: intended policy.
DV_CONSTRAINT_NAMES = [
    "delivery_monotonicity",
    "flood_zone_ordering",
]

#: POST-SIMULATION formal Borg constraints: computed from the objective vector
#: AFTER simulation, so they can only be evaluated inside the MM Borg objective
#: wrapper (src/mmborg.py), never by `make_constraint_function` (DV-only).
#: `nyc_reliability_floor` enforces the stakeholder floor on NYC weekly
#: delivery reliability (`config.NYC_RELIABILITY_FLOOR`, env
#: NYCOPT_NYC_RELIABILITY_FLOOR, default 0.5) directly in the search:
#: constraint-dominance excludes below-floor policies from search and archive,
#: replacing the post-hoc Pareto screen in src/pareto_filter.py for new runs.
POST_SIM_CONSTRAINT_NAMES = [
    "nyc_reliability_floor",
]

#: Full constraint order handed to Borg: DV-space first, post-simulation last.
CONSTRAINT_NAMES = DV_CONSTRAINT_NAMES + POST_SIM_CONSTRAINT_NAMES

#: Objective (by NAME — never a hard-coded index) that carries the raw
#: reliability value the `nyc_reliability_floor` constraint reads. This is
#: the BASE (§1) spelling used by config.ACTIVE_OBJECTIVES; resolved
#: objective sets report registry names (the annual-unit registry renames it
#: whenever a scenario design is wired — every search context), so lookups
#: must go through ``reliability_floor_objective_index``, never a bare
#: ``names.index``.
RELIABILITY_FLOOR_OBJECTIVE = "nyc_delivery_reliability_weekly"


def resolve_objective_index(names, objective: str) -> int:
    """Index of an objective in ``names``, accepting either registry spelling.

    The same underlying metric carries a base §1 name (single-trace registry,
    ``src.objectives``) and an annual-unit name (``src.objectives_ensemble``,
    active whenever a scenario design is wired — i.e. every search context).
    Post-processing that hard-codes one spelling breaks silently against the
    other, so any lookup by a literal objective name must come through here.
    The mapping is consulted in both directions, so either spelling resolves
    against either registry.

    Args:
        names: Ordered objective-name list, as returned by ``get_obj_names``.
        objective: Objective name in either spelling.

    Returns:
        Position of the objective in ``names``.

    Raises:
        ValueError: If neither spelling is present in ``names``.
    """
    from src.objectives_ensemble import _BASE_TO_ENSEMBLE

    names = list(names)
    ensemble_to_base = {v: k for k, v in _BASE_TO_ENSEMBLE.items()}
    candidates = [objective, _BASE_TO_ENSEMBLE.get(objective),
                  ensemble_to_base.get(objective)]
    for cand in candidates:
        if cand is not None and cand in names:
            return names.index(cand)
    raise ValueError(
        f"Objective '{objective}' (nor its alternate registry spelling) is "
        f"in the active set; got {names}."
    )


def reliability_floor_objective_index(names) -> int:
    """Index of the reliability objective the floor constraint reads.

    Accepts either spelling of the objective: the base §1 name
    (``RELIABILITY_FLOOR_OBJECTIVE``) or its annual-unit registry form
    (via ``src.objectives_ensemble._BASE_TO_ENSEMBLE``), so the lookup works
    against both the single-trace and the annual-unit registries.

    Args:
        names: Ordered objective-name list, as returned by ``get_obj_names``.

    Returns:
        Position of the reliability objective in ``names``.

    Raises:
        ValueError: If neither spelling is present — the floor cannot be
            enforced without the objective it reads.
    """
    try:
        return resolve_objective_index(names, RELIABILITY_FLOOR_OBJECTIVE)
    except ValueError:
        from src.objectives_ensemble import _BASE_TO_ENSEMBLE
        alias = _BASE_TO_ENSEMBLE.get(RELIABILITY_FLOOR_OBJECTIVE)
        raise ValueError(
            f"The nyc_reliability_floor constraint requires objective "
            f"'{RELIABILITY_FLOOR_OBJECTIVE}' (or its annual-unit form "
            f"'{alias}') in the active set; got {list(names)}. Fix "
            f"NYCOPT_OBJECTIVES — the floor cannot be enforced without it."
        ) from None


def get_n_constrs() -> int:
    """Number of formal Borg constraints (same for all FFMP formulations).

    Counts BOTH kinds: the DV-space constraints (pre-simulation, see
    ``DV_CONSTRAINT_NAMES``) and the post-simulation constraints
    (``POST_SIM_CONSTRAINT_NAMES``).
    """
    return len(CONSTRAINT_NAMES)


def get_constraint_names() -> list:
    """Ordered list of formal Borg constraint names (DV-space, then post-sim)."""
    return list(CONSTRAINT_NAMES)


def make_constraint_function(formulation_name: str = "ffmp"):
    """Return the DV-SPACE constraint-violation callable for a FFMP formulation.

    The returned callable is pure DV arithmetic (no simulation, no config
    deepcopy) and is safe to call before/instead of the objective function.
    It covers ONLY the ``DV_CONSTRAINT_NAMES`` entries; the post-simulation
    constraints (``POST_SIM_CONSTRAINT_NAMES``) need the computed objective
    vector and are appended by the MM Borg objective wrapper via
    ``make_post_sim_constraint_function``.

    Args:
        formulation_name: "ffmp" or "ffmp_N" (N-zone variable-resolution FFMP).

    Returns:
        Callable: dv_vector -> list of ``len(DV_CONSTRAINT_NAMES)`` violation
        floats, ordered per ``DV_CONSTRAINT_NAMES`` (0.0 = feasible).
    """
    from src.simulation import compute_constraint_violations

    def _constraint_fn(dv_vector):
        return compute_constraint_violations(
            np.asarray(dv_vector), formulation_name=formulation_name
        )

    return _constraint_fn


def make_post_sim_constraint_function(items=None):
    """Return the POST-SIMULATION constraint-violation callable.

    Consumes the Borg-oriented objective vector produced by
    ``make_objective_function`` (all values MINIMIZED, i.e. maximize
    objectives negated by ``compute_for_borg``) and returns the
    ``POST_SIM_CONSTRAINT_NAMES`` violations. Currently one constraint:

    ``nyc_reliability_floor`` — violation = max(0, floor - reliability) with
    the reliability recovered on the NATURAL 0-1 scale (the stored Borg value
    is un-negated using the objective's direction). The floor comes from
    ``config.NYC_RELIABILITY_FLOOR`` (env ``NYCOPT_NYC_RELIABILITY_FLOOR``),
    read at factory-call time so per-run env files take effect.

    Failed-simulation convention: penalty objective vectors (1e6/1e10
    sentinels from the eval wrappers) put the recovered "reliability" far
    outside [0, 1]. For such vectors the violation is reported as 0.0 —
    nothing was measured, and fabricating a violation magnitude would distort
    constraint-dominance ordering among genuinely infeasible solutions. The
    penalty objectives already guarantee the failed eval is Pareto-dominated
    by every real solution, so it is feasible-but-maximally-unattractive,
    matching the wrapper's exception path (see src/mmborg.py).

    Args:
        items: Objective names defining the active set; None reads
            ``config.ACTIVE_OBJECTIVES`` (same convention as
            ``get_obj_names``). The reliability objective is resolved by NAME
            via ``reliability_floor_objective_index`` (accepts the base §1
            spelling or its annual-unit registry form).

    Returns:
        Callable: borg_objective_list -> list of
        ``len(POST_SIM_CONSTRAINT_NAMES)`` violation floats (0.0 = feasible).

    Raises:
        ValueError: If the active objective set does not include
            ``RELIABILITY_FLOOR_OBJECTIVE`` — the floor cannot be enforced
            without the objective it reads.
    """
    from config import NYC_RELIABILITY_FLOOR

    names = get_obj_names(items)
    directions = get_obj_directions(items)
    idx = reliability_floor_objective_index(names)
    # Borg stores maximize objectives negated (+1 direction); un-negate to
    # recover the natural 0-1 reliability before comparing to the floor.
    sign = -1.0 if directions[idx] == 1 else 1.0
    floor = float(NYC_RELIABILITY_FLOOR)

    def _post_sim_fn(borg_objs):
        reliability = sign * float(borg_objs[idx])
        if not 0.0 <= reliability <= 1.0:
            # Penalty sentinel from a failed simulation — see docstring.
            return [0.0]
        return [max(0.0, floor - reliability)]

    return _post_sim_fn


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
    # Penalized-eval log budget: the first few failures carry the diagnosis
    # (a systematic misconfiguration fails identically on every eval); after
    # that stay quiet so a pathological DV region cannot flood the job log.
    _log_state = {"remaining": 5, "count": 0}

    from src.simulation import evaluate

    def _ffmp_fn(dv_vector):
        try:
            return evaluate(np.asarray(dv_vector),
                            formulation_name=formulation_name)
        except Exception:
            import sys
            import traceback
            _log_state["count"] += 1
            if _log_state["remaining"] > 0:
                _log_state["remaining"] -= 1
                msg = traceback.format_exc(limit=3).strip().splitlines()[-1]
                print(f"[objective_fn] WARNING: eval #{_log_state['count']} "
                      f"failed, returning 1e6 penalty ({msg})"
                      + ("" if _log_state["remaining"]
                         else " -- further eval-failure warnings suppressed"),
                      file=sys.stderr, flush=True)
            return _penalty

    return _ffmp_fn
