"""pareto_filter.py - Stakeholder-acceptability screening of a Pareto set.

Post-hoc screen in NATURAL units: drop Pareto-approximate solutions that
violate a hard stakeholder floor/ceiling on one or more objectives. It never
changes an objective value, only which solutions are carried forward.

The search enforces the same NYC reliability floor as the formal
post-simulation constraint ``nyc_reliability_floor``
(``src.formulations.make_post_sim_constraint_function``), so this screen is a
no-op on campaign archives. Keep ``DEFAULT_STAKEHOLDER_FLOORS`` equal to
``config.NYC_RELIABILITY_FLOOR``.

Orientation
-----------
Borg/MOEAFramework ``.set`` files store every objective MINIMIZED, so a maximize
objective (reliability, storage) is stored negated. :func:`to_natural` un-negates
using the formulation's direction vector, and every threshold here is expressed in
NATURAL, human-facing units (reliability in [0, 1], deficits in %, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Hard stakeholder floors (minimum acceptable NATURAL value), by objective name.
#: A solution is dropped if any listed objective falls BELOW its floor. The
#: floor is keyed under BOTH registry spellings because absent names are
#: ignored: ensemble archives carry the annual objective, single-trace (§1)
#: archives the weekly one.
DEFAULT_STAKEHOLDER_FLOORS: dict[str, float] = {
    "nyc_delivery_reliability_annual": 0.5,
    "nyc_delivery_reliability_weekly": 0.5,
}

#: Hard stakeholder ceilings (maximum acceptable NATURAL value), by objective name.
#: Empty by default; symmetric hook for minimize-objective screens (e.g. a deficit
#: cap) so the same function serves both directions.
DEFAULT_STAKEHOLDER_CEILINGS: dict[str, float] = {}


def to_natural(obj: np.ndarray, directions) -> np.ndarray:
    """Un-negate maximize objectives to NATURAL orientation.

    Args:
        obj: ``(n_solutions, n_objs)`` objective array as stored in a ``.set``
            (all-minimized, i.e. maximize objectives negated).
        directions: Per-objective direction ints (+1 maximize, -1 minimize),
            e.g. ``get_obj_directions()``.

    Returns:
        A copy in natural units (higher = better for maximize objectives).
    """
    natural = np.asarray(obj, dtype=float).copy()
    for k, d in enumerate(directions):
        if d == 1:
            natural[:, k] = -natural[:, k]
    return natural


def acceptability_mask(natural_obj: np.ndarray, obj_names,
                       floors: dict | None = None,
                       ceilings: dict | None = None) -> np.ndarray:
    """Boolean keep-mask over solutions meeting every floor and ceiling.

    A solution is kept iff, for each named objective, its NATURAL value is
    ``>= floor`` (floors) and ``<= ceiling`` (ceilings). Objectives absent from
    ``obj_names`` are ignored with no error (a floor for an inactive objective is
    a no-op), so one floor dict can serve several formulations.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units (see
            :func:`to_natural`).
        obj_names: Objective names aligned to the columns of ``natural_obj``.
        floors: ``{objective_name: min_acceptable}``; defaults to
            :data:`DEFAULT_STAKEHOLDER_FLOORS`. Pass ``{}`` for no floor.
        ceilings: ``{objective_name: max_acceptable}``; defaults to
            :data:`DEFAULT_STAKEHOLDER_CEILINGS`.

    Returns:
        Boolean array of length ``n_solutions`` (True = acceptable / keep).
    """
    floors = DEFAULT_STAKEHOLDER_FLOORS if floors is None else floors
    ceilings = DEFAULT_STAKEHOLDER_CEILINGS if ceilings is None else ceilings
    names = list(obj_names)
    natural_obj = np.asarray(natural_obj, dtype=float)
    mask = np.ones(natural_obj.shape[0], dtype=bool)
    for name, lo in floors.items():
        if name in names:
            mask &= natural_obj[:, names.index(name)] >= lo
    for name, hi in ceilings.items():
        if name in names:
            mask &= natural_obj[:, names.index(name)] <= hi
    return mask


@dataclass
class FilterResult:
    """Outcome of screening a reference set.

    Attributes:
        mask: Boolean keep-mask aligned to the ``.set`` rows (True = acceptable).
            The reference-set row index is the ``solution_id`` used by the re-eval
            cube, so ``accepted_ids`` selects scorecard rows directly.
        dv: ``(n_solutions, n_vars)`` decision variables (all rows, unfiltered).
        natural_obj: ``(n_solutions, n_objs)`` objectives in natural units.
        obj_names: Objective names aligned to ``natural_obj`` columns.
        directions: Per-objective direction ints (+1 max, -1 min).
        floors: The floors actually applied.
        ceilings: The ceilings actually applied.
    """

    mask: np.ndarray
    dv: np.ndarray
    natural_obj: np.ndarray
    obj_names: list
    directions: list
    floors: dict
    ceilings: dict

    @property
    def n_total(self) -> int:
        return int(self.mask.size)

    @property
    def n_accepted(self) -> int:
        return int(self.mask.sum())

    @property
    def accepted_ids(self) -> np.ndarray:
        """Reference-set row indices (== re-eval solution_ids) that pass the screen."""
        return np.where(self.mask)[0]

    def summary(self) -> str:
        floor_txt = ", ".join(f"{k} >= {v}" for k, v in self.floors.items()) or "none"
        return (f"stakeholder screen [{floor_txt}]: "
                f"{self.n_accepted}/{self.n_total} solutions acceptable "
                f"({100.0 * self.n_accepted / max(1, self.n_total):.0f}%)")


def filter_reference_set(set_file, formulation: str = "ffmp",
                         floors: dict | None = None,
                         ceilings: dict | None = None) -> FilterResult:
    """Load a ``.set``/``.ref`` and screen it against stakeholder floors.

    Args:
        set_file: Path to the reference set (whitespace-delimited vars + objs).
        formulation: Formulation name (fixes ``n_vars`` and the objective
            registry).
        floors / ceilings: See :func:`acceptability_mask`.

    Returns:
        A :class:`FilterResult`. ``result.accepted_ids`` are the re-eval
        ``solution_id``\\ s to keep in every downstream figure/scorecard.
    """
    from src.formulations import get_n_vars, get_obj_names, get_obj_directions
    from src.load.reference_set import load_reference_set

    n_vars = get_n_vars(formulation)
    dv, obj = load_reference_set(Path(set_file), n_vars)
    obj_names = get_obj_names()
    directions = get_obj_directions()
    natural = to_natural(obj, directions)
    mask = acceptability_mask(natural, obj_names, floors, ceilings)
    return FilterResult(
        mask=mask, dv=dv, natural_obj=natural, obj_names=obj_names,
        directions=directions,
        floors=DEFAULT_STAKEHOLDER_FLOORS if floors is None else floors,
        ceilings=DEFAULT_STAKEHOLDER_CEILINGS if ceilings is None else ceilings,
    )


