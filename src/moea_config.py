"""
moea_config.py - Registry of MOEA algorithm configurations (Borg settings only).

A *MOEA config* is a named bundle of Multi-Master Borg **algorithm settings** —
island count, worker count, evaluation budget, runtime-snapshot frequency, seed
count, and wall-time. It is one of two orthogonal axes that specify an
optimization run; the other is the scenario design (see
``src/scenario_designs.py``).

This module centralizes what is otherwise split between the former
``config.BORG_SETTINGS`` / ``config.MMBORG_SETTINGS`` dicts and value-carrying
CLI flags (``--nfe``, ``--islands``, ``--runtime-freq``, ``--time``). Moving
these into a versioned, named registry makes runs reproducible from config alone
rather than from shell history.

Deliberately **excluded** from a MOEA config (they live on the
problem-definition axis and are encoded in the moea slug, not here):
formulation, objectives, physics toggles (LSTM, salt-front mode), and epsilons.
Epsilons come from the active objective set via ``config.get_epsilons()``.

Status: ``smoke`` is a concrete, intentionally tiny dev-only config so the full
pipeline is runnable end-to-end. ``production`` carries the schema with every
campaign number left ``None``/TBD pending the compute-budget discussion (open
decision #5 in ``docs/notes/methods/experimental_design.md``).
"""

from __future__ import annotations

from dataclasses import dataclass


###############################################################################
# MOEAConfig
###############################################################################

@dataclass(frozen=True)
class MOEAConfig:
    """Immutable bundle of Multi-Master Borg algorithm settings.

    Attributes
    ----------
    name
        Single-string key used to select this config (via ``NYCOPT_MOEA_CONFIG``)
        and appended to the moea slug when it is not the default.
    n_islands
        Number of MM Borg islands. ``None`` until set.
    n_workers_per_island
        Worker ranks per island, used to size the MPI allocation. ``None`` until
        set (then the SLURM allocation is used as a fallback).
    max_evaluations
        Max NFE **per island** (total NFE = islands * max_evaluations). ``None``
        until set. Mutually informative with ``budget_scenario_years``.
    budget_scenario_years
        Alternative budget control expressed as total simulated scenario-years
        (function evaluations * ensemble size * realization length), per the
        budget-controlled comparison in ``experimental_design.md`` §Controls.
        ``None`` until the budget discussion fixes it.
    runtime_frequency
        NFE interval between Borg runtime-archive snapshots. ``None`` until set.
    n_seeds
        Number of random seeds (independent search replicates). ``None`` until set.
    max_time_hours
        Wall-time cap in hours, or ``None`` for NFE-bounded runs.
    notes
        Free-form notes.
    """

    name: str
    n_islands: int | None = None
    n_workers_per_island: int | None = None
    max_evaluations: int | None = None
    budget_scenario_years: int | None = None
    runtime_frequency: int | None = None
    n_seeds: int | None = None
    max_time_hours: int | None = None
    notes: str = ""

    @property
    def max_time_seconds(self) -> int | None:
        """Wall-time cap in seconds, or ``None`` when NFE-bounded."""
        if self.max_time_hours is None:
            return None
        return int(self.max_time_hours * 3600)

    @property
    def total_ntasks_mpi(self) -> int | None:
        """MPI rank count: ``1 + n_islands * (n_workers_per_island + 1)``.

        Returns ``None`` if either ``n_islands`` or ``n_workers_per_island`` is
        unset, in which case callers fall back to the SLURM allocation size.
        """
        if self.n_islands is None or self.n_workers_per_island is None:
            return None
        return 1 + self.n_islands * (self.n_workers_per_island + 1)


###############################################################################
# Registry
###############################################################################

MOEA_CONFIGS: dict[str, MOEAConfig] = {
    # Dev-only: tiny budget so the full optimization -> diagnostics -> reeval
    # pipeline runs to completion locally in minutes. NOT for production runs.
    "smoke": MOEAConfig(
        name="smoke",
        n_islands=2,
        n_workers_per_island=1,
        max_evaluations=200,      # per island
        runtime_frequency=50,
        n_seeds=1,
        max_time_hours=None,
        notes="Dev/smoke config. Plumbing exercise only — not a method choice.",
    ),
    # Production: schema only. Every campaign number is an open decision tied to
    # the total-simulated-scenario-years budget (experimental_design.md #5).
    "production": MOEAConfig(
        name="production",
        n_islands=None,
        n_workers_per_island=None,
        max_evaluations=None,
        budget_scenario_years=None,
        runtime_frequency=None,
        n_seeds=None,
        max_time_hours=None,
        notes="TBD — fixed against the compute-budget discussion "
              "(experimental_design.md open decision #5).",
    ),
}


###############################################################################
# Resolver + helpers
###############################################################################

def get_moea_config(name: str) -> MOEAConfig:
    """Resolve a MOEA-config name to its ``MOEAConfig``.

    Args:
        name: A key of ``MOEA_CONFIGS``.

    Returns:
        The matching ``MOEAConfig``.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return MOEA_CONFIGS[name]
    except KeyError:
        raise KeyError(
            f"Unknown MOEA config '{name}'. "
            f"Available configs: {list_moea_configs()}."
        ) from None


def list_moea_configs() -> list[str]:
    """Return the registered MOEA-config names in sorted order."""
    return sorted(MOEA_CONFIGS)
