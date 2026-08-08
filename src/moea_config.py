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
pipeline is runnable end-to-end. ``production`` is the campaign config, sized
against the measured Anvil cost surface and the 750,000-SU allocation
(``docs/notes/methods/scenario_design_methods.md`` §6).
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
    # Multi-Master Borg on Hopper, historic single-trace baseline. Sized to the
    # 5-node x 33-task = 165-rank budget: 4 islands x 40 workers + 4 island-
    # masters + 1 controller = 1 + 4*(40+1) = 165 ranks (160 parallel
    # evaluators). NFE is per-island, so total NFE = n_islands * max_evaluations.
    #
    #   mm_pilot:  4 * 1250  =   5,000 total NFE  (launch-verification pilot)
    #   mm_full:   4 * 12500 =  50,000 total NFE  (production run)
    #
    # NFE-bounded (max_time_hours=None); the SLURM --time wall cap is sized from
    # the pilot's measured per-eval cost. Single seed (n_seeds=1); the array
    # index supplies the Borg RNG seed.
    "mm_pilot": MOEAConfig(
        name="mm_pilot",
        n_islands=4,
        n_workers_per_island=40,
        max_evaluations=1250,     # per island -> 5,000 total NFE
        runtime_frequency=250,
        n_seeds=1,
        max_time_hours=None,
        notes="Hopper MM-Borg launch pilot: 5k NFE, 165 ranks (4x40+5). "
              "Verifies balance, per-eval cost, and output writing before the "
              "full 50k run.",
    ),
    # Scenario-design go/no-go pilot. Identical rank layout to mm_pilot (165
    # ranks, 5,000 total NFE). NFE is the SOLE binding constraint so all arms
    # execute the identical 5,000 NFE — the validity condition for the cross-
    # design Pareto comparison. Calibrated empirically: the slowest arm (N=64
    # hazard ensemble) measured ~153 s/warm-eval, so 5,000 NFE ≈ 1.7 h of eval
    # time on 160 workers — comfortably under the 3 h SLURM wall. No Borg
    # maxTime cap (it would risk truncating NFE unequally near the budget);
    # the SLURM --time=03:00:00 wall + 250-NFE runtime snapshots are the safety
    # net (a killed run is recoverable from the last snapshot).
    "pilot": MOEAConfig(
        name="pilot",
        n_islands=4,
        n_workers_per_island=40,
        max_evaluations=1250,     # per island -> 5,000 total NFE
        runtime_frequency=250,
        n_seeds=1,
        max_time_hours=None,      # NFE-bounded; SLURM --time is the wall safety
        notes="Go/no-go scenario-design pilot: 5k NFE, 165 ranks (4x40+5). "
              "NFE-binding (calibrated: N=64 arm ~153s/eval -> ~1.7h < 3h wall).",
    ),
    "mm_full": MOEAConfig(
        name="mm_full",
        n_islands=4,
        n_workers_per_island=40,
        max_evaluations=12500,    # per island -> 50,000 total NFE
        runtime_frequency=1000,
        n_seeds=10,               # recorded replicate count; submitted as
                                  # `sbatch --array=1-10 workflow/06_run_mmborg.sh`
        max_time_hours=None,
        notes="MM-Borg production run: 50k NFE, 165 ranks (4x40+5). Scenario "
              "design and objective set are the other run axes and are not fixed "
              "by this config.",
    ),
    # Moderate first full-workflow run, Anvil-shaped: 4 wholenode nodes = 512
    # cores, 2 islands x 254 workers + 2 island-masters + 1 controller =
    # 1 + 2*(254+1) = 511 ranks (1 idle core). Dense 128/node packing is the
    # measured Anvil choice (SI §S8.2 packing sweep: ~17-21% eval-time penalty
    # at full packing, already priced into the 173.8 s/eval cost surface, which
    # was measured AT 128 ranks/node; ~1.2 GB/rank << 256 GB node memory) —
    # the old 33/node rule was a Hopper measurement and does not transfer.
    # Island partitioning is throughput-free at fixed slot count (SI §S8.4),
    # so 2 islands is a search-reliability choice: the smallest multi-master
    # count, keeping per-island trajectories long (25k NFE/island, ~98
    # turnovers of the 254-worker pipeline) at the moderate budget. 50k total
    # NFE is 1/10 of the production budget; <=6.5 h / ~3,340 SU per seed at
    # the measured cost (upper bound — early DV-infeasible NFE return in
    # <0.1 s). Single seed (the array index supplies the Borg seed).
    "mm_moderate": MOEAConfig(
        name="mm_moderate",
        n_islands=2,
        n_workers_per_island=254,
        max_evaluations=25_000,   # per island -> 50,000 total NFE
        runtime_frequency=500,    # 50 runtime snapshots/island (restart safety)
        n_seeds=1,
        max_time_hours=None,      # NFE-bounded; SLURM --time is the wall safety
        notes="Moderate first full-workflow run: 50k NFE, 511 ranks (2x254+3) "
              "on 4 Anvil wholenode nodes. 1/10 of the production budget; "
              "<=6.5h/seed at the measured cost surface.",
    ),
    # Anvil scaling supplement (Stage B strong scaling; see
    # workflow/supplemental/anvil_scaling_borg.sh and supplemental_config.py).
    # Fixed TOTAL NFE = 1280 across all scale_* geometries so wall time is
    # directly comparable (max_evaluations is per island -> 1280/islands).
    # runtime_frequency scales with per-island NFE so every geometry logs the
    # same ~8 snapshots per island. Run ONLY with the historic design +
    # DEBUG_SIM=true (~13 s/eval short window) — these measure Borg
    # coordination overhead, not search quality, and are NOT for production.
    # scale_1x64 / scale_2x32 / scale_4x16 share 64 evaluation slots, giving
    # the island-decomposition comparison at fixed parallelism.
    "scale_smoke": MOEAConfig(
        name="scale_smoke",
        n_islands=1,
        n_workers_per_island=4,
        max_evaluations=80,       # per island -> 80 total NFE
        runtime_frequency=20,
        n_seeds=1,
        max_time_hours=None,
        notes="Anvil scaling supplement smoke: proves the Stage B path "
              "(shared partition, timing CSV, runtime files) in ~15 min.",
    ),
    "scale_1x8": MOEAConfig(
        name="scale_1x8",
        n_islands=1,
        n_workers_per_island=8,
        max_evaluations=1280,     # per island -> 1,280 total NFE
        runtime_frequency=160,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: strong-scaling baseline, 10 ranks.",
    ),
    "scale_1x16": MOEAConfig(
        name="scale_1x16",
        n_islands=1,
        n_workers_per_island=16,
        max_evaluations=1280,
        runtime_frequency=160,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: 18 ranks.",
    ),
    "scale_1x32": MOEAConfig(
        name="scale_1x32",
        n_islands=1,
        n_workers_per_island=32,
        max_evaluations=1280,
        runtime_frequency=160,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: 34 ranks.",
    ),
    "scale_1x64": MOEAConfig(
        name="scale_1x64",
        n_islands=1,
        n_workers_per_island=64,
        max_evaluations=1280,
        runtime_frequency=160,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: 66 ranks; 64-slot single-island arm "
              "of the island-decomposition comparison.",
    ),
    "scale_2x32": MOEAConfig(
        name="scale_2x32",
        n_islands=2,
        n_workers_per_island=32,
        max_evaluations=640,      # per island -> 1,280 total NFE
        runtime_frequency=80,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: 67 ranks; 64-slot two-island arm.",
    ),
    "scale_4x16": MOEAConfig(
        name="scale_4x16",
        n_islands=4,
        n_workers_per_island=16,
        max_evaluations=320,      # per island -> 1,280 total NFE
        runtime_frequency=40,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: 69 ranks; 64-slot four-island arm "
              "(320 NFE/island is a short Borg trajectory — overhead "
              "measurement only).",
    ),
    # Campaign production config, sized against the measured Anvil cost surface
    # (outputs/supplemental/ensemble_cost_experiment) and the 750,000-SU
    # allocation; derivation in scenario_design_methods.md §6 / SI §S8.5.
    #
    # Geometry: 8 Anvil wholenode nodes = 1,024 cores. 4 islands x 254 workers
    # + 4 island-masters + 1 controller = 1 + 4*(254+1) = 1,021 ranks (3 idle
    # cores). Island partitioning is throughput-free at fixed slot count
    # (measured), so 4 islands is a search-reliability choice that keeps
    # per-island trajectories long (125k NFE/island).
    #
    # Budget: 500,000 total NFE x 1,000 scenario-years/eval (N=100, L=10).
    # At the measured 173.8 s/eval and 0.729 scaling efficiency: ~32.6 h wall,
    # ~33,400 SU per matched-design search; ~6 h / ~6,100 SU for the historic
    # single-trace reference at the same NFE. SU is nearly flat in node count,
    # so 8 nodes (vs 4) is a wall-time choice with no cost penalty.
    #
    # NFE-bounded (max_time_hours=None): a Borg maxTime cap could truncate NFE
    # unequally across designs, breaking the equal-NFE validity condition. The
    # SLURM --time wall (sized ~40 h) plus 2,500-NFE/island runtime snapshots
    # (50 per island, one per ~40 min) are the safety net; a killed run resumes
    # from the last snapshot. n_seeds=2 (S=2), submitted via `sbatch --array`;
    # the array index supplies the Borg RNG seed.
    "production": MOEAConfig(
        name="production",
        n_islands=4,
        n_workers_per_island=254,
        max_evaluations=125_000,          # per island -> 500,000 total NFE
        budget_scenario_years=500_000_000,  # 500k NFE x 1,000 sc-yr/eval
        runtime_frequency=2_500,
        n_seeds=2,
        max_time_hours=None,              # NFE-bounded; SLURM --time is the wall
        notes="Campaign config: 500k NFE, 1,021 ranks (4x254+5) on 8 Anvil "
              "nodes, ~32.6 h / ~33,400 SU per matched search. Sized against "
              "the 750k-SU allocation (scenario_design_methods.md §6).",
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
