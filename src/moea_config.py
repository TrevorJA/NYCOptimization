"""
moea_config.py - Registry of MOEA algorithm configurations (Borg settings only).

A MOEA config bundles the Multi-Master Borg algorithm settings (islands,
workers, NFE, runtime-snapshot frequency, seeds, wall-time) and is selected by
``NYCOPT_MOEA_CONFIG``; the scenario design (``src/scenario_designs.py``) is the
other run axis. Formulation, objectives, physics toggles and epsilons live on
the problem-definition axis (the moea slug), not here.

``smoke`` is dev-only; ``production`` is the campaign config
(``docs/notes/methods/campaign_design.md``). There is no resume: the Borg
checkpoint is disabled, so every search must finish inside one SLURM job.
"""

from __future__ import annotations

from dataclasses import dataclass


###############################################################################
# MOEAConfig
###############################################################################

@dataclass(frozen=True)
class MOEAConfig:
    """Immutable bundle of Multi-Master Borg algorithm settings.

    Attributes:
        name: Key used to select this config (``NYCOPT_MOEA_CONFIG``); appended
            to the moea slug when it is not the production default.
        n_islands: Number of MM Borg islands.
        n_workers_per_island: Worker ranks per island, used to size the MPI
            allocation (``None`` falls back to the SLURM allocation).
        max_evaluations: NFE per island (total NFE = islands x this) for every
            seed without an entry in ``max_evaluations_by_seed``.
        max_evaluations_by_seed: Per-island NFE for specific seeds; entry ``i``
            applies to seed ``i + 1`` (the 1-indexed ``sbatch --array`` index).
            Lets one config hold seeds run to different budgets; resolve with
            :meth:`max_evaluations_for_seed`.
        budget_scenario_years: Alternative budget control in total simulated
            scenario-years; unused by every registered config.
        runtime_frequency: NFE interval between Borg runtime-archive snapshots.
        n_seeds: Number of random seeds (independent search replicates).
        max_time_hours: Wall-time cap in hours, or ``None`` for NFE-bounded runs.
        notes: Free-form notes.
    """

    name: str
    n_islands: int | None = None
    n_workers_per_island: int | None = None
    max_evaluations: int | None = None
    max_evaluations_by_seed: tuple[int, ...] = ()
    budget_scenario_years: int | None = None
    runtime_frequency: int | None = None
    n_seeds: int | None = None
    max_time_hours: int | None = None
    notes: str = ""

    def max_evaluations_for_seed(self, seed: int) -> int | None:
        """Per-island NFE for ``seed`` (1-indexed).

        Args:
            seed: The Borg RNG seed / ``sbatch --array`` index, starting at 1.

        Returns:
            ``max_evaluations_by_seed[seed - 1]`` when that entry exists,
            otherwise ``max_evaluations`` (which may be ``None`` for a
            schema-only config).

        Raises:
            ValueError: If ``seed`` is not a positive integer.
        """
        if seed < 1:
            raise ValueError(f"seed must be >= 1 (got {seed})")
        if seed <= len(self.max_evaluations_by_seed):
            return self.max_evaluations_by_seed[seed - 1]
        return self.max_evaluations

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
    # Hopper-era 165-rank configs (4 islands x 40 workers + 4 island masters +
    # 1 controller): pilot = 5k NFE, mm_full = the pre-campaign 50k-NFE config.
    # NFE-bounded; the sbatch array index supplies the Borg RNG seed.
    "pilot": MOEAConfig(
        name="pilot",
        n_islands=4,
        n_workers_per_island=40,
        max_evaluations=1250,     # per island -> 5,000 total NFE
        runtime_frequency=250,
        n_seeds=1,
        max_time_hours=None,      # NFE-bounded; SLURM --time is the wall safety
        notes="Pilot: 5k NFE, 165 ranks (4x40+5), NFE-bounded.",
    ),
    "mm_full": MOEAConfig(
        name="mm_full",
        n_islands=4,
        n_workers_per_island=40,
        max_evaluations=12500,    # per island -> 50,000 total NFE
        runtime_frequency=1000,
        n_seeds=10,               # submitted as `sbatch --array=1-10`
        max_time_hours=None,
        notes="Pre-campaign Hopper config: 50k NFE, 165 ranks (4x40+5).",
    ),
    # 2 islands x 254 workers = 511 ranks on 4 x 128; 25k NFE/island (50k
    # total); single seed.
    "mm_moderate": MOEAConfig(
        name="mm_moderate",
        n_islands=2,
        n_workers_per_island=254,
        max_evaluations=25_000,   # per island -> 50,000 total NFE
        runtime_frequency=500,    # 50 runtime snapshots/island (diagnostics)
        n_seeds=1,
        max_time_hours=None,      # NFE-bounded; SLURM --time is the wall safety
        notes="Moderate run: 50k NFE, 511 ranks (2x254+3) on 4 Anvil nodes.",
    ),
    # Anvil scaling supplement (Stage B strong scaling;
    # workflow/supplemental/anvil_scaling_borg.sh, supplemental_config.py).
    # Fixed total NFE = 1280 across all scale_* geometries (max_evaluations is
    # per island); runtime_frequency scales so every geometry logs ~8
    # snapshots per island. Run only with the historic design + DEBUG_SIM=true;
    # these measure coordination overhead, not search quality. scale_1x64 /
    # scale_2x32 / scale_4x16 share 64 evaluation slots.
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
        notes="Anvil scaling supplement: 66 ranks; 64-slot single-island "
              "geometry of the island-decomposition comparison.",
    ),
    "scale_2x32": MOEAConfig(
        name="scale_2x32",
        n_islands=2,
        n_workers_per_island=32,
        max_evaluations=640,      # per island -> 1,280 total NFE
        runtime_frequency=80,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: 67 ranks; 64-slot two-island geometry.",
    ),
    "scale_4x16": MOEAConfig(
        name="scale_4x16",
        n_islands=4,
        n_workers_per_island=16,
        max_evaluations=320,      # per island -> 1,280 total NFE
        runtime_frequency=40,
        n_seeds=2,
        max_time_hours=None,
        notes="Anvil scaling supplement: 69 ranks; 64-slot four-island "
              "geometry (320 NFE/island is a short Borg trajectory; overhead "
              "measurement only).",
    ),
    # Campaign production config (campaign_design.md is the specification and
    # budget). 4 islands x 382 workers + 4 island masters + 1 controller =
    # 1,533 ranks on 12 x 128 cores, with NYCOPT_SEARCH_REALIZATION_BATCH=150
    # in the matched env files. Seed 1 runs 187,500/island (750k total), later
    # seeds 125,000/island (500k); every seed is reported at 500k from seed 1's
    # 125,000/island runtime snapshot (scripts/main/extract_runtime_archive.py).
    # NFE-bounded (a Borg maxTime cap could truncate NFE unequally across
    # designs); the SLURM --time wall is the only cap and there is no resume.
    # S=2 seeds, one per `sbatch --array` index (the Borg RNG seed).
    "production": MOEAConfig(
        name="production",
        n_islands=4,
        n_workers_per_island=382,
        max_evaluations=125_000,              # per island -> 500,000 total NFE
        max_evaluations_by_seed=(187_500,),   # seed 1 -> 750,000 total NFE
        runtime_frequency=2_500,
        n_seeds=2,
        max_time_hours=None,              # NFE-bounded; SLURM --time is the wall
        notes="Campaign config: 1,533 ranks (4x382+5) on 12 Anvil nodes, "
              "N=300 batched at 150; seed 1 at 750k NFE, seed 2 at 500k, "
              "reported at equal NFE from seed 1's 125k/island snapshot "
              "(campaign_design.md).",
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
