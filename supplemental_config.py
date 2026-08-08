"""supplemental_config.py - Single source of truth for the project's
**supplemental** experiments (those outside the main MOEA workflow).

Each supplemental experiment gets its own clearly-labelled section below. The
module is intentionally the one place to look for every supplemental knob, so a
run is reproducible from version-controlled files alone — no CLI value flags,
no edits to the main ``config.py`` settings.

IMPORTANT — import order and per-experiment env. ``config.py`` reads several
``NYCOPT_*`` and ``PYWRDRB_*`` environment variables *at its own import* (e.g.
to decide whether the salinity/temperature LSTMs run, and the simulation
window). Different experiments need **different** values for those knobs, so
a single set of module-top ``os.environ`` writes cannot serve them all.

Each experiment instead exposes a ``configure_*_env()`` function that applies
its env knobs via ``os.environ.setdefault``. Entry-point scripts call the
relevant one **between** importing this module and importing ``config``::

    import supplemental_config as scfg   # stdlib only; sets no env on import
    scfg.configure_ensemble_env()        # now set this experiment's env
    from config import ...               # config reads the env we just set

To keep that guarantee this module never imports ``config`` (which would
either fire too late or create a cycle). Output paths are derived from
``__file__`` for the same reason.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_DIR: Path = Path(__file__).resolve().parent

#: Root for all supplemental outputs (gitignored, regenerable); each experiment
#: writes under its own subdirectory.
SUPPLEMENTAL_OUTPUT_ROOT: Path = _PROJECT_DIR / "outputs" / "supplemental"


def _apply_env(*, salinity: str, temperature: str,
               sim_start: "str | None" = None,
               sim_end: "str | None" = None) -> None:
    """Apply the shared LSTM / simulation-window env knobs via ``setdefault``.

    Using ``setdefault`` means a value already present in the environment (e.g.
    exported by a SLURM ``.env`` file) wins, so the experiment defaults never
    silently override an explicit operator choice.

    Args:
        salinity: ``"1"``/``"0"`` for ``NYCOPT_SALINITY_ON``.
        temperature: ``"1"``/``"0"`` for ``NYCOPT_TEMPERATURE_ON``.
        sim_start: Optional ``PYWRDRB_SIM_START_DATE`` override.
        sim_end: Optional ``PYWRDRB_SIM_END_DATE`` override.
    """
    os.environ.setdefault("NYCOPT_SALINITY_ON", salinity)
    os.environ.setdefault("NYCOPT_TEMPERATURE_ON", temperature)
    if sim_start is not None:
        os.environ.setdefault("PYWRDRB_SIM_START_DATE", sim_start)
    if sim_end is not None:
        os.environ.setdefault("PYWRDRB_SIM_END_DATE", sim_end)


###############################################################################
# Objective-sensitivity experiment (HISTORIC, single trace)
#
# Runs many random DV vectors through the model on a single historical
# reference trace and measures, per objective, discrimination across policies
# and redundancy (Spearman). One simulation per DV vector; no ensemble loop.
###############################################################################

# ---------------------------------------------------------------------------
# Mode switch
# ---------------------------------------------------------------------------
#: SMOKE=True is a tiny local dry-run (few samples, short simulation window) to
#: prove the code path and output structure. Set SMOKE=False for the HPC
#: campaign — that single edit restores the full sample count and the full
#: historical simulation period.
SMOKE: bool = False


def configure_historic_env() -> None:
    """Apply env knobs for the historic single-trace objective-sensitivity run.

    Salinity and temperature LSTMs stay off: the diagnostic calibrates the
    ANNUAL-UNIT (§2) objectives, and that registry has no salt-front or thermal
    objective (the salinity LSTM checkout is also not present on every host).
    Under SMOKE a short window keeps each simulation to ~10-15 s; the end stays
    within the trimmed model's pre-simulated release data (2022-09-30
    water-year end).
    """
    _apply_env(
        salinity="0",
        temperature="0",
        sim_start="2019-10-01" if SMOKE else None,
        sim_end="2022-09-30" if SMOKE else None,
    )


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
#: RNG seed for the Latin-hypercube DV sample (reproducibility).
SEED: int = 42

#: Formulation whose DV bounds define the sampling space ("ffmp" or "ffmp_N").
FORMULATION: str = "ffmp"

#: Number of random DV vectors; the FFMP baseline is added as an extra
#: reference row. N ~ 200-500 gives stable Spearman estimates for the
#: redundancy screen (use that on the HPC); the small default is sized for
#: laptop passes over the full window.
N_SAMPLES: int = 3 if SMOKE else 24

#: Objective-set selection (config setting, not a CLI flag):
#:   "full_registry" -> every objective in src.objectives.OBJECTIVES (default;
#:                      lets the redundancy screen compare each recommended
#:                      metric against the diagnostic it replaces).
#:   "active"        -> config.ACTIVE_OBJECTIVES (the current recommended set).
#:   list[str]       -> an explicit list of registry names, used verbatim.
OBJECTIVE_SET: "str | list[str]" = "full_registry"

#: Olden & Poff (2003) redundancy flag: |Spearman rho| above this marks a pair
#: as collinear in the redundancy screen.
RHO_FLAG_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Output tree (kept separate from main optimization outputs; gitignored)
# ---------------------------------------------------------------------------
OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "objective_sensitivity"
SAMPLES_DIR: Path = OUTPUT_ROOT / "samples"
CORRELATIONS_DIR: Path = OUTPUT_ROOT / "correlations"
FIGURES_DIR: Path = OUTPUT_ROOT / "figures"


def _stem() -> str:
    """Run-identifying filename stem shared by all artifacts of one run."""
    return f"{FORMULATION}_seed{SEED}_n{N_SAMPLES}"


def samples_csv_path() -> Path:
    """Path to the raw per-sample objective CSV produced by the run script."""
    return SAMPLES_DIR / f"objective_samples_{_stem()}.csv"


def discrimination_csv_path() -> Path:
    """Path to the per-objective discrimination summary table."""
    return CORRELATIONS_DIR / f"discrimination_summary_{_stem()}.csv"


def spearman_csv_path() -> Path:
    """Path to the full Spearman rank-correlation matrix."""
    return CORRELATIONS_DIR / f"spearman_matrix_{_stem()}.csv"


def flagged_pairs_csv_path() -> Path:
    """Path to the table of |rho| > threshold objective pairs."""
    return CORRELATIONS_DIR / f"flagged_pairs_{_stem()}.csv"


def figure_path(name: str, ext: str) -> Path:
    """Path for a named figure artifact (e.g. name='discrimination', ext='pdf')."""
    return FIGURES_DIR / f"{name}_{_stem()}.{ext}"


###############################################################################
# Epsilon-calibration experiment
# (docs/notes/methods/epsilon_calibration_experiment.md;
#  workflow/supplemental/epsilon_calibration.sh)
#
# Recalibrates the ANNUAL-UNIT (§2) search epsilons — the values Borg's
# ε-dominance archive actually uses — on the CAMPAIGN search measures, replacing
# the 24-policy historic-trace provenance. One sbatch job per scenario design
# (design selected by the sourced NYCOPT_ENV_FILE, exactly as the MM-Borg
# launcher does): sample EPS_N_POLICIES constraint-FEASIBLE random DV vectors
# (uniform-on-feasible via rejection; random vectors are ~1% feasible),
# evaluate each on that design's search ensemble through the same batched path
# Borg workers run, and persist the full per-unit annual-metric cube
# (n_dv x n_real x n_obj x n_units). The figures script then derives, per
# objective and per design: the signal scale (IQR/10 across feasible policies),
# the estimator noise floor (bootstrap over realizations; over unit-years for
# the single-trace historic design), the frequency-granularity floor, an
# ε-nondominated archive-size sweep, and a clean-rounded recommendation
# eps = ceil_clean(max(signal, noise, granularity)) — combined across designs
# into the single campaign vector (JARs and the Borg problem share one set).
###############################################################################

# ---------------------------------------------------------------------------
# Mode switch (independent of the historic SMOKE above)
# ---------------------------------------------------------------------------
#: EPS_SMOKE=True is a tiny dry-run (few policies, cheap bootstrap) to prove
#: the code path and output structure; flip to False for the HPC run.
EPS_SMOKE: bool = False


def configure_epsilon_env() -> None:
    """Apply env knobs for the epsilon-calibration run.

    Salinity and temperature LSTMs off (the annual-unit registry uses
    neither). No scenario-design default is set here: the design IS the run
    identity and must come from the sourced ``NYCOPT_ENV_FILE``
    (``NYCOPT_SCENARIO_DESIGN``), mirroring the MM-Borg launcher. No
    simulation-window override — the ensemble window self-derives from the
    realization length, and the historic design uses the full record.
    """
    _apply_env(salinity="0", temperature="0")


# ---------------------------------------------------------------------------
# Feasible-DV sample
# ---------------------------------------------------------------------------
#: RNG seed for the feasible-DV rejection sample (kept distinct from the
#: ensemble-generation seeds; every rank regenerates the identical sample).
EPS_SEED: int = 42

#: Formulation whose DV bounds + constraints define the feasible region.
EPS_FORMULATION: str = "ffmp"

#: Number of FEASIBLE random policies (the FFMP baseline is added as an extra
#: reference row, id -1). 512 sits in the doc's 200-500+ stable-IQR range and
#: clears one 128-rank wholenode in ~4-5 eval waves (~15 min at 173.8 s/eval).
EPS_N_POLICIES: int = 8 if EPS_SMOKE else 512

#: Hard cap on rejection draws (at ~1% acceptance, 512 feasible needs ~5e4).
EPS_MAX_DRAWS: int = 200_000 if EPS_SMOKE else 20_000_000

#: Realizations per simulation batch. 0 = all realizations as one pywr
#: scenario block — the CAMPAIGN default (config.SEARCH_REALIZATION_BATCH),
#: kept so the calibration measures exactly what Borg workers run.
EPS_REALIZATION_BATCH: int = 0

# ---------------------------------------------------------------------------
# Analysis grids (figures script only; no re-simulation)
# ---------------------------------------------------------------------------
#: Bootstrap resamples for the estimator-noise floor (resampling realizations
#: with replacement; unit-years for the single-trace historic design).
EPS_BOOTSTRAP_B: int = 50 if EPS_SMOKE else 1000

#: RNG seed for the bootstrap index draw (one shared draw per design, so every
#: objective/policy sees the same resampled realizations).
EPS_BOOTSTRAP_SEED: int = 7

#: Multipliers applied to the recommended epsilon vector for the archive-size
#: sweep (how strongly does epsilon resolution control Pareto-set cardinality).
EPS_SCALE_GRID: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)

#: Designs whose raw floors enter the campaign epsilon max (figures script).
#: The historic single-trace design is deliberately EXCLUDED: it is a
#: reported reference arm, not a matched-contrast arm, and
#: its 76-unit-year estimator's noise floor would coarsen the shared vector
#: ~3-4x beyond what the ensemble search measures need (e.g. reliability
#: epsilon 0.10 instead of 0.02 on the 0-1 scale). Its cube and per-design
#: diagnostics stay reported for context; the historic arm's archive
#: consequently resolves below its own noise floor (accepted, disclosed).
EPS_CAMPAIGN_DESIGNS: tuple = ("fixed_probabilistic", "hazard_filling_stationary")

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable)
# ---------------------------------------------------------------------------
EPS_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "epsilon_calibration"
EPS_CUBE_DIR: Path = EPS_OUTPUT_ROOT / "cube"
EPS_TABLES_DIR: Path = EPS_OUTPUT_ROOT / "tables"
EPS_FIGURES_DIR: Path = EPS_OUTPUT_ROOT / "figures"


def _eps_stem(design: str) -> str:
    """Run-identifying filename stem for one design's calibration artifacts."""
    return f"{EPS_FORMULATION}_{design}_seed{EPS_SEED}_n{EPS_N_POLICIES}"


def epsilon_cube_path(design: str) -> Path:
    """Path to one design's per-unit annual-metric cube HDF5 (run output)."""
    return EPS_CUBE_DIR / f"unit_cube_{_eps_stem(design)}.h5"


def epsilon_cube_glob() -> str:
    """Glob (relative to EPS_CUBE_DIR) matching every design's cube at the
    current sample settings — the figures script analyzes all it finds."""
    return f"unit_cube_{EPS_FORMULATION}_*_seed{EPS_SEED}_n{EPS_N_POLICIES}.h5"


def epsilon_table_path(name: str, design: "str | None" = None) -> Path:
    """Path for a named table CSV; per-design when ``design`` is given, else a
    cross-design combined artifact (e.g. the final recommendation table)."""
    stem = _eps_stem(design) if design else \
        f"{EPS_FORMULATION}_combined_seed{EPS_SEED}_n{EPS_N_POLICIES}"
    return EPS_TABLES_DIR / f"{name}_{stem}.csv"


def epsilon_figure_path(name: str, design: "str | None" = None,
                        ext: str = "png") -> Path:
    """Path for a named figure artifact (per-design or combined)."""
    stem = _eps_stem(design) if design else \
        f"{EPS_FORMULATION}_combined_seed{EPS_SEED}_n{EPS_N_POLICIES}"
    return EPS_FIGURES_DIR / f"{name}_{stem}.{ext}"


###############################################################################
# Anvil parallel-scaling experiment
# (workflow/supplemental/anvil_scaling_*.sh; manuscript supplement)
#
# Two measured stages plus a post-hoc analysis, following the Reed-group
# scaling-experiment conventions (strong scaling, speedup vs ideal, parallel
# efficiency = speedup/p, replicate bands):
#
#   Stage A (packing): on ONE exclusive 128-core Anvil node, sweep the number
#     of concurrent MPI ranks K, each rank timing cold+warm trimmed-model
#     ensemble evaluations (the exact `evaluate()` path Borg workers run).
#     Yields per-eval slowdown vs K, node throughput, SU cost per eval, and
#     per-rank peak RSS vs the 256 GB node memory — the ranks-per-node choice.
#   Stage B (Borg strong scaling): fixed total NFE, sweep island x worker
#     geometry (registered as `scale_*` MOEA configs in src/moea_config.py),
#     >=2 seeds per geometry, on the historic design with the DEBUG_SIM short
#     window (~13 s/eval; Borg coordination overhead is measured, not search
#     quality — the inflated overhead:eval ratio makes this a conservative
#     efficiency bound).
#
# IMPORTANT: Stage A must never combine with DEBUG_SIM / PYWRDRB_SIM_* date
# overrides — the ensemble window self-derives from the realization length
# (src/simulation.py::_ensemble_window) and a date override would shift it off
# the staged HDF5 axis. Stage B (historic single-trace) is where the short
# window is valid.
###############################################################################


def configure_anvil_scaling_env() -> None:
    """Apply env knobs for the Anvil scaling experiment (both stages).

    Salinity and temperature LSTMs off (the active objective set uses
    neither). No simulation-window override: Stage A's window self-derives
    from the ensemble realization length, and Stage B's short window is set
    by ``DEBUG_SIM=true`` in the SLURM script (via ``nycopt_read_run_identity``),
    not here.
    """
    _apply_env(salinity="0", temperature="0")


# ---------------------------------------------------------------------------
# Stage A — packing sweep
# ---------------------------------------------------------------------------
#: Steps per packing mode: (K concurrent ranks, warm evals per rank M,
#: realization batch B). B=0 is the production default (all realizations as
#: one pywr scenario block); B>0 exercises the memory-batched eval path.
#: "smoke" proves the code path in ~10 min; "ladder" is the full density
#: sweep; "spot" re-measures the candidate densities with more warm evals for
#: tighter statistics, plus one batched step for the memory-vs-time trade.
#: The spot K values are EDITED here after reviewing the ladder results — a
#: committed one-line change, the artifact of record (no shell flags).
#: M is larger at low K because those points normalize everything downstream:
#: the K=1 warm median is the slowdown/throughput baseline, so it gets 6 warm
#: samples (the extra evals are cheap on an otherwise-idle node).
PACKING_MODES: "dict[str, list[tuple[int, int, int]]]" = {
    "smoke":  [(1, 1, 0), (4, 1, 0)],
    "ladder": [(1, 6, 0), (8, 4, 0), (16, 2, 0), (32, 2, 0),
               (48, 2, 0), (64, 2, 0), (96, 2, 0), (128, 2, 0)],
    # Spot densities from the packing-ladder measurements: slowdown is
    # only ~1.17x at K=128 with ~89 GB projected node memory, so SU/eval is
    # minimized at full packing — re-measure the two densest points, plus one
    # batched step at K* for the memory-vs-time trade.
    "spot":   [(96, 4, 0), (128, 4, 0), (128, 4, 16)],
}

#: Batched-evaluation sweep (mode "batch"): B realizations per pywr
#: model.run() (``NYCOPT_SEARCH_REALIZATION_BATCH``), measured at K=1 (clean
#: per-run-overhead amortization curve, no contention) and at the chosen
#: packing density K* (the joint (K, B) operating point the campaign actually
#: runs at). B=0 is the production default: ALL realizations as one pywr
#: scenario block. Larger B amortizes model build/setup across scenarios but
#: holds more scenario state in memory — this sweep maps that trade so the
#: campaign picks (K, B) jointly rather than one axis at a time.
#: K* is env-overridable so the sweep can be (re)run once the ladder fixes
#: the real density: submit with NYCOPT_PACK_BATCH_KSTAR=<k>.
PACKING_BATCH_KSTAR: int = int(os.environ.get("NYCOPT_PACK_BATCH_KSTAR", "32"))
PACKING_BATCH_SIZES: "tuple[int, ...]" = (1, 2, 5, 10, 0)
PACKING_MODES["batch"] = [
    (k, 3, b) for k in (1, PACKING_BATCH_KSTAR) for b in PACKING_BATCH_SIZES
]

#: Cap on the per-rank start stagger (rank r sleeps min(r, cap) seconds before
#: its first eval) that decorrelates the ranks' memory-access phases.
PACKING_STAGGER_MAX_S: int = 30

#: Formulation whose baseline DVs are evaluated (per-eval cost is set by model
#: size x timesteps, not DV values).
PACKING_FORMULATION: str = "ffmp"

# ---------------------------------------------------------------------------
# Stage B — MM Borg strong scaling
# ---------------------------------------------------------------------------
#: Geometry table: MOEA config name -> (MPI ranks, sbatch --time). Ranks MUST
#: equal MOEAConfig.total_ntasks_mpi = 1 + islands*(workers+1); the submit
#: helper asserts this against src/moea_config.py before every sbatch. All
#: geometries fit one Anvil node (<=128 ranks) -> shared partition, per-core
#: SU charging. Times are sized from ~13 s/eval x 1280 total NFE with slack.
BORG_SCALE_GEOMETRIES: "dict[str, tuple[int, str]]" = {
    "scale_smoke": (6, "00:30:00"),
    "scale_1x8":  (10, "01:30:00"),
    "scale_1x16": (18, "01:00:00"),
    "scale_1x32": (34, "00:45:00"),
    "scale_1x64": (66, "00:30:00"),
    "scale_2x32": (67, "00:30:00"),
    "scale_4x16": (69, "00:30:00"),
}

#: Independent Borg RNG seed replicates per geometry (submitted as
#: ``sbatch --array=1-N``); seed variability bands in the scaling figures.
BORG_SCALE_SEEDS: int = 2

# ---------------------------------------------------------------------------
# Stage C — analysis / projection knobs
# ---------------------------------------------------------------------------
#: Anvil standard CPU node: 2x AMD EPYC 7763, 128 cores, 256 GB. wholenode
#: SU charging is per node-hour x 128 cores regardless of ranks used — the
#: quantity the packing sweep optimizes against.
SCALING_NODE_CORES: int = 128
SCALING_NODE_MEM_GB: int = 256

#: Production-campaign projection grid (figure F5): candidate node counts and
#: island counts at the chosen packing density K*, and the campaign NFE the
#: projection is expressed for (mm_full's 50k total NFE).
SCALING_PROJECTION_NODES: "tuple[int, ...]" = (2, 4, 8, 16)
SCALING_PROJECTION_ISLANDS: "tuple[int, ...]" = (2, 4, 8)
SCALING_PROJECTION_TOTAL_NFE: int = 50_000

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable; self-contained for the supplement)
# ---------------------------------------------------------------------------
SCALING_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "anvil_scaling_experiment"
SCALING_PACKING_DIR: Path = SCALING_OUTPUT_ROOT / "packing"
SCALING_BORG_DIR: Path = SCALING_OUTPUT_ROOT / "borg"
SCALING_FIGURES_DIR: Path = SCALING_OUTPUT_ROOT / "figures"
SCALING_TABLES_DIR: Path = SCALING_OUTPUT_ROOT / "tables"
SCALING_MANIFESTS_DIR: Path = SCALING_OUTPUT_ROOT / "manifests"


def packing_shard_path(k: int, batch: int, rank: int, job_id: str) -> Path:
    """Per-rank CSV shard path for one packing step (K, batch) of one job."""
    return (SCALING_PACKING_DIR
            / f"k{k:03d}_b{batch}_rank{rank:03d}_{job_id}.csv")


def packing_step_manifest_path(k: int, batch: int, job_id: str) -> Path:
    """JSON manifest path (exit code, epochs) for one packing step."""
    return SCALING_PACKING_DIR / f"step_k{k:03d}_b{batch}_{job_id}.json"


def borg_timing_csv_path(config_name: str, seed: int, job_id: str) -> Path:
    """One-row wall-time CSV path for one Stage B (geometry, seed) job."""
    return SCALING_BORG_DIR / f"timing_{config_name}_seed{seed:02d}_{job_id}.csv"


###############################################################################
# Ensemble-cost experiment — the t_eval(N, L, model) cost surface
# (docs/notes/methods/ensemble_cost_experiment.md;
#  workflow/supplemental/ensemble_cost_*.sh)
#
# The Anvil packing sweep measured ONE ensemble shape (kn_20yr_n20) across
# ranks-per-node. It says nothing about how a Borg evaluation's cost moves with
# the ensemble's SHAPE, and the campaign is sized in that shape: N realizations
# x L years, trimmed model for search AND re-evaluation (full model only for
# presim passes + the historic baseline). pywrdrb
# runs realizations as pywr SCENARIOS inside one model, so per-eval cost is
# sub-linear in N (vectorized per-timestep work) but ~linear in L (timesteps);
# a cost per scenario-year taken from one (N, L) point therefore misprices every
# other point, in the direction that matters most for the N-vs-L trade.
#
# This experiment measures that surface directly: for every cell (N, L, model),
# K concurrent ranks on one exclusive node each run 1 cold + M warm evaluations
# through the production ``evaluate()`` path (the same worker the packing sweep
# uses), recording wall time and peak RSS. The analysis derives the empirical N
# and L exponents, the full/trimmed ratio, and the SU projection for the search
# campaign and the held-out test-ensemble re-evaluation.
#
# DENSITY. Cells run at the largest memory-feasible ranks-per-node K <= 128.
# The packing sweep found SU/eval is minimized at full packing (128 ranks:
# 20.8 SU/1000 evals vs 71.7 at 32, only 1.17x per-eval slowdown), so 128 is the
# density the campaign would actually run at and therefore the density the cost
# surface must be priced at. Memory is the binding constraint, not contention:
# a 256 GB node gives ~2 GB/rank at K=128, and the large cells exceed that.
# ``ensemble_cost_cell_k`` derives each cell's K from the measured RSS model
# below; K is recorded in every shard and all SU math normalizes by it.
#
# IMPORTANT: never combine this experiment with DEBUG_SIM or PYWRDRB_SIM_* date
# overrides — the window self-derives from the realization length
# (src/simulation.py::_ensemble_window) and an override shifts it off the staged
# HDF5 date axis.
###############################################################################


def configure_ensemble_cost_env() -> None:
    """Apply env knobs for the ensemble-cost experiment.

    Salinity and temperature LSTMs off (the active objective set uses
    neither). No simulation-window override: the window self-derives from each
    cell's realization length. Deliberately does NOT touch
    ``NYCOPT_USE_TRIMMED_MODEL`` — the sweep script exports it per cell, and a
    ``setdefault`` here would make every "full" cell silently re-measure the
    trimmed model with no error.
    """
    _apply_env(salinity="0", temperature="0")


# ---------------------------------------------------------------------------
# The measured grid
# ---------------------------------------------------------------------------
#: Realizations N. Spans the campaign design point (100) by an order of
#: magnitude either side, so the sub-linear exponent is estimated over a decade
#: rather than interpolated between neighbours.
ENSEMBLE_COST_N_GRID: "tuple[int, ...]" = (1, 10, 20, 50, 100, 200)

#: Realization length L in years. 10 is the campaign design length; 30 bounds
#: the held-out test ensemble's L_test.
ENSEMBLE_COST_L_GRID: "tuple[int, ...]" = (5, 10, 20, 30)

#: Model variants: the search path and the re-evaluation path.
ENSEMBLE_COST_MODELS: "tuple[str, ...]" = ("trimmed", "full")

#: Formulation whose baseline DVs are evaluated. Per-eval cost is set by model
#: size x timesteps x scenarios, not by DV values; identical DVs across ranks
#: also make the objective vector byte-comparable (a free correctness check).
ENSEMBLE_COST_FORMULATION: str = "ffmp"

#: Cap on the per-rank start stagger (rank r sleeps min(r, cap) s before its
#: first eval), decorrelating the ranks' memory-access phases.
ENSEMBLE_COST_STAGGER_MAX_S: int = 30

# ---------------------------------------------------------------------------
# Cost model — sets each cell's packing density K and the sweep's time guard
# ---------------------------------------------------------------------------
# Both models below are CALIBRATION CONSTANTS, not results: they exist only to
# choose K per cell and to guard the job's walltime. They are committed edits
# (the artifact of record, per the packing sweep's convention), seeded from the
# packing sweep's measured point and updated from the "probe" mode's corner
# cells before the production sweeps run. Every reported number comes from the
# measurement, never from these.

#: Peak RSS per rank, MB: base + per_ry * (N * L). CALIBRATED from the probe
#: (K=1, no contention) at (N=1, L=5) and (N=200, L=10), plus the smoke cell at
#: (N=20, L=20). Fitted to the two LARGE cells, which makes the model
#: over-predict the small ones (603 MB vs an actual 441 MB at 5
#: realization-years) — the safe direction, since the error only costs a small
#: cell some packing density it does not need, while an under-prediction at the
#: large end would OOM a node.
#: The full model's memory is barely above the trimmed model's (+3% base, +13%
#: slope), NOT the 1.5-2x its extra live reservoirs suggested. Those reservoirs
#: are STARFIT release rules, not additional LP structure.
ENSEMBLE_COST_RSS_MB: "dict[str, tuple[float, float]]" = {
    "trimmed": (601.0, 0.394),
    "full": (617.0, 0.444),
}

#: Warm per-eval seconds: a + b * L * N**alpha. CALIBRATED from the same three
#: cells (a = the fixed per-eval overhead read off the N=1 cell; alpha from the
#: N=200 cell). Reproduces the smoke cell to within 3% (trimmed) and 12%
#: (full, over-predicting).
#: NOTE the exponents: ~0.96 trimmed, ~0.93 full — cost is very nearly LINEAR in
#: N, not the strong sub-linearity the pywr-scenario structure was assumed to
#: buy. Cost per scenario-year is almost flat (0.158 s/ry at 400 ry vs 0.145 s/ry
#: at 2000 ry). These constants only size the sweep's walltime guard; the
#: exponent is MEASURED properly by the N sweep at fixed L (see
#: ``scaling_fits.csv``), which is the deliverable.
ENSEMBLE_COST_T_EST_S: "dict[str, tuple[float, float, float]]" = {
    "trimmed": (1.5, 0.182, 0.956),
    "full": (1.5, 0.248, 0.929),
}

#: Fixed per-step cost beyond the evals themselves: interpreter + pywrdrb
#: import, model build, and the cold eval's extra model write/load.
ENSEMBLE_COST_STEP_OVERHEAD_S: float = 120.0

#: Safety factor on the step-time estimate used by the sweep's budget guard.
ENSEMBLE_COST_GUARD_MARGIN: float = 1.6

#: Fraction of node memory a cell may occupy at its chosen K. Headroom covers
#: the cold-build spike, the /dev/shm model JSONs, and RSS-model error.
ENSEMBLE_COST_MEM_SAFETY: float = 0.85

#: Candidate packing densities, densest first. 128 = full node (the SU-optimal
#: density measured by the packing sweep); the rest are fallbacks for cells whose
#: RSS will not fit 128 ranks in 256 GB.
ENSEMBLE_COST_K_LADDER: "tuple[int, ...]" = (128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1)


def ensemble_cost_rss_est_mb(n: int, ell: int, model: str) -> float:
    """Estimated peak RSS per rank (MB) for one cell, from the calibrated model.

    Args:
        n: Realizations.
        ell: Realization length in years.
        model: ``"trimmed"`` or ``"full"``.

    Returns:
        Estimated peak resident set size of one evaluating rank, in MB.
    """
    base, per_ry = ENSEMBLE_COST_RSS_MB[model]
    return base + per_ry * float(n * ell)


def ensemble_cost_t_est_s(n: int, ell: int, model: str) -> float:
    """Estimated warm per-eval wall time (s) for one cell, from the cost model.

    Used only to size the sweep's per-cell walltime guard; the measured value
    is the experiment's output.

    Args:
        n: Realizations.
        ell: Realization length in years.
        model: ``"trimmed"`` or ``"full"``.

    Returns:
        Estimated warm evaluation wall time in seconds.
    """
    a, b, alpha = ENSEMBLE_COST_T_EST_S[model]
    return a + b * float(ell) * float(n) ** alpha


def ensemble_cost_cell_k(n: int, ell: int, model: str) -> int:
    """Densest packing K a cell fits in node memory, from the RSS model.

    Walks ``ENSEMBLE_COST_K_LADDER`` densest-first and returns the first K whose
    projected node total ``K * RSS_est`` stays under
    ``ENSEMBLE_COST_MEM_SAFETY * SCALING_NODE_MEM_GB``. Cells that cannot reach
    128 are a finding in their own right: memory, not contention, is what caps
    the campaign's packing density at large N.

    Args:
        n: Realizations.
        ell: Realization length in years.
        model: ``"trimmed"`` or ``"full"``.

    Returns:
        Ranks per node for this cell (>= 1; 1 even if the estimate exceeds node
        memory, so the cell is still attempted and any OOM is recorded as a
        measurement rather than silently skipped).
    """
    budget_mb = ENSEMBLE_COST_MEM_SAFETY * SCALING_NODE_MEM_GB * 1024.0
    rss_mb = ensemble_cost_rss_est_mb(n, ell, model)
    for k in ENSEMBLE_COST_K_LADDER:
        if k * rss_mb <= budget_mb:
            return k
    return 1


def ensemble_cost_step_estimate_s(n: int, ell: int, model: str, m_warm: int) -> float:
    """Guard-sized wall time (s) for one sweep step: 1 cold + ``m_warm`` warm evals."""
    evals_s = (1 + m_warm) * ensemble_cost_t_est_s(n, ell, model)
    return ENSEMBLE_COST_GUARD_MARGIN * (evals_s + ENSEMBLE_COST_STEP_OVERHEAD_S)


# ---------------------------------------------------------------------------
# Sweep modes — the ordered cell lists each SLURM job runs
# ---------------------------------------------------------------------------
# A cell is ``(N, L, model, m_warm, k)``. ``k=0`` means "derive from the RSS
# model" (``ensemble_cost_cell_k``), which is what every production cell uses:
# recalibrating the RSS model then re-picks every density from one edit. smoke
# and probe pin k explicitly because they run on the shared partition.
#
# Priority order follows the budget question. The campaign design point
# (N=100, L=10) is measured first and with more warm evals than anything else,
# because it is the number that prices the whole campaign; then the N sweep at
# L=10 (the sub-linearity), the L sweep at N=100 (the linearity), and the
# full-model points needed for the re-evaluation ratio. Everything else is the
# factorial remainder, split trimmed/full so the memory-hungry full cells cannot
# take the trimmed surface down with them.

#: Cheap correctness gate on the already-staged kn_20yr_n20: proves the full
#: model builds and runs on a staged ensemble, and that NYCOPT_USE_TRIMMED_MODEL
#: actually reaches config (trimmed and full objective vectors must differ).
ENSEMBLE_COST_SMOKE: "list[tuple[int, int, str, int, int]]" = [
    (20, 20, "trimmed", 1, 4),
    (20, 20, "full", 1, 4),
]

#: Cells at K=1 (one rank, no contention) that calibrate
#: ``ENSEMBLE_COST_RSS_MB`` and ``ENSEMBLE_COST_T_EST_S`` before any wholenode
#: job runs. Both models at the cheapest cell and at a large one: the base is
#: read off the former, the slope off the lever arm between them.
#: The large end is (200, 10) = 2000 realization-years rather than the grid's
#: true corner (200, 30). Staging a 200-realization x 20-30 yr ensemble is by far
#: the slowest step in the experiment (step 04's predicted-inflow pass is
#: compute-bound and runs for hours at that size), and the calibration does not
#: need it: RSS is linear in realization-years, so a 5 -> 2000 lever arm pins the
#: model and extrapolates to 6000 fine. Blocking the whole sweep on those two
#: ensembles would buy nothing.
ENSEMBLE_COST_PROBE: "list[tuple[int, int, str, int, int]]" = [
    (1, 5, "trimmed", 1, 1),
    (1, 5, "full", 1, 1),
    (200, 10, "trimmed", 1, 1),
    (200, 10, "full", 1, 1),
]

#: The cells that unblock the budget, in the order they must be measured.
ENSEMBLE_COST_CORE: "list[tuple[int, int, str, int, int]]" = [
    # N=1 first: it is the only cell whose staging and scenario block are a
    # degenerate edge case, and it costs ~1 min — surface any failure in the
    # job's first minutes rather than after hours.
    (1, 10, "trimmed", 2, 0),
    # (1) The campaign design point, both models. More warm evals than anywhere
    # else: every SU number in the projection is this cell's median.
    (100, 10, "trimmed", 4, 0),
    (100, 10, "full", 4, 0),
    # (2) N sweep at L=10, trimmed — the sub-linearity in N.
    (10, 10, "trimmed", 2, 0),
    (20, 10, "trimmed", 2, 0),
    (50, 10, "trimmed", 2, 0),
    (200, 10, "trimmed", 2, 0),
    # (3) L sweep at N=100, trimmed — the linearity in L.
    (100, 5, "trimmed", 2, 0),
    (100, 20, "trimmed", 2, 0),
    (100, 30, "trimmed", 2, 0),
    # (4) Extra full-model points so the full/trimmed ratio is measured at >= 3
    # (N, L) cells rather than assumed constant, and so the re-eval projection
    # rests on a full-model point at the L_test=30 end.
    (20, 10, "full", 2, 0),
    (100, 30, "full", 2, 0),
]


def _remaining_cells(model: str, m_warm: int) -> "list[tuple[int, int, str, int, int]]":
    """Factorial cells of one model not already covered by core, cheapest first.

    Ordered by N*L (realization-years, the cost proxy) so a job that runs out of
    walltime loses the most expensive cells, which are also the ones whose
    absence the power-law fit tolerates best.
    """
    covered = {(n, ell, mdl) for n, ell, mdl, _, _ in ENSEMBLE_COST_CORE}
    cells = [
        (n, ell, model, m_warm, 0)
        for ell in ENSEMBLE_COST_L_GRID
        for n in ENSEMBLE_COST_N_GRID
        if (n, ell, model) not in covered
    ]
    return sorted(cells, key=lambda c: c[0] * c[1])


ENSEMBLE_COST_MODES: "dict[str, list[tuple[int, int, str, int, int]]]" = {
    "smoke": ENSEMBLE_COST_SMOKE,
    "probe": ENSEMBLE_COST_PROBE,
    "core": ENSEMBLE_COST_CORE,
    "rest_trimmed": _remaining_cells("trimmed", 2),
    "rest_full": _remaining_cells("full", 2),
}


def ensemble_cost_staging_cells() -> "list[tuple[int, int]]":
    """Every (N, L) ensemble the experiment needs staged, cheapest first."""
    return sorted(
        {(n, ell) for ell in ENSEMBLE_COST_L_GRID for n in ENSEMBLE_COST_N_GRID},
        key=lambda c: c[0] * c[1],
    )


# ---------------------------------------------------------------------------
# Campaign projection — what the measured surface is FOR
# ---------------------------------------------------------------------------
#: Search side: 6 scenario designs x K draws x S MOEA seeds independent Borg
#: runs, each of NFE evaluations at the campaign design point (N=100, L=10) on
#: the trimmed model. The grids are the open sizing decisions.
ENSEMBLE_COST_PROJ_DESIGNS: int = 6
ENSEMBLE_COST_PROJ_DRAWS: "tuple[int, ...]" = (5, 10)
ENSEMBLE_COST_PROJ_SEEDS: "tuple[int, ...]" = (2, 3)
ENSEMBLE_COST_PROJ_NFE: "tuple[int, ...]" = (25_000, 50_000, 100_000)

#: The campaign design point itself: the (N, L) whose measured trimmed cost
#: prices the search campaign.
ENSEMBLE_COST_DESIGN_POINT: "tuple[int, int]" = (100, 10)

#: MM-Borg geometry the search projection assumes: nodes per Borg run and
#: islands. The worker count is nodes*K - 1 - islands (one controller rank plus
#: one master per island), matching the Stage-B convention in analyze_scaling.py.
ENSEMBLE_COST_PROJ_NODES: int = 4
ENSEMBLE_COST_PROJ_ISLANDS: int = 2

#: Parallel efficiency applied to Borg search. Measured on the Anvil Stage-B
#: strong-scaling sweep (scale_1x64: speedup 5.83 vs ideal 8.0 -> 0.729,
#: outputs/supplemental/anvil_scaling_experiment/tables/borg_summary.csv). That
#: run used ~13 s evals, so coordination overhead is a LARGER share of wall time
#: there than at the campaign's minute-scale ensemble evals: applying it here
#: over-estimates walltime, i.e. the search projection is conservative.
ENSEMBLE_COST_PROJ_EFFICIENCY: float = 0.729

#: Re-evaluation side: n_policies archived policies re-simulated on the held-out
#: test ensemble E_test (N_theta forcing draws x R realizations each, L_test yr)
#: on the TRIMMED model. 1,200 = ~400 policies per design
#: at the calibrated epsilons, merged across seeds. The adopted E_test cell is
#: (N_theta, R, L_test) = (1000, 25, 50); the grid brackets it.
ENSEMBLE_COST_REEVAL_POLICIES: int = 1200
ENSEMBLE_COST_ETEST_NTHETA: "tuple[int, ...]" = (500, 1000, 1500)
ENSEMBLE_COST_ETEST_R: "tuple[int, ...]" = (10, 25)
ENSEMBLE_COST_ETEST_LTEST: "tuple[int, ...]" = (10, 50)

#: Re-evaluation is an embarrassingly parallel task farm, not a Borg search: no
#: island coordination, no synchronizing generations. Its only loss is a master
#: rank plus the ragged tail of the last wave, so it is priced at a utilization
#: factor, NOT at the Borg efficiency above.
ENSEMBLE_COST_REEVAL_UTILIZATION: float = 0.90

#: The allocation every projected cost is stated as a fraction of.
ENSEMBLE_COST_ALLOCATION_SU: int = 1_000_000

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable; self-contained for the supplement)
# ---------------------------------------------------------------------------
ENSEMBLE_COST_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "ensemble_cost_experiment"
ENSEMBLE_COST_CELLS_DIR: Path = ENSEMBLE_COST_OUTPUT_ROOT / "cells"
ENSEMBLE_COST_TABLES_DIR: Path = ENSEMBLE_COST_OUTPUT_ROOT / "tables"
ENSEMBLE_COST_FIGURES_DIR: Path = ENSEMBLE_COST_OUTPUT_ROOT / "figures"
ENSEMBLE_COST_MANIFESTS_DIR: Path = ENSEMBLE_COST_OUTPUT_ROOT / "manifests"


def ensemble_cost_shard_path(n: int, ell: int, model: str, k: int,
                             rank: int, job_id: str) -> Path:
    """Per-rank CSV shard path for one cell (N, L, model, K) of one job."""
    return (ENSEMBLE_COST_CELLS_DIR
            / f"n{n:03d}_L{ell:02d}_{model}_k{k:03d}_rank{rank:03d}_{job_id}.csv")


def ensemble_cost_step_manifest_path(n: int, ell: int, model: str, k: int,
                                     job_id: str) -> Path:
    """JSON manifest path (exit code, epochs) for one sweep step."""
    return (ENSEMBLE_COST_CELLS_DIR
            / f"step_n{n:03d}_L{ell:02d}_{model}_k{k:03d}_{job_id}.json")


def ensemble_cost_table_path(name: str) -> Path:
    """Path for a named ensemble-cost table CSV (e.g. name='cost_surface')."""
    return ENSEMBLE_COST_TABLES_DIR / f"{name}.csv"


def ensemble_cost_figure_path(name: str) -> Path:
    """Path stub for a named ensemble-cost figure (extension added by save_figure)."""
    return ENSEMBLE_COST_FIGURES_DIR / name


###############################################################################
# Objective-determinism experiment
# (scripts/supplemental/check_objective_determinism.py)
#
# The Pywr-DRB LP solver is mildly nondeterministic: repeated identical
# simulations can differ at the state-trajectory level. The campaign's working
# assumption is that the OBJECTIVES are deterministic — the solver jitter must
# not propagate through the metric window (6-month exclusion) and the
# annual-unit aggregation into the objective vector. This experiment measures
# that directly: DETERMINISM_N_REPEATS repeated evaluations of each policy on
# each simulation path the campaign uses, every repeat in a FRESH python
# process (fresh interpreter, fresh model build, fresh solver instance — an
# in-process rerun could be masked by module-level caching), comparing
# per-objective max absolute / relative deviation across repeats. One
# state-level series (daily aggregate NYC storage) is captured per repeat to
# document the underlying jitter the objectives are expected to absorb.
#
# VERDICT RULE (stated before running): an objective counts as deterministic
# on a path iff its across-repeat deviation is exactly zero or at
# floating-point noise scale (max relative deviation <= DETERMINISM_REL_TOL).
# Anything larger is reported as propagation, per objective, with the worst
# offender identified. Rerunnable after any model change; completed
# (path, repeat) runs are skipped, so delete outputs/supplemental/
# objective_determinism/runs/ to force a full re-measurement.
###############################################################################


def configure_determinism_env() -> None:
    """Apply env knobs for the objective-determinism experiment.

    Salinity and temperature LSTMs off (the active objective set uses
    neither). The scenario design defaults to ``historic`` so
    ``get_objective_set()`` resolves the annual-unit (§2) registry — the same
    objective function every wired design searches under. No simulation-window
    override: the historic paths run the full trace, the ensemble paths
    self-derive from the fixture's realization length. The per-path
    trimmed/full switch is NOT set here — the driver exports
    ``NYCOPT_USE_TRIMMED_MODEL`` per worker subprocess.
    """
    _apply_env(salinity="0", temperature="0")
    os.environ.setdefault("NYCOPT_SCENARIO_DESIGN", "historic")


#: Repeated evaluations per (policy, path); every repeat is a fresh process.
DETERMINISM_N_REPEATS: int = int(os.environ.get("NYCOPT_DETERMINISM_REPEATS", "5"))

#: Feasible non-baseline policies (the FFMP baseline is always policy 0), so
#: the check is not baseline-specific.
DETERMINISM_N_PERTURBED: int = int(os.environ.get("NYCOPT_DETERMINISM_PERTURBED", "3"))

#: Initial per-DV perturbation, as a fraction of each DV's bound range. Random
#: 36-DV vectors are ~1% feasible under the two DV-space formal constraints, so policies
#: are drawn as small perturbations of the (feasible) baseline and accepted
#: only when ``compute_constraint_violations`` is exactly [0, 0]; the fraction
#: halves when acceptance stalls.
DETERMINISM_PERTURB_FRAC: float = 0.05

#: RNG seed for the perturbed-policy draw (reproducible policy set).
DETERMINISM_SEED: int = 73

#: Formulation under test.
DETERMINISM_FORMULATION: str = "ffmp"

#: Staged ensemble fixture for the ensemble paths. Defaults to the local
#: 5-realization x 50-yr Kirsch-Nowak fixture (src/local_test_ensemble.py);
#: the experiment never generates ensembles — a missing fixture is an error.
DETERMINISM_ENSEMBLE_SLUG: str = os.environ.get(
    "NYCOPT_DETERMINISM_ENSEMBLE", "kn_50yr_n5")

#: Simulation paths measured, each for determinism against ITSELF (trimmed and
#: full need not agree with each other): the search/re-eval path (historic +
#: ensemble, trimmed) and the presim/baseline path (full model).
DETERMINISM_PATHS: "tuple[str, ...]" = tuple(
    p.strip() for p in os.environ.get(
        "NYCOPT_DETERMINISM_PATHS",
        "historic_trimmed,historic_full,ensemble_trimmed,ensemble_full",
    ).split(",") if p.strip()
)

#: Verdict threshold: max relative deviation at or below this is
#: floating-point noise; above it is reported as propagation.
DETERMINISM_REL_TOL: float = 1e-9

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable)
# ---------------------------------------------------------------------------
DETERMINISM_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "objective_determinism"
DETERMINISM_RUNS_DIR: Path = DETERMINISM_OUTPUT_ROOT / "runs"
DETERMINISM_FIGURES_DIR: Path = DETERMINISM_OUTPUT_ROOT / "figures"


def determinism_policies_path() -> Path:
    """Path to the JSON of DV vectors (baseline + perturbed) under test."""
    return DETERMINISM_OUTPUT_ROOT / "policies.json"


def determinism_run_path(path_name: str, repeat: int) -> Path:
    """Per-(path, repeat) worker result JSON path."""
    return DETERMINISM_RUNS_DIR / f"{path_name}_rep{repeat:02d}.json"


def determinism_summary_path(ext: str) -> Path:
    """Path to the aggregated summary table (``ext`` = 'csv' or 'json')."""
    return DETERMINISM_OUTPUT_ROOT / f"summary.{ext}"


def determinism_figure_path(name: str) -> Path:
    """Path stub for a named figure (extension added by ``save_figure``)."""
    return DETERMINISM_FIGURES_DIR / name


###############################################################################
# Framing-convention analysis (cube reductions; no simulation)
# (docs/notes/methods/framing_convention_diagnostics.md diagnostics 1 + 4,
#  plus the flood unit-operator comparison and the annual-unit redundancy
#  screen for the 8th objective)
#
# Pure post-processing of the epsilon-calibration per-unit annual-metric cubes
# (`epsilon_cube_glob()`): the same 512 constraint-feasible policies + FFMP
# baseline evaluated on each campaign design's own search ensemble already
# hold every stage-(i) annual metric these diagnostics reduce (failing-week
# counts, annual flood-day counts). Zero simulation; seconds of runtime.
###############################################################################

#: Candidate annual failure-week counts k for the saturation / ranking screen
#: (contains every shipped `_DEFAULT_FAILURE_K` value).
FRAMING_K_GRID: tuple = (1, 2, 3, 4)

#: Saturation band edges: a policy population is saturated at a criterion when
#: this fraction of policies scores <= band or >= 1 - band (Bonham et al. 2024).
FRAMING_SATURATION_BAND: float = 0.05

#: Bootstrap resamples for the flood-operator noise comparison (resampling
#: realizations with replacement; unit-years for the historic design).
FRAMING_BOOTSTRAP_B: int = 500

#: RNG seed for the bootstrap index draw.
FRAMING_BOOTSTRAP_SEED: int = 11

#: |Spearman rho| above which an objective pair is flagged collinear in the
#: annual-unit redundancy screen (Olden & Poff 2003).
FRAMING_RHO_FLAG_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable)
# ---------------------------------------------------------------------------
FRAMING_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "framing_convention"
FRAMING_TABLES_DIR: Path = FRAMING_OUTPUT_ROOT / "tables"
FRAMING_FIGURES_DIR: Path = FRAMING_OUTPUT_ROOT / "figures"


def framing_table_path(name: str, design: "str | None" = None) -> Path:
    """Path for a named table CSV (per-design or combined)."""
    stem = f"{name}_{design}" if design else name
    return FRAMING_TABLES_DIR / f"{stem}.csv"


def framing_figure_path(name: str) -> Path:
    """Path stub for a named figure (extension added by ``save_figure``)."""
    return FRAMING_FIGURES_DIR / name


###############################################################################
# Weekly satisfaction-factor sweep
# (docs/notes/methods/framing_convention_diagnostics.md diagnostic 2)
#
# The 0.99 weekly satisfaction factor sits inside the weekly reduction
# (src/objectives.py::_weekly_delivery_ok), UPSTREAM of the stored failing-week
# counts, so it cannot be recovered from the epsilon cubes. This sweep
# re-evaluates the SAME feasible-policy population (EPS_SEED / EPS_N_POLICIES,
# so rows align with the epsilon cubes) on the active design's search ensemble
# and stores, for the two delivery objectives (NYC, NJ), the per-unit
# failing-week counts AND the §1 weekly reliability at each candidate factor —
# one small extra cube axis computed inside a single simulation pass.
# One sbatch job per design, exactly like the epsilon calibration.
###############################################################################

#: SF_SMOKE=True is a tiny laptop dry-run (few policies on the historic
#: design) proving the code path; flip to False for the per-design HPC run.
SF_SMOKE: bool = os.environ.get("NYCOPT_SF_SMOKE", "0") == "1"

#: Candidate weekly satisfaction factors (contains the shipped 0.99).
SF_FACTOR_GRID: tuple = (0.95, 0.98, 0.99, 1.00)

#: Feasible policies: identical population to the epsilon calibration (same
#: seed + count -> same DV rows, so cubes are row-aligned across experiments).
SF_N_POLICIES: int = 4 if SF_SMOKE else EPS_N_POLICIES

#: Realizations per simulation batch (0 = one block, the campaign default).
SF_REALIZATION_BATCH: int = 0

#: Delivery objectives swept: (annual objective name, demand key, delivery
#: key, entitlement reset convention). Caps resolve from config at run time.
SF_DELIVERY_OBJECTIVES: tuple = (
    ("nyc_delivery_reliability_annual", "demand_nyc", "delivery_nyc", "annual"),
    ("nj_delivery_reliability_annual", "demand_nj", "delivery_nj", "monthly"),
)

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable)
# ---------------------------------------------------------------------------
SF_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "satisfaction_factor"
SF_CUBE_DIR: Path = SF_OUTPUT_ROOT / "cube"
SF_TABLES_DIR: Path = SF_OUTPUT_ROOT / "tables"
SF_FIGURES_DIR: Path = SF_OUTPUT_ROOT / "figures"


def _sf_stem(design: str) -> str:
    """Run-identifying filename stem for one design's sweep artifacts."""
    return f"{EPS_FORMULATION}_{design}_seed{EPS_SEED}_n{SF_N_POLICIES}"


def sf_cube_path(design: str) -> Path:
    """Path to one design's factor-sweep cube HDF5 (run output)."""
    return SF_CUBE_DIR / f"factor_cube_{_sf_stem(design)}.h5"


def sf_cube_glob() -> str:
    """Glob (relative to SF_CUBE_DIR) matching every design's cube at the
    current sample settings."""
    return f"factor_cube_{EPS_FORMULATION}_*_seed{EPS_SEED}_n{SF_N_POLICIES}.h5"


def sf_table_path(name: str, design: "str | None" = None) -> Path:
    """Path for a named table CSV (per-design or combined)."""
    stem = f"{name}_{_sf_stem(design)}" if design else name
    return SF_TABLES_DIR / f"{stem}.csv"


def sf_figure_path(name: str) -> Path:
    """Path stub for a named figure (extension added by ``save_figure``)."""
    return SF_FIGURES_DIR / name


###############################################################################
# Flood-objective definition diagnostics
# (docs/notes/methods/flood_objective_diagnostics.md)
#
# Decides the flood objective definition: the incumbent any-gauge minor-stage
# day count (`downstream_flood_days_minor`) vs magnitude-weighted exceedance
# candidates (stage-ft and flow bases). One local simulation pass evaluates a
# feasible-policy sample plus a flood-release-scale sweep ladder on the
# historic trace AND the local KN stationary fixture, persisting per-policy x
# realization x water-year-unit candidate values and per-gauge flood-day
# records; the figures script reduces that cube (discriminating power,
# monotone-response gate, sampling noise, epsilon proposal) and scores every
# candidate sim-vs-obs on the flood-gauge diagnostic experiment's 2000-2023
# output (zero re-simulation).
###############################################################################


def configure_floodobj_env() -> None:
    """Apply env knobs for the flood-objective diagnostics run.

    Salinity and temperature LSTMs off (the flood metrics use neither). The
    scenario design defaults to ``historic``; the KN-ensemble pass resolves its
    fixture spec by slug, not through the scenario design. No simulation-window
    override: the historic pass runs the full trace, the ensemble pass
    self-derives from the fixture realization length.
    """
    _apply_env(salinity="0", temperature="0")
    os.environ.setdefault("NYCOPT_SCENARIO_DESIGN", "historic")


# ---------------------------------------------------------------------------
# Policy set
# ---------------------------------------------------------------------------
#: RNG seed for the feasible-DV rejection sample (distinct experiment identity;
#: the sample does NOT need row alignment with the epsilon cubes).
FLOODOBJ_SEED: int = 19

#: Feasible random policies (the FFMP baseline is always policy 0), sized for
#: a laptop pass: n+1 policies + sweep ladder on historic + N=5 x 50 yr.
FLOODOBJ_N_POLICIES: int = 24

#: Points on the flood-release-scale sweep ladder: all six
#: ``flood_release_scale_*`` DVs move together as a fraction t in [0, 1] of
#: each DV's own [lower, upper] range (monotone-response gate).
FLOODOBJ_SWEEP_POINTS: int = 9

#: Formulation whose bounds/constraints define the policy population.
FLOODOBJ_FORMULATION: str = "ffmp"

#: Ensemble pass fixture: the local KN stationary test ensemble
#: (src/local_test_ensemble.py). Its flood-augmented inflows MUST postdate the
#: 2026-07-31 flood-node inflow fix; the run script audits and re-stages.
FLOODOBJ_ENSEMBLE_SLUG: str = "kn_50yr_n5"

#: Realizations per simulation batch (0 = one block, the campaign default).
FLOODOBJ_REALIZATION_BATCH: int = 0

# ---------------------------------------------------------------------------
# Sim-vs-obs block (reuses the Pywr-DRB flood-gauge diagnostic experiment)
# ---------------------------------------------------------------------------
#: The flood-gauge diagnostic experiment (read-only sibling repo). Its
#: post-fix default-policy output HDF5 and helper module
#: (`diagnostics.py`) provide the sim-vs-obs scoring at zero re-simulation.
FLOODOBJ_GAUGE_EXPERIMENT_DIR: Path = (
    _PROJECT_DIR.parent / "Pywr-DRB" / "experiments" / "nyc_flood_gauge_diagnostics"
)

#: Observed-comparison window (matches the gauge experiment's START/END).
FLOODOBJ_OBS_WINDOW: tuple = ("2000-01-01", "2023-12-31")

# ---------------------------------------------------------------------------
# Noise / epsilon reductions
# ---------------------------------------------------------------------------
#: Bootstrap resamples for the ensemble sampling-noise block (resampling
#: realizations and pooled unit-years with replacement).
FLOODOBJ_BOOTSTRAP_B: int = 1000

#: RNG seed for the bootstrap index draw.
FLOODOBJ_BOOTSTRAP_SEED: int = 7

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable)
# ---------------------------------------------------------------------------
FLOODOBJ_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "flood_objective"
FLOODOBJ_CUBE_DIR: Path = FLOODOBJ_OUTPUT_ROOT / "cube"
FLOODOBJ_TABLES_DIR: Path = FLOODOBJ_OUTPUT_ROOT / "tables"
FLOODOBJ_FIGURES_DIR: Path = FLOODOBJ_OUTPUT_ROOT / "figures"


def floodobj_cube_path(name: str) -> Path:
    """Path to a named run-output cube (.npz)."""
    stem = (f"{name}_{FLOODOBJ_FORMULATION}_seed{FLOODOBJ_SEED}"
            f"_n{FLOODOBJ_N_POLICIES}_s{FLOODOBJ_SWEEP_POINTS}")
    return FLOODOBJ_CUBE_DIR / f"{stem}.npz"


def floodobj_table_path(name: str) -> Path:
    """Path for a named table CSV."""
    return FLOODOBJ_TABLES_DIR / f"{name}.csv"


def floodobj_figure_path(name: str) -> Path:
    """Path stub for a named figure (extension added by ``save_figure``)."""
    return FLOODOBJ_FIGURES_DIR / name


###############################################################################
# Robustness satisficing-threshold diagnostics
# (docs/notes/methods/robustness_threshold_diagnostics.md)
#
# Places the satisficing thresholds (`objectives_ensemble._DEFAULT_THRESHOLDS`,
# PROVISIONAL since the 2026-08-07 substrate change) against measured
# evidence: the baseline FFMP policy's persisted E_test re-eval cube (step 05
# `--reeval`; per-SOW annual-unit objective values), the E_test DU forcing
# factors, and an apples-to-apples historic-trace anchor recomputed from the
# persisted baseline HDF5. Zero simulation — two scripts reduce persisted
# artifacts only:
#   robustness_threshold_anchor.py   annual-unit anchor -> JSON cache
#   robustness_threshold_figures.py  tables + SI figures + recommendation
###############################################################################


def configure_rtd_env() -> None:
    """Apply env knobs for the robustness-threshold diagnostics.

    Salinity and temperature LSTMs off (none of the 8 annual-unit objectives
    needs them). Scenario design defaults to ``historic`` so ``config`` imports
    as a pure lookup; every input below is pinned by absolute path, not
    resolved through the scenario wiring.
    """
    _apply_env(salinity="0", temperature="0")
    os.environ.setdefault("NYCOPT_SCENARIO_DESIGN", "historic")


# ---------------------------------------------------------------------------
# Inputs (persisted artifacts; nothing here triggers a simulation)
# ---------------------------------------------------------------------------
#: Baseline-policy raw re-eval cube on E_test (reeval_raw.parquet + meta),
#: written by step 05 `run_baseline.py --reeval`.
RTD_REEVAL_BASELINE_DIR: Path = (
    _PROJECT_DIR / "outputs" / "historic" / "ffmp_obj8_mm_full" / "reeval"
    / "etest_kn_50yr_n25000" / "baseline")

#: E_test forcing profiles: theta_params (25000, 3) + theta_param_names
#: ['m', 'r1', 'r2'] + realization_ids, one theta row per realization.
RTD_FORCING_NPZ: Path = (
    _PROJECT_DIR / "outputs" / "synthetic_ensembles" / "etest_kn_50yr_n25000"
    / "forcing_profiles.npz")

#: Persisted historic-trace baseline simulation (full model) and its
#: annual-unit objective vector. Since the 2026-08-07 substrate change the CSV
#: is in the SAME metric space as the cube (annual-unit search objectives), so
#: it serves as a cross-check on the anchor recompute.
RTD_BASELINE_HDF5: Path = _PROJECT_DIR / "outputs" / "baseline" / "ffmp_baseline.hdf5"
RTD_BASELINE_ANNUAL_CSV: Path = (
    _PROJECT_DIR / "outputs" / "baseline" / "ffmp_baseline_objectives.csv")

# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------
#: Dense points per objective on the natural-unit threshold sweep grid
#: (extended so the default and every candidate lie exactly on a sample).
RTD_SWEEP_POINTS: int = 201

#: DU factor names expected in the forcing npz (order-checked at load).
RTD_THETA_NAMES: tuple = ("m", "r1", "r2")

#: Objectives given theta-plane factor maps: all 8 criteria, in the single
#: informative plane (RTD_FACTOR_MAP_PLANE). The pass/fail boundary is
#: near-monotone in m for every objective and r2 is inert, so one plane per
#: criterion beats three planes for two criteria.
RTD_FACTOR_MAP_OBJECTIVES: tuple = (
    "nyc_delivery_reliability_annual",
    "nyc_delivery_deficit_p99_pct",
    "montague_flow_reliability_annual",
    "montague_flow_deficit_p99_pct",
    "trenton_flow_reliability_annual",
    "downstream_flood_exceedance_annual",
    "nyc_storage_min_p01_pct",
    "nj_delivery_reliability_annual",
)

#: The theta plane the factor maps are drawn in (names from RTD_THETA_NAMES).
RTD_FACTOR_MAP_PLANE: tuple = ("m", "r1")

#: External flood anchor in ft-days/yr: the observed basin experience, WY2001-2023
#: (outputs/supplemental/flood_objective/tables/A_sim_vs_obs.csv, row C4_max_ft,
#: column obs = 1.1722 -- the revealed-tolerated level of realized history).
#: The simulated baseline is NOT listed here: it is the anchor script's
#: runtime-recomputed historic value, never a hardcoded number.
RTD_FLOOD_ANCHORS: dict = {
    "observed_2000_2023": 1.17,
}

#: Distribution-feature candidates: quantiles of the baseline per-SOW value
#: distribution offered as threshold placements (reported, never adopted).
RTD_CANDIDATE_QUANTILES: tuple = (0.10, 0.50, 0.90)

#: |delta univariate SOW fraction| (current -> recommended) above which a
#: recommendation is flagged as changing a headline result; a degenerate
#: current fraction (< RTD_DEGENERACY_LIMIT or > 1 - RTD_DEGENERACY_LIMIT)
#: moving out of degeneracy also flags.
RTD_HEADLINE_IMPACT_DELTA: float = 0.10

#: A satisficing fraction within this distance of 0 or 1 is DEGENERATE: the
#: criterion is not discriminating on E_test (all-fail voids the Starr
#: conjunction; all-pass is a non-binding guardrail). Shared by the
#: recommendation table's degeneracy-exit flag and the figure shading.
RTD_DEGENERACY_LIMIT: float = 0.01

#: Confidence level for the Wilson score intervals on SOW-unit satisficing
#: fractions (n = 1,000 independent LHS draws — the only counting unit under
#: the per-SOW annual-unit substrate).
RTD_CI_CONFIDENCE: float = 0.95

#: Failure combinations reported individually in rtd_failure_combinations.csv
#: (the rest are pooled into a remainder row).
RTD_TOP_FAILURE_COMBOS: int = 8

#: Centered rolling-window width (in SOWs, odd) for the critical-m boundary:
#: local pass rate over the m-sorted SOWs; the boundary is the median m of the
#: 0.5-crossings. NaN when the criterion is degenerate.
RTD_CRITICAL_M_WINDOW: int = 101

#: Near-historic neighborhood size: the K SOWs closest to theta = 0 (per-factor
#: standardized distance) used to check the historic-trace anchor's consistency
#: with the generator at near-zero forcing change.
RTD_NEAR_HISTORIC_K: int = 25

#: FINAL recommended threshold vector, filled AFTER inspecting the pass-1
#: outputs (two-pass workflow; empty dicts on pass 1 leave the recommendation
#: columns NaN). Keys are ANNUAL objective names; basis strings are the short
#: per-objective justification carried into the summary table.
#:
#: SUPERSEDED 2026-08-07: the vector adopted that morning was measured on the
#: retired whole-trace substrate and does not carry over to the per-SOW
#: annual-unit objective values. The PROVISIONAL annual-space thresholds live
#: in objectives_ensemble._DEFAULT_THRESHOLDS; these dicts stay empty (pass 1
#: pending) until the diagnostic is re-run against the status-quo E_test cube
#: on the new substrate and the §0b placement rules are re-applied.
RTD_RECOMMENDED_THRESHOLDS: dict = {}
RTD_RECOMMENDATION_BASIS: dict = {}

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable)
# ---------------------------------------------------------------------------
RTD_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "robustness_threshold_diagnostics"
RTD_CACHE_DIR: Path = RTD_OUTPUT_ROOT / "cache"
RTD_TABLES_DIR: Path = RTD_OUTPUT_ROOT / "tables"
RTD_FIGURES_DIR: Path = RTD_OUTPUT_ROOT / "figures"

#: Historic-anchor ANNUAL-UNIT objective values recomputed from
#: RTD_BASELINE_HDF5 (JSON, the seam that keeps the figures script
#: pywrdrb-free). Refresh with NYCOPT_RTD_REFRESH=1 on the anchor script.
RTD_ANCHOR_CACHE: Path = RTD_CACHE_DIR / "historic_anchor_annual_metrics.json"


def rtd_table_path(name: str) -> Path:
    """Path for a named table CSV."""
    return RTD_TABLES_DIR / f"{name}.csv"


def rtd_figure_path(name: str) -> Path:
    """Path stub for a named figure (extension added by ``save_figure``)."""
    return RTD_FIGURES_DIR / name


###############################################################################
# Regret-tolerance diagnostics (RTOL)
# docs/notes/methods/regret_tolerance_diagnostics.md
###############################################################################
# Fixes the two free parameters of the incumbent-relative regret comparison
# BEFORE the campaign result is inspected: the no-harm tolerance
# ``tau_i = k * eps_i`` and the non-inferiority margin ``delta`` on
# ``no_harm_freq_tau``. Both are pre-registration quantities, so the admissible
# anchors are restricted by what they can bias (note section 1):
#
#   Tier A  estimator noise / measurement resolution  -> admissible for both
#   Tier B  external decision increments (Decree, observed record) -> tau only
#   Tier C  the candidate-policy regret distribution  -> INADMISSIBLE (circular)
#   Tier D  within-design nuisance variance (seed / draw pairs) -> delta only
#
# Tier C is the trap this block exists to prevent, and it is the same trap
# RTD_CANDIDATE_QUANTILES is reported-but-never-adopted for: a tolerance read off
# the distribution it is meant to test guarantees its own answer.

#: The incumbent's E_test cube. Same artifact the threshold diagnostics use; the
#: noise floor (Tier A) is a pure function of it and needs NO policy runs, so
#: pass A can be run as soon as step 05 lands and long before any search finishes.
RTOL_REEVAL_BASELINE_DIR: Path = RTD_REEVAL_BASELINE_DIR

#: Multipliers ``k`` on each objective's just-noticeable difference (the ANNUAL
#: epsilons from ``objectives_ensemble.ENSEMBLE_OBJECTIVES``). Must match
#: ``compare_designs.REGRET_TAU_GRID`` so the diagnostic and the comparison speak
#: the same coordinate.
RTOL_TAU_GRID: tuple = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)

#: One-sided normal deviate for the noise floor. 1.645 -> a policy operationally
#: identical to the incumbent is falsely flagged as harming a given objective in
#: a given SOW at most 5% of the time.
RTOL_FALSE_HARM_Z: float = 1.645
RTOL_TARGET_FALSE_HARM: float = 0.05

#: E_test forcing profiles (theta per realization), joined to the cube's SOW
#: labels via realization_id // realizations_per_sow. The dominant axis ``m``
#: orders the SOWs for the binned noise floor (see RTOL_M_BIN_SIZE).
RTOL_FORCING_NPZ: Path = RTD_FORCING_NPZ

#: Consecutive-SOW bin size on the m-sorted axis for the noise floor: the
#: floor is the median across bins of the within-bin SD of the incumbent's
#: per-SOW values. Narrow bins keep the forcing trend's within-bin variation
#: small; what remains of it only INFLATES the floor (the conservative,
#: upper-bound direction). 10 -> 100 bins at n_sow = 1,000.
RTOL_M_BIN_SIZE: int = 10

#: A tolerance is SATURATED when no_harm_freq_tau exceeds this for every design
#: (the non-inferiority claim becomes trivially true) and STARVED when it falls
#: below the lower bound for every design (nothing to compare). The defensible
#: reporting band lies between them.
RTOL_SATURATION_HI: float = 0.95
RTOL_SATURATION_LO: float = 0.05

#: SOW-level bootstrap resamples for the PAIRED between-design difference. The
#: designs are scored on the same SOWs, so the difference has a smaller standard
#: error than either margin and must be bootstrapped as a pair, not differenced
#: from two independent margins.
RTOL_BOOTSTRAP_N: int = 2000
RTOL_BOOTSTRAP_SEED: int = 7

#: Positive control for assay sensitivity. A non-inferiority claim is only
#: meaningful if the comparison could have detected a difference had one existed;
#: `historic` is the unmatched reference expected to be worse, so failure to
#: separate it is evidence the metric is insensitive at that tolerance, not
#: evidence that the designs agree.
RTOL_ASSAY_CONTROL_DESIGN: str = "historic"

#: The PRE-REGISTERED rules. These are recorded as strings so the note, the code,
#: and the manuscript cannot drift apart, and so that what was fixed in advance is
#: legible after the fact.
RTOL_TAU_RULE: str = (
    "k_headline = the smallest k on RTOL_TAU_GRID whose tau_i = k * eps_i clears "
    "the Tier-A noise floor for EVERY objective. Smallest, not largest: the "
    "hypothesis is a non-inferiority claim, which a loose tolerance flatters, so "
    "the most discriminating defensible tolerance is the conservative choice. The "
    "full k-curve is reported regardless; k_headline only fixes which rung carries "
    "the sentence."
)
RTOL_MARGIN_RULE: str = (
    "delta = max(2 x paired SOW-bootstrap SE of the between-design difference in "
    "no_harm_freq_tau, the within-design between-DRAW spread of the same quantity). "
    "The draw is the declared unit of analysis, so the draw-level term is the "
    "denominator the design contrast must beat. delta is a function of the NUISANCE "
    "variance only and never of the between-design contrast, so it cannot determine "
    "the direction of the answer; the pre-registration is of this rule, not of a "
    "number that could only be computed later."
)

#: Filled AFTER pass A, from the measured floors. Empty leaves the headline rung
#: unset and the note's checklist open.
RTOL_ADOPTED_K: float | None = None

RTOL_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "regret_tolerance_diagnostics"
RTOL_TABLES_DIR: Path = RTOL_OUTPUT_ROOT / "tables"
RTOL_FIGURES_DIR: Path = RTOL_OUTPUT_ROOT / "figures"


def rtol_table_path(name: str) -> Path:
    """Path for a named regret-tolerance table CSV."""
    return RTOL_TABLES_DIR / f"{name}.csv"


def rtol_figure_path(name: str) -> Path:
    """Path stub for a named regret-tolerance figure."""
    return RTOL_FIGURES_DIR / name
