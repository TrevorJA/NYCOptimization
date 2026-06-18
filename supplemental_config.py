"""supplemental_config.py - Single source of truth for the project's
**supplemental** experiments (those outside the main MOEA workflow).

Each supplemental experiment gets its own clearly-labelled section below. The
module is intentionally the one place to look for every supplemental knob, so a
run is reproducible from version-controlled files alone — no CLI value flags,
no edits to the main ``config.py`` settings.

IMPORTANT — import order and per-experiment env. ``config.py`` reads several
``NYCOPT_*`` and ``PYWRDRB_*`` environment variables *at its own import* (e.g.
to decide whether the salinity/temperature LSTMs run, and the simulation
window). Different experiments need **different** values for those knobs — the
historic objective-sensitivity diagnostic runs the salinity LSTM on a short
calendar window, while the ensemble diagnostic runs salinity *off* and derives
its window from the realization length. A single set of module-top
``os.environ`` writes therefore cannot serve both.

Each experiment instead exposes a ``configure_*_env()`` function that applies
its env knobs via ``os.environ.setdefault``. Entry-point scripts call the
relevant one **between** importing this module and importing ``config``::

    import supplemental_config as scfg   # stdlib only; sets no env on import
    scfg.configure_ensemble_env()        # now set this experiment's env
    from config import ...               # config reads the env we just set

To keep that guarantee this module imports only the stdlib and never imports
``config`` (which would either fire too late or create a cycle). Output paths
are derived from ``__file__`` for the same reason.
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
# (docs/notes/methods/objective_sensitivity_experiment.md)
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

    Salinity LSTM on so ``salt_front_intrusion_max_rm`` is a real number, not
    NaN (the full-registry redundancy screen compares it against the Trenton
    flow objective that replaced it). Temperature LSTM stays off — it is
    deferred, so ``lordville_temp_exceedance_days`` is reported as NaN. Under
    SMOKE a short window keeps each simulation to ~10-15 s; the end stays within
    the trimmed model's pre-simulated release data (2022-09-30 water-year end).
    """
    _apply_env(
        salinity="1",
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

#: Number of random DV vectors. The doc recommends N ~ 200-500 for stable
#: Spearman estimates; the FFMP baseline is added as an extra reference row.
N_SAMPLES: int = 3 if SMOKE else 500

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
# Ensemble objective-sensitivity experiment
# (docs/notes/methods/ensemble_objective_sensitivity_experiment.md)
#
# On ONE fixed probabilistic (Kirsch-Nowak) ensemble, evaluate many random DV
# vectors ONCE over the full ensemble, store the per-realization base-metric
# matrix (N_DV x N_realizations x N_objectives), and derive all diagnostics
# post-hoc: (1) ensemble-size (K) ranking convergence, (2) across-realization
# operator agreement, plus secondary redundancy and threshold-sensitivity.
#
# PROVISIONAL SIZES. The manuscript ensemble-design sizes and the realization
# length are open decisions (docs/notes/methods/experimental_design.md). The
# full-scale numbers below are placeholders chosen so the K-grid and rank
# correlations are estimable; revisit when those decisions are made. TODO.
###############################################################################

# ---------------------------------------------------------------------------
# Mode switch (independent of the historic SMOKE above)
# ---------------------------------------------------------------------------
#: ENS_SMOKE=True is a tiny laptop dry-run: N=5 x 20-yr ensemble, 3 DVs. The
#: full-scale HPC numbers (below, in the False branch) are ready to run
#: unchanged — flip this one flag.
ENS_SMOKE: bool = False


def configure_ensemble_env() -> None:
    """Apply env knobs for the ensemble objective-sensitivity run.

    Salinity and temperature LSTMs are **off**: the active objective set uses
    neither, and disabling them is a large speedup over the full ensemble. No
    simulation-window override is set — the ensemble window self-derives from
    the realization length (``src/simulation.py::_ensemble_window`` clips to
    ``config.START_DATE + realization_years``), and the staged ensemble is
    generated to match that same window.
    """
    _apply_env(salinity="0", temperature="0")


# ---------------------------------------------------------------------------
# Probabilistic ensemble (Kirsch-Nowak, historic reference fit)
# ---------------------------------------------------------------------------
#: Realizations and per-realization length. The inflow_type slug is the dynamic
#: ``kn_{Y}yr_n{N}`` grammar resolved by ``src.ensembles.get_ensemble_spec``,
#: so it matches exactly what the Step-1 generator stages.
ENS_N_REALIZATIONS: int = 5 if ENS_SMOKE else 256
ENS_REALIZATION_YEARS: int = 20
#: Generation seed for the Kirsch-Nowak ensemble (distinct from the DV seed).
ENS_KN_SEED: int = 1234


def ensemble_inflow_type() -> str:
    """Dynamic ``kn_{Y}yr_n{N}`` inflow-type slug for the experiment ensemble."""
    return f"kn_{ENS_REALIZATION_YEARS}yr_n{ENS_N_REALIZATIONS}"


# ---------------------------------------------------------------------------
# DV sweep
# ---------------------------------------------------------------------------
#: RNG seed for the Latin-hypercube DV sample.
ENS_SEED: int = 42
#: Formulation whose DV bounds define the sampling space.
ENS_FORMULATION: str = "ffmp"
#: Number of random DV vectors (FFMP baseline added as an extra row, id -1).
ENS_N_DV: int = 3 if ENS_SMOKE else 199
#: Realizations evaluated per simulation batch, to bound peak memory. Only the
#: scalar per-realization metrics are retained; each batch's timeseries are
#: freed before the next. <= ENS_N_REALIZATIONS.
ENS_REALIZATION_BATCH: int = 5 if ENS_SMOKE else 64

#: Base objectives whose per-realization metric is stored. "active" = the 7
#: recommended base metrics (salinity/temperature excluded; their LSTMs are
#: off). "full_registry" or an explicit list are also accepted.
ENS_OBJECTIVE_SET: "str | list[str]" = "active"

#: Predicted-inflow modes to stage. The model's default ``flow_prediction_mode``
#: is "perfect_foresight"; "regression_disagg" is also written so the staged
#: file works regardless of the mode the simulation requests. perfect_foresight
#: requires the presimulated-release HDF5, hence STARFIT prep runs first.
ENS_PREDICTION_MODES: tuple = ("regression_disagg", "perfect_foresight")

# ---------------------------------------------------------------------------
# Post-hoc diagnostic grids (figure script only; no re-simulation)
# ---------------------------------------------------------------------------
#: Ensemble sub-sample sizes K for the ranking-convergence diagnostic. The full
#: ensemble (ENS_N_REALIZATIONS) is the proxy-truth ranking.
ENS_K_GRID: list = [2, 3, 5] if ENS_SMOKE else [10, 25, 50, 100, 200, 256]
#: Random sub-sample repeats per K (the tau_b(K) band).
ENS_K_SUBSAMPLE_REPEATS: int = 3 if ENS_SMOKE else 20
#: RNG seed for the K sub-sampling (reproducible bands).
ENS_K_SUBSAMPLE_SEED: int = 7

#: Across-realization operators compared for the operator-agreement diagnostic.
#: "satisficing" uses the per-objective thresholds in
#: ``src.objectives_ensemble._DEFAULT_THRESHOLDS``; the others are distribution
#: summaries of the per-realization base metric.
ENS_OPERATORS: tuple = ("satisficing", "mean", "p90", "cvar90")

#: Multipliers applied to each objective's default satisficing threshold for the
#: threshold-sensitivity table (tau_b of rankings vs the default-threshold
#: ranking).
ENS_THRESHOLD_MULTIPLIERS: tuple = (0.8, 0.9, 1.0, 1.1, 1.2)

#: Olden & Poff (2003) redundancy flag for the ensemble-objective Spearman screen.
ENS_RHO_FLAG_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Output tree (gitignored, regenerable)
# ---------------------------------------------------------------------------
ENS_OUTPUT_ROOT: Path = SUPPLEMENTAL_OUTPUT_ROOT / "ensemble_objective_sensitivity"
ENS_MATRIX_DIR: Path = ENS_OUTPUT_ROOT / "matrix"
ENS_TABLES_DIR: Path = ENS_OUTPUT_ROOT / "tables"
ENS_FIGURES_DIR: Path = ENS_OUTPUT_ROOT / "figures"


def _ens_stem() -> str:
    """Run-identifying filename stem shared by all ensemble-experiment artifacts."""
    return (f"{ENS_FORMULATION}_{ensemble_inflow_type()}"
            f"_seed{ENS_SEED}_dv{ENS_N_DV}")


def ensemble_matrix_path() -> Path:
    """Path to the per-realization base-metric matrix HDF5 (run-script output)."""
    return ENS_MATRIX_DIR / f"per_realization_metrics_{_ens_stem()}.h5"


def ensemble_table_path(name: str) -> Path:
    """Path for a named diagnostic table CSV (e.g. name='operator_agreement')."""
    return ENS_TABLES_DIR / f"{name}_{_ens_stem()}.csv"


def ensemble_figure_path(name: str, ext: str) -> Path:
    """Path for a named ensemble figure artifact (e.g. name='tau_vs_k')."""
    return ENS_FIGURES_DIR / f"{name}_{_ens_stem()}.{ext}"
