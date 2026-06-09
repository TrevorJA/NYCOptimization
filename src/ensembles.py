"""
ensembles.py - Ensemble-evaluation source registry for multi-realization MOEA.

This module is the single source of truth for how the optimizer maps a *preset
name* (e.g., ``"historic_single"``, ``"wcu_kirsch_n5"``) to an immutable
``EnsembleSpec`` describing the inflow source, the realization indices to
draw, deeply-uncertain (DU) factor specs, and the slug-fragment that
identifies this ensemble in output paths.

The single-realization legacy path is itself a preset (``historic_single``)
with ``is_ensemble=False`` and ``realization_indices=(0,)``. Existing runs
continue to work unchanged when ``NYCOPT_ENSEMBLE_PRESET`` is unset, because
the slug fragment is empty and the simulation layer routes ``is_ensemble=False``
through the original ``run_simulation_inmemory`` path.

v1 ships:
    - ``historic_single``       — single-trace passthrough (default, legacy)
    - ``wcu_kirsch_n5``         — N=5 Kirsch–Nowak smoke ensemble (M2 stages)
    - ``reeval_wcu_kirsch_n300``— independent N=300 re-eval ensemble (M2 stages)

DU factors are intentionally absent from v1 presets — see
local_notes/followups/du_factor_design.md (TBD) for the deferred design notes;
the ``du_factors`` field is the forward hook.

Plan reference:
    /home/fs02/pmr82_0001/tja73/.claude/plans/in-this-session-i-tranquil-feigenbaum.md
"""

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


###############################################################################
# EnsembleSpec
###############################################################################

@dataclass(frozen=True)
class EnsembleSpec:
    """Immutable specification of an ensemble for an optimization or re-eval run.

    Attributes
    ----------
    preset_name
        Name used to look this spec up in the ``PRESETS`` registry. Persisted
        in slugs and output directory names.
    inflow_type
        The pywrdrb inflow-dataset key. For ``is_ensemble=True`` specs this
        names the staged HDF5 directory under
        ``Pywr-DRB/input_data/synthetic_ensembles/{inflow_type}/`` that
        ``FlowEnsemble`` will load. For ``is_ensemble=False`` it is the
        registered single-trace key (e.g. ``"pub_nhmv10_BC_withObsScaled"``).
    realization_indices
        Tuple of integer realization IDs. ``len(realization_indices)``
        equals the number of pywr scenarios. For ``historic_single`` this is
        ``(0,)`` (single trace, treated as a 1-realization scenario block).
    du_factors
        Mapping of factor-name -> per-realization value spec. **Empty in v1**;
        forward-hooked for the deferred DU-factor work. Treat as immutable.
    seed
        Optional seed used by the underlying generator. Carried so the staging
        pipeline can reproduce or re-stage the ensemble deterministically.
    is_ensemble
        ``True`` when the simulation layer should route through the
        ensemble-aware path (``run_simulation_ensemble_inmemory``) and use
        pywrdrb's ``inflow_ensemble_indices`` plumbing. ``False`` for the
        legacy single-trace passthrough.
    source_kind
        Short identifier for the generator family: ``"historic"``,
        ``"synhydro_kn"``, ``"moeafind"`` (M2+). Used for diagnostics and
        for dispatching the correct generator class in the staging pipeline.
    slug_fragment
        String inserted into the output slug (e.g. ``"wcu5"``). Empty for
        ``historic_single`` so legacy slugs remain unchanged.
    """

    preset_name: str
    inflow_type: str
    realization_indices: tuple[int, ...]
    du_factors: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    is_ensemble: bool = True
    source_kind: str = "synhydro_kn"
    slug_fragment: str = ""
    realization_years: int | None = None
    # Length (in years) of each generated synthetic realization. ``None`` means
    # span the full training window (currently 1945-10-01 → 2022-09-30 ≈ 78
    # years). Smaller values produce shorter realizations, which is faster to
    # generate and simulate — useful while the ensemble pipeline is in active
    # development. The simulation window for an ensemble run is automatically
    # clipped to the realization length (see
    # ``src/simulation.py::run_simulation_ensemble_inmemory``).

    @property
    def n_realizations(self) -> int:
        return len(self.realization_indices)

    @property
    def du_factor_signature(self) -> str:
        """Stable string representation of ``du_factors`` for cache keys.

        Empty string when no DU factors are active. Sorted by key so the
        signature is deterministic regardless of insertion order.
        """
        if not self.du_factors:
            return ""
        return "|".join(f"{k}={self.du_factors[k]}" for k in sorted(self.du_factors))


###############################################################################
# Preset registry
###############################################################################
# Add new presets here. Each entry is a complete EnsembleSpec. The keys of
# this dict are what users supply via NYCOPT_ENSEMBLE_PRESET / NYCOPT_REEVAL_-
# ENSEMBLE_PRESET.
#
# v1 inflow_type values for the WCU/reeval entries are forward-looking: the
# matching staged HDF5 files do not exist until M2 of the ensemble plan has
# been completed. M1 ships the registry plumbing only; resolving these specs
# at config import time is safe (no I/O). Trying to *run* a simulation with
# them before M2 will fail at pywrdrb's HDF5 load step with a clear error.

PRESETS: dict[str, EnsembleSpec] = {
    "historic_single": EnsembleSpec(
        preset_name="historic_single",
        inflow_type="pub_nhmv10_BC_withObsScaled",
        realization_indices=(0,),
        is_ensemble=False,
        source_kind="historic",
        slug_fragment="",  # legacy slug-preserving
    ),
    "wcu_kirsch_n5": EnsembleSpec(
        preset_name="wcu_kirsch_n5",
        # M2 KirschNowakGenerator stages this directory:
        inflow_type="syn_kirsch_drb_n100_seed42",
        realization_indices=tuple(range(5)),
        seed=42,
        is_ensemble=True,
        source_kind="synhydro_kn",
        slug_fragment="wcu5",
        # 20-year realizations during pipeline development for fast
        # iteration; promote to None (full 78-yr window) for production.
        realization_years=20,
    ),
    "reeval_wcu_kirsch_n300": EnsembleSpec(
        preset_name="reeval_wcu_kirsch_n300",
        # M2 KirschNowakGenerator stages this directory (independent seed
        # from the search preset to protect against selection bias per
        # Bonham 2024):
        inflow_type="syn_kirsch_drb_n300_seed1337",
        realization_indices=tuple(range(300)),
        seed=1337,
        is_ensemble=True,
        source_kind="synhydro_kn",
        slug_fragment="reeval_wcu300",
    ),
}


###############################################################################
# Resolver + helpers
###############################################################################

_KN_SLUG_RE = re.compile(r"^kn_(\d+)yr_n(\d+)$")


def kirsch_nowak_slug(n_years: int, n_realizations: int) -> str:
    """Build the canonical ``kn_{Y}yr_n{N}`` slug for a Kirsch-Nowak ensemble."""
    return f"kn_{n_years}yr_n{n_realizations}"


def _spec_from_kn_slug(slug: str) -> EnsembleSpec | None:
    """Build an ``EnsembleSpec`` from a ``kn_{Y}yr_n{N}`` slug, or None if it doesn't match."""
    m = _KN_SLUG_RE.match(slug)
    if m is None:
        return None
    n_years, n_realizations = int(m.group(1)), int(m.group(2))
    return EnsembleSpec(
        preset_name=slug,
        inflow_type=slug,
        realization_indices=tuple(range(n_realizations)),
        is_ensemble=True,
        source_kind="synhydro_kn",
        slug_fragment=slug,
        realization_years=n_years,
    )


def get_ensemble_spec(preset_name: str) -> EnsembleSpec:
    """Resolve a preset name to its ``EnsembleSpec``.

    First checks the static ``PRESETS`` registry; if no match, falls back to
    parsing the ``kn_{Y}yr_n{N}`` slug grammar for ensembles staged by
    ``scripts/main/generate_stochastic_ensemble.py``. Raises ``KeyError`` if
    neither resolves.
    """
    if preset_name in PRESETS:
        return PRESETS[preset_name]
    spec = _spec_from_kn_slug(preset_name)
    if spec is not None:
        return spec
    raise KeyError(
        f"Unknown ensemble preset '{preset_name}'. "
        f"Available presets: {list_presets()} "
        f"(or any 'kn_{{Y}}yr_n{{N}}' slug for a Step-1-staged ensemble)."
    )


def list_presets() -> list[str]:
    """Return the registered named preset names in sorted order.

    Does not enumerate ``kn_{Y}yr_n{N}`` slugs — those resolve lazily through
    ``get_ensemble_spec`` and the slug space is unbounded.
    """
    return sorted(PRESETS)


def with_indices_override(spec: EnsembleSpec, indices: list[int]) -> EnsembleSpec:
    """Return a copy of ``spec`` with ``realization_indices`` replaced.

    Used by the ``NYCOPT_ENSEMBLE_INDICES`` env hook to subset an ensemble
    for smoke testing without authoring a separate preset.
    """
    return replace(spec, realization_indices=tuple(indices))


###############################################################################
# Path registration with pywrdrb's path navigator
###############################################################################
# Pywr-DRB resolves ``flows/{inflow_type}`` via its path navigator. For staged
# ensembles, the directory lives under ``STAGED_ENSEMBLE_DIR/{inflow_type}/``
# (config.STAGED_ENSEMBLE_DIR), which pywrdrb does not know about by default.
# Both the staging script and the simulation entrypoint must register the
# directory before invoking pywrdrb. This helper wraps that registration so
# the path convention is centralized.
#
# Idempotent: calling multiple times for the same inflow_type is safe.

def staged_ensemble_dir(inflow_type: str):
    """Return the absolute path where ``inflow_type`` is staged under
    ``config.STAGED_ENSEMBLE_DIR/{inflow_type}/``.

    This is a thin helper that delays the import of ``config`` to avoid an
    import cycle (config imports from this module).
    """
    from pathlib import Path
    from config import STAGED_ENSEMBLE_DIR
    return Path(STAGED_ENSEMBLE_DIR).resolve() / inflow_type


def register_ensemble_path(inflow_type: str) -> None:
    """Register a staged ensemble directory with pywrdrb's path navigator.

    Adds ``flows/{inflow_type}`` to the pywrdrb shortcut namespace
    (``pn.sc``) pointing at the staged-ensemble directory. After calling
    this, ``pn.sc.get(f"flows/{inflow_type}")`` resolves correctly, which
    is the lookup ``FlowEnsemble`` / ``PredictionEnsemble`` /
    ``PredictedInflowEnsemblePreprocessor`` use.

    NOTE: ``FloodNodeInflowEnsemblePreprocessor`` (added to pywrdrb in
    commit 7d5e210 on the nyc_opt branch) uses a different API
    (``pn.flows.get_str(inflow_type)``) which only works for inflow
    types that physically live under pywrdrb's bundled ``flows/`` tree.
    A fix-up patch on the pywrdrb side is queued — see
    ``local_notes/followups/pywrdrb_flood_preproc_path_api.md`` — that
    will switch the new preprocessor to ``pn.sc.get`` for consistency
    with the other ensemble preprocessors.

    Idempotent. Safe to call multiple times.
    """
    import pywrdrb
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{inflow_type}"] = str(staged_ensemble_dir(inflow_type))
    pywrdrb.load_pn_config(pn_config)
