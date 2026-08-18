"""
ensembles.py - Ensemble-evaluation source registry for multi-realization MOEA.

This module is the single source of truth for how the optimizer maps a *preset
name* (e.g., ``"historic_single"``, ``"wcu_kirsch_n5"``) to an immutable
``EnsembleSpec`` describing the inflow source, the realization indices to
draw, deeply-uncertain (DU) factor specs, and the slug-fragment that
identifies this ensemble in output paths.

The single-realization path is itself a preset (``historic_single``) with
``is_ensemble=False`` and ``realization_indices=(0,)``: its slug fragment is
empty and the simulation layer routes it through ``run_simulation_inmemory``.

Static presets registered here:
    - ``historic_single``       — single-trace passthrough (default)
    - ``wcu_kirsch_n5``         — N=5 Kirsch–Nowak ensemble test fixture
    - ``reeval_wcu_kirsch_n300``— independent N=300 re-eval ensemble

Static presets carry no DU factors; the ``du_factors`` field is the forward
hook.
"""

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

#: Epoch of every synthetic realization: the date of day 0, and the anchor the
#: generator's synthetic index is built from (SynHydro's Kirsch generator
#: synthesizes calendar-year, January-start sequences, so this must be a
#: January 1; generation asserts it). Also the anchor month for the historic
#: hazard-window layers, which cut the record the way scenario windows are cut.
#: Re-exported by ``config`` (the public access point).
ENSEMBLE_START_DATE = "1945-01-01"


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
        Mapping of factor-name -> per-realization value spec. Empty for
        static presets; forward hook for DU-factor work. Treat as immutable.
    seed
        Optional seed used by the underlying generator. Carried so the staging
        pipeline can reproduce or re-stage the ensemble deterministically.
    is_ensemble
        ``True`` when the simulation layer should route through the
        ensemble-aware path (``run_simulation_ensemble_inmemory``) and use
        pywrdrb's ``inflow_ensemble_indices`` plumbing. ``False`` for the
        single-trace passthrough.
    source_kind
        Short identifier for the generator family: ``"historic"``,
        ``"synhydro_kn"``, ``"moeafind"``. Used for diagnostics and
        for dispatching the correct generator class in the staging pipeline.
    slug_fragment
        String inserted into the output slug (e.g. ``"wcu5"``). Empty for
        ``historic_single``.
    """

    preset_name: str
    inflow_type: str
    realization_indices: tuple[int, ...]
    du_factors: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    is_ensemble: bool = True
    source_kind: str = "synhydro_kn"
    slug_fragment: str = ""
    # Length (in years) of each generated synthetic realization. Required
    # (an int) for every ensemble spec: the simulation window is derived as
    # ``start_date + realization_years`` (see ``src/simulation.py::
    # _ensemble_window``). ``None`` is valid only for single-trace specs,
    # which simulate the historic window (config.START_DATE/END_DATE).
    realization_years: int | None = None
    # Date of day 0 of every staged realization, read from the staged
    # ``_meta.json``. A January 1 under the truthful stamping convention
    # (the generator synthesizes calendar-year sequences). ``None`` only for
    # single-trace specs.
    start_date: str | None = None
    # When True, this spec describes a resample pool: ``realization_indices``
    # is the full pool, and the simulation layer redraws ``resample_size``
    # indices from it at every function evaluation (the resampled-
    # probabilistic design, Trindade et al. 2017). False for all fixed
    # designs (the default).
    resample_per_eval: bool = False
    # Number of realizations to draw per evaluation when ``resample_per_eval``;
    # ``None`` for fixed designs.
    resample_size: int | None = None

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
# inflow_type values may name staged HDF5 directories that are not present
# in every checkout. Resolving a spec at config import time is safe (no I/O);
# running a simulation without the staged files fails at pywrdrb's HDF5 load
# step with a clear error.

PRESETS: dict[str, EnsembleSpec] = {
    "historic_single": EnsembleSpec(
        preset_name="historic_single",
        inflow_type="pub_nhmv10_BC_withObsScaled",
        realization_indices=(0,),
        is_ensemble=False,
        source_kind="historic",
        slug_fragment="",
    ),
    # Small N=5 ensemble used only as a fixture by tests/test_ensemble_simulation.py
    # to exercise the ensemble-aware simulation machinery (cache keys, batching,
    # end-to-end). Not referenced by any scenario design.
    "wcu_kirsch_n5": EnsembleSpec(
        preset_name="wcu_kirsch_n5",
        # Staged by KirschNowakGenerator:
        inflow_type="syn_kirsch_drb_n100_seed42",
        realization_indices=tuple(range(5)),
        seed=42,
        is_ensemble=True,
        source_kind="synhydro_kn",
        slug_fragment="wcu5",
        # 20-year realizations during pipeline development for fast
        # iteration (test fixture only).
        realization_years=20,
        start_date=ENSEMBLE_START_DATE,
    ),
}

# The held-out test ensemble E_test needs NO entry here: ``_spec_from_staged_dir``
# resolves any staged directory carrying a ``_meta.json`` by slug, so
# ``NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_30yr_n1000`` resolves once step 02 has
# staged it. E_test is an ``EnsembleSpec``, never a ``ScenarioDesign``: it never
# enters search, and no search ensemble is drawn from it.


###############################################################################
# Resolver + helpers
###############################################################################

_KN_SLUG_RE = re.compile(r"^kn_(\d+)yr_n(\d+)$")


def _verified_staged_start_date(meta: Mapping[str, Any], slug: str) -> str:
    """Return a staged ensemble's ``start_date``, enforcing the stamping convention.

    Every staged artifact must record the date of day 0 and it must match
    ``config.ENSEMBLE_START_DATE``. Metas stamped under the retired October
    convention (or lacking the key) identify stale artifacts that would
    silently rotate the statistical season against the simulation calendar;
    they fail here, at resolution time, rather than downstream. Set
    ``NYCOPT_ALLOW_STALE_STAMP=1`` to bypass for deliberate archaeology only.
    """
    import os

    start = meta.get("start_date")
    if start is None:
        raise ValueError(
            f"staged ensemble '{slug}' records no start_date in _meta.json: it predates "
            f"the truthful January stamping convention and must be regenerated."
        )
    if str(start) != ENSEMBLE_START_DATE and not os.environ.get("NYCOPT_ALLOW_STALE_STAMP"):
        raise ValueError(
            f"staged ensemble '{slug}' is stamped start_date={start!r}, but the stamping "
            f"convention is {ENSEMBLE_START_DATE!r}. The artifact predates the truthful "
            f"January convention and must be regenerated (set NYCOPT_ALLOW_STALE_STAMP=1 "
            f"to inspect it anyway)."
        )
    return str(start)


def kirsch_nowak_slug(n_years: int, n_realizations: int) -> str:
    """Build the canonical ``kn_{Y}yr_n{N}`` slug for a Kirsch-Nowak ensemble."""
    return f"kn_{n_years}yr_n{n_realizations}"


def _spec_from_kn_slug(slug: str) -> EnsembleSpec | None:
    """Build an ``EnsembleSpec`` from a ``kn_{Y}yr_n{N}`` slug, or None if it doesn't match.

    The slug grammar carries no dates; when the ensemble is already staged, the
    stamp is read (and convention-verified) from its ``_meta.json``, otherwise
    the configured convention is assumed for the yet-to-be-staged artifact.
    """
    import json

    m = _KN_SLUG_RE.match(slug)
    if m is None:
        return None
    n_years, n_realizations = int(m.group(1)), int(m.group(2))
    meta_path = staged_ensemble_dir(slug) / "_meta.json"
    if meta_path.exists():
        start_date = _verified_staged_start_date(json.loads(meta_path.read_text()), slug)
    else:
        start_date = ENSEMBLE_START_DATE
    return EnsembleSpec(
        preset_name=slug,
        inflow_type=slug,
        realization_indices=tuple(range(n_realizations)),
        is_ensemble=True,
        source_kind="synhydro_kn",
        slug_fragment=slug,
        realization_years=n_years,
        start_date=start_date,
    )


def _spec_from_staged_dir(slug: str) -> EnsembleSpec | None:
    """Build an ``EnsembleSpec`` from a staged ensemble's ``_meta.json``, or None.

    Any directory ``STAGED_ENSEMBLE_DIR/{slug}/`` that carries a ``_meta.json``
    written by a generator (the Step-1 Kirsch-Nowak generator or the scengen
    hazard-filling driver) resolves to an ensemble of ``n_realizations``
    realizations numbered ``0..N-1``. This is the generic handoff: scengen emits
    a final ensemble HDF5 + ``_meta.json``, NYCOptimization resolves it by slug
    with no manifest-as-contract and no realization-index override.
    """
    import json

    meta_path = staged_ensemble_dir(slug) / "_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    n = int(meta["n_realizations"])
    years = meta.get("realization_years", meta.get("n_years"))
    if years is None:
        raise ValueError(
            f"staged ensemble '{slug}' records no realization length "
            f"(realization_years/n_years) in _meta.json; the simulation window "
            f"cannot be derived."
        )
    return EnsembleSpec(
        preset_name=slug,
        inflow_type=slug,
        realization_indices=tuple(range(n)),
        # Provenance seed, in specificity order: the selector anchor seed of a
        # hazard-filled ensemble, else the generator root seed of a directly
        # generated ensemble or pool.
        seed=meta.get("selector_seed", meta.get("root_seed", meta.get("seed"))),
        is_ensemble=True,
        source_kind=meta.get("source_kind", "synhydro_kn"),
        slug_fragment=slug,
        realization_years=int(years),
        start_date=_verified_staged_start_date(meta, slug),
    )


def get_ensemble_spec(preset_name: str) -> EnsembleSpec:
    """Resolve a preset name to its ``EnsembleSpec``.

    Resolution order:
        1. the static ``PRESETS`` registry;
        2. the ``kn_{Y}yr_n{N}`` slug grammar (parsed from the name, no I/O), for
           ensembles staged by ``scripts/main/generate_stochastic_ensemble.py``;
        3. any other staged ensemble directory carrying a ``_meta.json`` (e.g.
           a scengen hazard-filling final ensemble ``hazfill_{L}yr_n{N}_s{seed}``).

    Raises ``KeyError`` if none resolves.
    """
    if preset_name in PRESETS:
        return PRESETS[preset_name]
    spec = _spec_from_kn_slug(preset_name)
    if spec is not None:
        return spec
    spec = _spec_from_staged_dir(preset_name)
    if spec is not None:
        return spec
    raise KeyError(
        f"Unknown ensemble preset '{preset_name}'. "
        f"Available presets: {list_presets()} "
        f"(or any 'kn_{{Y}}yr_n{{N}}' slug, or a staged ensemble dir with a "
        f"_meta.json under config.STAGED_ENSEMBLE_DIR)."
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
    for smoke testing without authoring a separate preset, and by the
    resampled-probabilistic per-evaluation draw to install the freshly drawn
    subset of master-pool indices.
    """
    return replace(spec, realization_indices=tuple(indices))


def as_resampling_pool(spec: EnsembleSpec, resample_size: int) -> EnsembleSpec:
    """Mark ``spec`` as a resample-per-eval pool.

    The returned spec keeps ``realization_indices`` as the full pool and
    sets ``resample_per_eval=True`` with ``resample_size`` realizations drawn
    per evaluation (see ``src/simulation.py::evaluate``). Used by the
    resampled-probabilistic scenario design.
    """
    if resample_size > spec.n_realizations:
        raise ValueError(
            f"resample_size ({resample_size}) cannot exceed the pool size "
            f"({spec.n_realizations}) of preset '{spec.preset_name}'."
        )
    return replace(spec, resample_per_eval=True, resample_size=resample_size)


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


#: Files a staged ensemble must carry before ``evaluate()`` can simulate on it,
#: mapped to the workflow step that writes each. The two step-02 files come from
#: the generator; the three step-04 files come from the pywrdrb preprocessors.
#: The full model does not read ``presimulated_releases_mgd.hdf5`` itself, but
#: step 04 bakes the ``perfect_foresight`` predicted inflows from it, so a
#: complete staging carries it regardless of which model variant will run.
STAGED_ENSEMBLE_FILES: dict[str, str] = {
    "gage_flow_mgd.hdf5": "02",
    "catchment_inflow_mgd.hdf5": "02",
    "catchment_inflow_with_flood_nodes_mgd.hdf5": "04",
    "presimulated_releases_mgd.hdf5": "04",
    "predicted_inflows_mgd.hdf5": "04",
}


def staged_ensemble_missing(slug: str) -> list[str]:
    """Return the required files ``slug`` is missing (empty = ready to simulate).

    A directory's mere existence is NOT a staging test: an interrupted or
    metadata-only run leaves a non-empty directory (``_meta.json`` plus
    diagnostics) that the step-02 generator's own "already staged" check will
    happily skip over, after which step 04 dies on the absent inflow HDF5. This
    checks for the files themselves, and treats a zero-byte file as missing.

    Args:
        slug: Staged ensemble directory name (e.g. ``"kn_10yr_n100"``).

    Returns:
        The missing (or empty) required filenames, in ``STAGED_ENSEMBLE_FILES``
        order. An empty list means the ensemble is fully staged.
    """
    d = staged_ensemble_dir(slug)
    return [
        name for name in STAGED_ENSEMBLE_FILES
        if not (d / name).is_file() or (d / name).stat().st_size == 0
    ]


def load_chunk_index(pool_slug: str) -> dict | None:
    """Load a pool's ``chunk_index.json`` (chunks -> global realization ranges), or None."""
    import json
    p = staged_ensemble_dir(pool_slug) / "chunk_index.json"
    return json.loads(p.read_text()) if p.exists() else None


def pool_chunk_specs(pool_slug: str) -> list[tuple["EnsembleSpec", list[int]]]:
    """Return ``[(chunk_spec, global_realization_ids), ...]`` for a chunked (or single-dir) pool.

    Each chunk resolves as a standalone staged ensemble with local realizations ``0..S-1``; the
    paired ``global_realization_ids`` re-key its rows to the pool's global index space for
    aggregation. A single-dir pool (no ``chunk_index.json``, or one whose only chunk is the pool
    dir itself) returns one chunk = the pool spec with identity global ids.
    """
    import json
    idx = load_chunk_index(pool_slug)
    if idx is None or not idx.get("chunks"):
        spec = get_ensemble_spec(pool_slug)
        return [(spec, list(range(spec.n_realizations)))]
    out: list[tuple[EnsembleSpec, list[int]]] = []
    for s in idx["chunks"]:
        slug = s["slug"]
        spec = get_ensemble_spec(slug)
        meta_path = staged_ensemble_dir(slug) / "_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        gids = meta.get("global_realization_ids") or list(range(s["global_start"], s["global_end"]))
        out.append((spec, [int(g) for g in gids]))
    return out


def materialize_subset(
    pool_slug: str, global_indices, out_slug: str, *,
    files=("gage_flow_mgd.hdf5", "catchment_inflow_mgd.hdf5"), extra_meta: dict | None = None,
) -> str:
    """Stage a reduced ensemble of selected pool realizations by reading only those from the chunks.

    Used by the hazard-filling designs, which are the ONLY designs that subsample: hazard
    coordinates are emergent properties of a realized sequence, so a hazard-space design must
    select from a candidate pool rather than generate to a target. Every other design generates
    its realizations directly.

    Groups the requested **global** indices by their containing chunk and reads only those columns
    from each chunk's HDF5 (memory-cheap ``from_hdf5(realization_subset=...)``), renumbers to local
    ``0..N-1``, and writes the reduced staged ensemble (+ ``_meta.json`` carrying
    ``global_realization_ids``). Works for a chunked or single-dir pool; peak memory scales with the
    selection size, not the pool. Requires the daily chunks to be stored (``store_daily``).

    Args:
        pool_slug: The candidate-pool slug (holds ``chunk_index.json`` / daily chunks).
        global_indices: Pool global realization ids to materialize, in the desired output order.
        out_slug: Slug for the staged reduced ensemble.
        files: HDF5 basenames to slice (default the pywrdrb gage + catchment pair).
        extra_meta: Extra provenance merged into the reduced ensemble's ``_meta.json``.

    Returns:
        ``out_slug``.
    """
    import json
    from synhydro.core.ensemble import Ensemble

    meta_path = staged_ensemble_dir(pool_slug) / "_meta.json"
    pool_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if pool_meta.get("store_daily") is False:
        # Stream-only pool (campaign-scale candidate pools store the hazard
        # image, not the daily traces): the selected members cannot be read
        # back, so regenerate them bit-for-bit from their global indices.
        return _materialize_subset_regenerated(
            pool_slug, pool_meta, [int(g) for g in global_indices], out_slug,
            files=files, extra_meta=extra_meta,
        )

    chunks = pool_chunk_specs(pool_slug)
    loc: dict[int, tuple[str, int]] = {}
    years = None
    for spec, gids in chunks:
        years = spec.realization_years if years is None else years
        for local, g in enumerate(gids):
            loc[int(g)] = (spec.inflow_type, local)

    requested = [int(g) for g in global_indices]
    missing = [g for g in requested if g not in loc]
    if missing:
        raise KeyError(f"global indices not in pool '{pool_slug}': {missing[:10]}...")

    out_dir = staged_ensemble_dir(out_slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        by_chunk: dict[str, list[tuple[int, int]]] = {}
        for out_i, g in enumerate(requested):
            sslug, local = loc[g]
            by_chunk.setdefault(sslug, []).append((out_i, local))
        reduced: dict[int, object] = {}
        for sslug, pairs in by_chunk.items():
            subset = [local for _out_i, local in pairs]
            ens = Ensemble.from_hdf5(
                str(staged_ensemble_dir(sslug) / fname), realization_subset=subset
            )
            d = ens.data_by_realization  # keyed 0..len(subset)-1 in `subset` order
            for k, (out_i, _local) in enumerate(pairs):
                reduced[out_i] = d[k]
        Ensemble(reduced).to_hdf5(str(out_dir / fname))

    meta = {
        "slug": out_slug,
        "n_realizations": len(requested),
        "realization_years": years,
        "n_years": years,
        "global_realization_ids": requested,
        "source_pool": pool_slug,
        "source_kind": "synhydro_kn",
        "start_date": _verified_staged_start_date(pool_meta, pool_slug),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return out_slug


def _materialize_subset_regenerated(
    pool_slug: str, pool_meta: dict, requested: list, out_slug: str, *,
    files, extra_meta: dict | None,
) -> str:
    """Stage selected members of a STREAM-ONLY pool by regenerating them.

    A campaign-scale candidate pool stores only its hazard image
    (``store_daily=False`` — the daily pool would be TB-scale), so the selected
    realizations are regenerated from the pool's recorded generation config,
    keyed to the same global-index RNG streams the pool generation used — the
    ``src.ensemble_generation.regenerate_realization`` determinism contract,
    validated by ``tests/test_master_ensemble_determinism.py``. Restricted to
    ``stationary`` + ``kn`` pools: theta is vacuous there, so ONE batched
    generator pass serves every index (per-index streams make the batch
    identical to per-index regeneration); a ``du_forced`` pool needs
    per-profile moment adjustment and an ``hmm`` pool is only block-exact.

    Args:
        pool_slug: The stream-only candidate-pool slug.
        pool_meta: The pool's parsed ``_meta.json``.
        requested: Pool global realization ids, in the desired output order.
        out_slug: Slug for the staged reduced ensemble.
        files: HDF5 basenames to write (must be within the gage/inflow pair —
            they are the only artifacts generation produces).
        extra_meta: Extra provenance merged into the output ``_meta.json``.

    Returns:
        ``out_slug``.
    """
    import json
    from scengen.forcing_ensemble import ForcingEnsembleConfig
    from synhydro.core.ensemble import Ensemble

    from src.ensemble_generation import (
        _disaggregate_fill_inflow,
        _generate_profile_monthly,
        _prepare_generators,
    )

    population = pool_meta.get("population")
    generator = pool_meta.get("generator", "kn")
    if population != "stationary" or generator != "kn":
        raise NotImplementedError(
            f"stream-only pool '{pool_slug}' has population={population!r}, "
            f"generator={generator!r}; regeneration staging is wired for "
            "stationary Kirsch-Nowak pools only."
        )
    known = {"gage_flow_mgd.hdf5", "catchment_inflow_mgd.hdf5"}
    if set(files) - known:
        raise ValueError(
            f"regeneration produces only {sorted(known)}, got {list(files)}"
        )
    n_pool = int(pool_meta["n_realizations"])
    bad = [g for g in requested if not 0 <= int(g) < n_pool]
    if bad:
        raise KeyError(
            f"global indices out of pool '{pool_slug}' range [0, {n_pool}): "
            f"{bad[:10]}..."
        )

    cfg = ForcingEnsembleConfig(
        root_seed=int(pool_meta["root_seed"]),
        n_forcing_profiles=int(pool_meta["n_forcing_profiles"]),
        realizations_per_profile=int(pool_meta.get("realizations_per_profile", 1)),
        realization_years=int(pool_meta["realization_years"]),
        population="stationary",
        generator="kn",
        seed_domain=pool_meta.get("seed_domain"),
        flowtype=pool_meta["flowtype"],
        # Convention-verified: a stale (pre-January-convention) pool must not be
        # silently rematerialized under its old stamp.
        start_date=_verified_staged_start_date(pool_meta, pool_slug),
        store_daily=False,
    )
    print(f"[materialize] pool '{pool_slug}' is stream-only: regenerating "
          f"{len(requested)} realizations (root_seed={cfg.root_seed}, "
          f"global-index streams)...")
    setup = _prepare_generators(cfg)
    # profile_idx is unused for a stationary population (no moment adjustment),
    # and each index draws from its own global-index stream, so one batched
    # call is exactly the per-index regeneration.
    monthly, gen_meta = _generate_profile_monthly(
        setup, cfg, 0, indices=[int(g) for g in requested]
    )
    gage_by_real, inflow_by_real, _sites = _disaggregate_fill_inflow(
        Ensemble(monthly, metadata=gen_meta),
        nowak=setup.nowak, kdes=setup.kdes,
        root_seed=cfg.root_seed, start_date=cfg.start_date,
    )

    out_dir = staged_ensemble_dir(out_slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_file = {"gage_flow_mgd.hdf5": gage_by_real,
               "catchment_inflow_mgd.hdf5": inflow_by_real}
    for fname in files:
        source = by_file[fname]
        # Rekey to local 0..N-1 in `requested` order AFTER daily generation
        # (rekeying before disaggregation would change the daily output).
        Ensemble({i: source[int(g)] for i, g in enumerate(requested)}).to_hdf5(
            str(out_dir / fname)
        )

    years = int(pool_meta["realization_years"])
    meta = {
        "slug": out_slug,
        "n_realizations": len(requested),
        "realization_years": years,
        "n_years": years,
        "global_realization_ids": [int(g) for g in requested],
        "source_pool": pool_slug,
        "source_kind": "synhydro_kn",
        "regenerated_from_stream_only_pool": True,
        "root_seed": int(pool_meta["root_seed"]),
        "start_date": cfg.start_date,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return out_slug


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
    A fix-up patch on the pywrdrb side would switch the new preprocessor
    to ``pn.sc.get`` for consistency with the other ensemble
    preprocessors.

    Idempotent. Safe to call multiple times.
    """
    import pywrdrb
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{inflow_type}"] = str(staged_ensemble_dir(inflow_type))
    pywrdrb.load_pn_config(pn_config)
