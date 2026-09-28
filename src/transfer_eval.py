"""transfer_eval.py - Evaluate a design's Pareto set on another design's search ensemble.

The driver behind the transfer-evaluation matrix
(``docs/notes/methods/transfer_evaluation.md``). One work unit is one policy
evaluated on one target ensemble; the MPI task farm in
``scripts/supplemental/transfer_evaluation_run.py`` distributes units over
ranks and this module owns everything about what a unit is, how it is
evaluated, how it is persisted and how the persisted units are merged.

Why this module exists at all
-----------------------------
``src.reeval_core.evaluate_solution_raw`` cannot be used for a design-to-design
transfer, for two independent reasons:

1. It calls ``resolve_reeval()`` with no arguments, so it always reads
   ``config.REEVAL_ENSEMBLE_SPEC`` and silently ignores any spec the caller
   meant to use. The same is true of ``reeval_raw_meta`` and
   ``persist_reeval_raw``.
2. Its ensemble branch raises when ``sow_grouping`` returns ``None``, which it
   does for every campaign search ensemble: ``fixprob_*`` and
   ``hazfill_stat_abs_*`` are ``population: "stationary"`` with no
   ``forcing_profiles.npz``, so they carry no state-of-the-world structure.
   That guard is correct - robustness is defined on DU-forced SOWs - and this
   instrument does not try to weaken it. A transfer cell has no SOW structure
   and nothing here feeds ``src.robustness``.

So the driver composes the two public primitives directly:
``src.simulation.evaluate_annual_units`` with an explicit ``ensemble_spec``,
then ``src.reeval_core.sow_objective_matrix`` with a single pooled group, which
applies the same unit operators over the same annual units the search used.
Passing the target as a value rather than mutating
``NYCOPT_REEVAL_ENSEMBLE_PRESET`` also leaves
``config.assert_search_test_seed_domains_disjoint`` at full strength.

Execution model
---------------
Deliberately the same shape as ``src.chunk_reeval``, which is the project's
tested MPI task farm: dynamic ``O_CREAT|O_EXCL`` claim scheduling, atomic
per-unit writes with a ``.failed`` sidecar, restart-resume reconstructed
entirely from the filesystem, and a separate merge stage. The helpers there are
private and keyed to ``(solution_id, chunk_idx)``; the public, generically
keyed equivalents below generalize them to this instrument's unit key.
``src.chunk_reeval`` itself is on the campaign path and is not modified.

Claim scheduling matters here specifically because the work is heterogeneous:
a ``historic_single`` unit costs about 31 s against about 155 s for an
N=100 x L=10 unit, a five-fold spread. Ranks pull work rather than being handed
a contiguous slice, so no rank is left holding only cheap units.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "TransferCell",
    "build_cells",
    "resolve_target_spec",
    "check_cell_preconditions",
    "evaluate_unit",
    "unit_stem",
    "atomic_write_unit",
    "read_unit",
    "completed_units",
    "try_claim",
    "out_of_wall_time",
    "cell_provenance",
    "merge_cell",
    "load_source_set",
    "write_moea_set_file",
    "run_indicator",
    "merge_reference_sets",
]

#: Number of decision variables and objectives is read from the formulation at
#: call time, never hardcoded here; these names exist only for error text.
_UNIT_COLUMNS = ("realization", "unit_year", "objective", "value")


###############################################################################
# Cells and units
###############################################################################

@dataclass(frozen=True)
class TransferCell:
    """One (source optimization, target ensemble) cell of the matrix.

    Attributes:
        source: Scenario-design name whose Pareto set supplies the policies.
        set_file: Absolute path of that design's adopted ``.set`` file.
        target_design: Scenario-design name supplying the evaluation ensemble.
        target_draw: Ensemble draw index of the target (0 = the searched one).
        target_slug: Resolved staged slug / preset name of the target.
        on_design: True when source and target are the same design and draw,
            i.e. a diagonal cell. Diagonal cells are read from the ``.set``
            objective columns and are only ever simulated by the
            path-consistency check.
    """

    source: str
    set_file: Path
    target_design: str
    target_draw: int
    target_slug: str
    on_design: bool

    @property
    def key(self) -> str:
        """Filesystem-safe two-indexed cell identifier."""
        return f"src-{self.source}__tgt-{self.target_slug}"


def resolve_target_spec(design: str, draw: int):
    """Resolve a target ensemble to an ``EnsembleSpec`` value.

    Args:
        design: Scenario-design name owning the ensemble.
        draw: Ensemble draw index.

    Returns:
        The ``EnsembleSpec``. For ``historic`` this is the single-trace
        ``historic_single`` preset, which has ``is_ensemble=False``.

    Raises:
        ValueError: Propagated from ``resolve_search_spec`` when a design has no
            such draw (``historic`` accepts draw 0 only).
    """
    from src.scenario_designs import get_scenario_design
    return get_scenario_design(design).resolve_search_spec(draw)


def build_cells(set_files: dict, targets, include_diagonal: bool = False) -> list:
    """Enumerate the cells of the matrix.

    Args:
        set_files: Source design -> adopted ``.set`` path.
        targets: Iterable of ``(target design, draw)`` pairs.
        include_diagonal: When False (the default) on-design cells are omitted,
            because their objective values are already stored in the source
            ``.set``. The path-consistency check passes True to evaluate a
            handful of diagonal units deliberately.

    Returns:
        List of :class:`TransferCell`, ordered source-major then target, which
        is also the order the work list is built in so ranks working the same
        cell share the target's staged HDF5 through the page cache and hit the
        ``src.simulation`` model-dict cache.
    """
    cells = []
    for source in sorted(set_files):
        for target_design, draw in targets:
            spec = resolve_target_spec(target_design, draw)
            on_design = (source == target_design and draw == 0)
            if on_design and not include_diagonal:
                continue
            cells.append(TransferCell(
                source=source,
                set_file=Path(set_files[source]),
                target_design=target_design,
                target_draw=int(draw),
                target_slug=spec.preset_name,
                on_design=on_design,
            ))
    return cells


###############################################################################
# Preconditions - fail loudly, never coerce
###############################################################################

def check_cell_preconditions(cell: TransferCell, n_vars: int, n_objs: int,
                             expected_n: int, expected_years: int) -> dict:
    """Validate one cell before any simulation is spent on it.

    Guards the substrate identity that the whole matrix rests on. The specific
    hazard: the adopted ``.set`` files were searched on the N=100 draw-0
    ensembles, but ``src.scenario_designs.SEARCH_ENSEMBLE_N`` now defaults to
    300. If ``NYCOPT_SEARCH_N`` is not pinned, the target resolves to an
    unstaged n300 slug - or, worse, to a staged one that would score N=100
    diagonal values against N=300 off-diagonal values and confound every
    readout with ensemble size.

    Args:
        cell: The cell to check.
        n_vars: Expected decision-variable count (36 for ffmp).
        n_objs: Expected objective count (8 for the campaign set).
        expected_n: Realization count the source set was searched at.
        expected_years: Realization length L the source set was searched at.

    Returns:
        Dict describing the verified target identity, for the provenance sidecar.

    Raises:
        FileNotFoundError: If the source ``.set`` is absent.
        RuntimeError: If the target is unstaged or its identity disagrees with
            the source-search configuration.
    """
    from src.ensembles import staged_ensemble_missing, staged_ensemble_dir

    if not cell.set_file.exists():
        raise FileNotFoundError(
            f"[tev] source set missing: {cell.set_file}\n"
            f"[tev] Expected the adopted epsilon-refiltered merged set; see "
            f"scripts/supplemental/write_refiltered_sets.py.")

    spec = resolve_target_spec(cell.target_design, cell.target_draw)
    identity = {
        "target_slug": spec.preset_name,
        "target_is_ensemble": bool(spec.is_ensemble),
        "target_n_realizations": int(spec.n_realizations),
        "target_realization_years": spec.realization_years,
        "target_start_date": spec.start_date,
        "target_seed": spec.seed,
    }

    if not spec.is_ensemble:
        # The historic single trace is a named, deliberate exception to the
        # matched-L precondition: it is one 78-year observed record, not an
        # ensemble of L-year realizations, and no coercion could make it one.
        # The methods note records that cross-substrate comparisons involving
        # historic are directional rather than quantitative.
        identity["length_exception"] = (
            "single observed trace; realization length is not comparable to the "
            "N x L ensembles and is reported as such")
        return identity

    missing = staged_ensemble_missing(spec.inflow_type)
    if missing:
        raise RuntimeError(
            f"[tev] target ensemble '{spec.inflow_type}' is not fully staged; "
            f"missing: {missing}\n"
            f"[tev] Run workflow/04_prep_pywrdrb_inputs.sh --preset {spec.inflow_type}.")

    if int(spec.n_realizations) != int(expected_n):
        raise RuntimeError(
            f"[tev] target '{spec.preset_name}' has N={spec.n_realizations} but the "
            f"source sets were searched at N={expected_n}. Pin NYCOPT_SEARCH_N="
            f"{expected_n} (supplemental_config.configure_tev_env does this) — "
            f"refusing to score different ensemble sizes against each other.")
    if int(spec.realization_years or 0) != int(expected_years):
        raise RuntimeError(
            f"[tev] target '{spec.preset_name}' has L={spec.realization_years} but "
            f"the source sets were searched at L={expected_years}; refusing to "
            f"compare different realization lengths.")

    meta_path = staged_ensemble_dir(spec.inflow_type) / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        identity["target_meta"] = {
            k: meta.get(k) for k in
            ("slug", "population", "seed_domain", "root_seed", "selector_seed",
             "source_pool", "n_realizations", "realization_years", "start_date")
        }
        identity["target_meta_sha256"] = hashlib.sha256(
            meta_path.read_bytes()).hexdigest()
    return identity


###############################################################################
# Unit evaluation
###############################################################################

def evaluate_unit(dv_vector, cell: TransferCell, obj_set, formulation: str,
                  realization_batch=None):
    """Evaluate one policy on one target ensemble.

    Reproduces the search-time aggregation exactly: per-realization stage-(i)
    annual metrics from ``evaluate_annual_units``, then every unit-year of
    every surviving realization pooled through the same unit operators by
    ``sow_objective_matrix`` with a single group. That is what
    ``compute_for_borg_ensemble`` does during search, so the composed vector is
    the search-equivalent one.

    Never raises: a failed unit is reported so the task farm records a
    ``.failed`` sidecar and keeps going, matching
    ``src.chunk_reeval._evaluate_unit``.

    Args:
        dv_vector: ``(n_vars,)`` decision variables.
        cell: The cell being evaluated (supplies the target spec).
        obj_set: The active annual-unit ``ObjectiveSet``.
        formulation: Formulation name (``"ffmp"``).
        realization_batch: Realizations per Pywr model build; None defers to
            ``config.SEARCH_REALIZATION_BATCH``.

    Returns:
        ``(payload, error)``. ``payload`` is a dict with ``units`` (the long
        per-realization annual-unit DataFrame), ``natural`` (the composed
        objective vector in natural units), ``n_survivors`` and ``n_realizations``;
        ``error`` is None on success. On failure ``(None, "ExcType: msg")``.
    """
    from src.reeval_core import sow_objective_matrix
    from src.simulation import evaluate_annual_units

    try:
        spec = resolve_target_spec(cell.target_design, cell.target_draw)
        units, obj_names = evaluate_annual_units(
            dv_vector, formulation_name=formulation, objective_set=obj_set,
            ensemble_spec=spec, realization_batch=realization_batch,
        )
        active = [o.name for o in obj_set]
        if list(obj_names) != active:
            raise RuntimeError(
                f"objective mismatch: driver returned {list(obj_names)} but the "
                f"active set is {active}")

        # One pooled group: every unit-year of every realization feeds one
        # objective vector, which is the search's own reduction.
        n_real = int(units.shape[0])
        matrix, _labels, survivors = sow_objective_matrix(
            units, obj_set, [0] * n_real)

        r_idx, o_idx, u_idx = np.meshgrid(
            np.arange(units.shape[0]), np.arange(units.shape[1]),
            np.arange(units.shape[2]), indexing="ij")
        frame = pd.DataFrame({
            "realization": r_idx.ravel().astype(np.int32),
            "objective": np.asarray(active, dtype=object)[o_idx.ravel()],
            "unit_year": u_idx.ravel().astype(np.int16),
            "value": units.ravel().astype(float),
        })
        return {
            "units": frame,
            "natural": matrix[0].astype(float),
            "n_survivors": int(survivors[0]),
            "n_realizations": n_real,
            "n_unit_years": int(units.shape[2]),
            "obj_names": active,
        }, None
    except Exception as exc:  # noqa: BLE001 - a failed unit contributes no rows
        return None, f"{type(exc).__name__}: {exc}"


def print_unit_line(cell_key: str, row: int, t0: float, rank: int) -> None:
    """Per-unit telemetry: wall time and the rank's peak RSS so far.

    ``ru_maxrss`` is the process high-water mark (kB on Linux), so the printed
    value is cumulative-peak rather than per-unit. It is what sizes
    ranks-per-node on the next submission.
    """
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[tev:unit] rank={rank} cell={cell_key} row={row} "
          f"elapsed_s={time.perf_counter() - t0:.1f} rss_gb={rss_gb:.2f}",
          flush=True)


###############################################################################
# Atomic persistence, claims and resume
# (public, generically keyed generalizations of the private helpers in
#  src/chunk_reeval.py, which stays untouched on the campaign path)
###############################################################################

def unit_stem(units_dir: Path, row: int) -> Path:
    """Canonical extension-less path for one unit's artifacts.

    One stem owns ``{.parquet, .csv.gz, .failed, .parquet.tmp}``, so the state
    of a unit is a question about which extensions exist.
    """
    return Path(units_dir) / f"sol{int(row):05d}"


def atomic_write_unit(frame: pd.DataFrame, stem: Path, meta: dict) -> None:
    """Write one unit's rows atomically (temp file plus rename).

    A killed rank can never leave a half-written unit that a resume would
    trust, because the visible filename only ever appears via ``os.replace``.
    Falls back to gzipped CSV when no parquet engine is available, mirroring
    ``src.chunk_reeval._flush_unit``.

    Args:
        frame: The unit's long-format annual-unit rows.
        stem: Extension-less output stem from :func:`unit_stem`.
        meta: Per-unit scalars (composed objectives, survivors, timing) stored
            as parquet key-value metadata, or as a sidecar JSON for the CSV
            fallback.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.copy()
    for key, value in meta.items():
        payload.attrs[key] = value
    tmp = stem.with_suffix(".parquet.tmp")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(payload, preserve_index=False)
        table = table.replace_schema_metadata({
            b"tev_meta": json.dumps(meta, default=float).encode("utf-8")})
        pq.write_table(table, tmp)
        os.replace(tmp, stem.with_suffix(".parquet"))
    except Exception:  # noqa: BLE001 - pyarrow missing or unusable
        tmp = stem.with_suffix(".csv.gz.tmp")
        payload.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, stem.with_suffix(".csv.gz"))
        stem.with_suffix(".meta.json").write_text(json.dumps(meta, default=float))


def read_unit(stem: Path) -> tuple:
    """Read one unit's rows and metadata, or ``(None, None)`` if absent."""
    pq_path = stem.with_suffix(".parquet")
    if pq_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(pq_path)
        raw = (table.schema.metadata or {}).get(b"tev_meta")
        meta = json.loads(raw.decode("utf-8")) if raw else {}
        return table.to_pandas(), meta
    csv_path = stem.with_suffix(".csv.gz")
    if csv_path.exists():
        meta_path = stem.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return pd.read_csv(csv_path), meta
    return None, None


def completed_units(units_dir: Path, *, retry_failed: bool = False) -> set:
    """Row indices that need no work: done, or failed and not being retried.

    State is reconstructed entirely from the filesystem - no journal, no lock
    file that could go stale - so resubmitting the same job *is* the resume.

    Args:
        units_dir: The cell's per-unit directory.
        retry_failed: When True a ``.failed`` sidecar does not count as done.

    Returns:
        Set of integer row indices.
    """
    done: set = set()
    units_dir = Path(units_dir)
    if not units_dir.exists():
        return done
    for path in units_dir.glob("sol*"):
        name = path.name
        try:
            row = int(name.split(".", 1)[0][len("sol"):])
        except ValueError:
            continue
        if name.endswith(".parquet") or name.endswith(".csv.gz"):
            done.add(row)
        elif name.endswith(".failed") and not retry_failed:
            done.add(row)
    return done


def try_claim(claims_dir: Path, cell_key: str, row: int) -> bool:
    """Atomically claim a unit for this rank.

    ``O_CREAT|O_EXCL`` is the whole mechanism: exactly one rank wins the
    create, every other rank sees ``FileExistsError`` and moves on. Claims are
    job-scoped, so a killed job's claims are ignored by the next one rather
    than blocking its own work.

    Returns:
        True if this rank now owns the unit.
    """
    claims_dir = Path(claims_dir)
    path = claims_dir / f"{cell_key}__sol{int(row):05d}.claim"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def out_of_wall_time(unit_seconds: float, stop_epoch: float) -> bool:
    """True when the next unit could not finish before the job's stop epoch.

    Stopping cleanly is better than being killed mid-unit: the atomic write
    means a killed unit is merely absent rather than corrupt, but a clean stop
    also prints how far the job got.
    """
    if not stop_epoch or not unit_seconds:
        return False
    return time.time() + 1.25 * float(unit_seconds) >= float(stop_epoch)


###############################################################################
# Provenance
###############################################################################

def _git_state() -> dict:
    """Current commit and dirty flag, or a reason the state is unknown."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, timeout=30)
        if commit.returncode != 0:
            return {"commit": None, "note": commit.stderr.strip()[:200]}
        return {"commit": commit.stdout.strip(),
                "dirty": bool(status.stdout.strip())}
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a run
        return {"commit": None, "note": f"{type(exc).__name__}: {exc}"}


def cell_provenance(cell: TransferCell, identity: dict, obj_set,
                    n_rows: int, extra: dict = None) -> dict:
    """Everything needed to read one cell back without guessing.

    The matrix is two-indexed, so a cell's identity cannot be inferred from its
    path alone: both the source set and the target ensemble have to be
    recorded, along with the objective definitions and the evaluation
    configuration that produced the numbers.
    """
    import config

    return {
        "instrument": "transfer_evaluation",
        "source_design": cell.source,
        "source_set_file": str(cell.set_file),
        "source_set_sha256": hashlib.sha256(cell.set_file.read_bytes()).hexdigest(),
        "source_set_rows": int(n_rows),
        "target_design": cell.target_design,
        "target_draw": cell.target_draw,
        "on_design": cell.on_design,
        "target_identity": identity,
        "objective_names": [o.name for o in obj_set],
        "objective_directions": [int(o.sign) for o in obj_set],
        "objective_epsilons": [float(o.epsilon) for o in obj_set],
        "unit_operators": [type(o.unit_operator).__name__ for o in obj_set],
        "search_ensemble_n": os.environ.get("NYCOPT_SEARCH_N"),
        "flow_prediction_mode": config.PYWRDRB_FLOW_PREDICTION_MODE,
        "use_trimmed_model": bool(config.USE_TRIMMED_MODEL),
        "nyc_nj_demand_source": config.NYC_NJ_DEMAND_SOURCE,
        "realization_batch": int(config.SEARCH_REALIZATION_BATCH),
        "git": _git_state(),
        "written": pd.Timestamp.utcnow().isoformat(),
        **(extra or {}),
    }


###############################################################################
# Merge
###############################################################################

def merge_cell(units_dir: Path, n_rows: int, obj_names) -> tuple:
    """Reassemble one cell's per-unit artifacts.

    Stateless over the unit files, so it is trivially resumable and a merge bug
    can never cost a re-simulation. Missing and failed units are distinguished:
    a failed unit was evaluated and produced no rows, a missing one was never
    reached and means the evaluate job is incomplete.

    Args:
        units_dir: The cell's per-unit directory.
        n_rows: Number of source-set rows the cell covers.
        obj_names: Objective names in column order.

    Returns:
        ``(natural, long_frame, status)``. ``natural`` is ``(n_rows, n_objs)``
        with NaN rows for absent units; ``long_frame`` concatenates every
        unit's annual-unit rows with a ``row`` column; ``status`` reports the
        missing and failed row indices and the per-unit timings.
    """
    units_dir = Path(units_dir)
    names = list(obj_names)
    natural = np.full((int(n_rows), len(names)), np.nan, dtype=float)
    frames, seconds, survivors = [], {}, {}
    missing, failed = [], []

    for row in range(int(n_rows)):
        stem = unit_stem(units_dir, row)
        frame, meta = read_unit(stem)
        if frame is None:
            if stem.with_suffix(".failed").exists():
                failed.append(row)
            else:
                missing.append(row)
            continue
        vec = meta.get("natural")
        if vec is not None:
            natural[row, :] = np.asarray(vec, dtype=float)
        if "seconds" in meta:
            seconds[row] = float(meta["seconds"])
        if "n_survivors" in meta:
            survivors[row] = int(meta["n_survivors"])
        frame = frame.copy()
        frame["row"] = np.int32(row)
        frames.append(frame)

    long_frame = (pd.concat(frames, ignore_index=True) if frames
                  else pd.DataFrame(columns=list(_UNIT_COLUMNS) + ["row"]))
    status = {
        "n_rows": int(n_rows),
        "n_present": int(n_rows) - len(missing) - len(failed),
        "n_missing": len(missing),
        "n_failed": len(failed),
        "missing_rows": missing[:50],
        "failed_rows": failed[:50],
        "seconds": seconds,
        "survivors": survivors,
    }
    return natural, long_frame, status


def recompose_from_units(long_frame: pd.DataFrame, obj_set, n_rows: int) -> np.ndarray:
    """Recompute composed objectives from the persisted per-realization units.

    The merge-integrity check: the same arithmetic over the same data must
    reproduce the per-unit composed vectors exactly. A mismatch means the
    persisted tensor does not support the offline recomposition the instrument
    promises, and the analysis stage refuses to run on it.

    Args:
        long_frame: Merged long-format annual-unit rows for one cell.
        obj_set: The active annual-unit ``ObjectiveSet``.
        n_rows: Number of source-set rows.

    Returns:
        ``(n_rows, n_objs)`` natural-unit objectives, NaN where a row is absent.
    """
    names = [o.name for o in obj_set]
    out = np.full((int(n_rows), len(names)), np.nan, dtype=float)
    if long_frame.empty:
        return out
    for row, chunk in long_frame.groupby("row", sort=True):
        for k, obj in enumerate(obj_set):
            vals = chunk.loc[chunk["objective"] == names[k], "value"].to_numpy(float)
            if vals.size:
                out[int(row), k] = float(obj.unit_operator(vals))
    return out


###############################################################################
# .set file input and output
###############################################################################

def load_source_set(set_file: Path, n_vars: int, n_objs: int) -> tuple:
    """Load a source ``.set``, returning decision variables and stored objectives.

    The ``n_objs`` argument turns on ``load_reference_set``'s fail-loud
    column-count guard, which is the cheapest defence against silently reading
    a file written for a different formulation.

    Returns:
        ``(dvs, stored_borg)`` - decision variables and the stored objective
        columns, which are in Borg orientation (every objective minimized).
    """
    from src.load.reference_set import load_reference_set

    dvs, objs = load_reference_set(Path(set_file), n_vars, n_objs=n_objs)
    if dvs.shape[0] == 0:
        raise RuntimeError(f"[tev] {set_file} parsed as zero solutions.")
    return dvs, objs


def write_moea_set_file(path: Path, dvs: np.ndarray, borg_obj: np.ndarray,
                        problem: str) -> Path:
    """Write a MOEAFramework v5 result file.

    Format is load-bearing and its failure mode is silent. MOEAFramework 5.0
    needs ``# Version=5`` as the first line - without it the reader assumes the
    v4 layout, expects an extra constraint column and throws - and a final line
    containing a lone ``#`` as the entry terminator; without that the file
    parses as zero entries. A legacy Borg-written ``.set`` (three comment lines,
    ``%.6e`` values, no terminator) passed as a reference does not error, it
    returns an indicator value of 0.000000.

    Values are written at full ``repr`` precision rather than ``%.6e`` because
    ``CalculateIndicator -i Contribution`` matches solutions by value; rounding
    makes every match fail and silently reports zero contribution.

    Args:
        path: Output path.
        dvs: ``(n, n_vars)`` decision variables.
        borg_obj: ``(n, n_objs)`` objectives, Borg orientation (all minimized).
        problem: MOEAFramework problem name (e.g. ``drb_ffmp``).

    Returns:
        The output path.
    """
    dvs = np.atleast_2d(np.asarray(dvs, dtype=float))
    objs = np.atleast_2d(np.asarray(borg_obj, dtype=float))
    if dvs.shape[0] != objs.shape[0]:
        raise ValueError(
            f"row mismatch: {dvs.shape[0]} decision vectors vs {objs.shape[0]} "
            f"objective vectors")
    n_vars, n_objs = dvs.shape[1], objs.shape[1]

    lines = [
        "# Version=5",
        f"# Problem={problem}",
        f"# NumberOfVariables={n_vars}",
        f"# NumberOfObjectives={n_objs}",
        "# NumberOfConstraints=0",
    ]
    for i in range(n_vars):
        lines.append(f"# Variable.{i + 1}.Definition=RealVariable(-1000000.0,1000000.0)")
    for i in range(n_objs):
        lines.append(f"# Objective.{i + 1}.Definition=Minimize")
    for dv_row, obj_row in zip(dvs, objs):
        lines.append(" ".join(repr(float(v)) for v in np.concatenate([dv_row, obj_row])))
    lines.append("#")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return path


###############################################################################
# MOEAFramework CLI
###############################################################################

def _cli_env() -> dict:
    """Environment for the MOEAFramework CLI, with Java on PATH.

    Java 17 lives in the project's conda env rather than a module, so a job
    that reached Python through an already-activated env may still not have
    ``java`` on PATH for a subprocess. Prepending the interpreter's own
    directory is enough and is a no-op when Java is already visible.
    """
    import sys

    env = dict(os.environ)
    bindir = str(Path(sys.executable).parent)
    if bindir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    return env


def merge_reference_sets(inputs, output: Path, problem: str, epsilons=None,
                         timeout_s: float = 1800.0) -> Path:
    """Merge result files into one reference set with MOEAFramework.

    ``epsilons`` defaults to None, which yields the **plain Pareto** union, and
    that is deliberate: the epsilon archive is applied separately, once, using
    the Borg box convention that ``src.sensitivity_common.epsilon_nondominated``
    implements and that ``src/diagnostics.py`` documents as validated to
    reproduce Borg's own seed-archive membership exactly. One epsilon
    definition, the project's own, applied in one place.

    A note on ``src/diagnostics.py``, which states that ResultFileMerger
    "merges by PLAIN Pareto dominance and ignores ``--epsilon`` for archiving
    (measured: identical output under two vectors)". On this installation that
    is **not** the case for ``.set`` inputs: merging the three campaign sets
    gives 1,330 rows with no ``--epsilon``, 1,248 under the adopted vector and
    144 under a ten-fold vector. The merger does apply epsilon dominance, in
    MOEAFramework's own formulation, which is not identical to Borg's box
    convention. Passing epsilon here would therefore stack two different
    epsilon definitions. Hence the default of None. (The campaign path is
    unaffected and untouched: filtering an already-epsilon-filtered union with
    the Borg convention still yields a valid Borg archive - only the stated
    reason in that docstring is wrong, not its result.)

    Returns:
        The output path.
    """
    from src.diagnostics import get_cli_path

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    cmd = [get_cli_path(), "ResultFileMerger", "--problem", problem]
    if epsilons is not None:
        cmd += ["--epsilon", ",".join(str(float(e)) for e in epsilons)]
    cmd += ["--output", str(output)] + [str(p) for p in inputs]
    subprocess.run(cmd, check=True, timeout=timeout_s, env=_cli_env())
    if not output.exists():
        raise RuntimeError(f"[tev] ResultFileMerger produced no output at {output}")
    return output


def run_indicator(indicator: str, reference: Path, inputs, problem: str,
                  epsilons=None, timeout_s: float = 1800.0) -> dict:
    """Run ``CalculateIndicator`` and parse its per-input values.

    ``Contribution`` is the MOEAFramework 5.0 replacement for the 3.x
    ``SetContribution`` tool: it reports the FRACTION of the reference set
    matched by each approximation set. Matching is exact unless ``--epsilon``
    is given, in which case solutions in the same epsilon box count as
    equivalent - which is the semantics this instrument wants, so epsilons are
    passed for every indicator.

    Args:
        indicator: MOEAFramework indicator name (``Hypervolume``,
            ``Contribution``, ``AdditiveEpsilonIndicator``, ...).
        reference: Reference set file. For hypervolume this also fixes the
            normalization bounds and the reference point, so passing ONE shared
            reference across every cell is what makes the values comparable.
        inputs: Approximation set files.
        problem: MOEAFramework problem name.
        epsilons: Optional per-objective epsilons.
        timeout_s: Wall limit; exact hypervolume in 8 dimensions can be
            prohibitive, and a timeout is an expected outcome rather than a bug.

    Returns:
        Mapping of input path string -> float value. Empty on timeout, with the
        reason printed.
    """
    from src.diagnostics import get_cli_path

    # ARGUMENT ORDER IS LOAD-BEARING, and getting it wrong fails silently.
    # With ``--reference`` placed before ``--epsilon``, the CLI writes NO
    # output and exits 0 - measured on this installation - so a wrong order
    # reads back as "no values" rather than as an error. ``--epsilon`` before
    # ``--reference``, with the approximation sets last, is the order that
    # works and is also the order src/diagnostics.py uses for MetricsEvaluator.
    cmd = [get_cli_path(), "CalculateIndicator",
           "--problem", problem, "--indicator", indicator]
    if epsilons is not None:
        cmd += ["--epsilon", ",".join(str(float(e)) for e in epsilons)]
    cmd += ["--reference", str(reference)]
    cmd += [str(p) for p in inputs]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True,
                              timeout=timeout_s, env=_cli_env())
    except subprocess.TimeoutExpired:
        print(f"[tev] {indicator}: timed out after {timeout_s:.0f}s; reported as "
              f"unavailable.", flush=True)
        return {}
    except subprocess.CalledProcessError as exc:
        print(f"[tev] {indicator}: CLI failed: {exc.stderr.strip()[:500]}", flush=True)
        return {}

    values = {}
    for line in proc.stdout.splitlines():
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            values[parts[0].strip()] = float(parts[1])
        except ValueError:
            continue

    # Exit 0 with nothing parsed is the silent-failure mode this CLI has
    # several routes into: a wrong argument order, a legacy-dialect .set passed
    # as the reference, a missing entry terminator. None of them error. Refuse
    # to report it as an absent result.
    if not values:
        raise RuntimeError(
            f"[tev] {indicator}: the CLI exited 0 but produced no parseable "
            f"values. This is the silent-failure path, not an empty answer. "
            f"Check the argument order, that every input carries the "
            f"'# Version=5' header and the trailing '#' terminator, and that "
            f"the reference set is non-empty.\n"
            f"[tev] command: {' '.join(cmd)}\n"
            f"[tev] stdout: {proc.stdout[:300]!r}\n"
            f"[tev] stderr: {proc.stderr[:300]!r}")
    return values
