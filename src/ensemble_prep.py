"""ensemble_prep.py - Stage a streamflow ensemble into pywrdrb HDF5 inputs.

Shared step-04 logic for the main workflow
(``scripts/main/prep_pywrdrb_inputs.py``) and the local test-fixture staging
(``src/local_test_ensemble.py``). Given an :class:`~src.ensembles.EnsembleSpec`
whose ``catchment_inflow_mgd.hdf5`` has already been staged by the step-02
generator (``src/ensemble_generation.py``), this produces the remaining files
the trimmed-model ensemble simulation path requires under
``STAGED_ENSEMBLE_DIR/{inflow_type}/``:

    catchment_inflow_with_flood_nodes_mgd.hdf5   (FloodNodeInflowEnsemblePreprocessor)
    presimulated_releases_mgd.hdf5  (+ _metadata.json)   (STARFITReleaseEnsemblePreprocessor)
    predicted_inflows_mgd.hdf5                   (PredictedInflowEnsemblePreprocessor)
    diversion_nyc_extrapolated_mgd.hdf5          (ExtrapolatedDiversionEnsemblePreprocessor)

The flood-augmented inflows are needed because the optimization model enables
NYC flood operations (``enable_nyc_flood_operations=True`` in
``src/simulation.py::_build_model_builder``); the presimulated releases feed the
trimmed model; the predicted inflows feed the Montague/Trenton release forecast
(``PredictionEnsemble``). STARFIT runs before predicted inflows because the
``perfect_foresight`` prediction mode reads the presimulated-release HDF5.

Exactly ONE forecast mode is staged: ``config.PYWRDRB_FLOW_PREDICTION_MODE``,
the project standard for every Pywr-DRB simulation, which is also what
``src/simulation.py::_build_model_builder`` pins at model build. Staging any
other mode writes columns no model in this project reads, at full
per-realization preprocessing cost.

The NYC demand series is the time-dynamic extrapolated diversion every
ensemble simulation reads (``config.NYC_DEMAND_SOURCE_ENSEMBLE``): for each
realization the quarterly diversion-on-log-inflow regressions, fitted once to
the 2000-2021 River Master record on the reconstruction's NYC reservoir inflows,
are applied to the same three-node inflow sum of that realization,
nearest-neighbour daily patterns are attached, and the 800 MGD
monthly-mean cap is enforced (:func:`stage_nyc_demand`). NJ demand is the
constant 100 MGD Decree quantity (``config.NJ_DEMAND_SOURCE``), so no NJ
diversion series or NJ-diversion forecast (``predicted_diversions_mgd.hdf5``)
is staged.

E_test chunks (``_meta.json`` ``kind == "forcing_pool_chunk"``) are the one
case where the series is scaled: each realization's whole regression-structured
series is multiplied by its SOW's demand scale ``k`` (the NYC entitlement
utilization axis, ``src.demand_utilization``; read from the parent's
``demand_utilization_nominal.csv``) BEFORE the cap, and the realized
utilization of every staged realization is written to
``demand_utilization_realized.csv`` in the chunk dir. Every other ensemble
(search ensembles, local fixtures) is staged unscaled, ``k = 1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

#: Filenames each preprocessor writes into the staged ensemble directory.
_CATCHMENT_INFLOW = "catchment_inflow_mgd.hdf5"
_FLOOD_INFLOW = "catchment_inflow_with_flood_nodes_mgd.hdf5"
_PRESIM_RELEASES = "presimulated_releases_mgd.hdf5"
_PREDICTED_INFLOWS = "predicted_inflows_mgd.hdf5"
_NYC_DEMAND = "diversion_nyc_extrapolated_mgd.hdf5"
#: Per-realization realized-utilization table written beside the demand HDF5 of
#: an E_test chunk (merged into the parent dir by
#: ``scripts/supplemental/merge_demand_utilization_realized.py``).
_REALIZED_UTILIZATION = "demand_utilization_realized.csv"
#: Per-SOW nominal table the append step writes into the parent E_test dir.
_NOMINAL_UTILIZATION = "demand_utilization_nominal.csv"


def chunk_demand_scales(spec) -> tuple[dict[str, float] | None, Any]:
    """Per-realization NYC demand scales of an E_test chunk, or ``None`` otherwise.

    A chunk's ``_meta.json`` (``kind == "forcing_pool_chunk"``) maps each local
    realization to its global id (``global_realization_ids``) and so to its
    SOW (``global_id // realizations_per_profile``), whose scale ``k`` comes
    from the parent slug's ``demand_utilization_nominal.csv``.

    Args:
        spec: The staged ensemble's :class:`~src.ensembles.EnsembleSpec`.

    Returns:
        ``(scales, mapping)``: ``scales`` keyed by local realization id as a
        string (what the Pywr-DRB ensemble preprocessor expects) and a frame
        with columns ``local_id, global_id, sow, u_nominal, k`` aligned to
        ``spec.realization_indices``; ``(None, None)`` for any ensemble that
        is not an E_test chunk (staged unscaled, ``k = 1``).

    Raises:
        FileNotFoundError: If the chunk's parent carries no nominal table
            (``append_demand_utilization.py`` has not run).
        KeyError: If a SOW of the chunk is absent from the parent's table.
    """
    import json

    import pandas as pd

    from src.ensembles import staged_ensemble_dir

    meta_path = staged_ensemble_dir(spec.inflow_type) / "_meta.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text())
    if meta.get("kind") != "forcing_pool_chunk":
        return None, None
    parent = str(meta["pool_slug"])
    rpp = int(meta["realizations_per_profile"])
    gids = [int(g) for g in meta["global_realization_ids"]]
    table_path = staged_ensemble_dir(parent) / _NOMINAL_UTILIZATION
    if not table_path.exists():
        raise FileNotFoundError(
            f"E_test chunk '{spec.inflow_type}' belongs to '{parent}', which carries no "
            f"{_NOMINAL_UTILIZATION}: the NYC utilization column has not been appended. "
            f"Run `python scripts/supplemental/append_demand_utilization.py --variant "
            f"<variant>` on the parent before staging any chunk's demand series."
        )
    table = pd.read_csv(table_path).set_index("sow")
    rows = []
    for local in spec.realization_indices:
        gid = gids[int(local)]
        sow = gid // rpp
        if sow not in table.index:
            raise KeyError(
                f"SOW {sow} (global realization {gid} of chunk '{spec.inflow_type}') is "
                f"absent from {table_path}; the parent's table and the chunk disagree."
            )
        rows.append({"local_id": int(local), "global_id": gid, "sow": int(sow),
                     "u_nominal": float(table.at[sow, "u"]), "k": float(table.at[sow, "k"])})
    mapping = pd.DataFrame(rows)
    scales = {str(r.local_id): float(r.k) for r in mapping.itertuples(index=False)}
    return scales, mapping


def write_realized_utilization(demand_h5: Path, mapping, out_csv: Path):
    """Compute each staged realization's realized utilization and persist the table.

    Args:
        demand_h5: The staged ``diversion_nyc_extrapolated_mgd.hdf5``.
        mapping: The frame returned by :func:`chunk_demand_scales`.
        out_csv: Destination (``demand_utilization_realized.csv`` in the chunk dir).

    Returns:
        The written frame (``global_id, sow, u_nominal, k, u_realized``).
    """
    import h5py
    import pandas as pd

    from src.demand_utilization import realized_utilization

    rows = []
    with h5py.File(demand_h5, "r") as f:
        for r in mapping.itertuples(index=False):
            group = f[str(r.local_id)]
            idx = pd.to_datetime(group["datetime"][:].astype(str))
            series = pd.Series(group["aggregate"][:], index=idx, dtype=float)
            rows.append({"global_id": int(r.global_id), "sow": int(r.sow),
                         "u_nominal": float(r.u_nominal), "k": float(r.k),
                         "u_realized": realized_utilization(series)})
    frame = pd.DataFrame(rows).sort_values("global_id").reset_index(drop=True)
    frame.to_csv(out_csv, index=False)
    return frame


def _rank_of(use_mpi: bool, comm) -> int:
    """Return this process's MPI rank (0 in serial mode)."""
    if not use_mpi:
        return 0
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    return comm.Get_rank()


def stage_nyc_demand(spec, *, use_mpi: bool = False, comm=None,
                     force: bool = False) -> Path:
    """Stage the per-realization NYC demand series of an ensemble.

    Runs Pywr-DRB's ``ExtrapolatedDiversionEnsemblePreprocessor`` over
    ``spec.realization_indices`` and writes ``diversion_nyc_extrapolated_mgd.hdf5``
    into the staged directory: each realization's monthly demand is the
    quarterly regression on its own log NYC inflow plus a seeded residual
    draw, disaggregated to days by nearest-neighbour month matching and capped
    at an 800 MGD monthly mean. This is the NYC demand every ensemble
    simulation reads (``config.NYC_DEMAND_SOURCE_ENSEMBLE``). Idempotent unless
    ``force``; callable on its own for ensembles whose other step-04 inputs are
    already staged (``scripts/supplemental/stage_nyc_demand.py``).

    For an E_test chunk the series is scaled per SOW before the cap by the
    utilization axis's ``k`` (:func:`chunk_demand_scales`; the parent must have
    been appended first, otherwise this fails loudly), and the realized
    utilization of every realization is written to
    ``demand_utilization_realized.csv`` beside the HDF5. Every other ensemble
    is staged unscaled.

    Args:
        spec: An :class:`~src.ensembles.EnsembleSpec` with ``is_ensemble=True``
            whose ``catchment_inflow_mgd.hdf5`` is staged.
        use_mpi: Distribute per-realization work across MPI ranks.
        comm: MPI communicator (defaults to ``MPI.COMM_WORLD`` when ``use_mpi``).
        force: Rebuild the output even if it already exists.

    Returns:
        Path of the staged demand HDF5.

    Raises:
        ValueError: If ``spec`` is not an ensemble spec.
        FileNotFoundError: If the base ``catchment_inflow_mgd.hdf5`` is missing.
    """
    from pywrdrb.pre.extrapolate_nyc_nj_diversions import (
        ExtrapolatedDiversionEnsemblePreprocessor,
    )

    from src.ensembles import register_ensemble_path, staged_ensemble_dir

    if not getattr(spec, "is_ensemble", False):
        raise ValueError(
            f"stage_nyc_demand requires an ensemble spec; got preset "
            f"'{spec.preset_name}' with is_ensemble=False."
        )
    rank = _rank_of(use_mpi, comm)
    inflow_type = spec.inflow_type
    register_ensemble_path(inflow_type)
    flows_dir = staged_ensemble_dir(inflow_type)
    base_inflow = flows_dir / _CATCHMENT_INFLOW
    if not base_inflow.exists():
        raise FileNotFoundError(
            f"Base ensemble inflows not found: {base_inflow}. Run the step-02 "
            "generator for this inflow_type first."
        )
    out = flows_dir / _NYC_DEMAND
    if out.exists() and not force:
        if rank == 0:
            print(f"[ensemble_prep:{inflow_type}] NYC demand exists, skipping "
                  f"({out.name})", flush=True)
        return out
    scales, mapping = chunk_demand_scales(spec)
    if rank == 0:
        scaled = ""
        if scales:
            ks = list(scales.values())
            scaled = (f" with per-SOW demand scales k in [{min(ks):.3f}, {max(ks):.3f}]"
                      f" ({mapping['sow'].nunique()} SOWs)")
        print(f"[ensemble_prep:{inflow_type}] staging per-realization NYC demand "
              f"({len(list(spec.realization_indices))} realizations){scaled} ...",
              flush=True)
    demand_pp = ExtrapolatedDiversionEnsemblePreprocessor(
        loc="nyc",
        flow_type=inflow_type,
        ensemble_hdf5_file=str(base_inflow),
        realization_ids=list(spec.realization_indices),
        use_mpi=use_mpi,
        comm=comm,
        demand_scales=scales,
    )
    demand_pp.load()
    demand_pp.process()
    demand_pp.save()
    if mapping is not None and rank == 0:
        realized = write_realized_utilization(out, mapping, flows_dir / _REALIZED_UTILIZATION)
        diff = realized["u_realized"] - realized["u_nominal"]
        print(f"[ensemble_prep:{inflow_type}] realized - nominal u: mean {diff.mean():+.4f}, "
              f"range [{diff.min():+.4f}, {diff.max():+.4f}] -> {_REALIZED_UTILIZATION}",
              flush=True)
    return out


def stage_pywrdrb_ensemble_inputs(
    spec,
    *,
    use_mpi: bool = False,
    comm=None,
    force: bool = False,
    enable_flood: bool = True,
    initial_volume_frac: "float | None" = None,
) -> dict:
    """Stage the pywrdrb HDF5 inputs a trimmed-model ensemble simulation needs.

    Runs the flood-node, STARFIT-release, predicted-inflow and NYC-demand
    ensemble preprocessors in order over ``spec.realization_indices``. Each preprocessor
    is idempotent: when its output already exists and ``force`` is ``False`` it
    is skipped (the same decision on every rank, since all ranks share the
    filesystem).

    Args:
        spec: An :class:`~src.ensembles.EnsembleSpec` with ``is_ensemble=True``.
            Its ``catchment_inflow_mgd.hdf5`` must already be staged.
        use_mpi: Distribute per-realization work across MPI ranks.
        comm: MPI communicator (defaults to ``MPI.COMM_WORLD`` when ``use_mpi``).
        force: Rebuild outputs even if they already exist.
        enable_flood: Stage the flood-augmented inflows. Must be ``True`` for the
            optimization model, which enables NYC flood operations.
        initial_volume_frac: STARFIT initial storage fraction. Must match the
            model's ``INITIAL_VOLUME_FRAC``. ``None`` reads it from ``config``.

    Returns:
        Manifest mapping each staged file's role to its absolute path.

    Raises:
        ValueError: If ``spec`` is not an ensemble spec.
        FileNotFoundError: If the base ``catchment_inflow_mgd.hdf5`` is missing.
    """
    from pywrdrb.pre.flood_node_inflows import FloodNodeInflowEnsemblePreprocessor
    from pywrdrb.pre.generate_presimulated_releases import (
        STARFITReleaseEnsemblePreprocessor,
    )
    from pywrdrb.pre.predict_inflows import PredictedInflowEnsemblePreprocessor

    import config
    from src.ensembles import register_ensemble_path, staged_ensemble_dir

    if not getattr(spec, "is_ensemble", False):
        raise ValueError(
            f"stage_pywrdrb_ensemble_inputs requires an ensemble spec; "
            f"got preset '{spec.preset_name}' with is_ensemble=False."
        )

    rank = _rank_of(use_mpi, comm)
    inflow_type = spec.inflow_type
    realization_ids = list(spec.realization_indices)
    if initial_volume_frac is None:
        initial_volume_frac = config.INITIAL_VOLUME_FRAC

    # pywrdrb's path navigator must resolve flows/{inflow_type} -> staged dir
    # before any preprocessor (they look the directory up via pn.sc.get).
    register_ensemble_path(inflow_type)
    flows_dir = staged_ensemble_dir(inflow_type)

    base_inflow = flows_dir / _CATCHMENT_INFLOW
    if not base_inflow.exists():
        raise FileNotFoundError(
            f"Base ensemble inflows not found: {base_inflow}. Run the step-02 "
            "generator for this inflow_type first."
        )

    def _log(msg: str) -> None:
        if rank == 0:
            print(f"[ensemble_prep:{inflow_type}] {msg}", flush=True)

    manifest: dict = {"inflow_type": inflow_type, "flows_dir": str(flows_dir),
                      "catchment_inflow": str(base_inflow)}

    # 1. Flood-augmented inflows (required by enable_nyc_flood_operations).
    if enable_flood:
        flood_out = flows_dir / _FLOOD_INFLOW
        if flood_out.exists() and not force:
            _log(f"flood inflows exist, skipping ({flood_out.name})")
        else:
            _log("staging flood-augmented inflows ...")
            flood_pp = FloodNodeInflowEnsemblePreprocessor(
                inflow_type=inflow_type,
                realization_ids=realization_ids,
                use_mpi=use_mpi,
                comm=comm,
                force=force,
            )
            flood_pp.load()
            flood_pp.process()
            flood_pp.save()
        manifest["flood_inflow"] = str(flood_out)

    # 2. Presimulated STARFIT releases (trimmed model; also read by the
    #    perfect_foresight predicted-inflow mode, so it must precede step 3).
    presim_out = flows_dir / _PRESIM_RELEASES
    if presim_out.exists() and not force:
        _log(f"presim releases exist, skipping ({presim_out.name})")
    else:
        _log("staging presimulated STARFIT releases ...")
        starfit_pp = STARFITReleaseEnsemblePreprocessor(
            inflow_type=inflow_type,
            realization_ids=realization_ids,
            use_mpi=use_mpi,
            comm=comm,
            force=force,
            initial_volume_frac=initial_volume_frac,
        )
        starfit_pp.run()
    manifest["presimulated_releases"] = str(presim_out)

    # 3. Predicted Montague/Trenton inflows (release-forecast PredictionEnsemble).
    predicted_out = flows_dir / _PREDICTED_INFLOWS
    if predicted_out.exists() and not force:
        _log(f"predicted inflows exist, skipping ({predicted_out.name})")
    else:
        _log(f"staging predicted inflows (mode={config.PYWRDRB_FLOW_PREDICTION_MODE}) ...")
        pred_pp = PredictedInflowEnsemblePreprocessor(
            flow_type=inflow_type,
            ensemble_hdf5_file=str(base_inflow),
            realization_ids=realization_ids,
            modes=(config.PYWRDRB_FLOW_PREDICTION_MODE,),
            use_mpi=use_mpi,
            comm=comm,
        )
        pred_pp.load()
        pred_pp.process()
        pred_pp.save()
    manifest["predicted_inflows"] = str(predicted_out)

    # 4. Per-realization NYC demand (the time-dynamic extrapolated series every
    #    ensemble simulation reads; NJ is a constant and needs no staging).
    manifest["nyc_demand"] = str(
        stage_nyc_demand(spec, use_mpi=use_mpi, comm=comm, force=force))

    _log("done.")
    return manifest
