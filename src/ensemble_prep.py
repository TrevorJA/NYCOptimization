"""ensemble_prep.py - Stage a streamflow ensemble into pywrdrb HDF5 inputs.

Shared Step-3 logic for both the main workflow
(``scripts/main/prep_pywrdrb_inputs.py``) and the supplemental ensemble
objective-sensitivity experiment. Given an :class:`~src.ensembles.EnsembleSpec`
whose ``catchment_inflow_mgd.hdf5`` has already been staged by the Step-1
generator (``src/ensemble_generation.py``), this produces the remaining files
the trimmed-model ensemble simulation path requires under
``STAGED_ENSEMBLE_DIR/{inflow_type}/``:

    catchment_inflow_with_flood_nodes_mgd.hdf5   (FloodNodeInflowEnsemblePreprocessor)
    presimulated_releases_mgd.hdf5  (+ _metadata.json)   (STARFITReleaseEnsemblePreprocessor)
    predicted_inflows_mgd.hdf5                   (PredictedInflowEnsemblePreprocessor)

The flood-augmented inflows are needed because the optimization model enables
NYC flood operations (``enable_nyc_flood_operations=True`` in
``src/simulation.py::_build_model_builder``); the presimulated releases feed the
trimmed model; the predicted inflows feed the Montague/Trenton release forecast
(``PredictionEnsemble``). STARFIT runs before predicted inflows because the
``perfect_foresight`` prediction mode reads the presimulated-release HDF5.

NJ-diversion predictions (``predicted_diversions_mgd.hdf5``) are **not** staged:
the optimization model uses ``nyc_nj_demand_source="constant_max"``, under which
the model builder wires constant NJ-demand parameters instead of a diversions
``PredictionEnsemble``. A clear error is raised if a non-``constant_max`` demand
source is configured, since that path additionally requires a per-realization
diversion input file that this pipeline does not generate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

#: Filenames each preprocessor writes into the staged ensemble directory.
_CATCHMENT_INFLOW = "catchment_inflow_mgd.hdf5"
_FLOOD_INFLOW = "catchment_inflow_with_flood_nodes_mgd.hdf5"
_PRESIM_RELEASES = "presimulated_releases_mgd.hdf5"
_PREDICTED_INFLOWS = "predicted_inflows_mgd.hdf5"


def _rank_of(use_mpi: bool, comm) -> int:
    """Return this process's MPI rank (0 in serial mode)."""
    if not use_mpi:
        return 0
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    return comm.Get_rank()


def stage_pywrdrb_ensemble_inputs(
    spec,
    *,
    use_mpi: bool = False,
    comm=None,
    force: bool = False,
    prediction_modes: tuple = ("regression_disagg", "perfect_foresight"),
    enable_flood: bool = True,
    initial_volume_frac: "float | None" = None,
) -> dict:
    """Stage the pywrdrb HDF5 inputs a trimmed-model ensemble simulation needs.

    Runs the flood-node, STARFIT-release, and predicted-inflow ensemble
    preprocessors in order over ``spec.realization_indices``. Each preprocessor
    is idempotent: when its output already exists and ``force`` is ``False`` it
    is skipped (the same decision on every rank, since all ranks share the
    filesystem).

    Args:
        spec: An :class:`~src.ensembles.EnsembleSpec` with ``is_ensemble=True``.
            Its ``catchment_inflow_mgd.hdf5`` must already be staged.
        use_mpi: Distribute per-realization work across MPI ranks.
        comm: MPI communicator (defaults to ``MPI.COMM_WORLD`` when ``use_mpi``).
        force: Rebuild outputs even if they already exist.
        prediction_modes: Predicted-inflow modes to stage. ``perfect_foresight``
            (the model default) requires the presimulated-release HDF5, which is
            staged first.
        enable_flood: Stage the flood-augmented inflows. Must be ``True`` for the
            optimization model, which enables NYC flood operations.
        initial_volume_frac: STARFIT initial storage fraction. Must match the
            model's ``INITIAL_VOLUME_FRAC``. ``None`` reads it from ``config``.

    Returns:
        Manifest mapping each staged file's role to its absolute path.

    Raises:
        ValueError: If ``spec`` is not an ensemble spec.
        FileNotFoundError: If the base ``catchment_inflow_mgd.hdf5`` is missing.
        NotImplementedError: If a non-``constant_max`` NJ-demand source is set
            (would require an unstaged per-realization diversion input).
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

    if config.NYC_NJ_DEMAND_SOURCE != "constant_max":
        raise NotImplementedError(
            f"NYC_NJ_DEMAND_SOURCE='{config.NYC_NJ_DEMAND_SOURCE}' needs a "
            "per-realization diversion input (predicted_diversions_mgd.hdf5), "
            "which this pipeline does not stage. Use 'constant_max' or extend "
            "stage_pywrdrb_ensemble_inputs with PredictedDiversionEnsemblePreprocessor."
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
            f"Base ensemble inflows not found: {base_inflow}. Run the Step-1 "
            "generator (generate_kirsch_nowak_ensemble) for this inflow_type first."
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
        _log(f"staging predicted inflows (modes={prediction_modes}) ...")
        pred_pp = PredictedInflowEnsemblePreprocessor(
            flow_type=inflow_type,
            ensemble_hdf5_file=str(base_inflow),
            realization_ids=realization_ids,
            modes=prediction_modes,
            use_mpi=use_mpi,
            comm=comm,
        )
        pred_pp.load()
        pred_pp.process()
        pred_pp.save()
    manifest["predicted_inflows"] = str(predicted_out)

    _log("done.")
    return manifest
