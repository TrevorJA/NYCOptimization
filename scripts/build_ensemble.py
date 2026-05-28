"""
build_ensemble.py - Stage a multi-realization streamflow ensemble for Pywr-DRB.

Resolves an ``EnsembleSpec`` from the registry and writes two HDF5 files to
``{config.STAGED_ENSEMBLE_DIR}/{spec.inflow_type}/``:

  catchment_inflow_mgd.hdf5    - per-realization catchment inflows
                                 (consumed by pywrdrb.parameters.FlowEnsemble)
  predicted_inflows_mgd.hdf5   - per-realization Montague/Trenton lag forecasts
                                 (consumed by pywrdrb.parameters.PredictionEnsemble;
                                 required when running with inflow_ensemble_indices)

A small ``manifest.json`` records the generator class, parameters, RNG seed,
training-data source, and timestamps so a stage can be reproduced.

Usage
-----
    # Stage the preset selected by NYCOPT_ENSEMBLE_PRESET (default: historic_single,
    # which is a no-op):
    python scripts/build_ensemble.py

    # Stage a specific preset (override env):
    python scripts/build_ensemble.py --preset wcu_kirsch_n5

    # Force regeneration even if HDF5s exist:
    python scripts/build_ensemble.py --preset wcu_kirsch_n5 --force

The script is idempotent: existing HDF5s are kept unless ``--force`` is set.
``historic_single`` is a no-op (the legacy single-trace inflow_type is loaded
directly from pywrdrb's bundled data; nothing to stage).

Generators are registered in ``GENERATORS`` keyed on ``EnsembleSpec.source_kind``.
v1 ships ``synhydro_kn`` (Kirsch-Nowak via SynHydro). ``moeafind`` is reserved
for the deferred MOEA-FIND drop-in.

See: local_notes/configuration/knob_reference.md (Ensemble evaluation table).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import order: config first (it sources STAGED_ENSEMBLE_DIR + simulation
# dates), then ensembles helpers. NYCOPT_ENSEMBLE_PRESET is read at config
# import time; --preset overrides via re-resolve.
from config import START_DATE, END_DATE, STAGED_ENSEMBLE_DIR  # noqa: E402
from src.ensembles import (  # noqa: E402
    EnsembleSpec,
    get_ensemble_spec,
    list_presets,
    register_ensemble_path,
    staged_ensemble_dir,
)


###############################################################################
# Generator protocol + registry
###############################################################################

class Generator(Protocol):
    """Generator interface. Concrete classes implement ``stage``."""

    source_kind: str

    def stage(self, spec: EnsembleSpec, force: bool = False) -> Path:
        """Stage ``spec`` to its target directory and return that path."""
        ...


GENERATORS: dict[str, "Generator"] = {}


def register_generator(generator: "Generator") -> None:
    GENERATORS[generator.source_kind] = generator


###############################################################################
# KirschNowakGenerator (SynHydro)
###############################################################################

class KirschNowakGenerator:
    """Stage a Kirsch-Nowak ensemble via SynHydro.

    Training data is the bundled Pywr-DRB historic catchment inflow CSV at
    ``flows/pub_nhmv10_BC_withObsScaled/catchment_inflow_mgd.csv`` (the
    Amestoy et al. 2026 reconstructed median trace for 1945-2023). The 31
    column names are already pywrdrb node identifiers, so no site renaming
    is required.

    Each realization spans the optimization simulation window (START_DATE
    through END_DATE) by water years. Synthetic flows are placed on a daily
    DatetimeIndex starting at START_DATE so pywrdrb's loader aligns the
    ensemble with its simulation timestepper.

    After generation, three pywrdrb preprocessor post-steps run in order:

      1. ``pywrdrb.pre.STARFITReleaseEnsemblePreprocessor`` — precomputes
         daily STARFIT releases per realization and writes
         ``presimulated_releases_mgd.hdf5``. This artifact is the single
         source of truth that both the trimmed model
         (``PresimulatedReleaseEnsemble`` parameter) and the
         perfect-foresight predicted-inflow preprocessor consume — so it
         must run **before** the predicted-inflow step (step 2).
      2. ``pywrdrb.pre.PredictedInflowEnsemblePreprocessor`` in
         ``perfect_foresight`` mode (matching pywrdrb's default
         ``flow_prediction_mode``) — reads the precomputed STARFIT
         releases from step 1 instead of recomputing them in-process,
         then produces the Montague/Trenton lag forecasts consumed by
         ``PredictionEnsemble`` during simulation.
      3. ``pywrdrb.pre.FloodNodeInflowEnsemblePreprocessor`` — adds the
         three flood-monitoring USGS-gauge nodes (01426500 Hale Eddy,
         01421000 Fishs Eddy, 01436690 Bridgeville) by drainage-area
         scaling and rebalances ``delLordville`` / ``delMontague`` for
         mass conservation. Required for ensemble runs with
         ``enable_nyc_flood_operations=True``; pywrdrb's ``ModelBuilder``
         routes ``FlowEnsemble`` to ``catchment_inflow_with_flood_nodes_mgd.hdf5``
         when that option is set.
    """

    source_kind = "synhydro_kn"
    training_inflow_type = "pub_nhmv10_BC_withObsScaled"

    def stage(self, spec: EnsembleSpec, force: bool = False) -> Path:
        out_dir = staged_ensemble_dir(spec.inflow_type)
        out_dir.mkdir(parents=True, exist_ok=True)

        catchment_h5 = out_dir / "catchment_inflow_mgd.hdf5"
        presim_h5 = out_dir / "presimulated_releases_mgd.hdf5"
        flood_aug_h5 = out_dir / "catchment_inflow_with_flood_nodes_mgd.hdf5"
        predicted_h5 = out_dir / "predicted_inflows_mgd.hdf5"
        manifest = out_dir / "manifest.json"

        if (
            catchment_h5.exists()
            and presim_h5.exists()
            and predicted_h5.exists()
            and flood_aug_h5.exists()
            and not force
        ):
            logging.info(
                f"[stage] {spec.preset_name}: already staged at {out_dir} "
                f"(use --force to regenerate)"
            )
            return out_dir

        n_realizations = max(spec.n_realizations, max(spec.realization_indices) + 1)
        if n_realizations != spec.n_realizations:
            # Sparse indices (e.g. [0, 2, 4]) — generate enough realizations to
            # cover the highest requested ID; the staging file then stores all,
            # and pywrdrb subsets at runtime via ``inflow_ensemble_indices``.
            logging.info(
                f"[stage] {spec.preset_name}: realization_indices={spec.realization_indices} "
                f"-> generating {n_realizations} realizations to cover the highest index"
            )

        logging.info(
            f"[stage] {spec.preset_name}: generating {n_realizations} K-N "
            f"realizations seeded {spec.seed} "
            f"(training data = {self.training_inflow_type})"
        )
        t0 = time.time()
        ensemble = self._generate(spec, n_realizations)
        logging.info(f"[stage] generation complete in {time.time() - t0:.1f}s")

        logging.info(f"[stage] writing {catchment_h5}")
        ensemble.to_hdf5(str(catchment_h5), compression="gzip", stored_by_node=True)

        # PredictedInflowEnsemblePreprocessor needs to find the catchment HDF5
        # via pywrdrb's path navigator under ``flows/{inflow_type}``.
        register_ensemble_path(spec.inflow_type)

        logging.info(f"[stage] precomputing STARFIT releases -> {presim_h5}")
        t0 = time.time()
        self._stage_starfit_releases(spec=spec, force=force)
        logging.info(
            f"[stage] STARFIT release post-step complete in {time.time() - t0:.1f}s"
        )

        logging.info(f"[stage] computing predicted inflows -> {predicted_h5}")
        t0 = time.time()
        self._stage_predicted_inflows(
            spec=spec,
            ensemble_hdf5_file=catchment_h5,
            n_generated=n_realizations,
        )
        logging.info(
            f"[stage] predicted-inflow post-step complete in {time.time() - t0:.1f}s"
        )

        logging.info(f"[stage] augmenting flood-monitoring nodes -> {flood_aug_h5}")
        t0 = time.time()
        self._stage_flood_node_inflows(spec=spec, force=force)
        logging.info(
            f"[stage] flood-node post-step complete in {time.time() - t0:.1f}s"
        )

        manifest.write_text(json.dumps({
            "preset_name": spec.preset_name,
            "inflow_type": spec.inflow_type,
            "source_kind": spec.source_kind,
            "n_generated_realizations": n_realizations,
            "preset_realization_indices": list(spec.realization_indices),
            "seed": spec.seed,
            "training_inflow_type": self.training_inflow_type,
            "training_window": {"start": str(START_DATE), "end": str(END_DATE)},
            "synthetic_window": {"start": str(START_DATE), "end": str(END_DATE)},
            "generator": "synhydro.pipelines.KirschNowakPipeline",
            "predicted_inflow_mode": "perfect_foresight",
            "starfit_releases_precomputed": True,
            "flood_augmented": True,
            "staged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2))
        logging.info(f"[stage] wrote {manifest}")
        return out_dir

    # -- internals ------------------------------------------------------------

    def _load_training_df(self) -> pd.DataFrame:
        """Read the historic single-trace catchment inflows from pywrdrb."""
        import pywrdrb
        from pywrdrb.path_manager import get_pn_object
        pn = get_pn_object()
        csv_path = Path(pn.sc.get(f"flows/{self.training_inflow_type}")) / "catchment_inflow_mgd.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training inflow CSV not found: {csv_path}. "
                f"Required for KirschNowakGenerator."
            )
        df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)
        # Constrain to the optimization window so the synthetic ensemble's
        # statistics reflect the window we'll run in.
        df = df.loc[str(START_DATE):str(END_DATE)]
        if df.empty:
            raise ValueError(
                f"Training data empty after slicing to [{START_DATE}, {END_DATE}]"
            )
        return df

    def _generate(self, spec: EnsembleSpec, n_realizations: int):
        """Run KirschNowakPipeline and align dates to the simulation window.

        Returns the SynHydro ``Ensemble`` instance with its DatetimeIndex
        rewritten to start at ``START_DATE`` (one row per simulation day).
        """
        from synhydro.pipelines import KirschNowakPipeline
        from synhydro.core.ensemble import Ensemble

        Q_obs = self._load_training_df()

        # n_years sized to span the simulation window; SynHydro's Nowak
        # disaggregator produces one synthetic day per training day so the
        # downstream realignment to START_DATE preserves day-of-year statistics.
        # Allow the preset to override with a shorter window
        # (EnsembleSpec.realization_years) for fast iteration during
        # pipeline development.
        if spec.realization_years is not None:
            n_years = int(spec.realization_years)
        else:
            n_years = int(np.ceil((Q_obs.index[-1] - Q_obs.index[0]).days / 365.25))

        pipeline = KirschNowakPipeline(
            generate_using_log_flow=True,
            n_neighbors=5,
        )
        pipeline.preprocessing(Q_obs)
        pipeline.fit()
        ensemble = pipeline.generate(
            n_realizations=n_realizations,
            n_years=n_years,
            seed=spec.seed,
        )

        # Rewrite the DatetimeIndex of every realization to start at START_DATE.
        # SynHydro's generate-time index is internally consistent but offset
        # from our calendar; pywrdrb's FlowEnsemble looks up by calendar date
        # so we need exact alignment with the sim timestepper.
        target_index = pd.date_range(start=START_DATE, periods=len(Q_obs), freq="D")
        rewritten = {}
        for real_id, real_df in ensemble.data_by_realization.items():
            n_rows = min(len(real_df), len(target_index))
            new_df = real_df.iloc[:n_rows].copy()
            new_df.index = target_index[:n_rows]
            new_df.index.name = "datetime"
            rewritten[real_id] = new_df
        return Ensemble(rewritten, metadata=ensemble.metadata)

    def _stage_predicted_inflows(
        self,
        spec: EnsembleSpec,
        ensemble_hdf5_file: Path,
        n_generated: int,
    ) -> None:
        from pywrdrb.pre import PredictedInflowEnsemblePreprocessor

        preprocessor = PredictedInflowEnsemblePreprocessor(
            flow_type=spec.inflow_type,
            ensemble_hdf5_file=str(ensemble_hdf5_file),
            realization_ids=[str(i) for i in range(n_generated)],
            start_date=None,
            end_date=None,
            modes=("perfect_foresight",),
            use_log=True,
            remove_zeros=True,
            use_const=False,
            use_mpi=False,
        )
        preprocessor.load()
        preprocessor.process()
        preprocessor.save()

    def _stage_starfit_releases(self, spec: EnsembleSpec, force: bool) -> None:
        """Run pywrdrb's ensemble STARFIT preprocessor as a post-step.

        Reads ``catchment_inflow_mgd.hdf5`` from the staged directory and
        writes ``presimulated_releases_mgd.hdf5`` (plus a JSON metadata
        sidecar) next to it. The output is the single source of truth for
        STARFIT releases consumed by:

          - ``PresimulatedReleaseEnsemble`` parameter (the ensemble
            trimmed-model release source — pywrdrb's ``ModelBuilder``
            wires it automatically when ``use_trimmed_model=True`` AND
            ``inflow_ensemble_indices`` is set).
          - ``PredictedInflowEnsemblePreprocessor`` in
            ``perfect_foresight`` mode (now reads precomputed releases
            from this HDF5 instead of recomputing STARFIT in-process).

        Therefore this step **must run before** ``_stage_predicted_inflows``.
        """
        from pywrdrb.pre import STARFITReleaseEnsemblePreprocessor
        from config import INITIAL_VOLUME_FRAC

        preprocessor = STARFITReleaseEnsemblePreprocessor(
            inflow_type=spec.inflow_type,
            realization_ids=None,    # process every realization in the HDF5
            use_mpi=False,
            force=force,
            initial_volume_frac=INITIAL_VOLUME_FRAC,
        )
        preprocessor.load()
        preprocessor.process()
        preprocessor.save()

    def _stage_flood_node_inflows(self, spec: EnsembleSpec, force: bool) -> None:
        """Run pywrdrb's ensemble flood-node preprocessor as a post-step.

        Reads ``catchment_inflow_mgd.hdf5`` from the staged directory and writes
        ``catchment_inflow_with_flood_nodes_mgd.hdf5`` next to it. The output
        adds three flood-monitoring nodes (01426500 Hale Eddy, 01421000 Fishs
        Eddy, 01436690 Bridgeville) via drainage-area scaling, and rebalances
        ``delLordville`` / ``delMontague`` for mass conservation. pywrdrb's
        ``ModelBuilder`` reads from this augmented HDF5 automatically when
        running with ``enable_nyc_flood_operations=True`` AND
        ``inflow_ensemble_indices`` set.
        """
        from pywrdrb.pre import FloodNodeInflowEnsemblePreprocessor

        preprocessor = FloodNodeInflowEnsemblePreprocessor(
            inflow_type=spec.inflow_type,
            realization_ids=None,   # process every realization in the HDF5
            use_mpi=False,
            force=force,
        )
        preprocessor.load()
        preprocessor.process()
        preprocessor.save()


register_generator(KirschNowakGenerator())


###############################################################################
# CLI
###############################################################################

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_preset = os.environ.get("NYCOPT_ENSEMBLE_PRESET", "historic_single")
    p.add_argument(
        "--preset",
        default=default_preset,
        help=(
            f"Preset name from src.ensembles.PRESETS (default: $NYCOPT_ENSEMBLE_PRESET "
            f"= {default_preset!r}). Available: {list_presets()}."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if staged HDF5s already exist.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="DEBUG-level logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    spec = get_ensemble_spec(args.preset)
    logging.info(
        f"[build_ensemble] preset={spec.preset_name} "
        f"is_ensemble={spec.is_ensemble} N={spec.n_realizations} "
        f"source_kind={spec.source_kind} inflow_type={spec.inflow_type}"
    )

    if not spec.is_ensemble:
        logging.info(
            f"[build_ensemble] preset {spec.preset_name!r} is a single-trace "
            f"passthrough (is_ensemble=False); nothing to stage."
        )
        return 0

    if spec.source_kind not in GENERATORS:
        logging.error(
            f"No generator registered for source_kind={spec.source_kind!r}. "
            f"Registered: {list(GENERATORS)}."
        )
        return 2

    out_dir = GENERATORS[spec.source_kind].stage(spec, force=args.force)
    logging.info(
        f"[build_ensemble] staged {spec.preset_name} -> {out_dir} "
        f"(STAGED_ENSEMBLE_DIR = {STAGED_ENSEMBLE_DIR})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
