"""
prep_pywrdrb_inputs.py - Format subsampled streamflow ensembles into
pywrdrb-ingestible HDF5 inputs that the trimmed optimization model can read
via FlowEnsemble / PredictionEnsemble.

Placeholder. For each ensemble preset (see config.ENSEMBLE_PRESETS) it
should stage:

    outputs/synthetic_ensembles/{inflow_type}/{preset}/catchment_inflow_mgd.hdf5
    outputs/synthetic_ensembles/{inflow_type}/{preset}/predicted_inflows_mgd.hdf5

so pywrdrb's path navigator can resolve them at simulation start
(see src/ensembles.py::register_ensemble_path).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Step 3: format subsampled ensembles into pywrdrb HDF5 inputs. "
        "Implementation deferred — see config.STAGED_ENSEMBLE_DIR for the "
        "expected output layout."
    )


if __name__ == "__main__":
    main()
