"""Acceptance test: vectorized perfect-foresight predicted inflows == scalar.

Runs pywrdrb's PredictedInflowEnsemblePreprocessor twice over the staged local
test ensemble (kn_50yr_n5, the fixture test_master_ensemble_determinism also
uses) — once on the scalar reference path, once on the vectorized kernel — and
bit-compares every dataset of the two output HDF5s. Same process, same code:
exact equality, no tolerance (the anvil cross-era 1%-range convention applies
only to cross-job comparisons).

Marked ``slow``: stages the local ensemble on first run and the scalar
reference path takes O(minutes) for 2 realizations x 50 yr.

    python -m pytest tests/test_predicted_inflow_vectorization.py -v -m slow
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

slow = pytest.mark.slow


@slow
def test_vectorized_predicted_inflows_match_scalar_on_staged_ensemble(tmp_path):
    from src.local_test_ensemble import ensure_local_test_ensemble

    spec = ensure_local_test_ensemble()

    import config
    from src.ensembles import register_ensemble_path, staged_ensemble_dir

    from pywrdrb.pre.predict_inflows import PredictedInflowEnsemblePreprocessor

    slug = spec.inflow_type
    register_ensemble_path(slug)
    flows_dir = staged_ensemble_dir(slug)
    base_inflow = flows_dir / "catchment_inflow_mgd.hdf5"
    assert base_inflow.exists()

    # Two realizations keep the scalar reference to a few minutes.
    realization_ids = list(spec.realization_indices)[:2]

    outputs = {}
    for label, vectorize in (("scalar", False), ("vector", True)):
        pp = PredictedInflowEnsemblePreprocessor(
            flow_type=slug,
            ensemble_hdf5_file=str(base_inflow),
            realization_ids=realization_ids,
            modes=(config.PYWRDRB_FLOW_PREDICTION_MODE,),
        )
        pp._vectorize_perfect_foresight = vectorize
        # Never touch the staged artifact: redirect the save target.
        out_path = tmp_path / f"predicted_inflows_{label}.hdf5"
        pp.output_dirs = {"predicted_inflows_mgd.hdf5": out_path}
        pp.load()
        pp.process()
        pp.save()
        assert out_path.exists()
        outputs[label] = out_path

    with h5py.File(outputs["scalar"], "r") as f_ref, \
            h5py.File(outputs["vector"], "r") as f_vec:
        assert set(f_ref.keys()) == set(f_vec.keys())
        for rid in f_ref.keys():
            g_ref, g_vec = f_ref[rid], f_vec[rid]
            assert set(g_ref.keys()) == set(g_vec.keys())
            for ds in g_ref.keys():
                a, b = g_ref[ds][:], g_vec[ds][:]
                if ds == "datetime":
                    assert (a == b).all(), f"{rid}/{ds}"
                else:
                    np.testing.assert_array_equal(a, b, err_msg=f"{rid}/{ds}")
