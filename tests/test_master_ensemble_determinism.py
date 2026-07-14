"""tests/test_master_ensemble_determinism.py - Forcing-ensemble regeneration determinism gate.

The regenerate-on-demand design (methods §3.2/§7) requires that realization ``k`` be reproducible
bit-for-bit from ``(root_seed, k)`` alone, invariant to how the realization range is partitioned.
This test generates a tiny CMIP6-forced candidate pool (``store_daily=True``), then regenerates one
realization in isolation and asserts it equals the staged HDF5 slice (post float32/HDF5 round-trip).
Because the full pool generates realizations in blocks while ``regenerate_realization`` generates a
single index alone, equality proves partition-invariance through the Kirsch -> Nowak ->
per-realization-KDE path.

Marked ``slow``: it fits the Kirsch/Nowak generators (needs the pywrdrb-bundled historical record).

    venv/Scripts/python.exe -m pytest tests/test_master_ensemble_determinism.py -v -m slow
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pytest

import config
from synhydro import Ensemble
from scengen.forcing_ensemble import ForcingEnsembleConfig

slow = pytest.mark.slow

# Tiny pool: 4 forcing profiles x 2 realizations = 8 realizations of 2-yr records.
_N_FORCING, _R, _L, _SEED = 4, 2, 2, 0
_REGEN_INDEX = 3  # a realization in the 2nd forcing profile (p = 3 // 2 = 1)


def _tiny_config(out_dir: Path) -> ForcingEnsembleConfig:
    csv = config.ENSEMBLE_FORCING_MEAN_FRAC_CSV
    if not Path(csv).exists():
        pytest.skip(f"CMIP6 forcing table not available: {csv}")
    return ForcingEnsembleConfig(
        root_seed=_SEED,
        n_forcing_profiles=_N_FORCING,
        realizations_per_profile=_R,
        realization_years=_L,
        output_dir=out_dir,
        mean_frac_csv=csv,
        store_daily=True,
        compute_hazard_image=True,
        hazard_block_size=2,  # force multi-block generation so partition-invariance is exercised
    )


@slow
def test_regenerated_realization_matches_staged(tmp_path):
    from src.ensemble_generation import generate_forcing_ensemble, regenerate_realization

    cfg = _tiny_config(tmp_path / "pool")
    manifest = generate_forcing_ensemble(cfg)

    # Manifest / staged artifacts describe the full pool.
    assert manifest.n_realizations == _N_FORCING * _R
    assert manifest.design == "du_forced_pool"
    out = tmp_path / "pool"
    for name in ("gage_flow_mgd.hdf5", "catchment_inflow_mgd.hdf5",
                 "forcing_profiles.npz", "hazard_image.npz", "_meta.json", "manifest.json"):
        assert (out / name).exists(), f"missing staged artifact: {name}"

    # Regenerate one realization in isolation and compare to the staged HDF5 slice. Read the full
    # ensemble and index by global key (the realization_subset= path re-keys the returned dict).
    staged = Ensemble.from_hdf5(str(out / "gage_flow_mgd.hdf5"))
    staged_df = staged.data_by_realization[_REGEN_INDEX]
    regen_df = regenerate_realization(_SEED, _REGEN_INDEX, config=cfg)

    cols = [c for c in staged_df.columns if c in regen_df.columns]
    assert cols, "no shared columns between staged and regenerated frames"
    # Marginal catchment inflows are a pure function of these gage flows, so gage equality
    # (which covers Kirsch generation, Nowak disaggregation, and the regressed downstream gages
    # from the per-realization KDE fill) implies inflow equality.
    np.testing.assert_array_equal(
        regen_df[cols].to_numpy(dtype=np.float32),
        staged_df[cols].to_numpy(dtype=np.float32),
    )


@slow
def test_chunked_generation(tmp_path):
    """A chunked pool writes K independent chunk ensembles (bounded memory), each global-id mapped."""
    import json
    from src.ensemble_generation import generate_forcing_ensemble, regenerate_realization
    from src.ensembles import pool_chunk_specs

    out = tmp_path / "synthetic_ensembles" / "pool_2yr_n16"
    cfg = ForcingEnsembleConfig(
        root_seed=_SEED, n_forcing_profiles=8, realizations_per_profile=2,
        realization_years=_L, output_dir=out,
        mean_frac_csv=config.ENSEMBLE_FORCING_MEAN_FRAC_CSV,
        store_daily=True, compute_hazard_image=True, hazard_block_size=4, chunk_size=8,
    )
    if not Path(cfg.mean_frac_csv).exists():
        pytest.skip("CMIP6 forcing table not available")
    generate_forcing_ensemble(cfg)

    # Pool dir holds only global artifacts (no daily HDF5); daily lives in sibling chunk dirs.
    assert not (out / "gage_flow_mgd.hdf5").exists()
    si = json.loads((out / "chunk_index.json").read_text())
    assert si["n_chunks"] == 2
    assert [(s["global_start"], s["global_end"]) for s in si["chunks"]] == [(0, 8), (8, 16)]

    # Point staging resolution at this temp dir so chunk slugs resolve via _spec_from_staged_dir.
    import config as _cfg
    _orig = _cfg.STAGED_ENSEMBLE_DIR
    _cfg.STAGED_ENSEMBLE_DIR = tmp_path / "synthetic_ensembles"
    try:
        chunks = pool_chunk_specs("pool_2yr_n16")
        assert len(chunks) == 2
        assert chunks[0][1] == list(range(0, 8)) and chunks[1][1] == list(range(8, 16))
        assert all(spec.n_realizations == 8 for spec, _ in chunks)

        # A realization in chunk 1 (global 11 -> local 3) regenerates to its stored chunk slice.
        from synhydro import Ensemble
        sh1 = out.parent / "pool_2yr_n16__chunk001"
        staged = Ensemble.from_hdf5(str(sh1 / "gage_flow_mgd.hdf5")).data_by_realization
        regen = regenerate_realization(_SEED, 11, config=cfg)
        cols = [c for c in staged[3].columns if c in regen.columns]
        np.testing.assert_array_equal(
            regen[cols].to_numpy(np.float32), staged[3][cols].to_numpy(np.float32))
    finally:
        _cfg.STAGED_ENSEMBLE_DIR = _orig


@slow
def test_hazard_image_and_forcing_profiles_shapes(tmp_path):
    from src.ensemble_generation import generate_forcing_ensemble
    from scengen.diagnostics import load_hazard_image

    cfg = _tiny_config(tmp_path / "pool")
    generate_forcing_ensemble(cfg)
    out = tmp_path / "pool"
    n_m = _N_FORCING * _R

    haz = load_hazard_image(out / "hazard_image.npz")
    assert haz["H"].shape[0] == n_m
    assert haz["H"].shape[1] == len(haz["hazard_axes"])
    assert list(haz["realization_ids"]) == list(range(n_m))

    prof = np.load(out / "forcing_profiles.npz")
    assert prof["mean_factor_a"].shape == (n_m, 12)  # one theta row per realization
    assert list(prof["realization_ids"]) == list(range(n_m))
