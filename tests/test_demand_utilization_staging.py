"""
tests/test_demand_utilization_staging.py - End-to-end staging of the utilization axis.

On the LOCAL E_test fixture ``etest_kn_10yr_n12`` (6 SOWs x R = 2 x 10 yr, three
chunks of four realizations, already staged under outputs/synthetic_ensembles/):
append the NYC utilization column, rebuild the prefix subset
``etest_kn_10yr_n12_first2ch``, stage chunk 000's demand series with the per-SOW
scales, and assert that

  1. the realizations of the higher-u SOW carry the higher mean demand, no
     calendar month exceeds the 800 MGD cap, and the HDF5 records each
     realization's scale;
  2. the realized-utilization table of the chunk and its parent-level merge
     are consistent with the nominal table;
  3. a trimmed-model simulation of the chunk reads exactly the staged series
     (``ibt_demands.demand_nyc``), so the factor reaches the objectives.

Slow (the Pywr-DRB extrapolation plus a four-realization simulation); it
mutates the local fixture, which is gitignored output. Skipped when the
fixture or the calibration table is absent.

Run:
    NYCOPT_SALINITY_ON=0 NYCOPT_TEMPERATURE_ON=0 \
        venv/Scripts/python.exe -m pytest tests/test_demand_utilization_staging.py -v -m slow
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("NYCOPT_SALINITY_ON", "0")
os.environ.setdefault("NYCOPT_TEMPERATURE_ON", "0")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src import demand_utilization as du  # noqa: E402

POOL = "etest_kn_10yr_n12"
SUBSET = f"{POOL}_first2ch"
CHUNK = f"{POOL}__chunk000"
FIXTURE_DIR = config.STAGED_ENSEMBLE_DIR / POOL
CHUNK_DIR = config.STAGED_ENSEMBLE_DIR / CHUNK

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (FIXTURE_DIR / "forcing_profiles.npz").exists()
        or not (CHUNK_DIR / "catchment_inflow_mgd.hdf5").exists(),
        reason="local E_test fixture etest_kn_10yr_n12 is not staged",
    ),
    pytest.mark.skipif(not du.CALIBRATION_CSV.exists(),
                       reason="calibration table absent (run the driver)"),
]


@pytest.fixture(scope="module")
def staged():
    """Append, rebuild the subset, stage chunk 000, merge; return the tables."""
    from scripts.supplemental import make_etest_subset
    from scripts.supplemental.append_demand_utilization import append_demand_utilization
    from scripts.supplemental.merge_demand_utilization_realized import merge_realized
    from src.ensemble_prep import stage_nyc_demand
    from src.ensembles import get_ensemble_spec, staged_ensemble_dir

    nominal = append_demand_utilization(POOL, force=True)
    # Stage every chunk's scaled series first (fast: four 10-yr realizations
    # each): the subset tool refuses a chunk whose demand HDF5 is missing, and
    # the merge needs every chunk's realized table.
    spec = get_ensemble_spec(CHUNK)
    h5 = stage_nyc_demand(spec, force=True)
    for other in (f"{POOL}__chunk001", f"{POOL}__chunk002"):
        stage_nyc_demand(get_ensemble_spec(other), force=True)
    realized = pd.read_csv(staged_ensemble_dir(CHUNK) / "demand_utilization_realized.csv")
    argv = sys.argv
    sys.argv = ["make_etest_subset", "--pool", POOL, "--n-chunks", "2",
                "--out-slug", SUBSET, "--force"]
    try:
        make_etest_subset.main()
    finally:
        sys.argv = argv
    per_real, per_sow = merge_realized(POOL)
    return {"spec": spec, "h5": Path(h5), "nominal": nominal, "realized": realized,
            "per_real": per_real, "per_sow": per_sow}


def _series(h5: Path, local_id: int) -> pd.Series:
    with h5py.File(h5, "r") as f:
        g = f[str(local_id)]
        idx = pd.to_datetime(g["datetime"][:].astype(str))
        return pd.Series(g["aggregate"][:], index=idx, dtype=float)


def test_append_is_idempotent_and_the_subset_inherits_the_prefix(staged):
    from scripts.supplemental.append_demand_utilization import append_demand_utilization

    again = append_demand_utilization(POOL, force=True)
    pd.testing.assert_frame_equal(again, staged["nominal"])
    with np.load(config.STAGED_ENSEMBLE_DIR / SUBSET / "forcing_profiles.npz",
                 allow_pickle=True) as zs, \
            np.load(FIXTURE_DIR / "forcing_profiles.npz", allow_pickle=True) as zp:
        rpp = int(zp["realizations_per_profile"])
        np.testing.assert_array_equal(zs["demand_utilization"],
                                      zp["demand_utilization"][: 4 * rpp])
    with pytest.raises(FileExistsError):
        append_demand_utilization(POOL, force=False)


def test_higher_u_sow_has_higher_demand_and_no_month_exceeds_the_cap(staged):
    realized = staged["realized"].set_index("global_id")
    assert sorted(realized.index) == [0, 1, 2, 3]
    assert realized["sow"].tolist() == [0, 0, 1, 1]
    means = {}
    with h5py.File(staged["h5"], "r") as f:
        for local in range(4):
            g = f[str(local)]
            assert float(g["demand_scale"][()]) == pytest.approx(realized.at[local, "k"])
            assert "demand_scale" not in list(g.attrs["column_labels"])
    for local in range(4):
        s = _series(staged["h5"], local)
        monthly = s.groupby([s.index.year, s.index.month]).mean()
        assert monthly.max() <= du.NYC_ALLOCATION_MGD + 1e-6
        means[local] = float(s.mean())
    u = realized["u_nominal"]
    lo_sow, hi_sow = (0, 1) if u.loc[0] < u.loc[2] else (1, 0)
    lo_mean = np.mean([means[i] for i in realized.index[realized["sow"] == lo_sow]])
    hi_mean = np.mean([means[i] for i in realized.index[realized["sow"] == hi_sow]])
    assert hi_mean > lo_mean, "the higher-u SOW must draw more"
    # Realized sits near nominal: a few percentage points either way is the design.
    assert np.all(np.abs(realized["u_realized"] - realized["u_nominal"]) < 0.12)


def test_merge_reports_nominal_versus_realized_per_sow(staged):
    per_sow = staged["per_sow"]
    assert len(per_sow) == 6 and per_sow["n_realizations"].tolist() == [2] * 6
    assert np.allclose(per_sow["u_nominal"], staged["nominal"]["u"])
    assert (per_sow["u_realized_min"] <= per_sow["u_realized_mean"]).all()
    assert (per_sow["u_realized_mean"] <= per_sow["u_realized_max"]).all()
    assert (FIXTURE_DIR / "demand_utilization_realized_by_sow.csv").exists()


def test_simulation_reads_the_staged_scaled_demand(staged):
    """The trimmed model's demand_nyc is the staged series, realization by realization."""
    from src.formulations import get_baseline_values
    from src.simulation import dvs_to_config, run_simulation_ensemble_inmemory

    nyc_config = dvs_to_config(np.array(get_baseline_values("ffmp")), formulation_name="ffmp")
    data = run_simulation_ensemble_inmemory(nyc_config, staged["spec"])
    assert len(data) == 4
    realized = staged["realized"].set_index("global_id")
    sim_means = {}
    for local, d in enumerate(data):
        demand = d["ibt_demands"]["demand_nyc"]
        staged_series = _series(staged["h5"], local).reindex(demand.index)
        np.testing.assert_allclose(demand.to_numpy(dtype=float),
                                   staged_series.to_numpy(dtype=float), rtol=1e-9)
        sim_means[local] = float(demand.mean())
    by_sow = pd.Series(sim_means).groupby(realized["sow"].reindex(range(4)).to_numpy()).mean()
    u_by_sow = realized.groupby("sow")["u_nominal"].first()
    assert by_sow.idxmax() == u_by_sow.idxmax(), "simulated demand orders by nominal u"
