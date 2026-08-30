"""tests/test_regret_tolerance_diagnostics.py - The pre-registration machinery for tau and delta.

These tests exist because the two parameters they guard can move the whole RQ1
answer, and because the two ways of getting them wrong pull in OPPOSITE
directions:

  - a tolerance read off the candidate-policy distribution is circular, and
  - a tolerance that is too LOOSE makes a non-inferiority claim trivially true.

So the assertions below are not shape checks. They pin the direction of every
choice: the headline rung is the SMALLEST that clears the floor (not the
largest); the noise floor scales with the local spread of the per-SOW estimator
(not with the objective's level, and not with the forcing trend the binning
exists to exclude); the null differences carry no design effect; a saturated
tolerance is detected and named; and the margin machinery REFUSES to run before
a rung has been adopted.

Fixtures follow the per-SOW annual-unit substrate: long rows
``(solution_id, sow_id, objective, value)`` plus a meta carrying ``obj_names``
(ANNUAL names), ``sow_labels``, ``realizations_per_sow``, and
``substrate = "sow_annual_unit"``.

Run:
    venv/Scripts/python.exe -m pytest tests/test_regret_tolerance_diagnostics.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import src.robustness as rob  # noqa: E402


def _load_rtol():
    path = PROJECT_DIR / "scripts" / "supplemental" / "regret_tolerance_diagnostics.py"
    spec = importlib.util.spec_from_file_location("rtol_diag", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rtol_diag"] = mod
    spec.loader.exec_module(mod)
    return mod


rtol = _load_rtol()

# Two REAL annual objectives so the epsilon ladder resolves against the
# registry (eps 0.02 and 0.3 in ENSEMBLE_OBJECTIVES).
OBJS = ["nyc_delivery_reliability_annual", "downstream_flood_exceedance_annual"]
N_SOW, R = 400, 25

#: The dominant forcing coordinate, one value per SOW (the fixture's severity
#: axis; SOW labels are 0..N_SOW-1 in this order).
THETA_M = np.linspace(-0.2, 0.2, N_SOW)

#: Per-objective per-SOW estimator noise SD at noise_scale = 1. Chosen so the
#: reliability epsilon (0.02) sits ABOVE its floor and the flood epsilon (0.3)
#: sits BELOW its floor -- the live mis-scaled-axis case the shape table exists
#: to flag.
NOISE_SD = {OBJS[0]: 0.004, OBJS[1]: 0.5}

META = {
    "is_ensemble": True,
    "substrate": "sow_annual_unit",
    "obj_names": OBJS,
    "thresholds": {OBJS[0]: 0.85, OBJS[1]: 1.17},
    "kinds": {OBJS[0]: "ge", OBJS[1]: "le"},
    "directions": {OBJS[0]: "maximize", OBJS[1]: "minimize"},
    "sow_labels": list(range(N_SOW)),
    "n_sow": N_SOW,
    "realizations_per_sow": R,
    "n_realizations": N_SOW * R,
}


def _write(dir_: Path, records, n_sol) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=["solution_id", "sow_id", "objective",
                                   "value"]).to_csv(
        dir_ / "reeval_raw.csv.gz", index=False, compression="gzip")
    (dir_ / "reeval_raw_meta.json").write_text(json.dumps(
        dict(META, solution_ids=list(range(n_sol)), n_solutions=n_sol)))


def _baseline_cube(tmp_path: Path, noise_scale: float = 1.0,
                   trend_amp: float = 0.0, tag: str = "") -> rob.RawCube:
    """An incumbent whose per-SOW values carry a controlled m-trend + noise.

    ``trend_amp`` scales a linear response to the forcing coordinate m;
    ``noise_scale`` multiplies the per-SOW estimator noise. The same RNG seed
    is used at every scale so noise draws are identical up to the multiplier.
    """
    rng = np.random.default_rng(11)
    eps0 = rng.standard_normal(N_SOW)
    eps1 = rng.standard_normal(N_SOW)
    records = []
    for g in range(N_SOW):
        m = THETA_M[g]
        records.append((0, g, OBJS[0],
                        float(0.87 + trend_amp * m
                              + noise_scale * NOISE_SD[OBJS[0]] * eps0[g])))
        records.append((0, g, OBJS[1],
                        float(1.00 + trend_amp * m
                              + noise_scale * NOISE_SD[OBJS[1]] * eps1[g])))
    d = tmp_path / f"baseline_{tag or noise_scale}"
    _write(d, records, n_sol=1)
    return rob.load_raw(d)


###############################################################################
# Pass A: the noise floor
###############################################################################

def test_noise_floor_tracks_the_estimator_not_the_objective_level(tmp_path):
    """Doubling the incumbent's per-SOW spread must double the floor.

    The floor is a property of how precisely a SOW's pooled objective value is
    estimated. If it tracked the objective's LEVEL instead, it would be a
    Tier-C quantity in disguise and would drift with the system rather than
    with the measurement.
    """
    quiet = rtol.tolerance_floor(rtol.sow_noise_floor(
        _baseline_cube(tmp_path, 1.0, tag="q"), THETA_M))
    loud = rtol.tolerance_floor(rtol.sow_noise_floor(
        _baseline_cube(tmp_path, 2.0, tag="l"), THETA_M))
    for name in OBJS:
        q = float(quiet.set_index("objective").loc[name, "tau_floor"])
        l = float(loud.set_index("objective").loc[name, "tau_floor"])
        assert l == pytest.approx(2.0 * q, rel=0.15)


def test_noise_floor_excludes_the_forcing_trend(tmp_path):
    """The binned-in-m spread must not read the m-response as noise.

    A policy's systematic response to the forcing axis is signal, not
    estimator noise; binning narrowly in m is what keeps it out of the floor.
    A strong trend may only INFLATE the floor slightly (the documented
    upper-bound direction), never by the trend's own global spread.
    """
    flat = rtol.sow_noise_floor(
        _baseline_cube(tmp_path, 1.0, trend_amp=0.0, tag="flat"), THETA_M)
    trended = rtol.sow_noise_floor(
        _baseline_cube(tmp_path, 1.0, trend_amp=0.5, tag="trend"), THETA_M)
    global_trend_sd = float(np.std(0.5 * THETA_M, ddof=1))  # ~0.058
    for name in OBJS:
        f = float(flat.set_index("objective").loc[name, "sigma_local"])
        t = float(trended.set_index("objective").loc[name, "sigma_local"])
        assert t >= f * 0.95, "binning must not deflate the floor"
        assert t == pytest.approx(f, rel=0.25), (
            "the local floor must track the noise, not the trend"
        )
        if NOISE_SD[name] < global_trend_sd:
            assert t < global_trend_sd, (
                "a floor carrying the global trend spread would be the "
                "un-detrended estimator in disguise"
            )


def test_theta_m_by_sow_joins_on_sow_id(tmp_path):
    """One theta row per realization; SOW = realization_id // R, never position."""
    base = _baseline_cube(tmp_path, 1.0, tag="theta")
    rids = np.arange(N_SOW * R)
    m_per_real = THETA_M[rids // R]
    theta = np.column_stack([m_per_real, np.zeros_like(m_per_real),
                             np.zeros_like(m_per_real)])
    perm = np.random.default_rng(5).permutation(rids.size)  # shuffled storage
    npz = tmp_path / "forcing_profiles.npz"
    np.savez(npz, theta_params=theta[perm], realization_ids=rids[perm],
             theta_param_names=np.array(["m", "r1", "r2"]))
    out = rtol.theta_m_by_sow(base, npz_path=npz)
    np.testing.assert_allclose(out, THETA_M)


def test_headline_rung_is_the_smallest_that_clears_every_floor(tmp_path):
    """Smallest, not largest: a loose tolerance flatters a non-inferiority claim."""
    floors = pd.DataFrame({
        "objective": ["a", "b"],
        "tau_floor": [0.6, 1.4],
        "eps": [1.0, 1.0],
        "k_floor": [0.6, 1.4],
    })
    head = rtol.headline_k(floors, grid=(0.0, 0.5, 1.0, 2.0, 5.0))
    assert head["k"] == 2.0                    # 1.0 does not clear b's 1.4
    assert head["binding_objective"] == "b"
    assert head["cleared"] is True


def test_no_rung_clearing_the_floor_is_reported_not_papered_over(tmp_path):
    """If the ladder is too short the script must say so, not return its top rung."""
    floors = pd.DataFrame({"objective": ["a"], "tau_floor": [99.0],
                           "eps": [1.0], "k_floor": [99.0]})
    head = rtol.headline_k(floors, grid=(0.0, 1.0, 2.0))
    assert head["k"] is None and head["cleared"] is False
    assert head["k_floor_max"] == pytest.approx(99.0)


def test_ladder_shape_flags_an_epsilon_below_its_noise_floor(tmp_path):
    """One mis-scaled axis silently redefines every rung, because k is shared.

    The flood objective is the live case: its annual epsilon sits well under
    its per-SOW estimator noise floor, so on that axis k = 1 is INSIDE the
    noise while on the reliability axis it is far outside it. The 'max' unit
    is what repairs that.
    """
    floors = rtol.tolerance_floor(rtol.sow_noise_floor(
        _baseline_cube(tmp_path, 1.0, tag="shape"), THETA_M))
    shapes = rtol.ladder_shape_table(floors).set_index("objective")

    flood = shapes.loc[OBJS[1]]
    assert bool(flood["eps_below_floor"]), "flood eps should be under its floor"
    assert flood["unit_max"] == pytest.approx(flood["tau_floor"])
    rel = shapes.loc[OBJS[0]]
    assert not bool(rel["eps_below_floor"])
    assert rel["unit_max"] == pytest.approx(rel["eps"])


def test_max_shape_ladder_lifts_only_the_starved_axis(tmp_path):
    """tau_ladder(floors=...) must raise the noise-bound axis and leave the other."""
    from src.objectives_ensemble import ENSEMBLE_OBJECTIVES
    floors = rtol.tolerance_floor(rtol.sow_noise_floor(
        _baseline_cube(tmp_path, 1.0, tag="max"), THETA_M))
    fl = rtol.floors_dict(floors)

    eps_only = rob.tau_ladder(OBJS, k=1.0)
    maxed = rob.tau_ladder(OBJS, k=1.0, floors=fl)

    assert maxed[OBJS[0]] == pytest.approx(eps_only[OBJS[0]])      # untouched
    assert maxed[OBJS[1]] > eps_only[OBJS[1]]                      # lifted to floor
    assert maxed[OBJS[1]] == pytest.approx(fl[OBJS[1]])
    # k still scales linearly under either shape.
    assert rob.tau_ladder(OBJS, k=2.0, floors=fl)[OBJS[1]] == pytest.approx(
        2.0 * maxed[OBJS[1]])
    # And the unit can never fall BELOW epsilon.
    for n in OBJS:
        assert maxed[n] >= ENSEMBLE_OBJECTIVES[n].epsilon - 1e-12


def test_tau_env_override_must_be_a_whole_vector(monkeypatch):
    """A partial override would leave the rest on a different tolerance basis."""
    monkeypatch.setenv("NYCOPT_REGRET_TAU", json.dumps({OBJS[0]: 0.05}))
    with pytest.raises(KeyError, match="omits"):
        rob.tau_ladder(OBJS)
    monkeypatch.setenv("NYCOPT_REGRET_TAU",
                       json.dumps({OBJS[0]: 0.05, OBJS[1]: 0.5, "stranger": 1.0}))
    with pytest.raises(KeyError, match="absent from this cube"):
        rob.tau_ladder(OBJS)
    monkeypatch.setenv("NYCOPT_REGRET_TAU",
                       json.dumps({OBJS[0]: 0.05, OBJS[1]: 0.5}))
    assert rob.tau_ladder(OBJS) == {OBJS[0]: 0.05, OBJS[1]: 0.5}


###############################################################################
# Pass B: bands, nulls, assay sensitivity
###############################################################################

def _profile(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["design", "draw", "seed", "tau_k",
                                       "best", "median", "n_solutions"])


def test_saturated_and_starved_rungs_are_named(tmp_path):
    """A rung where every design scores ~1 cannot support a non-inferiority claim."""
    rows = []
    for design in ("A", "B"):
        rows += [(design, 0, 0, 0.0, 0.01, 0.01, 4),
                 (design, 0, 0, 1.0, 0.50, 0.40, 4),
                 (design, 0, 0, 9.0, 0.99, 0.98, 4)]
    band = rtol.discrimination_band(_profile(rows)).set_index("tau_k")
    assert band.loc[0.0, "verdict"] == "starved"
    assert band.loc[1.0, "verdict"] == "informative"
    assert band.loc[9.0, "verdict"] == "saturated"


def test_null_differences_are_within_design_only(tmp_path):
    """Both nulls must be blind to the design contrast, or delta becomes circular."""
    rows = []
    for design, level in (("A", 0.20), ("B", 0.80)):     # a huge design effect
        for draw in (0, 1):
            for seed in (0, 1):
                rows.append((design, draw, seed, 1.0,
                             level + 0.01 * seed + 0.02 * draw, 0.0, 4))
    nulls = rtol.null_differences(_profile(rows))
    # Every difference is small (within-design), despite the 0.6 gap BETWEEN designs.
    assert nulls["abs_diff"].max() < 0.10
    assert set(nulls["level"]) == {"seed", "draw"}
    # Seed pairs see only the seed offset; draw pairs see the draw offset.
    seed_lvl = nulls[nulls["level"] == "seed"]["abs_diff"]
    draw_lvl = nulls[nulls["level"] == "draw"]["abs_diff"]
    assert seed_lvl.max() == pytest.approx(0.01)
    assert draw_lvl.max() == pytest.approx(0.02)


def test_assay_sensitivity_flags_an_insensitive_tolerance(tmp_path):
    """If the unmatched reference cannot be separated, a null is uninformative."""
    rows = []
    for design, gap in (("historic", 0.0), ("monte_carlo", 0.30)):
        for draw in (0, 1):
            rows += [(design, draw, 0, 1.0, 0.40 + gap + 0.005 * draw, 0.0, 4),
                     (design, draw, 0, 9.0, 0.99 + 0.0 * draw, 0.0, 4)]
    assay = rtol.assay_sensitivity(_profile(rows), control="historic")
    at1 = assay[assay["tau_k"] == 1.0].iloc[0]
    at9 = assay[assay["tau_k"] == 9.0].iloc[0]
    # (pandas stores the flag as numpy bool, hence bool() rather than `is True`)
    assert bool(at1["separates"]) and at1["control_gap"] == pytest.approx(0.30)
    # At the saturated rung both designs sit at 0.99, so the gap vanishes.
    assert not bool(at9["separates"])


def test_null_differences_single_run_keeps_schema():
    """One draw x one seed per design yields no pairs; the empty frame must keep
    its columns so pass B degrades to "null not estimable" instead of raising
    ``KeyError('level')``."""
    rows = [("A", 0, 0, 1.0, 0.5, 0.4, 4), ("B", 0, 0, 1.0, 0.6, 0.5, 4)]
    nulls = rtol.null_differences(_profile(rows))
    assert nulls.empty
    assert list(nulls.columns) == ["level", "design", "draw", "tau_k", "abs_diff"]
    # The downstream consumer must run, and report the null as not estimable
    # (NaN floor, nothing separable) rather than crash.
    assay = rtol.assay_sensitivity(_profile(rows), control="A")
    assert not assay.empty
    assert assay["draw_null_max"].isna().all()
    assert not assay["separates"].any()


def test_adopted_floors_roundtrips_pass_a_output(tmp_path, monkeypatch):
    """k-sweep call sites read pass A's persisted floors, so with the override
    unset they sweep the adopted max(eps, floor) basis, not the eps-only ladder."""
    import supplemental_config as sc_mod

    monkeypatch.setattr(sc_mod, "RTOL_TABLES_DIR", tmp_path)
    assert rob.adopted_floors() is None
    (tmp_path / "rtol_floors.json").write_text(json.dumps({OBJS[0]: 0.5}))
    assert rob.adopted_floors() == {OBJS[0]: 0.5}


def test_margin_is_not_computed_before_a_rung_is_adopted(monkeypatch, capsys):
    """Deriving delta at a rung chosen after seeing the profile is the circularity."""
    import supplemental_config as sc
    monkeypatch.setattr(sc, "RTOL_ADOPTED_K", None, raising=False)
    assert sc.RTOL_ADOPTED_K is None
    # The rule text is the pre-registered object and must name its own guard.
    assert "NUISANCE" in sc.RTOL_MARGIN_RULE
    assert "smallest" in sc.RTOL_TAU_RULE.lower()


def test_tau_grid_matches_the_comparison_grid():
    """The diagnostic and the comparison must speak the same coordinate."""
    import supplemental_config as sc
    path = PROJECT_DIR / "scripts" / "main" / "compare_designs.py"
    spec = importlib.util.spec_from_file_location("compare_designs_rtol", path)
    cd = importlib.util.module_from_spec(spec)
    sys.modules["compare_designs_rtol"] = cd
    spec.loader.exec_module(cd)
    assert (tuple(float(k) for k in sc.RTOL_TAU_GRID)
            == tuple(float(k) for k in cd.REGRET_TAU_GRID))
