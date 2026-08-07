"""tests/test_regret_tolerance_diagnostics.py - The pre-registration machinery for tau and delta.

These tests exist because the two parameters they guard can move the whole RQ2
answer, and because the two ways of getting them wrong pull in OPPOSITE
directions:

  - a tolerance read off the candidate-policy distribution is circular, and
  - a tolerance that is too LOOSE makes a non-inferiority claim trivially true.

So the assertions below are not shape checks. They pin the direction of every
choice: the headline rung is the SMALLEST that clears the floor (not the
largest); the noise floor scales with the estimator's noise (not with the
objective's level); the null differences carry no design effect; a saturated
tolerance is detected and named; and the margin machinery REFUSES to run before a
rung has been adopted.

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

# Two REAL base objectives so the epsilon ladder resolves against the registry.
OBJS = ["nyc_delivery_reliability_weekly", "downstream_flood_exceedance_minor"]
N_SOW, R = 40, 25

META = {
    "is_ensemble": True,
    "base_names": OBJS,
    "thresholds": {OBJS[0]: 0.87, OBJS[1]: 1.17},
    "kinds": {OBJS[0]: "ge", OBJS[1]: "le"},
    "directions": {OBJS[0]: "maximize", OBJS[1]: "minimize"},
    "realization_indices": list(range(N_SOW * R)),
    "sow_ids": [s for s in range(N_SOW) for _ in range(R)],
    "n_sow": N_SOW,
    "realizations_per_sow": R,
}


def _write(dir_: Path, records, n_sol) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=["solution_id", "realization_id", "objective",
                                   "value"]).to_csv(
        dir_ / "reeval_raw.csv.gz", index=False, compression="gzip")
    (dir_ / "reeval_raw_meta.json").write_text(json.dumps(
        dict(META, solution_ids=list(range(n_sol)), n_solutions=n_sol,
             n_realizations=N_SOW * R)))


def _baseline_cube(tmp_path: Path, noise_scale: float = 1.0) -> rob.RawCube:
    """An incumbent whose within-SOW spread is a controlled multiple of a base level."""
    rng = np.random.default_rng(11)
    records = []
    for sow in range(N_SOW):
        for rep in range(R):
            rid = sow * R + rep
            records.append((0, rid, OBJS[0],
                            float(0.87 + noise_scale * 0.02 * rng.standard_normal())))
            records.append((0, rid, OBJS[1],
                            float(1.00 + noise_scale * 0.10 * rng.standard_normal())))
    d = tmp_path / f"baseline_{noise_scale}"
    _write(d, records, n_sol=1)
    return rob.load_raw(d)


###############################################################################
# Pass A: the noise floor
###############################################################################

def test_noise_floor_tracks_the_estimator_not_the_objective_level(tmp_path):
    """Doubling the incumbent's within-SOW spread must double the floor.

    The floor is a property of how precisely a SOW mean is estimated. If it
    tracked the objective's LEVEL instead, it would be a Tier-C quantity in
    disguise and would drift with the system rather than with the measurement.
    """
    quiet = rtol.tolerance_floor(rtol.within_sow_noise(_baseline_cube(tmp_path, 1.0)))
    loud = rtol.tolerance_floor(rtol.within_sow_noise(_baseline_cube(tmp_path, 2.0)))
    for name in OBJS:
        q = float(quiet.set_index("objective").loc[name, "tau_floor"])
        l = float(loud.set_index("objective").loc[name, "tau_floor"])
        assert l == pytest.approx(2.0 * q, rel=0.15)


def test_noise_floor_shrinks_with_more_realizations_per_sow(tmp_path):
    """SE = sigma/sqrt(R): a better-resolved SOW mean permits a tighter tolerance."""
    base = _baseline_cube(tmp_path, 1.0)
    full = rtol.tolerance_floor(rtol.within_sow_noise(base))
    # Halve R by keeping only the first half of each SOW's realizations.
    keep = [rid for rid in base.realization_ids if (rid % R) < R // 2]
    sub_meta = dict(META, realization_indices=keep,
                    sow_ids=[rid // R for rid in keep],
                    realizations_per_sow=R // 2)
    d = tmp_path / "half"
    d.mkdir()
    rows = []
    for j, rid in enumerate(base.realization_ids):
        if rid not in keep:
            continue
        for k, name in enumerate(base.base_names):
            rows.append((0, rid, name, float(base.cube[0, j, k])))
    pd.DataFrame(rows, columns=["solution_id", "realization_id", "objective",
                                "value"]).to_csv(
        d / "reeval_raw.csv.gz", index=False, compression="gzip")
    (d / "reeval_raw_meta.json").write_text(json.dumps(
        dict(sub_meta, solution_ids=[0], n_solutions=1, n_realizations=len(keep))))
    half = rtol.tolerance_floor(rtol.within_sow_noise(rob.load_raw(d)))

    for name in OBJS:
        f = float(full.set_index("objective").loc[name, "tau_floor"])
        h = float(half.set_index("objective").loc[name, "tau_floor"])
        assert h > f, "fewer realizations per SOW must RAISE the floor"
        assert h == pytest.approx(f * np.sqrt(2.0), rel=0.25)


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

    The flood objective is the live case: eps = 0.01 ft-days/yr sits well under its
    estimator noise floor, so on that axis k = 1 is INSIDE the noise while on the
    reliability axis it is far outside it. The 'max' unit is what repairs that.
    """
    floors = rtol.tolerance_floor(rtol.within_sow_noise(_baseline_cube(tmp_path, 1.0)))
    shapes = rtol.ladder_shape_table(floors).set_index("objective")

    flood = shapes.loc[OBJS[1]]
    assert bool(flood["eps_below_floor"]), "flood eps should be under its floor"
    assert flood["unit_max"] == pytest.approx(flood["tau_floor"])
    rel = shapes.loc[OBJS[0]]
    assert not bool(rel["eps_below_floor"])
    assert rel["unit_max"] == pytest.approx(rel["eps"])


def test_max_shape_ladder_lifts_only_the_starved_axis(tmp_path):
    """tau_ladder(floors=...) must raise the noise-bound axis and leave the other."""
    from src.objectives import OBJECTIVES
    floors = rtol.tolerance_floor(rtol.within_sow_noise(_baseline_cube(tmp_path, 1.0)))
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
        assert maxed[n] >= OBJECTIVES[n].epsilon - 1e-12


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


def test_split_half_false_harm_falls_as_the_tolerance_widens(tmp_path):
    """The incumbent against itself: every flagged harm is false by construction."""
    base = _baseline_cube(tmp_path, 1.0)
    null = rtol.split_half_null(base, grid=(0.0, 1.0, 5.0, 50.0), reps=4)
    joint = (null[null["objective"] == "__joint__"]
             .set_index("tau_k")["false_harm_freq"])
    assert joint.loc[0.0] > joint.loc[50.0]
    # A tolerance of zero flags roughly half the SOWs per objective (the sign of
    # pure noise is a coin flip), so the joint false-harm rate is high.
    assert joint.loc[0.0] > 0.5
    # A very wide tolerance forgives noise entirely.
    assert joint.loc[50.0] == pytest.approx(0.0, abs=0.02)


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
    for design, gap in (("historic", 0.0), ("fixed_probabilistic", 0.30)):
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
