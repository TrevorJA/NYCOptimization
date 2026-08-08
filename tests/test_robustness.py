"""
tests/test_robustness.py - Unit tests for src.robustness offline scoring.

Uses an INDEPENDENT synthetic per-SOW matrix (hand-computed expected values,
not routed through the objective registry or the simulation), so the tests are
not tautological with the code under test. The substrate is the unified metric
currency: long rows ``(solution_id, sow_id, objective, value)`` of per-SOW
annual-unit objective values. Covers:

  1. load_raw densifies the long matrix into an (S, G, M) cube on the union of
     SOW labels, NaN-filling gaps, and REFUSES pre-substrate metas.
  2. Per-SOW satisficing (univariate + multivariate Starr domain criterion)
     against hand values and an inline reference, incl. NaN-as-unsatisfied.
  3. A missing threshold/kind is a HARD ERROR, never a silently-false column.
  4. Laplace (mean SOW) and maximin (worst SOW) anchors, natural units,
     respecting each objective's direction.
  5. The incumbent-relative regret family: signed and oriented (positive =
     better than current operations, both directions), design-independent, and
     joined on the SOW LABEL rather than position (a baseline missing a SOW
     contributes NaN, never a mis-pairing).
  6. gain_mean__ / regret_mean__ are the two one-sided halves of the signed
     incumbent advantage; natural units keep the SOWs a ratio form would drop.
  7. The tau ladder scales the ANNUAL-unit epsilons of ENSEMBLE_OBJECTIVES,
     honors noise floors, refuses unknown names, and the NYCOPT_REGRET_TAU env
     override is whole-vector-or-error.
  8. The attainability screen flags SOWs no policy can win, keyed by sow_id.
  9. The G == 1 gate NaNs the WHOLE scorecard (single-trace / one-SOW re-eval),
     and an all-NaN solution row is scored NaN, not 0.
 10. Retired metrics stay retired (realization-unit satisficing, the
     within-SOW aggregator knob, the normalized baseline improvement,
     regret-from-best, the overfitting gap).

Run:
    venv/Scripts/python.exe -m pytest tests/test_robustness.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import src.robustness as rob


def _satisficing_ref(values, threshold, kind):
    """Independent per-SOW satisficing reference (no project imports)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    finite = np.isfinite(arr)
    sat = (finite & (arr >= threshold)) if kind == "ge" \
        else (finite & (arr <= threshold))
    return float(sat.sum()) / float(arr.size)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------
# 3 solutions x 2 SOWs x 2 objectives; cells are per-SOW annual-unit values.
#   A: maximize, threshold 0.9, kind "ge"
#   B: minimize, threshold 10,  kind "le"
# Values (natural units); solution 2 / SOW 0 / A is missing (NaN).
_RECORDS = [
    # sid, sow_id, obj, value
    (0, 0, "A", 0.95), (0, 0, "B", 5.0),
    (0, 1, "A", 0.85), (0, 1, "B", 8.0),
    (1, 0, "A", 0.92), (1, 0, "B", 12.0),
    (1, 1, "A", 0.99), (1, 1, "B", 9.0),
    # (2, 0, "A", ...) intentionally absent -> NaN
    (2, 0, "B", 4.0),
    (2, 1, "A", 0.80), (2, 1, "B", 11.0),
]

_META = {
    "is_ensemble": True,
    "obj_names": ["A", "B"],
    "thresholds": {"A": 0.9, "B": 10.0},
    "kinds": {"A": "ge", "B": "le"},
    "directions": {"A": "maximize", "B": "minimize"},
    "sow_labels": [0, 1],
    "realizations_per_sow": 2,
    "substrate": "sow_annual_unit",
}


def _write_raw(tmp_path: Path, records, meta) -> Path:
    df = pd.DataFrame(records, columns=["solution_id", "sow_id",
                                        "objective", "value"])
    df.to_csv(tmp_path / "reeval_raw.csv.gz", index=False, compression="gzip")
    (tmp_path / "reeval_raw_meta.json").write_text(json.dumps(meta))
    return tmp_path


@pytest.fixture
def raw(tmp_path):
    _write_raw(tmp_path, _RECORDS, _META)
    return rob.load_raw(tmp_path)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

def test_load_raw_shape_and_nan(raw):
    assert raw.cube.shape == (3, 2, 2)
    assert raw.solution_ids == [0, 1, 2]
    assert raw.obj_names == ["A", "B"]
    assert raw.sow_labels == [0, 1]
    assert raw.n_sow == 2
    assert raw.realizations_per_sow == 2
    # Missing (sol2, SOW0, A) is NaN-filled.
    assert np.isnan(raw.cube[2, 0, 0])
    assert raw.cube[0, 0, 0] == pytest.approx(0.95)


def test_load_raw_refuses_a_pre_substrate_meta(tmp_path):
    """An old whole-trace cube (base_names, no obj_names) must be rejected.

    Scoring it under the new metric currency would silently mix two different
    metric definitions; the fix is regeneration, not tolerance.
    """
    meta = {k: v for k, v in _META.items() if k != "obj_names"}
    meta["base_names"] = ["A", "B"]
    _write_raw(tmp_path, _RECORDS, meta)
    with pytest.raises(ValueError, match="obj_names"):
        rob.load_raw(tmp_path)


def test_fully_failed_solution_survives_as_an_all_nan_slice(tmp_path):
    """A solution with NO rows must not vanish from the cube.

    meta['solution_ids'] carries every ATTEMPTED id, so an all-failed solution
    keeps its (NaN) place in the solution axis and the scorecard, instead of
    silently disappearing while still appearing in objectives_summary.csv.
    """
    meta = dict(_META, solution_ids=[0, 1, 2, 7])   # 7 attempted, no rows
    _write_raw(tmp_path, _RECORDS, meta)
    raw = rob.load_raw(tmp_path)
    assert raw.solution_ids == [0, 1, 2, 7]
    assert np.all(~np.isfinite(raw.cube[3, :, :]))

    scorecard, _ = rob.score_robustness(
        raw, metrics=("satisficing_multivariate_sow", "laplace_mean"))
    # The failed solution is NaN on EVERY metric (not satisficing = 0.0)...
    assert scorecard.loc[7].isna().all()
    # ...while the ran-but-imperfect solutions keep finite scores.
    assert np.isfinite(scorecard.loc[0, "sat_multivariate_sow"])


# ---------------------------------------------------------------------------
# Per-SOW satisficing (univariate + multivariate Starr domain criterion)
# ---------------------------------------------------------------------------

def test_satisficing_univariate_sow_handvalues(raw):
    df = rob.satisficing_univariate_sow(raw)
    a = df["sat_uni_sow__A"]
    b = df["sat_uni_sow__B"]
    assert a.loc[0] == pytest.approx(0.5)   # 0.95 T, 0.85 F
    assert a.loc[1] == pytest.approx(1.0)
    assert a.loc[2] == pytest.approx(0.0)   # NaN F, 0.80 F
    assert b.loc[0] == pytest.approx(1.0)
    assert b.loc[1] == pytest.approx(0.5)
    assert b.loc[2] == pytest.approx(0.5)


def test_satisficing_matches_inline_reference(raw):
    """Independent (no project imports) reference of the per-SOW criterion,
    including NaN-as-unsatisfied: a failed SOW can't masquerade as satisficing."""
    col = raw.cube[:, :, 0]
    df = rob.satisficing_univariate_sow(raw)
    for si, sid in enumerate(raw.solution_ids):
        assert df["sat_uni_sow__A"].loc[sid] == pytest.approx(
            _satisficing_ref(col[si, :], 0.9, "ge"))


def test_satisficing_multivariate_sow_joint_nan_fails(raw):
    s = rob.satisficing_multivariate_sow(raw)
    # sol0: SOW0 (T&T)=T, SOW1 (F&T)=F -> 0.5
    assert s.loc[0] == pytest.approx(0.5)
    # sol1: SOW0 (T&F)=F, SOW1 (T&T)=T -> 0.5
    assert s.loc[1] == pytest.approx(0.5)
    # sol2: SOW0 (NaN->F), SOW1 (F) -> 0.0
    assert s.loc[2] == pytest.approx(0.0)


def test_missing_threshold_is_a_hard_error(tmp_path):
    """A missing threshold must raise, never become an always-false column.

    The multivariate criterion is a conjunction over all objectives, so a
    silently-false column would zero the PRIMARY metric for every solution
    with no other symptom.
    """
    meta = dict(_META, thresholds={"A": 0.9, "B": None})
    _write_raw(tmp_path, _RECORDS, meta)
    raw = rob.load_raw(tmp_path)
    with pytest.raises(ValueError, match="B"):
        rob.satisficing_multivariate_sow(raw)
    with pytest.raises(ValueError, match="B"):
        rob.satisficing_univariate_sow(raw)
    with pytest.raises(ValueError, match="B"):
        rob.attainability_screen(raw)
    # An explicit override that covers every objective un-blocks it.
    s = rob.satisficing_multivariate_sow(raw, thresholds={"A": 0.9, "B": 10.0})
    assert s.loc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Risk-attitude anchors (McPhail T3): mean SOW / worst SOW
# ---------------------------------------------------------------------------

def test_laplace_mean_is_natural_units(raw):
    df = rob.laplace_mean(raw)
    # sol0 A = 0.95, 0.85 over the SOWs -> mean 0.90 (maximize, natural units).
    assert df["laplace__A"].loc[0] == pytest.approx(0.90)
    # sol0 B = 5.0, 8.0 -> mean 6.5 (minimize, natural units).
    assert df["laplace__B"].loc[0] == pytest.approx(6.5)


def test_maximin_picks_the_worst_sow_per_direction(raw):
    df = rob.maximin(raw)
    # A maximizes: the worst SOW is the SMALLEST value.
    assert df["maximin__A"].loc[0] == pytest.approx(0.85)
    # B minimizes: the worst SOW is the LARGEST value.
    assert df["maximin__B"].loc[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Retired metrics stay retired
# ---------------------------------------------------------------------------

def test_deleted_metrics_are_gone():
    """The retired metric families were removed, not renamed.

    regret_from_best is set-relative and design-coupled; the overfitting gap is
    undefined in Brodeur et al. (2020); improvement_vs_baseline was the
    |baseline|-normalized deviation whose signed information now lives in
    gain_mean__/regret_mean__; the realization-unit satisficing family and the
    within-SOW aggregator knob dissolved into the per-SOW substrate (pooling a
    SOW's unit-years through the unit operator IS the within-state collapse).
    """
    for name in (
        "regret_from_best", "overfitting_gap", "regret_from_baseline",
        "improvement_vs_baseline",
        "satisficing_multivariate", "satisficing_univariate",  # realization unit
        "collapse_within_sow", "WITHIN_SOW_AGGREGATORS",
        "aggregate_over_realizations", "realization_quantiles",
    ):
        assert not hasattr(rob, name), f"{name} should be deleted"

    import src.reeval_core as reeval_core
    assert not hasattr(reeval_core, "satisficing_from_raw")

    import src.objectives_ensemble as obj_ens
    assert not hasattr(obj_ens, "SatisficingAgg")

    assert set(rob._DEFAULT_METRICS) == {
        "satisficing_multivariate_sow", "satisficing_univariate_sow",
        "laplace_mean", "maximin", "regret_magnitudes", "regret_frequencies",
    }


# ---------------------------------------------------------------------------
# Attainability screen (per sow_id)
# ---------------------------------------------------------------------------

def test_attainability_flags_unwinnable_sows(raw):
    """Separates 'this design searched badly' from 'nobody can win this state'."""
    frame = rob.attainability_screen(raw)
    assert list(frame["sow_id"]) == list(raw.sow_labels)
    assert set(frame.columns) >= {"sow_id", "n_satisficing_solutions",
                                  "attainable", "anysat__A", "anysat__B"}
    # attainable iff at least one solution meets the joint criterion there.
    assert (frame["attainable"] == (frame["n_satisficing_solutions"] > 0)).all()
    # Hand check: SOW0 is won by sol0 only; SOW1 by sol1 only.
    assert frame.loc[frame["sow_id"] == 0, "n_satisficing_solutions"].iloc[0] == 1
    assert frame.loc[frame["sow_id"] == 1, "n_satisficing_solutions"].iloc[0] == 1


# ---------------------------------------------------------------------------
# G == 1 gating
# ---------------------------------------------------------------------------

def test_single_trace_nans_the_whole_scorecard(tmp_path):
    """EVERY metric is defined ACROSS SOWs, so all are N/A on a single trace.

    The gate must NaN the columns without COMPUTING the metrics: a single-trace
    re-eval may carry no satisficing thresholds at all, and _satisfaction_cube
    raises on a missing threshold.
    """
    meta = dict(_META, is_ensemble=False, sow_labels=[0],
                thresholds={"A": None, "B": None},
                realizations_per_sow=None)
    records = [(0, 0, "A", 0.95), (0, 0, "B", 5.0),
               (1, 0, "A", 0.85), (1, 0, "B", 12.0)]
    _write_raw(tmp_path, records, meta)
    raw = rob.load_raw(tmp_path)

    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, [(0, 0, "A", 0.80), (0, 0, "B", 10.0)], meta)
    baseline = rob.load_raw(bdir)

    scorecard, _ = rob.score_robustness(raw, baseline,
                                        metrics=rob._DEFAULT_METRICS)
    assert len(scorecard.columns) > 0
    assert scorecard.isna().all().all()


def test_one_sow_ensemble_is_gated_too(tmp_path):
    """G == 1 is the gate, not the is_ensemble flag alone: a one-SOW ensemble
    cannot support a fraction-of-SOWs criterion either."""
    meta = dict(_META, sow_labels=[0])
    records = [(0, 0, "A", 0.95), (0, 0, "B", 5.0)]
    _write_raw(tmp_path, records, meta)
    raw = rob.load_raw(tmp_path)
    scorecard, _ = rob.score_robustness(
        raw, metrics=("satisficing_multivariate_sow", "laplace_mean", "maximin"))
    assert scorecard.isna().all().all()


# ---------------------------------------------------------------------------
# Ranking stability
# ---------------------------------------------------------------------------

def test_ranking_stability_square_unit_diagonal(raw):
    scorecard, hb = rob.score_robustness(
        raw, metrics=("satisficing_univariate_sow", "laplace_mean", "maximin"))
    tau = rob.ranking_stability(scorecard, hb)
    assert tau.shape[0] == tau.shape[1] == scorecard.shape[1]
    assert np.allclose(np.diag(tau.to_numpy()), 1.0)


# ---------------------------------------------------------------------------
# Incumbent-relative regret
# ---------------------------------------------------------------------------
# Two solutions, two SOWs, two objectives; the cells are per-SOW values, so
# every expected value below is hand-computed, not read off the implementation.
#
#   incumbent (both SOWs):  A = 0.80   B = 10.0
#   sol 0:  SOW0  A = 0.90  B =  5.0   -> better on both
#           SOW1  A = 0.70  B = 13.0   -> WORSE on both
#   sol 1:  both  A = 1.00  B =  1.0   -> better on both, everywhere
#
# A maximizes and B minimizes, so D = sign * (policy - incumbent) is
#   sol0: SOW0 (+0.10, +5.0)   SOW1 (-0.10, -3.0)
#   sol1: both (+0.20, +9.0)

_REG_RECORDS = [
    (0, 0, "A", 0.90), (0, 0, "B", 5.0),
    (0, 1, "A", 0.70), (0, 1, "B", 13.0),
    (1, 0, "A", 1.00), (1, 0, "B", 1.0),
    (1, 1, "A", 1.00), (1, 1, "B", 1.0),
]

_REG_BASE_RECORDS = [
    (0, g, obj, val)
    for g in (0, 1)
    for obj, val in (("A", 0.80), ("B", 10.0))
]

_REG_TAU = {"A": 0.15, "B": 4.0}


def _regret_fixture(tmp_path, base_records=None):
    _write_raw(tmp_path, _REG_RECORDS, _META)
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, base_records or _REG_BASE_RECORDS, _META)
    return rob.load_raw(tmp_path), rob.load_raw(bdir)


def test_incumbent_advantage_is_oriented_positive_means_better(tmp_path):
    """D is signed so positive = better than current operations, both directions."""
    raw, base = _regret_fixture(tmp_path)
    D = rob.incumbent_advantage(raw, base)          # (S, G, M)
    assert D.shape == (2, 2, 2)
    # sol0, SOW0: A up 0.10 (maximize), B down 5.0 (minimize) -> BOTH positive.
    assert D[0, 0, 0] == pytest.approx(0.10)
    assert D[0, 0, 1] == pytest.approx(5.0)
    # sol0, SOW1: worse on both -> BOTH negative, whatever the direction.
    assert D[0, 1, 0] == pytest.approx(-0.10)
    assert D[0, 1, 1] == pytest.approx(-3.0)


def test_incumbent_advantage_joins_on_sow_label_not_position(tmp_path):
    """A baseline missing a SOW must NaN that SOW, never shift the pairing.

    The baseline here covers SOW labels {0, 2} while the policy cube covers
    {0, 1, 2}. A positional join would pair the baseline's second row (label 2)
    with the cube's SOW 1 and produce a finite, WRONG advantage there; the
    label join leaves SOW 1 NaN and scores SOW 2 exactly.
    """
    meta3 = dict(_META, sow_labels=[0, 1, 2])
    records = [(0, g, "A", 0.90) for g in (0, 1, 2)] + \
              [(0, g, "B", 5.0) for g in (0, 1, 2)]
    _write_raw(tmp_path, records, meta3)
    raw = rob.load_raw(tmp_path)

    bdir = tmp_path / "baseline"
    bdir.mkdir()
    base_records = [
        (0, 0, "A", 0.80), (0, 0, "B", 10.0),
        # SOW 1 never simulated for the baseline.
        (0, 2, "A", 0.60), (0, 2, "B", 8.0),
    ]
    _write_raw(bdir, base_records, meta3)
    base = rob.load_raw(bdir)
    assert base.sow_labels == [0, 2]

    D = rob.incumbent_advantage(raw, base)          # (1, 3, 2)
    assert D[0, 0, 0] == pytest.approx(0.10)        # vs the label-0 baseline
    assert np.isnan(D[0, 1, 0]) and np.isnan(D[0, 1, 1])  # NOT 0.90 - 0.60
    assert D[0, 2, 0] == pytest.approx(0.30)        # vs the label-2 baseline
    assert D[0, 2, 1] == pytest.approx(3.0)         # -1 * (5.0 - 8.0)

    # The magnitude means skip the NaN SOW; the frequencies count it as HARM
    # (a degenerate SOW must not read as "no harm").
    mags = rob.regret_magnitudes(raw, base)
    assert mags.loc[0, "gain_mean__A"] == pytest.approx(np.mean([0.10, 0.30]))
    freqs = rob.regret_frequencies(raw, base, tau={"A": 0.0, "B": 0.0})
    assert freqs.loc[0, "harm_freq__A"] == pytest.approx(1.0 / 3.0)


def test_regret_magnitudes_are_hand_computable_natural_units(tmp_path):
    raw, base = _regret_fixture(tmp_path)
    df = rob.regret_magnitudes(raw, base)

    # sol0 is worse in exactly one of two SOWs, by 0.10 (A) and 3.0 (B).
    assert df.loc[0, "regret_mean__A"] == pytest.approx(0.05)     # mean(0, 0.10)
    assert df.loc[0, "regret_mean__B"] == pytest.approx(1.5)      # mean(0, 3.0)
    assert df.loc[0, "gain_mean__A"] == pytest.approx(0.05)       # mean(0.10, 0)
    assert df.loc[0, "gain_mean__B"] == pytest.approx(2.5)        # mean(5.0, 0)
    # Conditional regret is the mean over the ADVERSE subset only (McPhail T2).
    assert df.loc[0, "regret_cond__A"] == pytest.approx(0.10)
    assert df.loc[0, "regret_cond__B"] == pytest.approx(3.0)
    # numpy's linear interpolation on [0, 0.10] at q=0.90.
    assert df.loc[0, "regret_q90__A"] == pytest.approx(0.09)


def test_regret_cond_is_nan_when_never_worse_not_zero(tmp_path):
    """"Never worse" and "worse by zero" are different facts and must read differently."""
    raw, base = _regret_fixture(tmp_path)
    df = rob.regret_magnitudes(raw, base)
    # sol1 beats the incumbent in every SOW on every objective.
    assert df.loc[1, "regret_mean__A"] == pytest.approx(0.0)
    assert df.loc[1, "regret_q90__B"] == pytest.approx(0.0)
    assert np.isnan(df.loc[1, "regret_cond__A"])
    assert np.isnan(df.loc[1, "regret_cond__B"])
    # ...and the gain side is what actually carries sol1's signal.
    assert df.loc[1, "gain_mean__A"] == pytest.approx(0.20)
    assert df.loc[1, "gain_mean__B"] == pytest.approx(9.0)


def test_gain_and_regret_are_the_one_sided_halves_of_the_advantage(tmp_path):
    """gain_mean - regret_mean IS the mean signed incumbent advantage.

    The retired improvement_vs_baseline's signed information survives as the
    two one-sided halves; this pins that they decompose the SAME quantity on
    the SAME unit rather than being two new unrelated numbers.
    """
    raw, base = _regret_fixture(tmp_path)
    D = rob.incumbent_advantage(raw, base)          # (S, G, M)
    df = rob.regret_magnitudes(raw, base)
    signed = np.nanmean(D, axis=1)                  # (S, M)
    for k, name in enumerate(raw.obj_names):
        recomposed = (df[f"gain_mean__{name}"]
                      - df[f"regret_mean__{name}"]).to_numpy()
        assert np.allclose(recomposed, signed[:, k])


def test_regret_frequencies_are_unit_free_and_hand_computable(tmp_path):
    raw, base = _regret_fixture(tmp_path)
    df = rob.regret_frequencies(raw, base, tau=_REG_TAU)

    # sol0 is worse in 1 of 2 SOWs on each objective, and both losses land in the
    # SAME SOW -- so the joint no-harm frequency is 0.5, not 0.
    assert df.loc[0, "harm_freq__A"] == pytest.approx(0.5)
    assert df.loc[0, "harm_freq__B"] == pytest.approx(0.5)
    assert df.loc[0, "no_harm_freq"] == pytest.approx(0.5)
    assert df.loc[0, "n_degraded_mean"] == pytest.approx(1.0)     # (0 + 2) / 2

    # sol1 never harms anyone.
    assert df.loc[1, "harm_freq__A"] == pytest.approx(0.0)
    assert df.loc[1, "no_harm_freq"] == pytest.approx(1.0)
    assert df.loc[1, "n_degraded_mean"] == pytest.approx(0.0)


def test_no_harm_tolerance_is_monotone_in_tau(tmp_path):
    """Pi_tau must be non-decreasing in tau; a wide enough tolerance forgives sol0."""
    raw, base = _regret_fixture(tmp_path)
    strict = rob.regret_frequencies(raw, base, tau={"A": 0.0, "B": 0.0})
    loose = rob.regret_frequencies(raw, base, tau=_REG_TAU)
    wider = rob.regret_frequencies(raw, base, tau={"A": 10.0, "B": 100.0})

    assert strict.loc[0, "no_harm_freq_tau"] == pytest.approx(0.5)
    # sol0's shortfalls (0.10, 3.0) both sit inside the tolerance (0.15, 4.0).
    assert loose.loc[0, "no_harm_freq_tau"] == pytest.approx(1.0)
    for a, b in ((strict, loose), (loose, wider)):
        assert (b["no_harm_freq_tau"] >= a["no_harm_freq_tau"] - 1e-12).all()

    # tau = 0 is exactly the strict weak-Pareto-improvement form.
    assert strict["no_harm_freq_tau"].equals(strict["no_harm_freq"])


def test_party_harm_is_a_disjunction_never_a_sum(tmp_path):
    """Under unanimity a party's loss is not compensable, so the party form unions.

    Summing would double-count sol0's single bad SOW (worse on BOTH objectives at
    once) and report a party harm frequency of 1.0 where the truth is 0.5.
    """
    raw, base = _regret_fixture(tmp_path)
    df = rob.regret_frequencies(raw, base, tau=_REG_TAU,
                                parties={"both": ("A", "B")})
    assert df.loc[0, "party_harm_freq__both"] == pytest.approx(0.5)
    summed = df.loc[0, "harm_freq__A"] + df.loc[0, "harm_freq__B"]
    assert summed == pytest.approx(1.0)          # the wrong answer, pinned
    assert df.loc[0, "party_harm_freq__both"] < summed


def test_natural_units_keep_the_sows_a_zero_baseline_would_drop(tmp_path):
    """Regression test for the defect that killed the |baseline|-normalized form.

    Where the incumbent's value is 0 a ratio metric divides by zero, NaNs the
    cell and silently drops it from the mean -- and the dropped cells are
    precisely the benign ones (flood exceedance in a year with no flooding), so
    the estimator is biased toward the adverse subset. Natural units have no
    denominator, so every SOW contributes and every magnitude is finite.
    """
    zero_base = [
        (0, 0, "A", 0.80), (0, 0, "B", 10.0),
        # B is EXACTLY 0 in SOW1, as flood exceedance behaves in a benign year.
        (0, 1, "A", 0.80), (0, 1, "B", 0.0),
    ]
    raw, base = _regret_fixture(tmp_path, base_records=zero_base)

    mags = rob.regret_magnitudes(raw, base)
    freqs = rob.regret_frequencies(raw, base, tau=_REG_TAU)
    # sol0 B: SOW0 +5.0 gain, SOW1 sign*(13.0 - 0.0) = -13.0 regret. BOTH count.
    assert mags.loc[0, "regret_mean__B"] == pytest.approx(6.5)    # mean(0, 13.0)
    assert mags.loc[0, "gain_mean__B"] == pytest.approx(2.5)      # mean(5.0, 0)
    assert freqs.loc[0, "harm_freq__B"] == pytest.approx(0.5)
    assert np.isfinite(mags.loc[:, mags.columns.str.startswith(
        ("regret_mean__", "regret_q90__", "gain_mean__"))].to_numpy()).all()


def test_regret_is_design_independent(tmp_path):
    """Dropping a solution must not move any other solution's regret.

    This is the property best-in-set regret lacks and the reason it is excluded;
    the incumbent reference is external and fixed, so it holds by construction --
    but it is the property the whole design rests on, so it is pinned.
    """
    raw_all, base = _regret_fixture(tmp_path)
    full_mag = rob.regret_magnitudes(raw_all, base)
    full_freq = rob.regret_frequencies(raw_all, base, tau=_REG_TAU)

    sub = tmp_path / "subset"
    sub.mkdir()
    _write_raw(sub, [r for r in _REG_RECORDS if r[0] != 1], _META)
    raw_sub = rob.load_raw(sub)
    sub_mag = rob.regret_magnitudes(raw_sub, base)
    sub_freq = rob.regret_frequencies(raw_sub, base, tau=_REG_TAU)

    for col in full_mag.columns:
        assert sub_mag.loc[0, col] == pytest.approx(full_mag.loc[0, col],
                                                    nan_ok=True)
    for col in full_freq.columns:
        assert sub_freq.loc[0, col] == pytest.approx(full_freq.loc[0, col])


def test_scorecard_orients_regret_low_and_gain_high(tmp_path):
    """A wrong orientation flag silently inverts the metric in every ranking."""
    raw, base = _regret_fixture(tmp_path)
    _, higher = rob.score_robustness(
        raw, base, metrics=("regret_magnitudes", "regret_frequencies"))
    assert higher["regret_mean__A"] is False
    assert higher["regret_q90__B"] is False
    assert higher["gain_mean__A"] is True
    assert higher["no_harm_freq"] is True
    assert higher["no_harm_freq_tau"] is True
    assert higher["harm_freq__A"] is False


def test_unresolvable_tau_gates_the_frequency_block_not_the_magnitudes(tmp_path):
    """Synthetic objective names have no registered epsilon, so tau_ladder cannot
    resolve: the frequency block must be NaN-gated (never computed at a silent
    tau = 0) while the tau-free magnitudes still compute."""
    raw, base = _regret_fixture(tmp_path)
    scorecard, _ = rob.score_robustness(
        raw, base, metrics=("regret_magnitudes", "regret_frequencies"))
    assert np.isfinite(scorecard["regret_mean__A"]).all()
    assert scorecard["no_harm_freq"].isna().all()
    assert scorecard["no_harm_freq_tau"].isna().all()
    assert scorecard["harm_freq__A"].isna().all()


# ---------------------------------------------------------------------------
# The tolerance ladder (annual-unit epsilons + env override)
# ---------------------------------------------------------------------------

def test_tau_ladder_refuses_to_default_an_unknown_objective_to_zero():
    """A missing epsilon must raise, not silently harden the criterion to tau = 0."""
    with pytest.raises(KeyError):
        rob.tau_ladder(["A", "B"])


def test_tau_ladder_scales_the_registered_annual_epsilon():
    """tau_i = k * eps_i on the ANNUAL-unit epsilons -- the same calibration the
    search resolution uses, so the ladder reads in search-resolution steps."""
    from src.objectives_ensemble import ENSEMBLE_OBJECTIVES
    for name in ("nyc_delivery_reliability_annual",
                 "downstream_flood_exceedance_annual",
                 "nyc_storage_min_p01_pct"):
        eps = ENSEMBLE_OBJECTIVES[name].epsilon
        assert rob.tau_ladder([name], k=2.0)[name] == pytest.approx(2.0 * eps)
        assert rob.tau_ladder([name], k=0.0)[name] == pytest.approx(0.0)


def test_tau_ladder_floors_bind_only_where_noise_exceeds_epsilon():
    from src.objectives_ensemble import ENSEMBLE_OBJECTIVES
    name = "nyc_delivery_reliability_annual"
    eps = ENSEMBLE_OBJECTIVES[name].epsilon
    # A floor above the epsilon binds; one below it leaves the epsilon in charge.
    assert rob.tau_ladder([name], k=1.0, floors={name: 10 * eps})[name] \
        == pytest.approx(10 * eps)
    assert rob.tau_ladder([name], k=1.0, floors={name: eps / 10})[name] \
        == pytest.approx(eps)


def test_tau_env_override_is_whole_vector_or_error(monkeypatch):
    """NYCOPT_REGRET_TAU replaces the WHOLE vector; a partial override would
    silently leave the rest on a different tolerance basis."""
    monkeypatch.setenv("NYCOPT_REGRET_TAU", json.dumps({"A": 0.5, "B": 2.0}))
    assert rob.tau_ladder(["A", "B"]) == {"A": 0.5, "B": 2.0}

    monkeypatch.setenv("NYCOPT_REGRET_TAU", json.dumps({"A": 0.5}))
    with pytest.raises(KeyError, match="omits"):
        rob.tau_ladder(["A", "B"])

    monkeypatch.setenv("NYCOPT_REGRET_TAU",
                       json.dumps({"A": 0.5, "B": 2.0, "Z": 1.0}))
    with pytest.raises(KeyError, match="absent"):
        rob.tau_ladder(["A", "B"])


def test_incumbent_spread_rejects_a_degenerate_scale(tmp_path):
    """The optional normalization must refuse a zero denominator, not emit infinities."""
    raw, base = _regret_fixture(tmp_path)
    with pytest.raises(ValueError, match="spread is zero"):
        rob.incumbent_spread(base)      # the incumbent is constant in this fixture


def test_run_records_the_regret_tolerance(tmp_path):
    """tau moves no_harm_freq_tau, so it is recorded next to the numbers."""
    raw, base = _regret_fixture(tmp_path)
    rob.run(tmp_path, baseline_dir=tmp_path / "baseline",
            metrics=("regret_magnitudes", "regret_frequencies"))
    meta = json.loads((tmp_path / "robustness_meta.json").read_text())
    assert meta["regret_unit"] == "sow"
    assert meta["regret_available"] is True
    assert meta["regret_tau_k"] == rob.REGRET_TAU_K
    assert meta["substrate"] == "sow_annual_unit"
    assert meta["n_sow"] == 2 and meta["realizations_per_sow"] == 2
    # Synthetic objective names are not in the registry, so the ladder is None and
    # the tau column is gated off rather than silently computed at tau = 0.
    assert meta["regret_tau"] is None
