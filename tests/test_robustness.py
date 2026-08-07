"""
tests/test_robustness.py - Unit tests for src.robustness offline scoring.

Uses an INDEPENDENT synthetic raw matrix (hand-computed expected values, not
routed through _resolve_thresholds or the simulation), so the tests are not
tautological with the code under test. Covers:
  1. load_raw densifies the long matrix into an (S, R, M) cube on the union of
     realization ids, NaN-filling gaps.
  2. Univariate satisficing reproduces SatisficingAgg incl. NaN-as-unsatisfied.
  3. Multivariate (Starr) domain criterion: NaN in any objective fails the joint
     criterion for that realization.
  4. Laplace (mean) and maximin (worst-case) anchors, in natural units and
     respecting each objective's direction.
  5. Improvement-over-status-quo joins on realization_id and clips improvements
     to 0 -- and is DESIGN-INDEPENDENT: dropping a solution does not change any
     other solution's score. That is the property regret-from-best lacked, and
     the reason it was deleted.
  6. The deleted metrics stay deleted (regret_from_best, overfitting_gap).
  7. The attainability screen flags realizations no policy can win.
  8. R==1 / single-trace gating: EVERY metric is realization-defined, so all are
     N/A on a single trace.
  9. ranking_stability returns a square matrix with unit diagonal.
 10. The SOW unit (Herman 2014; Trindade 2017; Gold 2022): sow_ids round-trip through
     the meta; the SOW-unit criterion is a DIFFERENT quantity from the realization-unit
     one when the within-SOW spread is large; and with no sow_ids it is N/A, never a
     silent fallback to the realization unit.

Run:
    venv/bin/python -m pytest tests/test_robustness.py -v
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
    """Independent reference for SatisficingAgg (no project imports)."""
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
# 3 solutions x 2 realizations x 2 objectives.
#   A: maximize, threshold 0.9, kind "ge"
#   B: minimize, threshold 10,  kind "le"
# Values (natural units); solution 2 / realization 0 / A is missing (NaN).
_RECORDS = [
    # sid, rid, obj, value
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
    "base_names": ["A", "B"],
    "thresholds": {"A": 0.9, "B": 10.0},
    "kinds": {"A": "ge", "B": "le"},
    "directions": {"A": "maximize", "B": "minimize"},
    "realization_indices": [0, 1],
}


def _write_raw(tmp_path: Path, records, meta) -> Path:
    df = pd.DataFrame(records, columns=["solution_id", "realization_id",
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
    assert raw.base_names == ["A", "B"]
    # Missing (sol2, real0, A) is NaN-filled.
    assert np.isnan(raw.cube[2, 0, 0])
    assert raw.cube[0, 0, 0] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Univariate satisficing
# ---------------------------------------------------------------------------

def test_satisficing_univariate_handvalues(raw):
    df = rob.satisficing_univariate(raw)
    a = df["sat_uni__A"]
    b = df["sat_uni__B"]
    assert a.loc[0] == pytest.approx(0.5)   # 0.95 T, 0.85 F
    assert a.loc[1] == pytest.approx(1.0)
    assert a.loc[2] == pytest.approx(0.0)   # NaN F, 0.80 F
    assert b.loc[0] == pytest.approx(1.0)
    assert b.loc[1] == pytest.approx(0.5)
    assert b.loc[2] == pytest.approx(0.5)


def test_satisficing_matches_satisficingagg(raw):
    # Consistency gate: column A must equal the SEARCH-time aggregator on the
    # same values (so re-eval scoring reproduces what search would compute).
    from src.objectives_ensemble import SatisficingAgg
    agg = SatisficingAgg(threshold=0.9, kind="ge")
    col = raw.cube[:, :, 0]
    df = rob.satisficing_univariate(raw)
    for si, sid in enumerate(raw.solution_ids):
        assert df["sat_uni__A"].loc[sid] == pytest.approx(agg(col[si, :]))


def test_satisficing_matches_inline_reference(raw):
    # Independent (no project imports) reference, so the gate above is not the
    # only check of the satisficing formula.
    col = raw.cube[:, :, 0]
    df = rob.satisficing_univariate(raw)
    for si, sid in enumerate(raw.solution_ids):
        assert df["sat_uni__A"].loc[sid] == pytest.approx(
            _satisficing_ref(col[si, :], 0.9, "ge"))


# ---------------------------------------------------------------------------
# Multivariate (Starr) domain criterion
# ---------------------------------------------------------------------------

def test_satisficing_multivariate_joint_nan_fails(raw):
    s = rob.satisficing_multivariate(raw)
    # sol0: r0 (T&T)=T, r1 (F&T)=F -> 0.5
    assert s.loc[0] == pytest.approx(0.5)
    # sol1: r0 (T&F)=F, r1 (T&T)=T -> 0.5
    assert s.loc[1] == pytest.approx(0.5)
    # sol2: r0 (NaN->F), r1 (F) -> 0.0
    assert s.loc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Risk-attitude anchors (McPhail T3)
# ---------------------------------------------------------------------------

def test_laplace_mean_is_natural_units(raw):
    df = rob.laplace_mean(raw)
    # sol0 A = 0.95, 0.85 -> mean 0.90 (maximize, natural units).
    assert df["laplace__A"].loc[0] == pytest.approx(0.90)
    # sol0 B = 5.0, 8.0 -> mean 6.5 (minimize, natural units).
    assert df["laplace__B"].loc[0] == pytest.approx(6.5)


def test_maximin_picks_the_worst_realization_per_direction(raw):
    df = rob.maximin(raw)
    # A maximizes: the worst realization is the SMALLEST value.
    assert df["maximin__A"].loc[0] == pytest.approx(0.85)
    # B minimizes: the worst realization is the LARGEST value.
    assert df["maximin__B"].loc[0] == pytest.approx(8.0)


def test_deleted_metrics_are_gone():
    """regret_from_best and overfitting_gap were removed, not renamed.

    regret_from_best is set-relative and design-coupled (dropping one scenario
    design would change every other design's score) and does not converge on a
    tail objective (Bonham et al. 2024). The overfitting gap is undefined in
    Brodeur et al. (2020) and is structurally invalid under a measure change.
    """
    assert not hasattr(rob, "regret_from_best")
    assert not hasattr(rob, "overfitting_gap")
    assert not hasattr(rob, "regret_from_baseline")  # renamed
    assert "regret_from_best" not in rob._DEFAULT_METRICS


# ---------------------------------------------------------------------------
# Improvement over the status quo
# ---------------------------------------------------------------------------

def _with_baseline(tmp_path):
    """Stage the fixture plus a status-quo baseline both objectives can be scored against."""
    _write_raw(tmp_path, _RECORDS, _META)
    raw = rob.load_raw(tmp_path)
    base_records = [
        (0, 0, "A", 0.80), (0, 0, "B", 15.0),
        (0, 1, "A", 0.80), (0, 1, "B", 15.0),
    ]
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, base_records, _META)
    return raw, rob.load_raw(bdir)


def test_improvement_vs_baseline_is_signed_and_positive_means_better(tmp_path):
    """Positive = better than the status quo, for BOTH objective directions.

    The metric is signed, not clipped. Clipping would credit a policy with nothing
    for beating the baseline -- and since optimized policies are expected to
    dominate the status quo nearly everywhere, the clipped quantity collapses to
    ~0 for every policy of every design and discriminates nothing.
    """
    raw, baseline = _with_baseline(tmp_path)
    df = rob.improvement_vs_baseline(raw, baseline, normalize="none")

    # B MINIMIZES; baseline 15 on both realizations. sol0 B = 5, 8 -> both better,
    # so the improvement is POSITIVE (and equals the size of the gap).
    assert df["vs_baseline__B"].loc[0] == pytest.approx(np.mean([15.0 - 5.0,
                                                                 15.0 - 8.0]))
    # A MAXIMIZES; baseline 0.80. sol0 A = 0.95, 0.85 -> both better -> positive.
    assert df["vs_baseline__A"].loc[0] == pytest.approx(np.mean([0.95 - 0.80,
                                                                 0.85 - 0.80]))
    # sol1 B = 12, 9 -> better than 15 on both -> positive.
    assert df["vs_baseline__B"].loc[1] > 0


def test_improvement_vs_baseline_goes_negative_when_worse(tmp_path):
    """A policy WORSE than the status quo scores negative, not zero.

    This is the discrimination the old clipped version destroyed.
    """
    _write_raw(tmp_path, _RECORDS, _META)
    raw = rob.load_raw(tmp_path)
    # A status quo that BEATS every solution: A very high, B very low.
    base_records = [
        (0, 0, "A", 1.00), (0, 0, "B", 1.0),
        (0, 1, "A", 1.00), (0, 1, "B", 1.0),
    ]
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, base_records, _META)
    baseline = rob.load_raw(bdir)

    df = rob.improvement_vs_baseline(raw, baseline, normalize="none")
    assert (df["vs_baseline__A"].dropna() < 0).all()
    assert (df["vs_baseline__B"].dropna() < 0).all()


def test_improvement_vs_baseline_is_oriented_higher_is_better(tmp_path):
    """The scorecard must declare it higher-is-better for EVERY objective.

    The value is direction-oriented, so a minimize objective's improvement is
    positive too. Getting this flag wrong would silently invert that objective in
    every ranking-stability correlation.
    """
    raw, baseline = _with_baseline(tmp_path)
    _, higher_better = rob.score_robustness(
        raw, baseline, metrics=("improvement_vs_baseline",))
    cols = [c for c in higher_better if c.startswith("vs_baseline__")]
    assert cols
    assert all(higher_better[c] for c in cols)


def test_improvement_vs_baseline_is_design_independent(tmp_path):
    """Dropping a solution must not change any other solution's score.

    This is the property regret-from-best lacks and the reason it was deleted:
    its reference is the best value in the POOLED set, so removing one scenario
    design's policies changes every other design's regret.
    """
    _write_raw(tmp_path, _RECORDS, _META)
    raw_all = rob.load_raw(tmp_path)
    base_records = [
        (0, 0, "A", 0.80), (0, 0, "B", 15.0),
        (0, 1, "A", 0.80), (0, 1, "B", 15.0),
    ]
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, base_records, _META)
    baseline = rob.load_raw(bdir)
    full = rob.improvement_vs_baseline(raw_all, baseline)

    # Re-score with solution 0 removed entirely.
    sub_dir = tmp_path / "subset"
    sub_dir.mkdir()
    _write_raw(sub_dir, [r for r in _RECORDS if r[0] != 0], _META)
    subset = rob.improvement_vs_baseline(rob.load_raw(sub_dir), baseline)

    for sid in subset.index:
        for col in subset.columns:
            a, b = full.loc[sid, col], subset.loc[sid, col]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Attainability screen
# ---------------------------------------------------------------------------

def test_attainability_flags_unwinnable_realizations(raw):
    """Separates 'this design searched badly' from 'nobody can win this scenario'."""
    frame = rob.attainability_screen(raw)
    assert list(frame["realization_id"]) == list(raw.realization_ids)
    assert set(frame.columns) >= {"realization_id", "n_satisficing_solutions",
                                  "attainable"}
    # attainable iff at least one solution meets the joint criterion there.
    assert (frame["attainable"] == (frame["n_satisficing_solutions"] > 0)).all()


# ---------------------------------------------------------------------------
# R == 1 gating
# ---------------------------------------------------------------------------

def test_single_trace_scores_every_metric_na(tmp_path):
    """EVERY metric is realization-defined, so all are N/A on a single trace.

    Previously only satisficing was gated, so the baseline-relative metric was
    still computed at R == 1 and written -- a meaningless number that looked
    meaningful.
    """
    meta = dict(_META, is_ensemble=False, realization_indices=[0])
    records = [(0, 0, "A", 0.95), (0, 0, "B", 5.0),
               (1, 0, "A", 0.85), (1, 0, "B", 12.0)]
    _write_raw(tmp_path, records, meta)
    raw = rob.load_raw(tmp_path)
    scorecard, _ = rob.score_robustness(
        raw, metrics=("satisficing_univariate", "satisficing_multivariate",
                      "laplace_mean", "maximin"))
    assert scorecard.isna().all().all()


# ---------------------------------------------------------------------------
# The SOW unit (Herman 2014; Trindade 2017; Gold 2022, 2023)
# ---------------------------------------------------------------------------
# 2 SOWs x 2 realizations. Objective A (maximize, >= 0.9) is deliberately built with a
# LARGE WITHIN-SOW SPREAD: solution 0 passes exactly one realization in each SOW.
#   realization: 0    1  | 2    3
#   SOW:         0    0  | 1    1
#   sol 0 / A:   1.00 0.50 | 1.00 0.50   -> realization unit: 2/4 = 0.5
#                                          SOW mean  -> 0.75, 0.75 -> both FAIL  -> 0.0
#                                          SOW worst -> 0.50, 0.50 -> both FAIL  -> 0.0
#   sol 1 / A:   0.95 0.92 | 0.95 0.92   -> realization unit: 1.0; SOW mean 1.0; worst 1.0
# B (minimize, <= 10) is non-binding for both, so A alone drives the joint criterion.
_SOW_RECORDS = [
    (0, 0, "A", 1.00), (0, 0, "B", 1.0),
    (0, 1, "A", 0.50), (0, 1, "B", 1.0),
    (0, 2, "A", 1.00), (0, 2, "B", 1.0),
    (0, 3, "A", 0.50), (0, 3, "B", 1.0),
    (1, 0, "A", 0.95), (1, 0, "B", 1.0),
    (1, 1, "A", 0.92), (1, 1, "B", 1.0),
    (1, 2, "A", 0.95), (1, 2, "B", 1.0),
    (1, 3, "A", 0.92), (1, 3, "B", 1.0),
]

_SOW_META = dict(
    _META,
    realization_indices=[0, 1, 2, 3],
    sow_ids=[0, 0, 1, 1],
    n_sow=2,
    realizations_per_sow=2,
)


@pytest.fixture
def sow_raw(tmp_path):
    _write_raw(tmp_path, _SOW_RECORDS, _SOW_META)
    return rob.load_raw(tmp_path)


def test_load_raw_recovers_sow_ids(sow_raw):
    assert sow_raw.sow_ids == [0, 0, 1, 1]
    assert sow_raw.n_sow == 2
    assert sow_raw.realizations_per_sow == 2
    groups = sow_raw.sow_groups()
    assert [s for s, _ in groups] == [0, 1]
    assert list(groups[0][1]) == [0, 1] and list(groups[1][1]) == [2, 3]


def test_sow_ids_align_on_realization_id_not_position(tmp_path):
    """A missing realization must not shift every later realization into the wrong SOW."""
    records = [r for r in _SOW_RECORDS if r[1] != 1]   # realization 1 never ran
    _write_raw(tmp_path, records, _SOW_META)
    raw = rob.load_raw(tmp_path)
    assert raw.realization_ids == [0, 2, 3]
    assert raw.sow_ids == [0, 1, 1]                    # NOT [0, 0, 1]


def test_sow_unit_is_a_different_quantity_from_the_realization_unit(sow_raw):
    """The whole point: collapsing within a SOW first changes the answer.

    Solution 0 satisfies half the traces in EVERY state of the world. On the
    realization unit that reads as 50% robust; on the SOW unit it is robust in NO state
    of the world, because no state's collapsed performance meets the criterion.
    """
    per_realization = rob.satisficing_multivariate(sow_raw)
    per_sow_mean = rob.satisficing_multivariate_sow(sow_raw, within_sow_agg="mean")
    per_sow_worst = rob.satisficing_multivariate_sow(sow_raw, within_sow_agg="worst")

    assert per_realization.loc[0] == pytest.approx(0.5)
    assert per_sow_mean.loc[0] == pytest.approx(0.0)
    assert per_sow_worst.loc[0] == pytest.approx(0.0)
    assert per_sow_mean.loc[0] != pytest.approx(per_realization.loc[0])

    # A solution that clears the bar in every trace clears it under both units.
    assert per_realization.loc[1] == pytest.approx(1.0)
    assert per_sow_mean.loc[1] == pytest.approx(1.0)
    assert per_sow_worst.loc[1] == pytest.approx(1.0)


def test_within_sow_aggregator_changes_the_collapsed_vector(sow_raw):
    """mean = risk-neutral inside the SOW; worst = risk-averse. Both are real choices."""
    mean_cube, labels = rob.collapse_within_sow(sow_raw, "mean")
    worst_cube, _ = rob.collapse_within_sow(sow_raw, "worst")
    assert labels == [0, 1]
    assert mean_cube.shape == (2, 2, 2)          # (solutions, SOWs, objectives)
    # A maximizes: SOW 0 of solution 0 holds 1.00 and 0.50.
    assert mean_cube[0, 0, 0] == pytest.approx(0.75)
    assert worst_cube[0, 0, 0] == pytest.approx(0.50)
    # B minimizes: its worst is the LARGEST value (both 1.0 here).
    assert worst_cube[0, 0, 1] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        rob.collapse_within_sow(sow_raw, "median")


def test_sow_metric_is_na_without_a_grouping_never_the_realization_unit(tmp_path):
    """No sow_ids -> the SOW column is NaN. Substituting the realization unit would be
    reporting a different quantity under the SOW name."""
    meta = {k: v for k, v in _SOW_META.items()
            if k not in ("sow_ids", "n_sow", "realizations_per_sow")}
    _write_raw(tmp_path, _SOW_RECORDS, meta)
    raw = rob.load_raw(tmp_path)
    assert raw.sow_ids is None
    assert raw.n_sow is None

    with pytest.raises(ValueError, match="no sow_ids"):
        rob.collapse_within_sow(raw)

    scorecard, higher_better = rob.score_robustness(
        raw, metrics=("satisficing_multivariate", "satisficing_multivariate_sow"))
    assert scorecard["sat_multivariate_sow"].isna().all()
    assert scorecard["sat_multivariate"].notna().all()   # the realization unit still works
    assert higher_better["sat_multivariate_sow"] is True


def test_scorecard_carries_both_units_when_grouped(sow_raw):
    scorecard, _ = rob.score_robustness(
        sow_raw, metrics=("satisficing_multivariate", "satisficing_multivariate_sow"))
    assert list(scorecard.columns) == ["sat_multivariate", "sat_multivariate_sow"]
    assert scorecard.loc[0, "sat_multivariate"] == pytest.approx(0.5)
    assert scorecard.loc[0, "sat_multivariate_sow"] == pytest.approx(0.0)


def test_run_records_the_within_sow_aggregator(tmp_path):
    """The aggregator moves the number, so it must be recorded next to it."""
    _write_raw(tmp_path, _SOW_RECORDS, _SOW_META)
    rob.run(tmp_path, metrics=("satisficing_multivariate_sow",), within_sow_agg="worst")
    meta = json.loads((tmp_path / "robustness_meta.json").read_text())
    assert meta["within_sow_aggregator"] == "worst"
    assert meta["sow_metrics_available"] is True
    assert meta["n_sow"] == 2 and meta["realizations_per_sow"] == 2


# ---------------------------------------------------------------------------
# Ranking stability
# ---------------------------------------------------------------------------

def test_ranking_stability_square_unit_diagonal(raw):
    scorecard, hb = rob.score_robustness(
        raw, metrics=("satisficing_univariate", "laplace_mean", "maximin"))
    tau = rob.ranking_stability(scorecard, hb)
    assert tau.shape[0] == tau.shape[1] == scorecard.shape[1]
    assert np.allclose(np.diag(tau.to_numpy()), 1.0)


# ---------------------------------------------------------------------------
# Incumbent-relative regret
# ---------------------------------------------------------------------------
# Two solutions, two SOWs of two realizations each, two objectives. SOW means are
# whole numbers so every expected value below is hand-computed, not read off the
# implementation.
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
    (0, 0, "A", 0.90), (0, 0, "B", 4.0),
    (0, 1, "A", 0.90), (0, 1, "B", 6.0),      # SOW0 mean B = 5.0
    (0, 2, "A", 0.70), (0, 2, "B", 12.0),
    (0, 3, "A", 0.70), (0, 3, "B", 14.0),     # SOW1 mean B = 13.0
    (1, 0, "A", 1.00), (1, 0, "B", 1.0),
    (1, 1, "A", 1.00), (1, 1, "B", 1.0),
    (1, 2, "A", 1.00), (1, 2, "B", 1.0),
    (1, 3, "A", 1.00), (1, 3, "B", 1.0),
]

_REG_BASE_RECORDS = [
    (0, r, obj, val)
    for r in (0, 1, 2, 3)
    for obj, val in (("A", 0.80), ("B", 10.0))
]

_REG_TAU = {"A": 0.15, "B": 4.0}


def _regret_fixture(tmp_path, base_records=None):
    _write_raw(tmp_path, _REG_RECORDS, _SOW_META)
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, base_records or _REG_BASE_RECORDS, _SOW_META)
    return rob.load_raw(tmp_path), rob.load_raw(bdir)


def test_incumbent_advantage_is_oriented_positive_means_better(tmp_path):
    """D is signed so positive = better than current operations, both directions."""
    raw, base = _regret_fixture(tmp_path)
    D = rob.incumbent_advantage(raw, base)          # (S, n_sow, M)
    assert D.shape == (2, 2, 2)
    # sol0, SOW0: A up 0.10 (maximize), B down 5.0 (minimize) -> BOTH positive.
    assert D[0, 0, 0] == pytest.approx(0.10)
    assert D[0, 0, 1] == pytest.approx(5.0)
    # sol0, SOW1: worse on both -> BOTH negative, whatever the direction.
    assert D[0, 1, 0] == pytest.approx(-0.10)
    assert D[0, 1, 1] == pytest.approx(-3.0)


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


def test_gain_minus_regret_reproduces_the_signed_improvement(tmp_path):
    """mean(P) - mean(G) IS the signed improvement. Ties the new family to the old."""
    raw, base = _regret_fixture(tmp_path)
    D = rob.incumbent_advantage(raw, base, unit="realization")
    gain = np.nanmean(np.maximum(0.0, D), axis=1)
    regret = np.nanmean(np.maximum(0.0, -D), axis=1)
    signed = rob.improvement_vs_baseline(raw, base, normalize="none")
    for k, name in enumerate(raw.base_names):
        assert np.allclose(gain[:, k] - regret[:, k],
                           signed[f"vs_baseline__{name}"].to_numpy())


def test_natural_units_keep_the_sows_a_zero_baseline_would_drop(tmp_path):
    """Regression test for the defect that motivated dropping the ratio form.

    Where the incumbent's value is 0 the RELATIVE metric divides by zero, NaNs the
    cell and silently drops it from the mean -- and the dropped cells are the
    benign ones, so the estimator is biased toward the adverse subset. Natural
    units have no denominator, so every SOW contributes.
    """
    zero_base = [
        (0, r, obj, val)
        for r in (0, 1, 2, 3)
        # B is 10.0 in SOW0 (realizations 0, 1) and EXACTLY 0 in SOW1 (2, 3),
        # exactly as flood exceedance behaves in a year with no flooding.
        for obj, val in (("A", 0.80), ("B", 10.0 if r in (0, 1) else 0.0))
    ]
    raw, base = _regret_fixture(tmp_path, base_records=zero_base)

    # The legacy ratio form sees only the two realizations with a non-zero
    # denominator: sol0's B is 4.0 and 6.0 there, against a baseline of 10.0.
    legacy = rob.improvement_vs_baseline(raw, base, normalize="best")
    assert legacy.loc[0, "vs_baseline__B"] == pytest.approx(
        np.mean([(10.0 - 4.0) / 10.0, (10.0 - 6.0) / 10.0]))

    # Natural units use ALL of them: +5.0 in SOW0, -13.0 in SOW1.
    mags = rob.regret_magnitudes(raw, base)
    freqs = rob.regret_frequencies(raw, base, tau=_REG_TAU)
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
    _write_raw(sub, [r for r in _REG_RECORDS if r[0] != 1], _SOW_META)
    raw_sub = rob.load_raw(sub)
    sub_mag = rob.regret_magnitudes(raw_sub, base)
    sub_freq = rob.regret_frequencies(raw_sub, base, tau=_REG_TAU)

    for col in full_mag.columns:
        assert sub_mag.loc[0, col] == pytest.approx(full_mag.loc[0, col],
                                                    nan_ok=True)
    for col in full_freq.columns:
        assert sub_freq.loc[0, col] == pytest.approx(full_freq.loc[0, col])


def test_sow_unit_regret_needs_a_grouping_on_both_cubes(tmp_path):
    """The realization unit is a DIFFERENT quantity and is never substituted."""
    _write_raw(tmp_path, _REG_RECORDS, _META)          # no sow_ids
    raw = rob.load_raw(tmp_path)
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, _REG_BASE_RECORDS, _META)
    base = rob.load_raw(bdir)
    with pytest.raises(ValueError, match="sow_ids"):
        rob.incumbent_advantage(raw, base, unit="sow")
    # ...and the realization unit still works, explicitly asked for.
    assert rob.incumbent_advantage(raw, base, unit="realization").shape[1] == 4


def test_regret_columns_are_na_without_a_sow_grouping(tmp_path):
    """The scorecard NaNs the regret block rather than falling back to realizations."""
    _write_raw(tmp_path, _REG_RECORDS, _META)          # no sow_ids
    raw = rob.load_raw(tmp_path)
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, _REG_BASE_RECORDS, _META)
    scorecard, _ = rob.score_robustness(raw, rob.load_raw(bdir),
                                        metrics=("regret_magnitudes",))
    assert "regret_mean__A" in scorecard.columns
    assert scorecard["regret_mean__A"].isna().all()


def test_scorecard_orients_regret_low_and_gain_high(tmp_path):
    """A wrong orientation flag silently inverts the metric in every ranking."""
    raw, base = _regret_fixture(tmp_path)
    _, higher = rob.score_robustness(
        raw, base, metrics=("regret_magnitudes", "regret_frequencies"))
    assert higher["regret_mean__A"] is False
    assert higher["regret_q90__B"] is False
    assert higher["gain_mean__A"] is True
    assert higher["no_harm_freq"] is True
    assert higher["harm_freq__A"] is False


def test_tau_ladder_refuses_to_default_an_unknown_objective_to_zero(tmp_path):
    """A missing epsilon must raise, not silently harden the criterion to tau = 0."""
    with pytest.raises(KeyError):
        rob.tau_ladder(["A", "B"])


def test_tau_ladder_scales_the_registered_epsilon():
    from src.objectives import OBJECTIVES
    name = "nyc_delivery_reliability_weekly"
    assert rob.tau_ladder([name], k=2.0)[name] == pytest.approx(
        2.0 * OBJECTIVES[name].epsilon)
    assert rob.tau_ladder([name], k=0.0)[name] == pytest.approx(0.0)


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
    # Synthetic objective names are not in the registry, so the ladder is None and
    # the tau column is gated off rather than silently computed at tau = 0.
    assert meta["regret_tau"] is None
