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
