"""
tests/test_robustness.py - Unit tests for src.robustness offline scoring.

Uses an INDEPENDENT synthetic raw matrix (hand-computed expected values, not
routed through _resolve_thresholds or the simulation), so the tests are not
tautological with the code under test (plan verification P3.1). Covers:
  1. load_raw densifies the long matrix into an (S, R, M) cube on the union of
     realization ids, NaN-filling gaps.
  2. Univariate satisficing reproduces SatisficingAgg incl. NaN-as-unsatisfied.
  3. Multivariate (Starr) domain criterion: NaN in any objective fails the joint
     criterion for that realization.
  4. Regret-from-best uses the POOLED per-realization best and excludes failed
     (NaN) cells; range-normalized.
  5. Regret-from-baseline joins on realization_id and clips improvements to 0.
  6. R==1 / single-trace gating yields N/A satisficing.
  7. ranking_stability returns a square matrix with unit diagonal.

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
# Regret-from-best
# ---------------------------------------------------------------------------

def test_regret_from_best_pooled_and_nan(raw):
    df = rob.regret_from_best(raw, normalize="range")
    a = df["regret_best__A"]
    # real0 A best 0.95 range 0.03; real1 A best 0.99 range 0.19
    # sol0: mean(0, (0.99-0.85)/0.19)
    assert a.loc[0] == pytest.approx(np.mean([0.0, (0.99 - 0.85) / 0.19]))
    assert a.loc[1] == pytest.approx(0.5)               # mean(1.0, 0.0)
    # sol2 real0 is NaN -> excluded; mean over the single finite realization.
    assert a.loc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Regret-from-baseline
# ---------------------------------------------------------------------------

def test_regret_from_baseline_clips_improvement(tmp_path):
    _write_raw(tmp_path, _RECORDS, _META)
    raw = rob.load_raw(tmp_path)
    # Baseline policy: A worse and B worse than most solutions on both reals.
    base_records = [
        (0, 0, "A", 0.80), (0, 0, "B", 15.0),
        (0, 1, "A", 0.80), (0, 1, "B", 15.0),
    ]
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    _write_raw(bdir, base_records, _META)
    baseline = rob.load_raw(bdir)

    df = rob.regret_from_baseline(raw, baseline, normalize="none")
    # B minimize, baseline 15 both reals. sol0 B = 5,8 -> both better -> regret 0.
    assert df["regret_baseline__B"].loc[0] == pytest.approx(0.0)
    # sol1 B = 12,9 -> both < 15 -> better -> regret 0.
    assert df["regret_baseline__B"].loc[1] == pytest.approx(0.0)
    # A maximize, baseline 0.80. sol2 A = NaN,0.80 -> not worse -> 0.
    assert df["regret_baseline__A"].loc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# R == 1 gating
# ---------------------------------------------------------------------------

def test_single_trace_satisficing_is_na(tmp_path):
    meta = dict(_META, is_ensemble=False, realization_indices=[0])
    records = [(0, 0, "A", 0.95), (0, 0, "B", 5.0),
               (1, 0, "A", 0.85), (1, 0, "B", 12.0)]
    _write_raw(tmp_path, records, meta)
    raw = rob.load_raw(tmp_path)
    scorecard, hb = rob.score_robustness(
        raw, metrics=("satisficing_univariate", "satisficing_multivariate",
                      "regret_from_best"))
    assert scorecard["sat_multivariate"].isna().all()
    assert scorecard.filter(like="sat_uni__").isna().all().all()
    # Regret across solutions is still defined at R==1.
    assert scorecard.filter(like="regret_best__").notna().any().any()


# ---------------------------------------------------------------------------
# Ranking stability
# ---------------------------------------------------------------------------

def test_ranking_stability_square_unit_diagonal(raw):
    scorecard, hb = rob.score_robustness(
        raw, metrics=("satisficing_univariate", "regret_from_best"))
    tau = rob.ranking_stability(scorecard, hb)
    assert tau.shape[0] == tau.shape[1] == scorecard.shape[1]
    assert np.allclose(np.diag(tau.to_numpy()), 1.0)
