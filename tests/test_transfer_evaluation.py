"""Tests for the transfer-evaluation instrument.

Coverage claims, in order:

  1. Orientation. Natural units and Borg orientation round-trip, and a baseline
     read in natural units yields the correct dominance verdict against
     objective columns stored in Borg orientation. This is the instrument's
     most likely silent error: the two conventions meet inside one function.
  2. Readout 1. Dominance and epsilon-dominance counts on hand-computed cases,
     including exact ties (which must NOT count as dominance) and the
     "not dominated by the baseline" converse.
  3. Readout 2. Merged-set attribution on a constructed pool whose answer is
     known by hand, including a solution shared by two sources, and the
     distinction between composition share and contribution rate.
  4. Path consistency. The epsilon-fraction table on a known difference.
  5. Execution. Claim files are exclusive; atomic writes leave no temp file and
     survive a round trip; resume classifies done / failed / missing correctly.
  6. Guards. Cell enumeration omits the diagonal by default; the .set writer
     emits the MOEAFramework v5 header and terminator that the CLI requires,
     and round-trips through the project's own reader.

Nothing here simulates. Run (compute node; bare srun pytest dies in MPI_Init
on Anvil):

    mpirun -np 1 python -m pytest tests/test_transfer_evaluation.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import transfer_stats as tstats  # noqa: E402

# Two maximize objectives and one minimize, which is enough to break any
# implementation that assumes a single orientation.
DIRECTIONS = [1, -1, 1]
EPSILONS = [0.1, 1.0, 0.5]
NAMES = ["reliability", "deficit_pct", "storage_pct"]


###############################################################################
# 1. Orientation
###############################################################################

def test_borg_and_natural_round_trip():
    natural = np.array([[0.9, 12.0, 30.0], [0.5, 40.0, 10.0]])
    borg = tstats.to_borg(natural, DIRECTIONS)
    # Maximize columns are negated, the minimize column is untouched.
    assert borg[0].tolist() == [-0.9, 12.0, -30.0]
    assert np.allclose(tstats.to_natural(borg, DIRECTIONS), natural)


def test_stored_borg_columns_compare_correctly_against_a_natural_baseline():
    """The sign trap, end to end.

    Objective columns arrive from a ``.set`` in Borg orientation; the
    scenario-matched baseline CSV arrives in natural units. Converting the
    former and not the latter (or neither) silently inverts every maximize
    axis, which would read a worse-than-baseline solution as dominating.
    """
    baseline_natural = np.array([0.60, 30.0, 15.0])
    # One solution better on every axis, one worse on every axis.
    stored_borg = np.array([[-0.80, 20.0, -25.0],
                            [-0.40, 45.0, -5.0]])
    natural = tstats.to_natural(stored_borg, DIRECTIONS)
    summary = tstats.dominance_summary(natural, baseline_natural, DIRECTIONS,
                                       EPSILONS)
    assert summary["n_dominating"] == 1
    assert summary["n_solutions"] == 2

    # Forgetting the conversion must change the answer, or the test proves nothing.
    wrong = tstats.dominance_summary(stored_borg, baseline_natural, DIRECTIONS,
                                     EPSILONS)
    assert wrong["n_dominating"] != summary["n_dominating"]


###############################################################################
# 2. Readout 1 - dominance over the baseline
###############################################################################

def test_dominance_summary_hand_case():
    baseline = np.array([0.60, 30.0, 15.0])
    natural = np.array([
        [0.80, 20.0, 25.0],   # better on all three -> dominates
        [0.80, 30.0, 15.0],   # better on one, tied on two -> dominates (weak)
        [0.60, 30.0, 15.0],   # identical -> does NOT dominate itself
        [0.40, 45.0, 5.0],    # worse on all three -> dominated BY the baseline
        [0.80, 45.0, 25.0],   # better on two, worse on one -> neither
    ])
    s = tstats.dominance_summary(natural, baseline, DIRECTIONS, EPSILONS)
    assert s["n_solutions"] == 5
    assert s["n_dominating"] == 2
    assert s["frac_dominating"] == pytest.approx(2 / 5)
    # Only the all-worse row is dominated by the baseline.
    assert s["n_not_dominated"] == 4
    # Objectives strictly beaten: 3, 1, 0, 0, 2.
    assert s["objectives_beaten_hist"] == [2, 1, 1, 1]
    assert s["mean_objectives_beaten"] == pytest.approx(6 / 5)


def test_epsilon_dominance_ignores_sub_epsilon_differences():
    """A difference the optimizer could not resolve is not an improvement.

    Note the baseline is deliberately 0.63 rather than a round 0.60. Epsilon
    boxes are ``floor(value / eps)``, so a baseline sitting exactly on a box
    boundary puts any improvement at all into the next box up, and the test
    would assert the opposite of what it means. That boundary sensitivity is a
    real property of epsilon dominance, not an artifact of this code - the
    campaign's own epsilon calibration records flood-objective cardinality
    effects arising from exactly this - so the test pins the interior case and
    this comment records why.
    """
    baseline = np.array([0.63, 30.0, 15.0])
    # Better on reliability by 0.01 against eps = 0.1: same epsilon box
    # (floor(-0.64/0.1) == floor(-0.63/0.1) == -7), so it must not count.
    natural = np.array([[0.64, 30.0, 15.0]])
    assert tstats.dominance_summary(natural, baseline, DIRECTIONS,
                                    EPSILONS)["n_dominating"] == 1
    assert tstats.dominance_summary(natural, baseline, DIRECTIONS,
                                    EPSILONS)["n_eps_dominating"] == 0

    # A full box better does count.
    natural = np.array([[0.75, 30.0, 15.0]])
    assert tstats.dominance_summary(natural, baseline, DIRECTIONS,
                                    EPSILONS)["n_eps_dominating"] == 1


def test_epsilon_dominance_rejects_non_positive_epsilon():
    with pytest.raises(ValueError, match="finite and positive"):
        tstats.epsilon_dominance_mask(np.array([[0.5, 1.0, 2.0]]),
                                      np.array([0.5, 1.0, 2.0]),
                                      DIRECTIONS, [0.1, 0.0, 0.5])


def test_dominance_summary_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        tstats.dominance_summary(np.zeros((4, 3)), np.zeros(2), DIRECTIONS, EPSILONS)


###############################################################################
# 3. Readout 2 - merged-set attribution
###############################################################################

def test_attribution_counts_shared_members_both_ways():
    """A solution in two source sets is the one case attribution must disclose."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[3.0, 4.0], [5.0, 6.0]])   # row 0 is shared with a
    keys = {"alpha": tstats.dv_key_array(a), "beta": tstats.dv_key_array(b)}
    members = tstats.dv_key_array(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    credited, disjoint, n_shared = tstats.attribute_members(members, keys)
    assert credited == {"alpha": 2, "beta": 2}      # shared member credited twice
    assert sum(disjoint.values()) == 3              # tie-break sums to the merged size
    assert disjoint == {"alpha": 2, "beta": 1}      # "alpha" wins by sorted order
    assert n_shared == 1


def test_contribution_rate_is_size_invariant_where_share_is_not():
    """The reason both numbers are reported.

    A source three times larger contributes three times as many members while
    each of its solutions is equally likely to survive. Composition share
    therefore favours the larger archive; contribution rate does not.
    """
    credited = {"small": 10, "large": 30}
    disjoint = {"small": 10, "large": 30}
    sizes = {"small": 100, "large": 300}
    rows = {r["source"]: r for r in
            tstats.contribution_table(credited, disjoint, sizes, 40)}
    assert rows["large"]["composition_share"] == pytest.approx(0.75)
    assert rows["small"]["composition_share"] == pytest.approx(0.25)
    assert rows["large"]["contribution_rate"] == pytest.approx(0.10)
    assert rows["small"]["contribution_rate"] == pytest.approx(0.10)


def test_dv_keys_distinguish_near_identical_rows():
    a = tstats.dv_key_array(np.array([[1.0, 2.0]]))
    b = tstats.dv_key_array(np.array([[1.0, 2.0 + 1e-12]]))
    assert a[0] != b[0]
    assert tstats.dv_key_array(np.array([[1.0, 2.0]]))[0] == a[0]


def test_dv_key_array_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        tstats.dv_key_array(np.array([1.0, 2.0]))


###############################################################################
# 4. Path consistency
###############################################################################

def test_eps_fraction_table_reports_the_expected_ratio():
    driver = np.array([[0.500, 10.0, 20.0], [0.500, 10.0, 20.0]])
    stored = np.array([[0.510, 10.5, 20.0], [0.500, 10.0, 20.0]])
    rows = {r["objective"]: r for r in
            tstats.eps_fraction_table(driver, stored, EPSILONS, NAMES)}
    # 0.010 against eps = 0.1
    assert rows["reliability"]["max_diff_over_eps"] == pytest.approx(0.1)
    # 0.5 against eps = 1.0
    assert rows["deficit_pct"]["max_diff_over_eps"] == pytest.approx(0.5)
    assert rows["storage_pct"]["max_diff_over_eps"] == pytest.approx(0.0)
    assert rows["reliability"]["n_compared"] == 2


def test_eps_fraction_table_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        tstats.eps_fraction_table(np.zeros((2, 3)), np.zeros((3, 3)),
                                  EPSILONS, NAMES)


###############################################################################
# 5. Execution - claims, atomic writes, resume
###############################################################################

def _tev():
    """Import the driver library lazily; it imports pandas and src.* helpers."""
    from src import transfer_eval
    return transfer_eval


def test_claim_is_exclusive(tmp_path):
    tev = _tev()
    claims = tmp_path / "claims"
    claims.mkdir()
    assert tev.try_claim(claims, "cellA", 7) is True
    assert tev.try_claim(claims, "cellA", 7) is False
    # A different unit in the same cell, and the same row in another cell, are free.
    assert tev.try_claim(claims, "cellA", 8) is True
    assert tev.try_claim(claims, "cellB", 7) is True


def test_atomic_write_round_trips_and_leaves_no_temp_file(tmp_path):
    import pandas as pd

    tev = _tev()
    frame = pd.DataFrame({"realization": [0, 0, 1], "objective": ["a", "b", "a"],
                          "unit_year": [0, 0, 0], "value": [1.0, 2.0, 3.0]})
    meta = {"natural": [1.5, 2.5], "n_survivors": 2, "seconds": 12.5}
    stem = tev.unit_stem(tmp_path, 3)
    tev.atomic_write_unit(frame, stem, meta)

    assert not list(tmp_path.glob("*.tmp"))
    back, back_meta = tev.read_unit(stem)
    assert back is not None
    assert len(back) == 3
    assert back_meta["n_survivors"] == 2
    assert back_meta["natural"] == [1.5, 2.5]


def test_completed_units_classifies_done_failed_and_missing(tmp_path):
    import pandas as pd

    tev = _tev()
    tev.atomic_write_unit(pd.DataFrame({"value": [1.0]}),
                          tev.unit_stem(tmp_path, 0), {"natural": [0.0]})
    tev.unit_stem(tmp_path, 1).with_suffix(".failed").write_text("RuntimeError: x\n")
    # Row 2 is simply absent.

    done = tev.completed_units(tmp_path, retry_failed=False)
    assert done == {0, 1}
    # With retry enabled the failed unit becomes work again; the written one does not.
    assert tev.completed_units(tmp_path, retry_failed=True) == {0}


def test_completed_units_on_a_missing_directory_is_empty(tmp_path):
    tev = _tev()
    assert tev.completed_units(tmp_path / "nope") == set()


def test_wall_guard_stops_only_when_a_unit_could_not_finish():
    import time as _time

    tev = _tev()
    assert tev.out_of_wall_time(0, 0) is False              # guard disabled
    assert tev.out_of_wall_time(100, _time.time() + 3600) is False
    assert tev.out_of_wall_time(100, _time.time() + 10) is True


###############################################################################
# 6. Guards - cells and .set format
###############################################################################

def test_moea_set_file_has_the_v5_header_and_terminator(tmp_path):
    """Both are load-bearing and both fail silently when absent.

    Without ``# Version=5`` the reader assumes the v4 layout and expects an
    extra constraint column; without the trailing lone ``#`` the file parses as
    zero entries and an indicator returns 0.000000 rather than erroring.
    """
    tev = _tev()
    dvs = np.arange(8, dtype=float).reshape(2, 4)
    objs = np.array([[-0.9, 12.0], [-0.5, 40.0]])
    path = tev.write_moea_set_file(tmp_path / "x.set", dvs, objs, "drb_ffmp")

    lines = path.read_text().splitlines()
    assert lines[0] == "# Version=5"
    assert "# Problem=drb_ffmp" in lines
    assert "# NumberOfVariables=4" in lines
    assert "# NumberOfObjectives=2" in lines
    assert "# NumberOfConstraints=0" in lines
    assert lines[-1] == "#"
    assert not list(tmp_path.glob("*.tmp"))


def test_moea_set_file_round_trips_through_the_project_reader(tmp_path):
    """Full precision matters: Contribution matches solutions by value."""
    from src.load.reference_set import load_reference_set

    tev = _tev()
    rng = np.random.default_rng(20260911)
    dvs = rng.random((5, 4))
    objs = rng.random((5, 2))
    path = tev.write_moea_set_file(tmp_path / "y.set", dvs, objs, "drb_ffmp")

    back_dv, back_obj = load_reference_set(path, n_vars=4, n_objs=2)
    assert np.array_equal(back_dv, dvs)
    assert np.array_equal(back_obj, objs)


def test_moea_set_file_rejects_row_mismatch(tmp_path):
    tev = _tev()
    with pytest.raises(ValueError, match="row mismatch"):
        tev.write_moea_set_file(tmp_path / "z.set", np.zeros((2, 4)),
                                np.zeros((3, 2)), "drb_ffmp")


def test_unit_stem_is_stable_and_zero_padded(tmp_path):
    tev = _tev()
    assert tev.unit_stem(tmp_path, 7).name == "sol00007"
    assert tev.unit_stem(tmp_path, 1234).name == "sol01234"


###############################################################################
# 7. Readout 2 diagnostics
###############################################################################

def test_enrichment_is_one_under_the_no_effect_null():
    """A set contributing in proportion to its share of the pool has no advantage.

    This is the whole reason enrichment is reported: the raw composition share
    of a set holding 70 % of the pool is 70 % even when every solution is
    equally likely to survive, so the share alone cannot distinguish a design
    effect from set-size arithmetic.
    """
    pool = {"big": 700, "small": 300}
    members = ["big"] * 70 + ["small"] * 30
    rows = {r["source"]: r for r in tstats.enrichment_table(members, pool)}
    assert rows["big"]["merged_share"] == pytest.approx(0.70)
    assert rows["small"]["merged_share"] == pytest.approx(0.30)
    assert rows["big"]["enrichment"] == pytest.approx(1.0)
    assert rows["small"]["enrichment"] == pytest.approx(1.0)


def test_enrichment_detects_a_real_advantage_despite_a_smaller_set():
    """The small set contributes fewer members but is enriched."""
    pool = {"big": 900, "small": 100}
    members = ["big"] * 50 + ["small"] * 50   # equal counts from unequal pools
    rows = {r["source"]: r for r in tstats.enrichment_table(members, pool)}
    assert rows["small"]["enrichment"] == pytest.approx(5.0)
    assert rows["big"]["enrichment"] == pytest.approx(50 / 90)


def test_occupied_boxes_counts_distinct_boxes_not_solutions():
    eps = [1.0, 1.0]
    # Three solutions, two of which share a box.
    borg = np.array([[0.1, 0.1], [0.9, 0.9], [2.5, 2.5]])
    assert tstats.occupied_boxes(borg, eps) == 2
    assert tstats.occupied_boxes(np.empty((0, 2)), eps) == 0


def test_leave_one_out_localises_a_single_driving_objective():
    """A source carried by one axis loses its enrichment when that axis goes."""
    def archive_fn(keep):
        # Synthetic: "alpha" is enriched only while objective 1 is retained.
        return {"alpha": 4.0 if 1 in keep else 1.0, "beta": 1.0}

    rows = {r["dropped"]: r for r in
            tstats.leave_one_out_enrichment(archive_fn, 3, ["o0", "o1", "o2"])}
    assert rows["none"]["alpha"] == 4.0
    assert rows["o1"]["alpha"] == 1.0        # the driving axis
    assert rows["o0"]["alpha"] == 4.0
    assert rows["o2"]["alpha"] == 4.0
