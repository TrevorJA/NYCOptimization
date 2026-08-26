"""tests/test_hazard_support_decomposition.py - HSD pure-function tests.

Covers the pure computation helpers of
``scripts/supplemental/hazard_support_run.py`` with hand-computed synthetic
inputs (nothing routed through the real pool images, E_test, or any cube):

  1. robust_bounds / scale_to_bounds: percentile bounds, unclipped linear
     scaling (out-of-box points land outside [0, 1]), degenerate-range error;
  2. self_nn_distances / nn_distances: hand-checked geometry;
  3. sow_aggregate: label-keyed (never positional) within-SOW means, NaN for
     an uncovered SOW;
  4. assign_strata: the pre-registered cut points, inclusive edges, NaN raises;
  5. tercile_labels: quantile binning matches the compare_designs convention;
  6. axis_excursions: zero inside the box, symmetric beyond either face;
  7. paired_design_delta: draw-level nesting (seeds are pseudoreplicates) and
     NaN when a design is absent.

Run (compute node; see docs/notes - bare srun pytest dies in MPI_Init):
    mpirun -np 1 python -m pytest tests/test_hazard_support_decomposition.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.supplemental.hazard_support_run import (  # noqa: E402
    assign_strata, axis_excursions, nn_distances, paired_design_delta,
    robust_bounds, scale_to_bounds, self_nn_distances, sow_aggregate,
    tercile_labels,
)


class TestBoundsAndScaling:
    def test_percentile_bounds(self):
        H = np.linspace(0.0, 100.0, 101).reshape(-1, 1)
        lo, hi = robust_bounds(H, 1.0, 99.0)
        assert lo[0] == pytest.approx(1.0)
        assert hi[0] == pytest.approx(99.0)

    def test_scaling_is_unclipped(self):
        lo, hi = np.array([0.0]), np.array([10.0])
        Z = scale_to_bounds(np.array([[-5.0], [0.0], [10.0], [20.0]]), lo, hi)
        assert Z.ravel() == pytest.approx([-0.5, 0.0, 1.0, 2.0])

    def test_degenerate_range_raises(self):
        H = np.ones((50, 2))
        H[:, 1] = np.arange(50)
        with pytest.raises(ValueError):
            robust_bounds(H, 1.0, 99.0)


class TestNearestNeighbour:
    def test_self_nn_excludes_self(self):
        Z = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        d = self_nn_distances(Z, workers=1)
        assert d == pytest.approx([1.0, 1.0, 4.0])

    def test_query_distances(self):
        ref = np.array([[0.0, 0.0], [2.0, 0.0]])
        d = nn_distances(np.array([[0.5, 0.0], [3.0, 4.0]]), ref, workers=1)
        assert d == pytest.approx([0.5, np.hypot(1.0, 4.0)])


class TestAggregation:
    def test_label_keyed_means(self):
        # SOW 0 has windows at rows 0 and 3 (shuffled order), SOW 2 has row 1;
        # SOW 1 has no windows and must come back NaN, not 0.
        values = np.array([1.0, 4.0, 0.0, 3.0])
        theta_index = np.array([0, 2, 2, 0])
        out = sow_aggregate(values, theta_index, 3)
        assert out[0] == pytest.approx(2.0)
        assert np.isnan(out[1])
        assert out[2] == pytest.approx(2.0)

    def test_boolean_fraction(self):
        out = sow_aggregate(np.array([True, False, True, True]),
                            np.array([0, 0, 0, 0]), 1)
        assert out[0] == pytest.approx(0.75)


class TestStrata:
    def test_pre_registered_cuts_inclusive(self):
        names = ("in_support", "boundary", "out_of_support")
        strata = assign_strata(np.array([0.0, 0.05, 0.051, 0.499, 0.50, 1.0]),
                               (0.05, 0.50), names)
        assert list(strata) == ["in_support", "in_support", "boundary",
                                "boundary", "out_of_support", "out_of_support"]

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            assign_strata(np.array([0.1, np.nan]), (0.05, 0.5), ("a", "b", "c"))


class TestTerciles:
    def test_matches_searchsorted_convention(self):
        x = np.arange(9, dtype=float)
        labels, edges = tercile_labels(x, 3)
        assert len(edges) == 4
        assert labels.min() == 0 and labels.max() == 2
        # Equal thirds of a uniform grid.
        assert (np.bincount(labels) == np.array([3, 3, 3])).all()


class TestExcursions:
    def test_zero_inside_box(self):
        Z = np.array([[0.0, 0.5, 1.0]])
        assert axis_excursions(Z) == pytest.approx(np.zeros((1, 3)))

    def test_beyond_either_face(self):
        Z = np.array([[-0.25, 1.5]])
        assert axis_excursions(Z).ravel() == pytest.approx([0.25, 0.5])


class TestPairedDelta:
    def test_draw_level_nesting(self):
        # HF draw 0 seeds mean (0.75 + 0.5)/2 = 0.625; HF draw 1 seed = 0.25
        # -> HF draw-level mean 0.4375. PS single draw mean 0.25. Seeds are
        # averaged WITHIN a draw first, so the two-seed draw does not count
        # twice against the one-seed draw.
        idx = np.arange(4)
        vecs = {
            ("hazard_filling_stationary", 0, 1): np.array([1, 1, 1, 0], bool),
            ("hazard_filling_stationary", 0, 2): np.array([1, 1, 0, 0], bool),
            ("hazard_filling_stationary", 1, 1): np.array([1, 0, 0, 0], bool),
            ("fixed_probabilistic", 0, 1): np.array([1, 0, 0, 0], bool),
        }
        delta = paired_design_delta(vecs, idx)
        assert delta == pytest.approx(0.4375 - 0.25)

    def test_absent_design_is_nan(self):
        vecs = {("hazard_filling_stationary", 0, 1): np.ones(3, bool)}
        assert np.isnan(paired_design_delta(vecs, np.arange(3)))

    def test_subset_indexing(self):
        vecs = {
            ("hazard_filling_stationary", 0, 1): np.array([1, 0, 0, 0], bool),
            ("fixed_probabilistic", 0, 1): np.array([0, 0, 1, 1], bool),
        }
        # Restricted to rows {0, 1}: HF = 0.5, PS = 0.0.
        assert paired_design_delta(vecs, np.array([0, 1])) == pytest.approx(0.5)
