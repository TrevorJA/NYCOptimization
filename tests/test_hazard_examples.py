"""
tests/test_hazard_examples.py - Example selection for the hazard-metric illustration.

Pure checks on ``src.plotting.hazard_examples``: percentile targets resolve to
the nearest member in rank space, members are never chosen twice, ties on a
zero-inflated axis share one percentile, and target validation rejects
non-selection axes and out-of-range percentiles. No staged data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import config  # noqa: E402
from src.plotting import hazard_examples as hx  # noqa: E402

AXES = list(config.HAZARD_SELECTION_AXES)


def _image(n: int = 50, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = rng.gamma(2.0, 1.0, size=(n, len(AXES)))
    H[: n // 5, AXES.index("drought_magnitude")] = 0.0   # no-event atom
    return H


def test_percentile_targets_pick_nearest_member():
    H = _image()
    m = AXES.index("drought_magnitude")
    top = int(np.argmax(H[:, m]))
    bottom_d = int(np.argmin(H[:, AXES.index("flood_peak_discharge")]))
    chosen = hx.select_examples(H, AXES, [{"drought_magnitude": 1.0},
                                          {"flood_peak_discharge": 0.0}])
    assert chosen == [top, bottom_d]


def test_members_are_chosen_at_most_once():
    H = _image()
    chosen = hx.select_examples(H, AXES, [{"drought_magnitude": 1.0}] * 3)
    assert len(set(chosen)) == 3
    order = np.argsort(H[:, AXES.index("drought_magnitude")])[::-1]
    assert sorted(chosen) == sorted(int(i) for i in order[:3])


def test_zero_atom_shares_one_percentile():
    H = _image()
    pct = hx.ensemble_percentile(H, AXES, "drought_magnitude")
    zeros = H[:, AXES.index("drought_magnitude")] == 0.0
    assert np.allclose(pct[zeros], pct[zeros][0])
    assert pct[zeros][0] < pct[~zeros].min()


def test_validate_targets_rejects_bad_input():
    with pytest.raises(ValueError, match="non-selection"):
        hx.validate_targets([{"flood_rise_rate": 0.5}])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        hx.validate_targets([{"drought_magnitude": 1.5}])
    with pytest.raises(ValueError, match="distinct identity"):
        hx.validate_targets([{"drought_magnitude": 0.5}] * (len(hx.EXAMPLE_COLORS) + 1))
    ok = [{"drought_magnitude": 0.9, "flood_peak_discharge": 0.1}]
    assert hx.validate_targets(ok) is ok
