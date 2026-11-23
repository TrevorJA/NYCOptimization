"""tests/test_dv_distributions.py - The decision-variable distribution figure."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.formulations import get_bounds, get_var_names
from src.plotting import style
from src.plotting.dv_distributions import (GROUPS, _stack_offsets,
                                           describe_variable,
                                           plot_dv_distributions)


def test_every_ffmp_variable_has_a_rule_group():
    """A renamed or new DV must be given a group and a label, not fall to 'other'."""
    described = [describe_variable(n) for n in get_var_names("ffmp")]
    assert all(key in GROUPS and key != "other" for key, _, _ in described)
    labels = [(key, label) for key, label, _ in described]
    assert len(set(labels)) == len(labels)


def test_stack_offsets_separate_coincident_dots_and_leave_lone_dots():
    off = _stack_offsets(np.array([0.0, 0.0, 0.0, 5.0]), step=1.0, reach=1.0)
    assert off[3] == 0.0
    assert sorted(off[:3]) == [-1.0, 0.0, 1.0]


def test_figure_builds_with_and_without_highlight():
    lo, hi = get_bounds("ffmp")
    rng = np.random.default_rng(0)
    dv = lo + rng.random((40, len(lo))) * (hi - lo)
    mask = np.zeros(40, dtype=bool)
    mask[:5] = True
    style.apply_style()
    for m in (mask, None):
        fig = plot_dv_distributions(dv, "ffmp", m, title="t\nu")
        assert style.enforce_min_fontsize(fig) == 0
        plt.close(fig)
