"""tests/test_ensemble_size_diagnostics.py - ensemble-size statistics, pure functions.

Covers ``src/ensemble_size_stats.py`` with hand-built synthetic libraries
(nothing routed through a pool image, a staged ensemble, or a simulation):

  1. compose_objectives reproduces the registered operators' scalar values
     exactly, including the failure-frequency / worst-sentinel semantics;
  2. replicate construction: disjoint prefix blocks partition the reference,
     supplementation flags the overlapping subsets, and is deterministic;
  3. level / paired SE: the paired SE of two policies sharing a common
     realization component is far below their level SE (the Linderoth
     precision argument the binding criterion rests on);
  4. epsilon-dominance relations: codes agree with the Borg box convention,
     majority relation tie-break, flip rate of an identical replicate = 0;
  5. optimism sign convention and the n_eff ratio;
  6. the closed form 1 - q^N.

Run (compute node; bare srun pytest dies in MPI_Init on Anvil):
    mpirun -np 1 python -m pytest tests/test_ensemble_size_diagnostics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ensemble_size_stats import (  # noqa: E402
    borg_form, compose_objectives, disjoint_prefix_blocks, epsilon_relations,
    flip_rate, level_se, majority_relation, n_eff_ratio, optimism, p_at_least_one_beyond,
    paired_se, pair_index, supplemented_replicates, summarize_over_pairs,
)
from src.objectives_ensemble import (  # noqa: E402
    FailureFrequencyOp, PooledMeanOp, PooledPercentileOp,
)


def _library(n_policy=3, n_real=40, n_unit=9, seed=0):
    rng = np.random.default_rng(seed)
    units = rng.random((n_policy, n_real, 3, n_unit))
    units[:, :, 0, :] = rng.integers(0, 5, size=(n_policy, n_real, n_unit))  # failing weeks
    units[:, :, 1, :] *= 100.0                                              # deficit %
    return units


OPS = [FailureFrequencyOp(k=3), PooledPercentileOp(99.0, worst_value=100.0),
       PooledMeanOp(worst_value=5490.0)]
DIRS = ["maximize", "minimize", "minimize"]


class TestComposition:
    def test_matches_scalar_operators(self):
        units = _library()
        rows = np.array([1, 5, 7, 12])
        out = compose_objectives(units, rows, OPS)
        for p in range(units.shape[0]):
            for k, op in enumerate(OPS):
                pooled = units[p, rows, k, :].reshape(-1)
                assert out[p, k] == pytest.approx(float(op(pooled)), rel=0, abs=0)

    def test_nan_sentinels(self):
        units = _library()
        units[0, 3, :, :] = np.nan
        rows = np.array([3, 4])
        out = compose_objectives(units, rows, OPS)
        pooled = units[0, rows, 0, :].reshape(-1)
        assert out[0, 0] == pytest.approx(float(OPS[0](pooled)))
        assert out[0, 1] == pytest.approx(float(OPS[1](units[0, rows, 1, :].reshape(-1))))

    def test_borg_form_negates_maximize(self):
        v = np.array([[0.9, 20.0, 1.0]])
        f = borg_form(v, DIRS)
        assert f[0, 0] == -0.9 and f[0, 1] == 20.0 and f[0, 2] == 1.0


class TestReplicates:
    def test_disjoint_blocks_partition(self):
        blocks = disjoint_prefix_blocks(1000, 100)
        assert len(blocks) == 10
        assert np.array_equal(np.concatenate(blocks), np.arange(1000))

    def test_supplemented_flags_and_determinism(self):
        reps, flags = supplemented_replicates(1000, 300, 5, seed=1)
        assert len(reps) == 5 and flags.sum() == 2 and not flags[:3].any()
        reps2, _ = supplemented_replicates(1000, 300, 5, seed=1)
        for a, b in zip(reps, reps2):
            assert np.array_equal(a, b)
        assert all(len(r) == 300 for r in reps)

    def test_too_small_reference_raises(self):
        with pytest.raises(ValueError):
            disjoint_prefix_blocks(50, 100)


class TestPrecision:
    def test_paired_se_below_level_se_under_common_noise(self):
        rng = np.random.default_rng(3)
        R = 200
        common = rng.normal(0.0, 1.0, size=(R, 1))          # shared realization effect
        a = 10.0 + common[:, 0] + rng.normal(0, 0.05, R)
        b = 10.5 + common[:, 0] + rng.normal(0, 0.05, R)
        values = np.stack([a, b], axis=1)[:, :, None]         # (R, 2 policies, 1 obj)
        lse = level_se(values)
        pse = paired_se(values)
        assert lse[0, 0] > 0.8 and lse[1, 0] > 0.8
        assert pse[0, 0] < 0.15
        summ = summarize_over_pairs(pse)
        assert summ["max"][0] == pytest.approx(pse[0, 0])

    def test_level_se_single_replicate_is_nan(self):
        assert np.isnan(level_se(np.zeros((1, 2, 3)))).all()

    def test_pair_index_count(self):
        assert len(pair_index(10)) == 45


class TestEpsilonRelations:
    eps = [0.05, 10.0, 0.3]

    def test_dominance_codes(self):
        # policy 0 better on every objective by more than one box than policy 1;
        # policy 2 shares every box with 0 (0.88 and 0.90 both floor to box -18
        # at eps 0.05 in Borg form; 10 and 12 to box 1; 1.0 and 1.1 to box 3)
        # -> incomparable, while it still dominates policy 1.
        v = np.array([[0.90, 10.0, 1.0],
                      [0.70, 40.0, 2.0],
                      [0.88, 12.0, 1.1]])
        codes = epsilon_relations(v, self.eps, DIRS)
        pairs = pair_index(3)
        assert codes[pairs.index((0, 1))] == 1
        assert codes[pairs.index((1, 2))] == -1
        assert codes[pairs.index((0, 2))] == 0

    def test_non_finite_incomparable(self):
        v = np.array([[0.9, np.nan, 1.0], [0.7, 40.0, 2.0]])
        assert epsilon_relations(v, self.eps, DIRS)[0] == 0

    def test_majority_and_flip_rate(self):
        codes = np.array([[1, 0, -1], [1, 1, -1], [0, 1, 0]])
        maj = majority_relation(codes)
        assert list(maj) == [1, 1, -1]
        assert flip_rate(codes, maj) == pytest.approx(3 / 9)
        assert flip_rate(np.array([[1, 0, -1]]), np.array([1, 0, -1])) == 0.0

    def test_majority_tie_prefers_incomparable(self):
        assert majority_relation(np.array([[1], [0]]))[0] == 0


class TestBiasAndNeff:
    def test_optimism_sign(self):
        ref = np.array([[0.8, 20.0, 1.0]])
        rep = np.array([[[0.9, 30.0, 0.5]]])                   # (1, 1, 3)
        g = optimism(rep, ref, DIRS)[0, 0]
        assert g[0] > 0 and g[1] < 0 and g[2] > 0

    def test_n_eff_ratio(self):
        assert n_eff_ratio(1.0, 2.0) == pytest.approx(0.25)
        assert np.isnan(n_eff_ratio(1.0, 0.0))

    def test_closed_form(self):
        assert p_at_least_one_beyond(0.99, 100) == pytest.approx(1 - 0.99 ** 100)
        assert p_at_least_one_beyond(0.99, 100) == pytest.approx(0.634, abs=1e-3)
