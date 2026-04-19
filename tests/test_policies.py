"""
tests/test_policies.py - Unit and integration tests for policy architectures and
external policy evaluation.

Categories:
  1. Policy unit tests (fast, no simulation)
  2. State config tests (no simulation)
  3. Integration tests (require simulation — marked @pytest.mark.slow)

Run fast tests only:
    venv/Scripts/python -m pytest tests/test_policies.py -v -m "not slow"

Run all tests:
    venv/Scripts/python -m pytest tests/test_policies.py -v
"""

# ---------------------------------------------------------------------------
# IMPORTANT: env vars must be set before any project imports because
# config.py and src.simulation read them at import time.
# ---------------------------------------------------------------------------
import os
os.environ['PYWRDRB_SIM_START_DATE'] = '2018-01-01'
os.environ['PYWRDRB_SIM_END_DATE'] = '2022-09-30'

import sys
sys.path.insert(0, '.')

import numpy as np
import pytest

from src.policies import PolicyBase, RBFPolicy, ObliqueTreePolicy, ANNPolicy

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rbf_policy():
    return RBFPolicy(n_inputs=15, n_outputs=1, n_rbf=6, output_max=3000.0)


@pytest.fixture
def tree_policy():
    return ObliqueTreePolicy(n_inputs=15, n_outputs=1, depth=3, output_max=3000.0)


@pytest.fixture
def ann_policy():
    return ANNPolicy(n_inputs=15, n_outputs=1, h1=8, h2=8, output_max=3000.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_valid_params(policy, rng):
    """Draw uniform random params within bounds."""
    lb, ub = policy.get_bounds()
    return lb + rng.random(len(lb)) * (ub - lb)


def _random_state(n_inputs, rng):
    return rng.random(n_inputs)


# ===========================================================================
# 1. RBFPolicy unit tests
# ===========================================================================

class TestRBFPolicy:

    def test_rbf_n_params(self, rbf_policy):
        p = rbf_policy
        expected = p._n_rbf * (p._n_inputs + 1 + p._n_outputs)
        assert p.n_params == expected

    def test_rbf_bounds_shape(self, rbf_policy):
        lb, ub = rbf_policy.get_bounds()
        assert len(lb) == rbf_policy.n_params
        assert len(ub) == rbf_policy.n_params

    def test_rbf_bounds_ordering(self, rbf_policy):
        lb, ub = rbf_policy.get_bounds()
        assert np.all(lb <= ub)

    def test_rbf_set_params_wrong_size(self, rbf_policy):
        with pytest.raises(ValueError):
            rbf_policy.set_params(np.zeros(rbf_policy.n_params + 1))

    def test_rbf_output_shape(self, rbf_policy):
        rng = np.random.default_rng(42)
        params = _random_valid_params(rbf_policy, rng)
        rbf_policy.set_params(params)
        out = rbf_policy(_random_state(rbf_policy.n_inputs, rng))
        assert out.shape == (rbf_policy.n_outputs,)

    def test_rbf_output_bounds(self, rbf_policy):
        rng = np.random.default_rng(0)
        for _ in range(100):
            params = _random_valid_params(rbf_policy, rng)
            rbf_policy.set_params(params)
            state = _random_state(rbf_policy.n_inputs, rng)
            out = rbf_policy(state)
            assert np.all(out >= rbf_policy._output_min), f"output {out} below min"
            assert np.all(out <= rbf_policy._output_max), f"output {out} above max"

    def test_rbf_deterministic(self, rbf_policy):
        rng = np.random.default_rng(7)
        params = _random_valid_params(rbf_policy, rng)
        state = _random_state(rbf_policy.n_inputs, rng)
        rbf_policy.set_params(params)
        out1 = rbf_policy(state)
        out2 = rbf_policy(state)
        assert np.allclose(out1, out2)


# ===========================================================================
# 2. ObliqueTreePolicy unit tests
# ===========================================================================

class TestObliqueTreePolicy:

    def test_tree_n_params(self, tree_policy):
        p = tree_policy
        n_internal = 2 ** p._depth - 1
        n_leaves = 2 ** p._depth
        expected = n_internal * (p._n_inputs + 1) + n_leaves * p._n_outputs
        assert p.n_params == expected

    def test_tree_bounds_shape(self, tree_policy):
        lb, ub = tree_policy.get_bounds()
        assert len(lb) == tree_policy.n_params
        assert len(ub) == tree_policy.n_params

    def test_tree_bounds_ordering(self, tree_policy):
        lb, ub = tree_policy.get_bounds()
        assert np.all(lb <= ub)

    def test_tree_set_params_wrong_size(self, tree_policy):
        with pytest.raises(ValueError):
            tree_policy.set_params(np.zeros(tree_policy.n_params - 1))

    def test_tree_output_shape(self, tree_policy):
        rng = np.random.default_rng(42)
        params = _random_valid_params(tree_policy, rng)
        tree_policy.set_params(params)
        out = tree_policy(_random_state(tree_policy.n_inputs, rng))
        assert out.shape == (tree_policy.n_outputs,)

    def test_tree_output_bounds(self, tree_policy):
        rng = np.random.default_rng(1)
        for _ in range(100):
            params = _random_valid_params(tree_policy, rng)
            tree_policy.set_params(params)
            state = _random_state(tree_policy.n_inputs, rng)
            out = tree_policy(state)
            assert np.all(out >= tree_policy._output_min), f"output {out} below min"
            assert np.all(out <= tree_policy._output_max), f"output {out} above max"

    def test_tree_deterministic(self, tree_policy):
        rng = np.random.default_rng(8)
        params = _random_valid_params(tree_policy, rng)
        state = _random_state(tree_policy.n_inputs, rng)
        tree_policy.set_params(params)
        out1 = tree_policy(state)
        out2 = tree_policy(state)
        assert np.allclose(out1, out2)

    def test_tree_different_leaves(self):
        """States that differ on x[0] sign should route to different leaves."""
        # depth=1: single root node, 2 leaves
        # Set split weight = [1, 0, 0, ...] with bias = 0.
        # x[0] < 0  → left leaf; x[0] >= 0 → right leaf.
        n_inputs = 5
        policy = ObliqueTreePolicy(n_inputs=n_inputs, n_outputs=1, depth=1,
                                   output_max=100.0)
        # n_internal=1, n_leaves=2
        # param layout: [w0..w4, bias, leaf0_val, leaf1_val]
        w = np.zeros(n_inputs)
        w[0] = 1.0
        bias = 0.0
        leaf0 = np.array([3.0])   # left leaf  (x[0] < 0)
        leaf1 = np.array([-3.0])  # right leaf (x[0] >= 0)
        params = np.concatenate([w, [bias], leaf0, leaf1])
        policy.set_params(params)

        # x[0] < 0 → should go left → leaf0 → sigmoid(3.0) * 100 ≈ 95
        state_left = np.array([-1.0] + [0.5] * (n_inputs - 1))
        out_left = policy(state_left)

        # x[0] >= 0 → should go right → leaf1 → sigmoid(-3.0) * 100 ≈ 5
        state_right = np.array([1.0] + [0.5] * (n_inputs - 1))
        out_right = policy(state_right)

        assert out_left[0] > 50.0, f"Left leaf output {out_left[0]} expected > 50"
        assert out_right[0] < 50.0, f"Right leaf output {out_right[0]} expected < 50"
        assert not np.allclose(out_left, out_right)


# ===========================================================================
# 2b. ANNPolicy unit tests
# ===========================================================================

class TestANNPolicy:

    def test_ann_n_params(self, ann_policy):
        p = ann_policy
        expected = p._n_inputs * p._h1 + p._h1 + p._h1 * p._h2 + p._h2 + p._h2 * p._n_outputs + p._n_outputs
        assert p.n_params == expected

    def test_ann_bounds_shape(self, ann_policy):
        lb, ub = ann_policy.get_bounds()
        assert len(lb) == ann_policy.n_params
        assert len(ub) == ann_policy.n_params

    def test_ann_bounds_ordering(self, ann_policy):
        lb, ub = ann_policy.get_bounds()
        assert np.all(lb <= ub)

    def test_ann_set_params_wrong_size(self, ann_policy):
        with pytest.raises(ValueError):
            ann_policy.set_params(np.zeros(ann_policy.n_params + 1))

    def test_ann_output_shape(self, ann_policy):
        rng = np.random.default_rng(42)
        params = _random_valid_params(ann_policy, rng)
        ann_policy.set_params(params)
        out = ann_policy(_random_state(ann_policy.n_inputs, rng))
        assert out.shape == (ann_policy.n_outputs,)

    def test_ann_output_bounds(self, ann_policy):
        rng = np.random.default_rng(0)
        for _ in range(100):
            params = _random_valid_params(ann_policy, rng)
            ann_policy.set_params(params)
            out = ann_policy(_random_state(ann_policy.n_inputs, rng))
            assert np.all(out >= 0.0), f"Output below min: {out}"
            assert np.all(out <= 3000.0), f"Output above max: {out}"

    def test_ann_deterministic(self, ann_policy):
        rng = np.random.default_rng(77)
        params = _random_valid_params(ann_policy, rng)
        ann_policy.set_params(params)
        state = _random_state(ann_policy.n_inputs, rng)
        out1 = ann_policy(state)
        out2 = ann_policy(state)
        assert np.allclose(out1, out2)


# ===========================================================================
# 3. Parametrized interface tests (all policy types)
# ===========================================================================

@pytest.mark.parametrize("policy_fixture", ["rbf_policy", "tree_policy", "ann_policy"])
class TestPolicyInterface:

    def test_policy_interface(self, policy_fixture, request):
        """All PolicyBase abstract methods/properties exist and are callable."""
        policy = request.getfixturevalue(policy_fixture)
        assert callable(policy)                     # __call__
        assert callable(policy.set_params)
        assert callable(policy.get_bounds)
        assert isinstance(policy.n_params, int)
        assert isinstance(policy.n_inputs, int)
        assert isinstance(policy.n_outputs, int)
        assert isinstance(policy.name, str)

    def test_policy_name(self, policy_fixture, request):
        policy = request.getfixturevalue(policy_fixture)
        assert len(policy.name) > 0


# ===========================================================================
# 4. State config tests (no simulation)
# ===========================================================================

class TestStateConfig:

    def test_default_matches_config(self):
        from src.external_policy import build_state_config
        from config import STATE_FEATURES
        cfg = build_state_config()
        assert len(cfg) == len(STATE_FEATURES)

    def test_explicit_feature_list(self):
        from src.external_policy import build_state_config
        cfg = build_state_config([
            "combined_nyc_storage_frac",
            "montague_flow_lag2",
            "trenton_flow_lag4",
            "nj_demand_lag4",
        ])
        assert len(cfg) == 4
        assert "combined_nodes" in cfg[0]
        for entry in cfg[1:]:
            assert "parameter" in entry

    def test_unknown_feature_raises(self):
        import pytest
        from src.external_policy import build_state_config
        with pytest.raises(KeyError):
            build_state_config(["not_a_real_feature"])

    def test_inline_dict_entry(self):
        from src.external_policy import build_state_config
        custom = {
            "parameter": "predicted_demand_nj_lag1",
            "normalize_by": 105.0,
        }
        cfg = build_state_config(["combined_nyc_storage_frac", custom])
        assert len(cfg) == 2
        assert cfg[1]["parameter"] == "predicted_demand_nj_lag1"


# ===========================================================================
# 5. Integration tests (require simulation)
# ===========================================================================

@slow
class TestIntegration:

    @pytest.fixture(autouse=True)
    def _obj_set(self):
        from config import get_objective_set
        self.obj_set = get_objective_set()

    def test_evaluate_constant_policy(self):
        """evaluate_with_policy with constant policy returns finite objectives."""
        from src.external_policy import evaluate_with_policy

        def constant_policy(state):
            return np.array([600.0])

        objs = evaluate_with_policy(constant_policy, mode="aggregate")
        assert len(objs) == self.obj_set.n_objs
        assert all(np.isfinite(v) for v in objs), f"Non-finite objectives: {objs}"

    def test_evaluate_rbf_policy(self, rbf_policy):
        """RBFPolicy evaluation returns 6 finite objectives."""
        from src.external_policy import evaluate_with_policy
        rng = np.random.default_rng(99)
        params = _random_valid_params(rbf_policy, rng)
        rbf_policy.set_params(params)
        objs = evaluate_with_policy(rbf_policy, mode="aggregate")
        assert len(objs) == self.obj_set.n_objs
        assert all(np.isfinite(v) for v in objs), f"Non-finite objectives: {objs}"

    def test_evaluate_tree_policy(self, tree_policy):
        """ObliqueTreePolicy evaluation returns finite objectives."""
        from src.external_policy import evaluate_with_policy
        rng = np.random.default_rng(55)
        params = _random_valid_params(tree_policy, rng)
        tree_policy.set_params(params)
        objs = evaluate_with_policy(tree_policy, mode="aggregate")
        assert len(objs) == self.obj_set.n_objs
        assert all(np.isfinite(v) for v in objs), f"Non-finite objectives: {objs}"

    def test_evaluate_ann_policy(self, ann_policy):
        """ANNPolicy evaluation returns finite objectives."""
        from src.external_policy import evaluate_with_policy
        rng = np.random.default_rng(33)
        params = _random_valid_params(ann_policy, rng)
        ann_policy.set_params(params)
        objs = evaluate_with_policy(ann_policy, mode="aggregate")
        assert len(objs) == self.obj_set.n_objs
        assert all(np.isfinite(v) for v in objs), f"Non-finite objectives: {objs}"

    def test_make_objective_function_ffmp(self):
        """make_objective_function dispatches correctly for FFMP."""
        from config import make_objective_function, get_baseline_values
        fn = make_objective_function("ffmp")
        objs = fn(get_baseline_values())
        assert len(objs) == self.obj_set.n_objs
        assert all(np.isfinite(v) for v in objs)

    def test_make_objective_function_rbf(self):
        """make_objective_function dispatches correctly for RBF."""
        from config import make_objective_function, get_bounds
        fn = make_objective_function("rbf")
        lb, ub = get_bounds("rbf")
        dvs = (lb + ub) / 2.0
        objs = fn(dvs)
        assert len(objs) == self.obj_set.n_objs
        assert all(np.isfinite(v) for v in objs)

    def test_run_policy_simulation_returns_data(self):
        """run_policy_simulation returns dict with expected keys."""
        from src.external_policy import run_policy_simulation

        def constant_policy(state):
            return np.array([600.0])

        data = run_policy_simulation(constant_policy, mode="aggregate")
        expected_keys = {"res_storage", "major_flow", "ibt_demands",
                         "ibt_diversions", "mrf_target"}
        assert expected_keys.issubset(set(data.keys())), (
            f"Missing keys: {expected_keys - set(data.keys())}"
        )

    def test_aggregate_mode_balances_storage(self):
        """Aggregate mode: final storage fractions across 3 NYC reservoirs
        should be within 0.05 of each other (balancing logic working)."""
        from src.external_policy import run_policy_simulation
        from config import NYC_RESERVOIRS, NYC_RESERVOIR_CAPACITIES

        def constant_policy(state):
            return np.array([400.0])

        data = run_policy_simulation(constant_policy, mode="aggregate")
        storage = data["res_storage"]

        final_fracs = []
        for res in NYC_RESERVOIRS:
            cap = NYC_RESERVOIR_CAPACITIES[res]
            frac = storage[res].iloc[-1] / cap
            final_fracs.append(frac)

        max_diff = max(final_fracs) - min(final_fracs)
        assert max_diff <= 0.05, (
            f"Storage fractions not balanced: {dict(zip(NYC_RESERVOIRS, final_fracs))}, "
            f"max_diff={max_diff:.4f}"
        )

    def test_individual_vs_aggregate_differ(self, rbf_policy):
        """Same policy in individual vs aggregate mode should give different objectives."""
        from src.external_policy import evaluate_with_policy
        rng = np.random.default_rng(77)
        params = _random_valid_params(rbf_policy, rng)

        # Individual mode: 3-output policy (one per reservoir)
        rbf_individual = RBFPolicy(n_inputs=15, n_outputs=3, n_rbf=6,
                                   output_max=1000.0)
        lb, ub = rbf_individual.get_bounds()
        ind_params = lb + rng.random(len(lb)) * (ub - lb)
        rbf_individual.set_params(ind_params)

        rbf_aggregate = RBFPolicy(n_inputs=15, n_outputs=1, n_rbf=6,
                                  output_max=3000.0)
        agg_params = _random_valid_params(rbf_aggregate, rng)
        rbf_aggregate.set_params(agg_params)

        objs_ind = evaluate_with_policy(rbf_individual, mode="individual")
        objs_agg = evaluate_with_policy(rbf_aggregate, mode="aggregate")

        # Both should be valid
        assert len(objs_ind) == self.obj_set.n_objs
        assert len(objs_agg) == self.obj_set.n_objs
        # They should differ (different policies entirely)
        assert not np.allclose(objs_ind, objs_agg), (
            "Individual and aggregate objectives are identical — unexpected"
        )

    def test_feature_list_override(self):
        """Custom state_features list produces finite objectives."""
        from src.external_policy import evaluate_with_policy

        def moderate_policy(state):
            return np.array([500.0])

        minimal = ["combined_nyc_storage_frac", "montague_flow_lag2"]
        objs = evaluate_with_policy(
            moderate_policy, mode="aggregate", state_features=minimal
        )
        assert len(objs) == self.obj_set.n_objs
        assert all(np.isfinite(v) for v in objs)
