"""Tests for the post-simulation `nyc_reliability_floor` Borg constraint.

Covers `make_post_sim_constraint_function` (violation magnitude/direction on
the natural 0-1 scale, Borg sign un-negation, name-resolved objective lookup,
the config-registry floor + env override, the penalty-sentinel guard) and the
MM Borg objective wrapper composition (`src.mmborg.make_borg_objective`):
DV-space + post-sim constraint lists, the DV-infeasible skip-simulation path,
and the failed-evaluation zero-violation convention.

The DV-space constraints themselves are covered in tests/test_constraints.py.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.formulations import (
    POST_SIM_CONSTRAINT_NAMES,
    RELIABILITY_FLOOR_OBJECTIVE,
    get_baseline_values,
    get_n_constrs,
    get_obj_directions,
    get_obj_names,
    get_var_names,
    make_post_sim_constraint_function,
    reliability_floor_objective_index,
)

import src.mmborg as mmborg


def _borg_vector(natural_reliability, items=None, fill=0.0):
    """Borg-oriented objective vector with the reliability column set.

    Maximize objectives are stored negated in Borg space, so the reliability
    column carries ``-natural`` when its direction is +1.
    """
    names = get_obj_names(items)
    directions = get_obj_directions(items)
    idx = reliability_floor_objective_index(names)
    objs = [fill] * len(names)
    objs[idx] = (-natural_reliability if directions[idx] == 1
                 else natural_reliability)
    return objs


###############################################################################
# Violation magnitude and direction (natural 0-1 scale)
###############################################################################

def test_below_floor_violation_magnitude():
    fn = make_post_sim_constraint_function()
    # Default floor 0.5: reliability 0.4 -> violation 0.1.
    cons = fn(_borg_vector(0.4))
    assert len(cons) == len(POST_SIM_CONSTRAINT_NAMES) == 1
    assert cons[0] == pytest.approx(0.1)


@pytest.mark.parametrize("reliability", [0.5, 0.75, 1.0])
def test_at_or_above_floor_is_feasible(reliability):
    fn = make_post_sim_constraint_function()
    assert fn(_borg_vector(reliability)) == [0.0]


def test_violation_scales_linearly_with_shortfall():
    fn = make_post_sim_constraint_function()
    assert fn(_borg_vector(0.0))[0] == pytest.approx(0.5)
    assert fn(_borg_vector(0.25))[0] == pytest.approx(0.25)


###############################################################################
# Floor configuration: config registry + env override (never CLI flags)
###############################################################################

def test_floor_reads_config_registry_at_factory_time(monkeypatch):
    import config
    monkeypatch.setattr(config, "NYC_RELIABILITY_FLOOR", 0.8)
    fn = make_post_sim_constraint_function()
    assert fn(_borg_vector(0.75))[0] == pytest.approx(0.05)
    assert fn(_borg_vector(0.85)) == [0.0]


def test_floor_env_override():
    # End-to-end: NYCOPT_NYC_RELIABILITY_FLOOR must reach the constraint via
    # config's import-time env parsing, so exercise it in a fresh interpreter.
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(PROJECT_DIR)!r})
        from src.formulations import (
            RELIABILITY_FLOOR_OBJECTIVE,
            get_obj_directions,
            get_obj_names,
            make_post_sim_constraint_function,
            reliability_floor_objective_index,
        )
        fn = make_post_sim_constraint_function()
        names = get_obj_names()
        idx = reliability_floor_objective_index(names)
        objs = [0.0] * len(names)
        objs[idx] = -0.6 if get_obj_directions()[idx] == 1 else 0.6
        print(fn(objs)[0])
    """)
    env = dict(os.environ, NYCOPT_NYC_RELIABILITY_FLOOR="0.7")
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=str(PROJECT_DIR),
        capture_output=True, text=True, check=True,
    )
    assert float(out.stdout.strip()) == pytest.approx(0.7 - 0.6)


###############################################################################
# Name-resolved objective lookup (never a hard-coded index)
###############################################################################

def test_objective_resolved_by_name_not_index():
    # Reordered set: reliability is NOT column 0. The decoy column 0 carries
    # an above-floor reliability-like value, so reading the wrong column
    # would report feasible.
    items = ["montague_flow_reliability_weekly", RELIABILITY_FLOOR_OBJECTIVE]
    fn = make_post_sim_constraint_function(items)
    objs = _borg_vector(0.3, items=items)
    decoy_idx = 0
    assert reliability_floor_objective_index(get_obj_names(items)) == 1
    objs[decoy_idx] = -0.9  # feasible-looking decoy in Borg space
    assert fn(objs)[0] == pytest.approx(0.2)


def test_missing_reliability_objective_fails_loudly():
    with pytest.raises(ValueError, match=RELIABILITY_FLOOR_OBJECTIVE):
        make_post_sim_constraint_function(["montague_flow_reliability_weekly"])


###############################################################################
# Failed-simulation penalty sentinels: zero violation, penalty objs exclude
###############################################################################

@pytest.mark.parametrize("penalty", [1e6, 1e10])
def test_penalty_sentinel_reports_zero_violation(penalty):
    fn = make_post_sim_constraint_function()
    n = len(get_obj_names())
    assert fn([penalty] * n) == [0.0]


###############################################################################
# MM Borg objective wrapper composition
###############################################################################

def _infeasible_dv():
    # Delivery-monotonicity violation (same construction as test_constraints).
    names = get_var_names("ffmp")
    dv = get_baseline_values("ffmp").copy()
    dv[names.index("nyc_drought_factor_L3")] = 0.60
    dv[names.index("nyc_drought_factor_L4")] = 0.95
    dv[names.index("nyc_drought_factor_L5")] = 0.90
    return dv


def _stub_eval_factory(objs, calls):
    def _factory(formulation_name):
        def _fn(dv):
            calls.append(np.asarray(dv))
            return list(objs)
        return _fn
    return _factory


def test_wrapper_returns_full_constraint_list(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mmborg, "make_objective_function",
        _stub_eval_factory(_borg_vector(0.4), calls),
    )
    objective = mmborg.make_borg_objective("ffmp")
    objs, cons = objective(list(get_baseline_values("ffmp")), 0)
    assert len(calls) == 1
    assert len(cons) == get_n_constrs() == 3
    assert cons[:2] == [0.0, 0.0]                 # baseline is DV-feasible
    assert cons[2] == pytest.approx(0.1)          # reliability 0.4 vs 0.5
    assert objs == _borg_vector(0.4)


def test_wrapper_feasible_when_above_floor(monkeypatch):
    monkeypatch.setattr(
        mmborg, "make_objective_function",
        _stub_eval_factory(_borg_vector(0.9), []),
    )
    objective = mmborg.make_borg_objective("ffmp")
    _, cons = objective(list(get_baseline_values("ffmp")), 0)
    assert cons == [0.0] * 3


def test_wrapper_dv_infeasible_skips_simulation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mmborg, "make_objective_function",
        _stub_eval_factory(_borg_vector(0.9), calls),
    )
    objective = mmborg.make_borg_objective("ffmp")
    objs, cons = objective(list(_infeasible_dv()), 0)
    assert calls == []                            # no simulation ran
    assert all(o == 1e10 for o in objs)
    assert cons[0] > 0.0                          # delivery monotonicity
    assert cons[2] == 0.0                         # post-sim slot: unmeasured
    assert len(cons) == 3


def test_wrapper_failed_eval_is_feasible_with_penalty(monkeypatch):
    # Layer 1: make_objective_function's internal catch returns its 1e6
    # penalty vector; the post-sim [0, 1] guard must report zero violation.
    monkeypatch.setattr(
        mmborg, "make_objective_function",
        _stub_eval_factory([1e6] * len(get_obj_names()), []),
    )
    objective = mmborg.make_borg_objective("ffmp")
    objs, cons = objective(list(get_baseline_values("ffmp")), 0)
    assert cons == [0.0] * 3
    assert all(o == 1e6 for o in objs)

    # Layer 2: an exception anywhere in the wrapper returns the 1e10 penalty
    # with zero violations (feasible-but-maximally-unattractive convention).
    def _raising_factory(formulation_name):
        def _fn(dv):
            raise RuntimeError("boom")
        return _fn
    monkeypatch.setattr(mmborg, "make_objective_function", _raising_factory)
    objective = mmborg.make_borg_objective("ffmp")
    objs, cons = objective(list(get_baseline_values("ffmp")), 0)
    assert objs == [1e10] * len(get_obj_names())
    assert cons == [0.0] * 3
