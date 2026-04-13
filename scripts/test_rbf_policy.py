"""
scripts/test_rbf_policy.py - End-to-end test of RBFPolicy.

Usage:
    PYWRDRB_SIM_START_DATE=2018-01-01 PYWRDRB_SIM_END_DATE=2022-09-30 \
        python scripts/test_rbf_policy.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policies import RBFPolicy
from src.external_policy import evaluate_with_policy

# ------------------------------------------------------------------
# 1. Construct policy
# ------------------------------------------------------------------
N_INPUTS  = 15   # state with predictions
N_OUTPUTS = 1    # aggregate mode
N_RBF     = 6
OUTPUT_MAX = 3000.0  # MGD, approx total NYC release max

policy = RBFPolicy(
    n_inputs=N_INPUTS,
    n_outputs=N_OUTPUTS,
    n_rbf=N_RBF,
    output_max=OUTPUT_MAX,
    output_min=0.0,
)

print(f"Policy: {policy.name}")
print(f"n_params: {policy.n_params}")
assert policy.n_params == N_RBF * (N_INPUTS + 1 + N_OUTPUTS), "Parameter count mismatch"

# ------------------------------------------------------------------
# 2. Set random parameters within bounds
# ------------------------------------------------------------------
lb, ub = policy.get_bounds()
rng = np.random.default_rng(42)
params = rng.uniform(lb, ub)
policy.set_params(params)

# ------------------------------------------------------------------
# 3. Test __call__ and verify output bounds
# ------------------------------------------------------------------
state = rng.uniform(0.0, 1.0, size=N_INPUTS)
action = policy(state)

print(f"Sample state:  {state[:5]}...")
print(f"Sample action: {action}")
assert action.shape == (N_OUTPUTS,), f"Wrong action shape: {action.shape}"
assert np.all(action >= 0.0),         f"Action below output_min: {action}"
assert np.all(action <= OUTPUT_MAX),  f"Action above output_max: {action}"
print("Bounds check passed.")

# ------------------------------------------------------------------
# 4. Full pipeline test
# ------------------------------------------------------------------
print("\nRunning evaluate_with_policy (5-year period)...")
objs = evaluate_with_policy(policy, mode="aggregate", include_predictions=True)
print(f"Objectives: {objs}")
assert len(objs) > 0, "No objectives returned"
assert all(np.isfinite(o) for o in objs), f"Non-finite objectives: {objs}"
print("End-to-end test passed.")
