# Implementation Examples — Critical Code Snippets

**Concrete code for the core components. Copy-paste ready for implementation.**

---

## 1. ExternalPolicyParameter (pywrdrb/parameters/external_policy.py)

This is the centerpiece. It must be in pywrdrb/parameters/ (new file).

```python
"""External Policy Parameter for Pywr.

Delegates release decisions to an external policy function at each timestep.
Used for Architectures C, D, E (RBF, Tree, ANN).
"""

import numpy as np
from typing import Callable, List, Dict, Optional
from pywr.parameters import Parameter
import logging

logger = logging.getLogger(__name__)

# Global cache: cleared at start of each timestep by CacheClearParameter
_POLICY_CACHE: Dict[tuple, np.ndarray] = {}


class ExternalPolicyParameter(Parameter):
    """Pywr Parameter that delegates to an external policy function.
    
    At each timestep:
    1. Reads system state from model nodes
    2. Calls policy_fn(state) → action vector
    3. Returns action[output_index]
    4. Caches result to avoid redundant calls (multi-output policies)
    """
    
    def __init__(
        self,
        model,
        policy_fn: Callable,
        state_config: List[Dict],
        output_index: int,
        output_min: float = 0.0,
        output_max: Optional[float] = None,
        name: str = None,
        **kwargs
    ):
        """
        Args:
            model: Pywr Model instance
            policy_fn: Callable[[np.ndarray], np.ndarray]
                      Input: normalized state vector
                      Output: action vector (releases in MGD)
            state_config: List of dicts, each with keys:
                - 'node': Pywr node name (str)
                - 'attribute': Node attribute name (str, e.g. 'volume')
                - 'normalize_by': Divisor for normalization (float, optional)
            output_index: Which action element to return (int, 0-indexed)
            output_min: Lower clipping bound (float, default 0.0)
            output_max: Upper clipping bound (float, optional, default None)
            name: Parameter name for debugging (str)
        """
        super().__init__(model, name=name, **kwargs)
        self.policy_fn = policy_fn
        self.state_config = state_config
        self.output_index = output_index
        self.output_min = output_min
        self.output_max = output_max
    
    def _extract_state(self, timestep, scenario_index) -> np.ndarray:
        """Extract state vector from Pywr model nodes.
        
        Returns:
            np.ndarray of normalized state + temporal features
        """
        state = []
        
        # Extract state from configured nodes
        for cfg in self.state_config:
            node_name = cfg['node']
            attribute = cfg['attribute']
            normalize_by = cfg.get('normalize_by')
            
            # Get node
            if node_name not in self.model.nodes:
                raise KeyError(
                    f"Node '{node_name}' not found in model. "
                    f"Available nodes: {sorted(list(self.model.nodes.keys())[:10])}..."
                )
            
            node = self.model.nodes[node_name]
            
            # Get attribute (handle scenario indexing)
            try:
                raw_val = getattr(node, attribute)
            except AttributeError:
                raise AttributeError(
                    f"Node '{node_name}' has no attribute '{attribute}'. "
                    f"Available attributes: {[a for a in dir(node) if not a.startswith('_')][:10]}..."
                )
            
            # Convert to scalar
            if isinstance(raw_val, np.ndarray):
                # Scenario-indexed array
                try:
                    val = float(raw_val[scenario_index.global_id])
                except (IndexError, TypeError) as e:
                    raise IndexError(
                        f"Failed to index node.{attribute} with scenario_index.global_id={scenario_index.global_id}. "
                        f"Array shape: {raw_val.shape}. Error: {e}"
                    )
            else:
                val = float(raw_val)
            
            # Normalize
            if normalize_by is not None and normalize_by != 0:
                val /= normalize_by
            elif normalize_by == 0:
                logger.warning(f"normalize_by is 0 for node '{node_name}'; skipping normalization")
            
            state.append(val)
        
        # Add temporal features (day of year encoded as sin/cos)
        doy = timestep.datetime.timetuple().tm_yday
        state.append(np.sin(2.0 * np.pi * float(doy) / 366.0))
        state.append(np.cos(2.0 * np.pi * float(doy) / 366.0))
        
        return np.array(state, dtype=np.float64)
    
    def value(self, timestep, scenario_index):
        """Return the policy output value.
        
        Caches policy computation to handle multi-output policies efficiently.
        """
        global _POLICY_CACHE
        
        cache_key = (timestep.index, scenario_index.global_id)
        
        if cache_key not in _POLICY_CACHE:
            # First call for this timestep: extract state and call policy
            try:
                state = self._extract_state(timestep, scenario_index)
            except Exception as e:
                logger.error(f"State extraction failed at timestep {timestep.index}: {e}")
                raise
            
            try:
                action = self.policy_fn(state)
            except Exception as e:
                logger.error(
                    f"Policy function failed at timestep {timestep.index} "
                    f"(scenario {scenario_index.global_id}): {e}"
                )
                raise
            
            # Ensure action is numpy array
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float64)
            
            _POLICY_CACHE[cache_key] = action
        
        # Retrieve from cache
        try:
            action = _POLICY_CACHE[cache_key]
        except KeyError:
            logger.error(f"Cache miss for key {cache_key}. This should not happen.")
            raise
        
        # Extract requested output
        if self.output_index >= len(action):
            raise IndexError(
                f"output_index {self.output_index} is out of range for action vector "
                f"of length {len(action)}"
            )
        
        val = float(action[self.output_index])
        
        # Clip to bounds
        if self.output_max is not None:
            val = np.clip(val, self.output_min, self.output_max)
        else:
            val = max(val, self.output_min)
        
        return val


class CacheClearParameter(Parameter):
    """Dummy parameter that clears the policy cache at start of each timestep.
    
    Pywr evaluates parameters in definition order. By placing this parameter
    first, we ensure the cache is cleared before any ExternalPolicyParameter
    is evaluated.
    
    IMPORTANT: Must be added to the model BEFORE any ExternalPolicyParameter.
    """
    
    def __init__(self, model, name: str = "__cache_clear__", **kwargs):
        super().__init__(model, name=name, **kwargs)
    
    def value(self, timestep, scenario_index):
        """Clear cache and return dummy value."""
        global _POLICY_CACHE
        _POLICY_CACHE.clear()
        return 0.0


def clear_policy_cache():
    """Utility to manually clear the policy cache (for debugging)."""
    global _POLICY_CACHE
    _POLICY_CACHE.clear()
```

---

## 2. Model Replacement Utilities (src/model_builder_utils.py)

```python
"""Utilities for post-load model modification.

Replaces FFMP release parameters with external policy parameters.
Also handles fixed MRF target replacement.
"""

import logging
from typing import Callable, List, Dict, Optional

logger = logging.getLogger(__name__)


def replace_release_parameters(
    model,
    policy_fn: Callable,
    state_config: List[Dict],
    output_indices: Optional[Dict[str, int]] = None,
    reservoirs: Optional[List[str]] = None
) -> None:
    """Replace release link max_flow parameters with ExternalPolicyParameter.
    
    Args:
        model: Pywr Model instance (already loaded from JSON)
        policy_fn: Policy function: state → action vector
        state_config: List of dicts specifying how to extract state
        output_indices: Dict mapping reservoir name -> output index in action vector.
                       Default: {'cannonsville': 0, 'pepacton': 1, 'neversink': 2}
        reservoirs: List of reservoir names to control.
                   Default: ['cannonsville', 'pepacton', 'neversink']
    
    Raises:
        KeyError: If required parameter or node is not in model
    """
    from pywrdrb.parameters.external_policy import ExternalPolicyParameter, CacheClearParameter
    
    if output_indices is None:
        output_indices = {
            "cannonsville": 0,
            "pepacton": 1,
            "neversink": 2,
        }
    
    if reservoirs is None:
        reservoirs = ["cannonsville", "pepacton", "neversink"]
    
    # Mapping of reservoir name to parameter key in model.parameters
    param_keys = {
        "cannonsville": "rel_max_release_cannonsville",
        "pepacton": "rel_max_release_pepacton",
        "neversink": "rel_max_release_neversink",
    }
    
    # Insert cache-clear parameter first (must run before any ExternalPolicyParameter)
    cache_clear_param = CacheClearParameter(model, name="__cache_clear__")
    
    # Replace release parameters
    for res_name in reservoirs:
        if res_name not in param_keys:
            logger.warning(f"Unknown reservoir '{res_name}'. Skipping.")
            continue
        
        param_key = param_keys[res_name]
        
        if param_key not in model.parameters:
            raise KeyError(
                f"Release parameter '{param_key}' not found for reservoir '{res_name}'. "
                f"Available parameters: {sorted(list(model.parameters.keys())[:10])}..."
            )
        
        output_idx = output_indices.get(res_name, len(reservoirs) - 1)
        
        # Create ExternalPolicyParameter
        policy_param = ExternalPolicyParameter(
            model=model,
            policy_fn=policy_fn,
            state_config=state_config,
            output_index=output_idx,
            output_min=0.0,
            output_max=None,  # LP solver enforces capacity
            name=f"policy_release_{res_name}",
        )
        
        # Replace in model
        old_param = model.parameters[param_key]
        model.parameters[param_key] = policy_param
        
        logger.info(
            f"Replaced '{param_key}' (was {type(old_param).__name__}) "
            f"with ExternalPolicyParameter (output_index={output_idx})"
        )


def set_fixed_mrf_targets(
    model,
    montague_mgd: float = 1131.05,
    trenton_mgd: float = 1938.95
) -> None:
    """Replace dynamic MRF targets with fixed constants.
    
    For fair policy comparison, all architectures evaluated against same targets.
    
    Args:
        model: Pywr Model instance
        montague_mgd: Fixed Montague flow target (MGD)
        trenton_mgd: Fixed Trenton flow target (MGD)
    
    Raises:
        KeyError: If MRF target parameters not found
    """
    from pywrdrb.parameters.fixed_target import FixedMRFParameter
    
    # Mapping of location to parameter key
    param_keys = {
        "delMontague": "mrf_target_delMontague",
        "delTrenton": "mrf_target_delTrenton",
    }
    
    targets = {
        "delMontague": montague_mgd,
        "delTrenton": trenton_mgd,
    }
    
    for loc, param_key in param_keys.items():
        if param_key not in model.parameters:
            logger.warning(
                f"MRF target parameter '{param_key}' not found. "
                f"Available: {sorted(list(model.parameters.keys())[:10])}. "
                f"Skipping MRF target replacement."
            )
            continue
        
        target_val = targets[loc]
        
        # Create fixed parameter
        fixed_param = FixedMRFParameter(
            model=model,
            value=target_val,
            name=f"fixed_mrf_{loc}",
        )
        
        # Replace
        old_param = model.parameters[param_key]
        model.parameters[param_key] = fixed_param
        
        logger.info(
            f"Replaced '{param_key}' (was {type(old_param).__name__}) "
            f"with FixedMRFParameter (value={target_val} MGD)"
        )


def inspect_model_parameters(model, pattern: str = None) -> Dict[str, str]:
    """Debug utility: inspect model parameters.
    
    Args:
        model: Pywr Model instance
        pattern: Optional substring to filter parameter names (case-insensitive)
    
    Returns:
        Dict mapping parameter name -> class name
    """
    result = {}
    for key in sorted(model.parameters.keys()):
        if pattern is None or pattern.lower() in key.lower():
            param = model.parameters[key]
            result[key] = type(param).__name__
    
    print(f"\nModel has {len(model.parameters)} parameters:")
    print(f"Matching pattern '{pattern}':" if pattern else "All:")
    for key, cls in sorted(result.items()):
        print(f"  {key:50} → {cls}")
    
    return result


def inspect_model_nodes(model, pattern: str = None) -> Dict[str, str]:
    """Debug utility: inspect model nodes.
    
    Args:
        model: Pywr Model instance
        pattern: Optional substring to filter node names
    
    Returns:
        Dict mapping node name -> class name
    """
    result = {}
    for key in sorted(model.nodes.keys()):
        if pattern is None or pattern.lower() in key.lower():
            node = model.nodes[key]
            result[key] = type(node).__name__
    
    print(f"\nModel has {len(model.nodes)} nodes:")
    print(f"Matching pattern '{pattern}':" if pattern else "All:")
    for key, cls in sorted(result.items()):
        print(f"  {key:50} → {cls}")
    
    return result
```

---

## 3. Fixed MRF Parameter (pywrdrb/parameters/fixed_target.py)

```python
"""Fixed MRF Target Parameter for Pywr.

Simple parameter that always returns a constant value.
Used to replace monthlyprofile MRF targets with fixed targets.
"""

from pywr.parameters import Parameter


class FixedMRFParameter(Parameter):
    """Parameter that always returns a fixed constant value.
    
    Used for fixed-target evaluation mode where all policies are measured
    against the same downstream flow targets (Montague, Trenton) regardless
    of drought level.
    """
    
    def __init__(self, model, value: float, name: str = None, **kwargs):
        """
        Args:
            model: Pywr Model instance
            value: Constant value to return at every timestep
            name: Parameter name for debugging
        """
        super().__init__(model, name=name, **kwargs)
        self.value_const = float(value)
    
    def value(self, timestep, scenario_index):
        """Return the fixed constant value."""
        return self.value_const
```

---

## 4. PolicyBase ABC (optimization/policies/base.py)

```python
"""Base class for all policy architectures."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple, Optional


class PolicyBase(ABC):
    """Abstract base class for optimization-compatible policies.
    
    All policies must implement this interface to be used with Borg MOEA
    and ExternalPolicyParameter.
    """
    
    @abstractmethod
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy: state -> action.
        
        Args:
            state: np.ndarray of shape (n_features,)
                  Normalized state vector. Default contains:
                  - 3 reservoir storage fractions (0-1)
                  - sin(2π DOY / 366), cos(2π DOY / 366) (temporal features)
        
        Returns:
            np.ndarray of shape (n_outputs,), typically (3,) for 3 reservoirs
            Each element is a release in MGD (must be >= 0)
        """
        pass
    
    @abstractmethod
    def set_params(self, flat_vector: np.ndarray) -> None:
        """Set policy parameters from a flat decision variable vector.
        
        Called by Borg at each evaluation with new DV values.
        Must update internal policy state to reflect the new parameters.
        
        Args:
            flat_vector: 1D np.ndarray of decision variable values,
                        length must equal self.n_params
        
        Raises:
            ValueError: If length doesn't match self.n_params
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return decision variable bounds for Borg.
        
        Returns:
            (lower_bounds, upper_bounds): Tuple of two 1D arrays,
                                         each of length self.n_params
        """
        pass
    
    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of decision variables for this policy."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Architecture identifier (e.g., 'RBF', 'Tree', 'ANN')."""
        pass
    
    @property
    def state_config(self) -> List[Dict]:
        """Configuration for extracting state from Pywr nodes.
        
        Defines which nodes and attributes to read for the state vector.
        Default implementation: read aggregate NYC storage.
        
        Subclasses may override for different state features.
        
        Returns:
            List of dicts with keys: 'node', 'attribute', 'normalize_by'
        """
        return [
            {
                'node': 'cannonsville',
                'attribute': 'volume',
                'normalize_by': 95706.0,  # Capacity in MG
            },
            {
                'node': 'pepacton',
                'attribute': 'volume',
                'normalize_by': 140190.0,
            },
            {
                'node': 'neversink',
                'attribute': 'volume',
                'normalize_by': 34941.0,
            },
        ]
```

---

## 5. Example: RBFPolicy (optimization/policies/rbf_policy.py)

```python
"""RBF (Radial Basis Function) Policy for NYC reservoir management."""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import Rbf
from .base import PolicyBase


class RBFPolicy(PolicyBase):
    """Policy based on radial basis function interpolation.
    
    Maintains N RBF functions (one per output), each learned from training data.
    Interpolates smoothly between training points in state space.
    """
    
    def __init__(
        self,
        n_rbf_centers: int = 20,
        output_range: Tuple[float, float] = (0.0, 500.0),
        n_outputs: int = 3,
        name: str = "RBF",
    ):
        """
        Args:
            n_rbf_centers: Number of RBF centers (basis functions)
            output_range: (min, max) release values in MGD
            n_outputs: Number of policy outputs (reservoirs, typically 3)
            name: Architecture name
        """
        self.n_rbf_centers = n_rbf_centers
        self.output_min, self.output_max = output_range
        self.n_outputs = n_outputs
        self.arch_name = name
        
        # RBF functions (one per output)
        self.rbf_functions = [None] * n_outputs
        
        # Decision variables: weights for each RBF center, per output
        # Total: n_rbf_centers * n_outputs DVs
        self._dv_vector = None
        self._n_params = n_rbf_centers * n_outputs
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate RBF policy."""
        if self._dv_vector is None:
            raise RuntimeError("Policy not initialized. Call set_params() first.")
        
        action = np.zeros(self.n_outputs, dtype=np.float64)
        
        # Evaluate each RBF
        for i in range(self.n_outputs):
            if self.rbf_functions[i] is None:
                # RBF not yet created; return default
                action[i] = self.output_min
            else:
                # Evaluate RBF at state
                try:
                    val = self.rbf_functions[i](*state)  # Unpack state dimensions
                except Exception:
                    # RBF evaluation failed; return default
                    action[i] = self.output_min
                
                # Clip to bounds
                action[i] = np.clip(val, self.output_min, self.output_max)
        
        return action
    
    def set_params(self, flat_vector: np.ndarray) -> None:
        """Set RBF weights from flat DV vector.
        
        This is a simplified implementation. In reality, you would:
        1. Use some pre-computed RBF center locations (learned from offline training)
        2. Set the weights at each center to the DV values
        3. Rebuild the RBF interpolators
        
        For this stub, we just store the vector and assume RBFs exist.
        """
        if len(flat_vector) != self._n_params:
            raise ValueError(
                f"Expected {self._n_params} DVs, got {len(flat_vector)}"
            )
        
        self._dv_vector = np.array(flat_vector, dtype=np.float64)
        
        # TODO: Reshape DVs and rebuild RBF functions
        # weights = self._dv_vector.reshape(self.n_outputs, self.n_rbf_centers)
        # For each output i:
        #   self.rbf_functions[i] = Rbf(centers[0], centers[1], centers[2],
        #                                 weights[i], function='thin_plate')
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return decision variable bounds."""
        lower = np.full(self._n_params, 0.0)
        upper = np.full(self._n_params, self.output_max)
        return lower, upper
    
    @property
    def n_params(self) -> int:
        """Number of decision variables."""
        return self._n_params
    
    @property
    def name(self) -> str:
        """Architecture name."""
        return self.arch_name
```

---

## 6. Integration in simulation.py

Add this dispatch logic to `src/simulation.py`:

```python
# At the top of the evaluate() function
def evaluate(dv_vector, formulation_name="ffmp", objective_set=None):
    """Full evaluation pipeline: DVs -> simulation -> objectives.
    
    Dispatches based on formulation_name to the appropriate evaluation path.
    """
    global _EVAL_COUNT, _EVAL_START_TIME
    
    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = time.time()
    
    _EVAL_COUNT += 1
    t0 = time.time()
    
    # Dispatch based on formulation type
    if formulation_name in ("ffmp", "ffmp_vr_3", "ffmp_vr_6", "ffmp_vr_10", "ffmp_vr_15"):
        # Path 1: FFMP variants
        objectives = evaluate_ffmp(dv_vector, formulation_name, objective_set)
    
    elif formulation_name in ("rbf", "tree", "ann"):
        # Path 2: External policy variants
        from src.simulation_policy import evaluate_with_policy
        from optimization.policies import RBFPolicy, SoftTreePolicy, ANNPolicy
        
        # Instantiate the policy
        if formulation_name == "rbf":
            policy = RBFPolicy()
        elif formulation_name == "tree":
            policy = SoftTreePolicy()
        elif formulation_name == "ann":
            policy = ANNPolicy()
        else:
            raise ValueError(f"Unknown policy: {formulation_name}")
        
        # Set parameters from DVs
        policy.set_params(np.array(dv_vector))
        
        # Evaluate
        objectives = evaluate_with_policy(policy, objective_set=objective_set)
    
    else:
        raise ValueError(
            f"Unknown formulation: {formulation_name}. "
            f"Available: ffmp, ffmp_vr_3, ffmp_vr_6, ffmp_vr_10, "
            f"rbf, tree, ann"
        )
    
    # Log progress
    t_elapsed = time.time() - t0
    if _EVAL_COUNT % _EVAL_LOG_INTERVAL == 0:
        avg_time = (time.time() - _EVAL_START_TIME) / _EVAL_COUNT
        print(f"[Eval {_EVAL_COUNT:6d}] {formulation_name:12s} "
              f"time={t_elapsed:6.2f}s avg={avg_time:6.2f}s")
    
    return objectives
```

---

## 7. Testing: Minimal Unit Test

```python
# tests/test_external_policy.py
import numpy as np
from pywrdrb.parameters.external_policy import ExternalPolicyParameter

def test_external_policy_parameter_constant_policy():
    """Test ExternalPolicyParameter with a simple constant policy."""
    from unittest.mock import MagicMock
    
    # Create mock model and objects
    model = MagicMock()
    
    # Mock nodes
    mock_node_can = MagicMock()
    mock_node_can.volume = np.array([50000.0])  # 50,000 MG
    
    model.nodes = {'cannonsville': mock_node_can}
    
    # Simple constant policy
    def const_policy(state):
        return np.array([100.0, 80.0, 60.0])  # Release 100, 80, 60 MGD
    
    # State config
    state_config = [
        {'node': 'cannonsville', 'attribute': 'volume', 'normalize_by': 95706.0},
    ]
    
    # Create parameter
    param = ExternalPolicyParameter(
        model=model,
        policy_fn=const_policy,
        state_config=state_config,
        output_index=0,  # Return first output
        name="test_param",
    )
    
    # Mock timestep and scenario_index
    mock_timestep = MagicMock()
    mock_timestep.index = 0
    mock_timestep.datetime.timetuple.return_value.tm_yday = 100
    
    mock_scenario = MagicMock()
    mock_scenario.global_id = 0
    
    # Evaluate
    val = param.value(mock_timestep, mock_scenario)
    
    # Check
    assert val == 100.0, f"Expected 100.0, got {val}"
    print("✓ test_external_policy_parameter_constant_policy PASSED")


if __name__ == "__main__":
    test_external_policy_parameter_constant_policy()
```

---

## Summary

These code snippets provide the foundation for all new components. Key points:

1. **ExternalPolicyParameter** is the core: it reads state and calls the policy function
2. **CacheClearParameter** must be inserted first to clear cache each timestep
3. **Model replacement utilities** find and replace parameters post-load
4. **PolicyBase** defines the interface; RBF/Tree/ANN are subclasses
5. **dispatch logic** in `evaluate()` routes to the correct evaluation path

All pieces fit together cleanly with NO modifications to existing pywrdrb source code.
